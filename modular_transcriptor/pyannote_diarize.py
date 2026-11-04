from __future__ import annotations

import gc
import json
import os
from pathlib import Path

from modular_transcriptor import cache
from modular_transcriptor._logging import get_logger
from modular_transcriptor.types import PhaseCb

log = get_logger("pyannote_diarize")

# v2 diarization.json: carries per-cluster speaker centroid embeddings beside the
# turn list, so later stages can give diarized voices stable cross-stream
# identities. Bumping this invalidates any on-disk v1 (bare-list) artifact via the
# cache `extra` key, forcing a regenerate rather than a silent stale reuse.
DIAR_SCHEMA_VERSION = 2
EMBEDDING_MODEL = "pyannote-community-1"
_PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"
# Rows pyannote zero-pads for under-populated clusters have ~0 L2 norm; drop them.
_MIN_EMBEDDING_NORM = 1e-8

# Identity-based gap-fill: an ASR word span with no overlapping diarization turn
# ("gap") gets its own voice embedding, compared to the per-cluster centroids.
# Accept the nearest cluster only if the cosine clears GAP_MATCH_THRESH AND the
# coalesced span is at least MIN_GAP_EMBED_S long (too-short crops embed poorly).
GAP_MATCH_THRESH = 0.80
MIN_GAP_EMBED_S = 1.0
# Consecutive uncovered words within this many seconds are one contiguous speech
# span; a wider inter-word gap is silence and breaks the run.
_GAP_COALESCE_S = 0.5


def _flatten_word_spans(words_path: Path) -> list[tuple[float, float]]:
    """Flatten every parakeet.json word to a sorted (start, end) span."""
    try:
        data = json.loads(words_path.read_text())
    except (OSError, ValueError):
        return []
    spans: list[tuple[float, float]] = []
    for seg in data.get("segments") or []:
        for w in seg.get("words") or []:
            try:
                s = float(w["start"])
                e = float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if e > s:
                spans.append((s, e))
    spans.sort()
    return spans


def _uncovered_gap_spans(
    word_spans: list[tuple[float, float]],
    turns: list[dict],
) -> list[tuple[float, float]]:
    """Words that overlap no diarization turn, coalesced into contiguous runs."""
    intervals = [(t["start"], t["end"]) for t in turns]
    uncovered = [
        (s, e)
        for (s, e) in word_spans
        if not any(s < te and e > ts for ts, te in intervals)
    ]
    coalesced: list[tuple[float, float]] = []
    for s, e in uncovered:  # word_spans is pre-sorted, so uncovered is too
        if coalesced and s - coalesced[-1][1] <= _GAP_COALESCE_S:
            coalesced[-1] = (coalesced[-1][0], max(coalesced[-1][1], e))
        else:
            coalesced.append((s, e))
    return coalesced


def _embed_span(pipeline, waveform, sr: int, start: float, end: float):
    """Embed the amplified-waveform crop [start, end] with the SAME embedding
    component that produced the Task-1 cluster centroids, so the cosine is
    meaningful. Kept tiny and monkeypatchable: tests swap in a fake.

    Assumption: pyannote's SpeakerDiarization pipeline exposes its embedder at
    `pipeline._embedding` (a PretrainedSpeakerEmbedding), callable on a
    (batch, channel, samples) tensor and returning a (batch, dim) embedding.
    """
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]

    n = int(waveform.shape[-1])
    i0 = max(0, int(round(start * sr)))
    i1 = min(n, int(round(end * sr)))
    crop = waveform[:, i0:i1]
    if crop.dim() == 2:  # (channel, samples) -> (batch, channel, samples)
        crop = crop.unsqueeze(0)
    with torch.no_grad():
        vec = pipeline._embedding(crop)
    arr = vec.detach().cpu().numpy() if hasattr(vec, "detach") else np.asarray(vec)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _best_centroid_match(np, vec, centroids: dict):
    """Return (label, cosine) of the nearest centroid, or (None, -1.0)."""
    nv = float(np.linalg.norm(vec))
    if nv < _MIN_EMBEDDING_NORM:
        return None, -1.0
    best_label, best_cos = None, -1.0
    for label, cvec in centroids.items():
        nc = float(np.linalg.norm(cvec))
        if nc < _MIN_EMBEDDING_NORM:
            continue
        cos = float(np.dot(vec, cvec) / (nv * nc))
        if cos > best_cos:
            best_cos, best_label = cos, label
    return best_label, best_cos


def _gapfill_turns(
    np,
    pipeline,
    waveform,
    sr: int,
    words_path: Path | None,
    turns: list[dict],
    embeddings: dict[str, list[float]],
) -> list[dict]:
    """Best-effort: assign ASR gaps to the nearest known voice centroid."""
    if words_path is None or not embeddings:
        return []
    wp = Path(words_path)
    if not wp.exists():
        return []
    word_spans = _flatten_word_spans(wp)
    if not word_spans:
        return []
    centroids = {lbl: np.asarray(v, dtype=np.float64) for lbl, v in embeddings.items()}
    gap_turns: list[dict] = []
    for s, e in _uncovered_gap_spans(word_spans, turns):
        if e - s < MIN_GAP_EMBED_S:
            continue
        vec = _embed_span(pipeline, waveform, sr, s, e)
        label, cos = _best_centroid_match(np, vec, centroids)
        if label is not None and cos >= GAP_MATCH_THRESH:
            gap_turns.append(
                {"start": s, "end": e, "speaker": label, "source": "gapfill"}
            )
    return gap_turns


def _empty_v2() -> dict:
    return {
        "schema": DIAR_SCHEMA_VERSION,
        "turns": [],
        "embeddings": {},
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": 0,
    }


def run(
    audio_path: Path,
    out_path: Path,
    *,
    device: str = "cuda",
    phase_cb: PhaseCb | None = None,
    words_path: Path | None = None,
) -> Path:
    # words_path (parakeet.json) drives identity-based gap-fill below.
    extra = {"diar_schema": DIAR_SCHEMA_VERSION}
    if cache.is_fresh(out_path, audio_path, extra):
        log.info("diarize_skip_fresh", out=str(out_path))
        return out_path

    def emit(label, pct=None):
        if phase_cb:
            phase_cb(label, pct)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required for pyannote diarization")

    payload = _empty_v2()
    try:
        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        import torchaudio  # type: ignore[import-not-found]
        from pyannote.audio import Pipeline  # type: ignore[import-not-found]

        emit("loading diarization model", 5.0)
        pipeline = Pipeline.from_pretrained(_PYANNOTE_MODEL_ID, token=token)
        pipeline.to(torch.device(device))
        embedding_dim = int(pipeline._embedding.dimension)

        emit("loading audio", 15.0)
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        emit("diarizing", 20.0)
        out = pipeline.apply({"waveform": waveform, "sample_rate": sr})

        annotation = out.speaker_diarization
        labels = list(annotation.labels())
        raw_emb = out.speaker_embeddings

        embeddings: dict[str, list[float]] = {}
        for i, label in enumerate(labels):
            if raw_emb is None or i >= len(raw_emb):
                continue
            vec = np.asarray(raw_emb[i], dtype=np.float64)
            if float(np.linalg.norm(vec)) < _MIN_EMBEDDING_NORM:
                continue  # zero-padded under-cluster row
            embeddings[str(label)] = [float(x) for x in vec]

        turns = [
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "speaker": str(label),
            }
            for seg, _track, label in annotation.itertracks(yield_label=True)
        ]

        # Best-effort gap-fill: recover ASR speech that diarization missed by
        # matching its voice to a known centroid. Isolated so a gap-fill failure
        # degrades to "no extra turns", never discarding the diarization itself.
        try:
            emit("resolving gaps", 90.0)
            turns.extend(
                _gapfill_turns(
                    np, pipeline, waveform, sr, words_path, turns, embeddings
                )
            )
        except Exception as ge:  # noqa: BLE001 — gap-fill is non-essential
            log.warning("gapfill_soft_skip", err=str(ge))

        payload = {
            "schema": DIAR_SCHEMA_VERSION,
            "turns": turns,
            "embeddings": embeddings,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": embedding_dim,
        }

        # CUDA/GC cleanup is best-effort: a failure here must NOT fall through
        # to the outer soft-degrade and discard the good payload above (that
        # would force an expensive full re-transcribe downstream).
        try:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cleanup error must not lose payload
            pass
    except (
        Exception
    ) as e:  # noqa: BLE001 — soft-degrade: no speakers, transcript still made
        log.warning("diarize_soft_degrade", err=str(e))
        payload = _empty_v2()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    tmp.write_text(json.dumps(payload))
    tmp.replace(out_path)
    cache.stamp(out_path, audio_path, extra)
    return out_path
