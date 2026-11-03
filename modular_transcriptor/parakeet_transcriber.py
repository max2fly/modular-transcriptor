from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

from rapidfuzz import fuzz

from modular_transcriptor import cache
from modular_transcriptor._logging import get_logger
from modular_transcriptor.types import PhaseCb

log = get_logger("parakeet_transcriber")

# Speech-aware chunking. The model's punctuation+capitalization head was trained
# on sequences <=40s and its accuracy plateaus/collapses beyond that (see
# "Longer is (Not Necessarily) Stronger", arXiv:2409.05601). So we aim for ~40s
# chunks but ALWAYS cut in a pause, never mid-utterance/mid-verse: around the
# target we cut at the quietest frame if it is a genuine pause; otherwise (e.g.
# continuous singing) we extend forward to the next real pause. A chunk only
# ever exceeds the target when there's no pause to cut on, and never exceeds the
# 5-minute safety cap. Normal speech pauses often, so it still chunks ~40s and
# keeps punctuation healthy; long no-pause runs stay whole (and are split into
# captions downstream in finalize).
TARGET_CHUNK_S = 40  # preferred chunk length (== PnC training max)
CHUNK_HARD_MAX_S = 300  # 5-minute safety net: force a cut if no pause is found
MIN_CHUNK_S = 15  # never cut a chunk shorter than this
CHUNK_SEARCH_S = 12  # search +/- this around the target for a pause
CHUNK_OVERLAP_S = 0  # emit regions are contiguous (cut at the pause midpoint)
_SILENCE_FRAME_S = 0.03  # 30ms RMS frames for the silence envelope
# A frame counts as a "pause" if its RMS is at/below this fraction of the LOCAL
# speech level (75th percentile of the ~next 2*target seconds of frame energy).
# Local + percentile, not global or median: a global threshold misses pauses in
# loud passages, and a median-based one fails in silence-dominated passages
# (median == silence, so a fraction of it lands below the gaps).
_PAUSE_FRACTION = 0.35
# A cut only lands in a SUSTAINED pause (>= this long) — a brief intra-word
# energy dip is shorter and must not be mistaken for a gap (that split
# "accident" -> "...an ice" at a boundary). We cut at the pause's midpoint.
_MIN_PAUSE_S = 0.12
MIN_DEDUP_OVERLAP_RATIO = 0.30
# A time-overlapping segment is a true duplicate (chunk-overlap re-transcription
# or a boundary-truncated fragment) only if its text is contained in the other's.
# Sequential sentences that merely share a short bridge phrase score below this
# and are both kept — otherwise real speech gets deleted (see tests).
DEDUP_TEXT_CONTAINMENT = 90
# Above this time-overlap ratio, two segments cover essentially the SAME audio
# (a chunk-seam re-transcription), so they are duplicates even when their text
# diverges — a boundary-truncated chunk garbles the shared utterance's edge
# (e.g. "...build your own Shamdo" vs a truncated "...build your own Sheptur").
# We then keep the longer (more complete) one. Sequential distinct sentences
# overlap far less than this, so they are unaffected.
HIGH_OVERLAP_RATIO = 0.5

# Hole-repair (second pass). A span with speech-level energy but NO decoded word
# is usually a cold-start drop (a chunk dropped its own opening). We re-decode it
# as a window centered on the hole (speech now mid-clip / warm) and merge back
# any recovered words. Non-speech holes (music/laughter) recover nothing — the
# model self-gates — so no extra gate is needed.
REPAIR_MIN_HOLE_S = 2.5    # ignore gaps shorter than this
REPAIR_CTX_S = 12.0        # context each side of the hole in the re-decode window
REPAIR_RMS_SPEECH = 0.05   # 0.25s-frame RMS above this counts as speech-ish


def speech_aware_bounds(audio, sr: int) -> list[tuple[int, int]]:
    """Sample-index (start, end) chunk pairs cut at PAUSES, not fixed offsets.

    Builds a coarse RMS envelope and, around each target boundary
    (``start + TARGET_CHUNK_S``), cuts at the MIDPOINT of the nearest SUSTAINED
    pause (a run of >= ``_MIN_PAUSE_S`` of frames at/below the per-clip
    local ``_PAUSE_FRACTION`` of nearby energy) — so a cut never splits a word on a brief intra-word
    dip. If no sustained pause is near the target (e.g. continuous singing), it
    scans FORWARD for the next one rather than cutting mid-verse — bounded by
    the 5-minute ``CHUNK_HARD_MAX_S`` safety cap. Because the cut sits in the
    MIDDLE of a real pause, chunks are contiguous (``CHUNK_OVERLAP_S`` = 0): an
    overlap would re-enter the previous utterance, re-transcribing (and often
    mis-hearing) its tail and starting the next chunk mid-phrase — which both
    duplicated words and degraded that chunk's punctuation.
    `numpy` is imported by the caller (`run`); `audio` is a 1-D float32 array.
    """
    import numpy as np  # type: ignore[import-not-found]

    n = int(len(audio))
    if n == 0:
        return []
    frame = max(1, int(_SILENCE_FRAME_S * sr))
    nf = n // frame
    if nf < 2:
        return [(0, n)]
    rms = np.sqrt(
        (audio[: nf * frame].reshape(nf, frame).astype(np.float64) ** 2).mean(axis=1)
        + 1e-12
    )
    min_pause = max(1, int(_MIN_PAUSE_S / _SILENCE_FRAME_S))  # frames of real gap

    def _runs(f_lo: int, f_hi: int, thr: float):
        """Yield the mid-frame of each sustained run (>= min_pause) of frames at
        or below ``thr`` in [f_lo, f_hi), so a cut lands in the MIDDLE of a
        genuine pause (never a one-frame intra-word dip that splits a word)."""
        i = f_lo
        while i < f_hi:
            if rms[i] <= thr:
                j = i
                while j < f_hi and rms[j] <= thr:
                    j += 1
                if j - i >= min_pause:
                    yield (i + j) // 2
                i = j
            else:
                i += 1

    target = int(TARGET_CHUNK_S * sr)
    mn = int(MIN_CHUNK_S * sr)
    search = int(CHUNK_SEARCH_S * sr)
    hard = int(CHUNK_HARD_MAX_S * sr)
    overlap = int(CHUNK_OVERLAP_S * sr)
    ref_frames = (2 * target) // frame  # local window for the adaptive threshold

    bounds: list[tuple[int, int]] = []
    start = 0
    while start < n:
        if n - start <= target + search:
            bounds.append((start, n))
            break
        # LOCAL pause threshold: a fraction of the SPEECH level just ahead,
        # estimated as a high percentile of local frame energy. Keyed to the
        # local speech level (not the whole clip, and not the median): a global
        # threshold misses pauses in loud passages, and a median-based one fails
        # in silence-dominated passages (the median IS the silence, so a
        # fraction of it falls below the real gaps). The high percentile tracks
        # how loud the speech is, so gaps reliably fall below the threshold.
        sf = start // frame
        local = float(np.percentile(rms[sf : min(nf, sf + ref_frames)], 75))
        thr = max(1e-6, _PAUSE_FRACTION * local)
        tgt_f = (start + target) // frame
        lo_f = max((start + mn) // frame, (start + target - search) // frame)
        hi_f = min(nf, (start + target + search) // frame)
        cands = list(_runs(lo_f, hi_f, thr))
        if cands:
            cut_frame = min(cands, key=lambda m: abs(m - tgt_f))
        else:
            # no pause near target (continuous singing): extend forward to the
            # next pause, up to the 5-min safety cap.
            scan_hi = min(nf, (start + hard) // frame)
            cut_frame = next(_runs(hi_f, scan_hi, thr), scan_hi)
        cut = min(max(cut_frame * frame, start + mn), n)
        bounds.append((start, cut))
        if cut >= n:
            break
        start = max(cut - overlap, start + 1)
    return bounds


def parakeet_cache_extra(
    boost_file: Path | None,
    boost_alpha: float,
    context_score: float = 0.3,
    depth_scaling: float = 4.0,
) -> dict:
    """Cache-key `extra` for the parakeet pass.

    Folds the chunking geometry and the boost parameters into the freshness
    fingerprint so that changing the chunk size, the boost word list, the boost
    strength, or the gentleness knobs invalidates a stale `parakeet.json`.
    `boost_sha` is the sha256 hex of the boost file's bytes (None when no boost
    file)."""
    boost_sha: str | None = None
    if boost_file is not None:
        boost_sha = hashlib.sha256(Path(boost_file).read_bytes()).hexdigest()
    else:
        # All boost params are inert without a boost file, so normalize them out
        # of the key: an un-boosted parakeet.json must not re-invalidate when a
        # boost default changes.
        boost_alpha = 0.0
        context_score = 0.0
        depth_scaling = 0.0
    return {
        "target_chunk_s": TARGET_CHUNK_S,
        "hard_max_chunk_s": CHUNK_HARD_MAX_S,
        "boost_sha": boost_sha,
        "boost_alpha": boost_alpha,
        "boost_context_score": context_score,
        "boost_depth_scaling": depth_scaling,
    }


def build_decoding_cfg(
    boost_file: Path | None,
    boost_alpha: float,
    context_score: float = 0.3,
    depth_scaling: float = 4.0,
) -> dict | None:
    """Greedy-batch RNNT decoding config enabling NeMo context biasing.

    Returns a plain dict shaped for ``model.change_decoding_strategy`` (merged
    onto the model's own ``cfg.decoding`` in ``run``), or ``None`` when biasing
    is disabled (no boost file, or non-positive alpha) so callers decode
    unchanged.

    The boosting-tree lives under ``greedy`` because NeMo's TDT/RNNT greedy-batch
    decoder reads ``boosting_tree`` / ``boosting_tree_alpha`` from the
    ``GreedyBatchedRNNTInferConfig`` at ``decoding.greedy``.

    ``context_score`` is the per-arc push on each boosted token (lower = gentler,
    less likely to override a confident acoustic decision) and ``depth_scaling``
    rewards deeper/longer phrase matches (higher = favor multi-token names over
    single near-homophone tokens). Together they let the boost break ties for
    real phrases without force-injecting single tokens; both are tuned on the
    full stream."""
    if boost_file is None or boost_alpha <= 0:
        return None
    return {
        "strategy": "greedy_batch",
        "greedy": {
            "boosting_tree": {
                "key_phrases_file": str(boost_file),
                "context_score": context_score,
                "depth_scaling": depth_scaling,
            },
            "boosting_tree_alpha": boost_alpha,
        },
    }


def run(
    audio_path: Path,
    out_path: Path,
    *,
    model_id: str = "nvidia/parakeet-tdt-0.6b-v3",
    device: str = "cuda",
    phase_cb: PhaseCb | None = None,
    boost_file: Path | None = None,
    # Chosen by the GPU alpha sweep on 317640725476 (task-C2 Step 5): 2.0
    # recovers near-homophone names (Shonda->Shondo, Horgan->Hargan) with zero
    # collateral, while higher values violate the plausibility rule (alpha 4.0+
    # forces "medical"->garbled "MITACLEMARK" and corrupts neighbors).
    boost_alpha: float = 2.0,
    # Gentleness knobs chosen by the full-stream knob sweep on 317640725476:
    # low context_score (per-token push) + high depth_scaling makes the boost
    # BREAK TIES for genuine multi-token phrases without force-injecting single
    # near-homophone tokens. This cut near-homophone stealing 11->1 and garbage
    # 78->50 (==unboosted) at the highest fidelity to the unboosted transcript
    # (0.978), i.e. it only changes the transcript where it is confident.
    boost_context_score: float = 0.3,
    boost_depth_scaling: float = 4.0,
) -> Path:
    if cache.is_fresh(
        out_path,
        audio_path,
        extra=parakeet_cache_extra(
            boost_file, boost_alpha, boost_context_score, boost_depth_scaling
        ),
    ):
        log.info("parakeet_skip_fresh", out=str(out_path))
        return out_path

    def emit(label: str, pct: float | None = None) -> None:
        if phase_cb:
            phase_cb(label, pct)

    import nemo.collections.asr as nemo_asr  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    import torchaudio  # type: ignore[import-not-found]
    from omegaconf import OmegaConf, open_dict  # type: ignore[import-not-found]

    emit("loading audio", 2.0)
    waveform, sr = torchaudio.load(str(audio_path))
    assert sr == 16000, f"expected 16k, got {sr}"
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio_np = waveform.squeeze(0).numpy().astype(np.float32)

    emit(f"loading {model_id}", 6.0)
    model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    model.eval()
    model = model.to(device)

    boost_cfg = build_decoding_cfg(
        boost_file, boost_alpha, boost_context_score, boost_depth_scaling
    )
    # Install the decoding strategy via a helper so a chunk that trips NeMo's
    # fragile word-confidence boundary reconstruction (a hard RuntimeError on a
    # token-vs-word off-by-one, seen around contractions/apostrophes) can be
    # re-decoded WITHOUT confidence rather than killing the whole transcription.
    # with_confidence=True turns on per-word confidence -> each word's `score`
    # (a debug signal that flags low-confidence / likely-wrong words near zero);
    # boosting rides the same greedy_batch strategy and is always applied.
    # Snapshot the model's PRISTINE decoding cfg once. Every install rebuilds
    # from this, so a with_confidence=False install genuinely drops confidence —
    # rebuilding from the live model.cfg.decoding would re-read the confidence
    # already installed, and the fallback decode would crash all over again.
    _base_decoding = OmegaConf.to_container(model.cfg.decoding)

    def _install_decoding(with_confidence: bool) -> None:
        cfg = OmegaConf.create(_base_decoding)
        with open_dict(cfg):
            if with_confidence:
                OmegaConf.update(
                    cfg, "confidence_cfg.preserve_frame_confidence", True, force_add=True
                )
                OmegaConf.update(
                    cfg, "confidence_cfg.preserve_token_confidence", True, force_add=True
                )
                OmegaConf.update(
                    cfg, "confidence_cfg.preserve_word_confidence", True, force_add=True
                )
                # min over a word's sub-tokens: the weakest token sets the score.
                OmegaConf.update(
                    cfg, "confidence_cfg.aggregation", "min", force_add=True
                )
            if boost_cfg is not None:
                OmegaConf.update(cfg, "strategy", boost_cfg["strategy"])
                OmegaConf.update(
                    cfg, "greedy.boosting_tree.key_phrases_file",
                    boost_cfg["greedy"]["boosting_tree"]["key_phrases_file"],
                    force_add=True,
                )
                OmegaConf.update(
                    cfg, "greedy.boosting_tree.context_score",
                    boost_cfg["greedy"]["boosting_tree"]["context_score"],
                    force_add=True,
                )
                OmegaConf.update(
                    cfg, "greedy.boosting_tree.depth_scaling",
                    boost_cfg["greedy"]["boosting_tree"]["depth_scaling"],
                    force_add=True,
                )
                OmegaConf.update(
                    cfg, "greedy.boosting_tree_alpha",
                    boost_cfg["greedy"]["boosting_tree_alpha"], force_add=True,
                )
        model.change_decoding_strategy(cfg)

    _install_decoding(with_confidence=True)
    if boost_cfg is not None:
        log.info(
            "parakeet_biasing_enabled",
            boost_file=str(boost_file),
            boost_alpha=boost_alpha,
        )

    def _transcribe_clip(clip):
        """Decode one clip, surviving NeMo's fragile word-confidence boundary
        reconstruction (hard RuntimeError on a token/word off-by-one) by
        re-decoding WITHOUT confidence, then re-arming. Shared by the main chunk
        loop and the hole-repair pass."""
        try:
            return model.transcribe([clip], timestamps=True, verbose=False)[0]
        except RuntimeError as e:
            if "confidence" not in str(e).lower():
                raise
            log.warning("parakeet_word_confidence_fallback", err=str(e))
            _install_decoding(with_confidence=False)
            try:
                return model.transcribe([clip], timestamps=True, verbose=False)[0]
            finally:
                _install_decoding(with_confidence=True)

    n_total = audio_np.shape[0]
    bounds = speech_aware_bounds(audio_np, sr)
    n_chunks = len(bounds)

    all_segments: list[dict] = []
    for ci, (chunk_start_sample, chunk_end_sample) in enumerate(bounds):
        clip = audio_np[chunk_start_sample:chunk_end_sample]
        if clip.shape[0] < 1600:
            continue
        chunk_start_s = chunk_start_sample / sr
        hyp = _transcribe_clip(clip)
        all_segments.extend(_convert_to_canonical(hyp, time_offset=chunk_start_s))
        emit("transcribing", 6.0 + 90.0 * (ci + 1) / n_chunks)
        # NB: do NOT call torch.cuda.empty_cache() inside this loop — NeMo's
        # TDT decoder retains references to internal buffers between
        # transcribe() calls; emptying the allocator frees memory the decoder
        # still points at -> "CUDA error: illegal memory access" after enough
        # successive calls (observed at chunk 270/270).

    all_segments = _dedup_overlapping(all_segments)
    # Second pass: repair cold-start DROPS. A chunk occasionally drops its own
    # opening (a knife-edge TDT instability sensitive to the exact boundary
    # sample); the dropped span then has speech-level audio energy but no words.
    # Re-decode each such hole as a window CENTERED on it — the speech is now
    # mid-clip (warm), not a cold start — and merge back any recovered words.
    # Non-speech holes (music/laughter) self-gate: the model emits nothing.
    emit("repairing dropped speech", 96.0)
    all_segments = _repair_holes(audio_np, sr, all_segments, _transcribe_clip)

    try:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except (torch.AcceleratorError, RuntimeError) as e:
        log.warning("parakeet_cleanup_ignored", err=str(e))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    tmp.write_text(
        json.dumps(
            {
                "engine": model_id,
                "duration_s": n_total / sr,
                "segments": all_segments,
            }
        )
    )
    tmp.replace(out_path)
    cache.stamp(
        out_path, audio_path, extra=parakeet_cache_extra(
            boost_file, boost_alpha, boost_context_score, boost_depth_scaling
        ),
    )
    return out_path


def _dedup_overlapping(segs: list[dict]) -> list[dict]:
    if not segs:
        return []
    sorted_segs = sorted(segs, key=lambda s: (s["start"], s["end"]))
    kept: list[dict] = [sorted_segs[0]]
    for s in sorted_segs[1:]:
        prev = kept[-1]
        overlap = min(prev["end"], s["end"]) - max(prev["start"], s["start"])
        if overlap <= 0:
            kept.append(s)
            continue
        shorter_span = max(
            1e-3, min(prev["end"] - prev["start"], s["end"] - s["start"])
        )
        ratio = overlap / shorter_span
        # Heavy time-overlap => same utterance transcribed twice at a chunk
        # seam. Keep the longer (more complete) one even if the boundary-
        # truncated copy garbled the edge and the texts diverge.
        if ratio >= HIGH_OVERLAP_RATIO:
            if len(s["text"]) > len(prev["text"]):
                kept[-1] = s
            continue
        shorter, longer = (
            (s, prev) if len(s["text"]) <= len(prev["text"]) else (prev, s)
        )
        contained = (
            fuzz.partial_ratio(shorter["text"].lower(), longer["text"].lower())
            >= DEDUP_TEXT_CONTAINMENT
        )
        if ratio >= MIN_DEDUP_OVERLAP_RATIO and contained:
            if len(s["text"]) > len(prev["text"]):
                kept[-1] = s
            continue
        kept.append(s)
    return kept


def _convert_to_canonical(hyp, time_offset: float = 0.0) -> list[dict]:
    ts = getattr(hyp, "timestamp", None) or {}
    seg_entries = ts.get("segment", [])
    word_entries = ts.get("word", [])

    # Per-word confidence is a SEPARATE list on the hypothesis (index-aligned
    # with word_entries), NOT a key on each word dict. None/short when decoding
    # ran without confidence enabled -> fall back to 1.0 so score stays a float.
    wconf = list(getattr(hyp, "word_confidence", None) or [])
    scores = [
        float(wconf[i]) if i < len(wconf) else 1.0 for i in range(len(word_entries))
    ]

    segments: list[dict] = []
    for sent in seg_entries:
        s_start = float(sent["start"]) + time_offset
        s_end = float(sent["end"]) + time_offset
        words_in = [
            {
                "word": w["word"],
                "start": float(w["start"]) + time_offset,
                "end": float(w["end"]) + time_offset,
                "score": sc,
            }
            for w, sc in zip(word_entries, scores)
            if float(sent["start"]) <= float(w["start"]) <= float(sent["end"])
        ]
        # Segment confidence = mean of its word scores (None when wordless), a
        # quick per-caption weakness signal for the debug views.
        seg_scores = [x["score"] for x in words_in]
        avg_conf = sum(seg_scores) / len(seg_scores) if seg_scores else None
        segments.append(
            {
                "start": s_start,
                "end": s_end,
                "text": sent["segment"],
                "language": "en",
                "avg_logprob": avg_conf,
                "words": words_in,
            }
        )
    return segments


def _repair_holes(audio, sr, segments, transcribe_clip):
    """Second pass: recover cold-start DROPS. A span with speech-level energy but
    no decoded word is re-decoded as a window CENTERED on it (so the dropped
    speech sits mid-clip / warm, not at a cold chunk start) and any recovered
    in-hole words are merged back. Non-speech holes (music/laughter) recover
    nothing — the model self-gates — so no extra gate is needed.

    `audio` is the 1-D float32 waveform, `transcribe_clip` decodes one clip to a
    hypothesis. Returns the merged, de-duplicated segment list."""
    import numpy as np  # type: ignore[import-not-found]

    n = int(len(audio))
    dur = n / sr
    fr = max(1, int(0.25 * sr))
    nf = n // fr
    if nf < 2:
        return segments
    rms = np.sqrt(
        (audio[: nf * fr].reshape(nf, fr).astype(np.float64) ** 2).mean(axis=1) + 1e-12
    )
    covered = np.zeros(nf, dtype=bool)
    for seg in segments:
        for w in seg.get("words") or []:
            covered[int(w["start"] / 0.25): int(w["end"] / 0.25) + 1] = True
    hole = (rms > REPAIR_RMS_SPEECH) & ~covered

    holes: list[tuple[float, float]] = []
    i = 0
    while i < nf:
        if hole[i]:
            j = i
            while j < nf and hole[j]:
                j += 1
            if (j - i) * 0.25 >= REPAIR_MIN_HOLE_S:
                holes.append((i * 0.25, j * 0.25))
            i = j
        else:
            i += 1
    if not holes:
        return segments

    added: list[dict] = []
    for (h0, h1) in holes:
        t0 = max(0.0, h0 - REPAIR_CTX_S)
        t1 = min(dur, h1 + REPAIR_CTX_S)
        clip = audio[int(t0 * sr): int(t1 * sr)]
        if clip.shape[0] < 1600:
            continue
        for seg in _convert_to_canonical(transcribe_clip(clip), time_offset=t0):
            inhole = [
                w for w in seg.get("words") or [] if h0 - 0.3 <= w["start"] <= h1 + 0.3
            ]
            if inhole:
                added.append(
                    {
                        "start": inhole[0]["start"],
                        "end": inhole[-1]["end"],
                        "text": " ".join(w["word"] for w in inhole).strip(),
                        "language": "en",
                        "avg_logprob": seg.get("avg_logprob"),
                        "words": inhole,
                    }
                )
    if not added:
        return segments
    log.info("parakeet_holes_repaired", candidates=len(holes), recovered=len(added))
    return _dedup_overlapping(sorted(segments + added, key=lambda s: s["start"]))
