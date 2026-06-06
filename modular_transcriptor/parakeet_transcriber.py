from __future__ import annotations

import gc
import json
from pathlib import Path

from modular_transcriptor._logging import get_logger
from modular_transcriptor.types import PhaseCb

log = get_logger("parakeet_transcriber")

CHUNK_S = 60
OVERLAP_S = 10
MIN_DEDUP_OVERLAP_RATIO = 0.30


def run(
    audio_path: Path,
    out_path: Path,
    *,
    model_id: str = "nvidia/parakeet-tdt-0.6b-v3",
    device: str = "cuda",
    phase_cb: PhaseCb | None = None,
) -> Path:
    if out_path.exists():
        log.info("parakeet_skip_exists", out=str(out_path))
        return out_path

    def emit(label: str, pct: float | None = None) -> None:
        if phase_cb:
            phase_cb(label, pct)

    import nemo.collections.asr as nemo_asr  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    import torchaudio  # type: ignore[import-not-found]

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

    chunk_samples = CHUNK_S * sr
    overlap_samples = OVERLAP_S * sr
    step = chunk_samples - overlap_samples
    n_total = audio_np.shape[0]
    n_chunks = max(1, (n_total - overlap_samples + step - 1) // step)

    all_segments: list[dict] = []
    for ci in range(n_chunks):
        chunk_start_sample = ci * step
        chunk_end_sample = min(chunk_start_sample + chunk_samples, n_total)
        clip = audio_np[chunk_start_sample:chunk_end_sample]
        if clip.shape[0] < 1600:
            continue
        chunk_start_s = chunk_start_sample / sr
        hyps = model.transcribe([clip], timestamps=True, verbose=False)
        all_segments.extend(_convert_to_canonical(hyps[0], time_offset=chunk_start_s))
        emit("transcribing", 6.0 + 90.0 * (ci + 1) / n_chunks)
        # NB: do NOT call torch.cuda.empty_cache() inside this loop — NeMo's
        # TDT decoder retains references to internal buffers between
        # transcribe() calls; emptying the allocator frees memory the decoder
        # still points at -> "CUDA error: illegal memory access" after enough
        # successive calls (observed at chunk 270/270).

    try:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except (torch.AcceleratorError, RuntimeError) as e:
        log.warning("parakeet_cleanup_ignored", err=str(e))

    all_segments = _dedup_overlapping(all_segments)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    tmp.write_text(json.dumps({"engine": model_id, "segments": all_segments}))
    tmp.replace(out_path)
    return out_path


def _dedup_overlapping(segs: list[dict]) -> list[dict]:
    if not segs:
        return []
    sorted_segs = sorted(segs, key=lambda s: (s["start"], s["end"]))
    kept: list[dict] = [sorted_segs[0]]
    for s in sorted_segs[1:]:
        prev = kept[-1]
        overlap = min(prev["end"], s["end"]) - max(prev["start"], s["start"])
        shorter_span = max(
            1e-3, min(prev["end"] - prev["start"], s["end"] - s["start"])
        )
        ratio = overlap / shorter_span
        if overlap > 0 and ratio >= MIN_DEDUP_OVERLAP_RATIO:
            if len(s["text"]) > len(prev["text"]):
                kept[-1] = s
            continue
        kept.append(s)
    return kept


def _convert_to_canonical(hyp, time_offset: float = 0.0) -> list[dict]:
    ts = getattr(hyp, "timestamp", None) or {}
    seg_entries = ts.get("segment", [])
    word_entries = ts.get("word", [])

    segments: list[dict] = []
    for sent in seg_entries:
        s_start = float(sent["start"]) + time_offset
        s_end = float(sent["end"]) + time_offset
        words_in = [
            {
                "word": w["word"],
                "start": float(w["start"]) + time_offset,
                "end": float(w["end"]) + time_offset,
                "score": float(w.get("confidence", 1.0)),
            }
            for w in word_entries
            if float(sent["start"]) <= float(w["start"]) <= float(sent["end"])
        ]
        segments.append(
            {
                "start": s_start,
                "end": s_end,
                "text": sent["segment"],
                "language": "en",
                "avg_logprob": None,
                "words": words_in,
            }
        )
    return segments
