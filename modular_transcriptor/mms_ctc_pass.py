from __future__ import annotations

import gc
import json
from pathlib import Path

from modular_transcriptor._logging import get_logger
from modular_transcriptor.types import PhaseCb

log = get_logger("mms_ctc_pass")

CHUNK_S = 30
OVERLAP_S = 2


def _decode_frames(
    predicted_ids,
    frame_dur_s,
    chunk_start_s,
    core_start,
    core_end,
    pad_id,
    word_delim_id,
    id_to_char,
):
    out_chars: list[dict] = []
    prev = -1
    for fi, idx in enumerate(predicted_ids):
        idx_int = int(idx)
        if idx_int == prev:
            continue
        prev = idx_int
        if idx_int == pad_id:
            continue
        ch = id_to_char.get(idx_int)
        if not ch:
            continue
        t_s = chunk_start_s + fi * frame_dur_s
        if not (core_start <= t_s < core_end):
            continue
        if idx_int == word_delim_id:
            out_chars.append({"char": "|", "t": t_s})
        else:
            out_chars.append({"char": ch.lower(), "t": t_s})
    return out_chars


def run(
    audio_path: Path,
    out_path: Path,
    *,
    model_id: str = "facebook/mms-1b-all",
    lang_adapter: str = "eng",
    device: str = "cuda",
    phase_cb: PhaseCb | None = None,
) -> Path:
    if out_path.exists():
        log.info("mms_skip_exists", out=str(out_path))
        return out_path

    def emit(label, pct=None):
        if phase_cb:
            phase_cb(label, pct)

    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    import torchaudio  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCTC,
        AutoProcessor,
    )

    emit("loading audio", 2.0)
    waveform, sr = torchaudio.load(str(audio_path))
    assert sr == 16000, f"expected 16k, got {sr}"
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio_np = waveform.squeeze(0).numpy().astype(np.float32)

    emit(f"loading {model_id}", 6.0)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).to(device).eval()
    processor.tokenizer.set_target_lang(lang_adapter)
    model.load_adapter(lang_adapter)
    pad_id = processor.tokenizer.pad_token_id
    word_delim_id = processor.tokenizer.encoder.get("|")
    id_to_char = {v: k for k, v in processor.tokenizer.encoder.items()}

    chunk_samples = CHUNK_S * sr
    overlap_samples = OVERLAP_S * sr
    step = chunk_samples - overlap_samples
    n_total = audio_np.shape[0]
    n_chunks = max(1, (n_total - overlap_samples + step - 1) // step)

    out_chars: list[dict] = []
    for ci in range(n_chunks):
        chunk_start_sample = ci * step
        chunk_end_sample = min(chunk_start_sample + chunk_samples, n_total)
        clip = audio_np[chunk_start_sample:chunk_end_sample]
        if clip.shape[0] < 1600:
            continue
        inputs = processor(clip, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        n_frames = logits.shape[1]
        clip_duration_s = clip.shape[0] / sr
        frame_dur_s = clip_duration_s / n_frames
        chunk_start_s = chunk_start_sample / sr
        chunk_end_s = chunk_end_sample / sr
        core_start = chunk_start_s + (OVERLAP_S / 2 if ci > 0 else 0)
        core_end = chunk_end_s - (OVERLAP_S / 2 if ci < n_chunks - 1 else 0)
        out_chars.extend(
            _decode_frames(
                predicted_ids,
                frame_dur_s,
                chunk_start_s,
                core_start,
                core_end,
                pad_id,
                word_delim_id,
                id_to_char,
            )
        )
        emit("transcribing", 6.0 + 90.0 * (ci + 1) / n_chunks)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    tmp.write_text(json.dumps(out_chars))
    tmp.replace(out_path)
    return out_path
