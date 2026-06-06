from __future__ import annotations

import gc
import json
import os
from pathlib import Path

from modular_transcriptor._logging import get_logger
from modular_transcriptor.types import PhaseCb

log = get_logger("pyannote_diarize")


def _turns_from_records(records) -> list[dict]:
    return [
        {
            "start": float(r["start"]),
            "end": float(r["end"]),
            "speaker": str(r["speaker"]),
        }
        for r in records
    ]


def run(
    audio_path: Path,
    out_path: Path,
    *,
    device: str = "cuda",
    phase_cb: PhaseCb | None = None,
) -> Path:
    if out_path.exists():
        log.info("diarize_skip_exists", out=str(out_path))
        return out_path

    def emit(label, pct=None):
        if phase_cb:
            phase_cb(label, pct)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required for pyannote diarization")

    turns: list[dict] = []
    try:
        import whisperx  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from whisperx.diarize import DiarizationPipeline  # type: ignore[import-not-found]

        emit("loading diarization model", 5.0)
        diar = None
        for kw in ("use_auth_token", "token"):
            try:
                diar = DiarizationPipeline(**{kw: token}, device=device)
                break
            except TypeError:
                continue
        if diar is None:
            diar = DiarizationPipeline(device=device)
        emit("loading audio", 15.0)
        audio = whisperx.load_audio(str(audio_path))

        def _diar_progress(pct: float) -> None:
            # whisperx maps segmentation -> 0-50%, embeddings -> 50-99%.
            emit("diarizing", float(pct))

        try:
            df = diar(audio, progress_callback=_diar_progress)
        except TypeError:
            # older whisperx without progress_callback — keep diarization, lose
            # only the progress detail.
            emit("diarizing", 20.0)
            df = diar(audio)
        records = df.to_dict("records") if hasattr(df, "to_dict") else list(df)
        turns = _turns_from_records(records)
        del diar
        gc.collect()
        torch.cuda.empty_cache()
    except (
        Exception
    ) as e:  # noqa: BLE001 — soft-degrade: no speakers, transcript still made
        log.warning("diarize_soft_degrade", err=str(e))
        turns = []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    tmp.write_text(json.dumps(turns))
    tmp.replace(out_path)
    return out_path
