from __future__ import annotations

import subprocess
from pathlib import Path

from modular_transcriptor._logging import get_logger

log = get_logger("transcription_audio")


def amplify_to_wav(
    src: Path, dst: Path, *, gain_db: int = 6, timeout_s: int = 1800
) -> Path:
    """Flat-gain amplification + brick-wall limiter to bring quiet speech up
    for the acoustic models WITHOUT dynamic compression (which lifts BGM /
    causes fishbowl artifacts). Output is 16 kHz mono pcm_s16le. Idempotent:
    if `dst` already exists it is reused."""
    if dst.exists():
        log.info("amplify_reuse", dst=str(dst))
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info("amplify_run", src=str(src), dst=str(dst), gain_db=gain_db)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-af",
            f"volume={gain_db}dB,alimiter=limit=0.95",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        check=True,
        timeout=timeout_s,
    )
    return dst
