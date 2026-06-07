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
    if `dst` already exists it is reused.

    `aresample=async=1` fills audio PTS gaps with silence so the extracted
    waveform stays on the *container* (player/PTS) timeline. Without it, a
    sequential decode collapses the per-VOD-segment AAC-boundary gaps
    (~0.04s each, one per ~10s segment), drifting the transcript timeline
    progressively ahead of the canonical video (measured up to ~3.7s of
    drift across a 48-min VOD)."""
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
            f"volume={gain_db}dB,alimiter=limit=0.95,aresample=async=1",
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
