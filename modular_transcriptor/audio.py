from __future__ import annotations

import subprocess
from pathlib import Path

from modular_transcriptor import cache
from modular_transcriptor._logging import get_logger

log = get_logger("transcription_audio")


def amplify_to_wav(
    src: Path,
    dst: Path,
    *,
    gain_db: int = 6,
    track: int | None = None,
    timeout_s: int = 1800,
) -> Path:
    """Flat-gain amplification + brick-wall limiter to bring quiet speech up
    for the acoustic models WITHOUT dynamic compression (which lifts BGM /
    causes fishbowl artifacts). Output is 16 kHz mono pcm_s16le.

    The WAV is cached so re-running transcription on an unchanged video skips
    the minutes-long extraction, but the cache is keyed on the source's
    identity (see `cache`), NOT mere existence: the source video can be
    re-merged / re-downloaded later — even to the SAME length but different
    content (a re-cut seam, a replaced segment). A plain exists() check kept the
    stale WAV, transcribing an obsolete timeline and drifting the transcript
    progressively ahead of the video. Any change to the source forces a
    re-extract; a legacy WAV with no sidecar is re-extracted once.

    `aresample=async=1` fills audio PTS gaps with silence so the extracted
    waveform stays on the *container* (player/PTS) timeline. Without it, a
    sequential decode collapses the per-VOD-segment AAC-boundary gaps
    (~0.04s each, one per ~10s segment), drifting the transcript timeline
    progressively ahead of the canonical video (measured up to ~3.7s of
    drift across a 48-min VOD)."""
    extra = {"track": track} if track is not None else None
    if dst.exists():
        if cache.is_fresh(dst, src, extra):
            log.info("amplify_reuse", dst=str(dst), src=str(src))
            return dst
        log.warning(
            "amplify_stale_reextract",
            dst=str(dst),
            src=str(src),
            recorded=cache.read_sidecar(dst),
            current=cache.fingerprint(src, extra),
        )
        dst.unlink()
        cache.clear(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info("amplify_run", src=str(src), dst=str(dst), gain_db=gain_db, track=track)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if track is not None:
        cmd += ["-map", f"0:a:{track}"]
    cmd += [
        "-af",
        f"volume={gain_db}dB,alimiter=limit=0.95,aresample=async=1",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, timeout=timeout_s)
    cache.stamp(dst, src, extra)
    return dst
