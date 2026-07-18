"""Freshness tracking for derived pipeline artifacts.

The transcription pipeline is a chain of cached artifacts:

    source video --amplify--> audio.amplified.wav --engine--> parakeet.json
                                                   --engine--> mms_ctc.json
                                                   --engine--> diarization.json

Each stage skips work when its output already exists, so a failed run can
resume cheaply. But "exists" is not "still valid": when an upstream artifact is
regenerated (video re-merged, WAV re-extracted), a stale downstream output sits
on a *different timeline*. Reusing it silently drifts the transcript — e.g. a
parakeet.json computed on a pre-assembly WAV ends up ~7s offset from a whisper
pass on the re-extracted WAV, and the merge, unable to align them, emits both
copies (duplicated / out-of-order subtitles).

Freshness is tracked by fingerprinting the *input* an output was derived from
(size + mtime, stored in a `<output>.src.json` sidecar) and re-checking it on
reuse. Any change to the input file forces a rebuild; we err toward caution.
"""

from __future__ import annotations

import json
from pathlib import Path


def sidecar_path(out: Path) -> Path:
    """Path of the fingerprint sidecar sitting beside `out`."""
    return out.with_name(out.name + ".src.json")


def fingerprint(src: Path) -> dict | None:
    """Identity of `src`: size + mtime (ns). Any re-merge / re-extract rewrites
    the file (new mtime, usually new size), so this changes whenever the content
    the output was derived from changes — even a same-length re-cut. None if
    `src` can't be stat'd."""
    try:
        st = src.stat()
        return {"src_size": st.st_size, "src_mtime_ns": st.st_mtime_ns}
    except OSError:
        return None


def read_sidecar(out: Path) -> dict | None:
    try:
        return json.loads(sidecar_path(out).read_text())
    except (OSError, ValueError):
        return None


def is_fresh(out: Path, src: Path) -> bool:
    """True only when `out` exists and its sidecar proves it was derived from
    the current `src` (byte-identical: same size + mtime)."""
    if not out.exists():
        return False
    recorded = read_sidecar(out)
    current = fingerprint(src)
    if recorded is None or current is None:
        return False
    return (
        recorded.get("src_size") == current["src_size"]
        and recorded.get("src_mtime_ns") == current["src_mtime_ns"]
    )


def stamp(out: Path, src: Path) -> None:
    """Record the fingerprint of `src` as the provenance of `out`."""
    fp = fingerprint(src)
    if fp is not None:
        sidecar_path(out).write_text(json.dumps(fp))


def clear(out: Path) -> None:
    """Remove the fingerprint sidecar for `out` (if any)."""
    sidecar_path(out).unlink(missing_ok=True)
