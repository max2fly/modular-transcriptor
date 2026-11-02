from __future__ import annotations

import json
from pathlib import Path

from modular_transcriptor.formats import write_transcript_csv
from modular_transcriptor.speakers import assign_speakers
from modular_transcriptor.types import PhaseCb

MODEL_NAME = "parakeet-tdt-0.6b-v3 (local-attn)"
PRODUCED_BY_VERSION = "0.4.0"


def _lean_segments(parakeet: dict) -> list[dict]:
    """Project each raw Parakeet segment onto the lean transcript schema,
    dropping per-word `score`/`language` and tagging source + kind."""
    lean: list[dict] = []
    for seg in parakeet.get("segments") or []:
        words = [
            {
                "start": float(w["start"]),
                "end": float(w["end"]),
                "word": w["word"],
            }
            for w in (seg.get("words") or [])
        ]
        lean.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg.get("text", ""),
                "source": "parakeet",
                "kind": "speech",
                "words": words,
            }
        )
    return lean


def _render_debug_md(segments: list[dict]) -> str:
    """One line per segment: [start-end] SPEAKER: text."""
    lines = ["# Transcript — parakeet-alone finalize\n"]
    for seg in segments:
        spk = seg.get("speaker") or "-"
        text = (seg.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{seg['start']:.2f}-{seg['end']:.2f}] {spk}: {text}")
    return "\n".join(lines) + "\n"


def finalize_run(work: Path, emit: PhaseCb | None = None) -> None:
    """Turn `work/parakeet.json` (+ optional `work/diarization.json`) into the
    final `transcript.json` and its sidecars in `work.parent`.

    Reads ONLY parakeet.json and diarization.json; any stale `whisper.json` /
    `mms_ctc.json` left in the work dir is ignored. Idempotent: returns
    immediately if `work.parent/transcript.json` already exists.
    """
    work = Path(work)
    out = work.parent
    if (out / "transcript.json").exists():
        return  # idempotent

    def _emit(label: str, pct: float | None = None) -> None:
        if emit is not None:
            emit(label, pct)

    _emit("finalizing transcript", 5.0)
    parakeet = json.loads((work / "parakeet.json").read_text())
    lean = _lean_segments(parakeet)

    diar_path = work / "diarization.json"
    turns = json.loads(diar_path.read_text()) if diar_path.exists() else []

    _emit("assigning speakers", 80.0)
    assign_speakers(lean, turns)

    # Prefer Parakeet's true decoded media length; a VOD ending in silence or
    # music runs past its last segment end, so max-seg-end under-reports it.
    pk_duration = parakeet.get("duration_s")
    duration_s = (
        float(pk_duration)
        if pk_duration is not None
        else max((seg["end"] for seg in lean), default=0.0)
    )
    diarized = bool(turns)

    _emit("writing outputs", 95.0)
    # Write the sidecars FIRST, then transcript.json LAST and atomically. The
    # idempotency guard keys on transcript.json's existence, so it must only
    # appear once every sidecar is on disk — otherwise a crash mid-write leaves
    # the guard believing the run finished while sidecars are missing.
    write_transcript_csv(out / "transcript.csv", lean)
    (out / "transcript.debug.json").write_text(
        json.dumps({"segments": lean}, ensure_ascii=False)
    )
    (out / "transcript.debug.md").write_text(_render_debug_md(lean))

    tmp = out / "transcript.json.partial"
    tmp.write_text(
        json.dumps(
            {
                "language": "en",
                "model": MODEL_NAME,
                "compute_type": "float16",
                "diarized": diarized,
                "duration_s": duration_s,
                "produced_by_version": PRODUCED_BY_VERSION,
                "segments": lean,
            },
            ensure_ascii=False,
        )
    )
    tmp.replace(out / "transcript.json")
