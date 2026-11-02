from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _emit_phase(label: str, pct: float | None) -> None:
    print(json.dumps({"phase": label, "pct": pct}), flush=True)


def _p(work: Path, name: str) -> Path:
    return work / name


def _stage_amplify(work, audio, emit, track=None):
    from modular_transcriptor.audio import amplify_to_wav

    # ffmpeg runs as one blocking call with no incremental progress, so emit a
    # single indeterminate label up front (otherwise the infobox shows nothing
    # for the minutes-long amplification of a full VOD).
    emit("amplifying audio", None)
    amplify_to_wav(
        Path(audio), _p(work, "audio.amplified.wav"), gain_db=6, track=track
    )


def _stage_parakeet(work, audio, emit):
    from modular_transcriptor.parakeet_transcriber import run

    run(_p(work, "audio.amplified.wav"), _p(work, "parakeet.json"), phase_cb=emit)


def _stage_pyannote(work, audio, emit):
    from modular_transcriptor.pyannote_diarize import run

    run(_p(work, "audio.amplified.wav"), _p(work, "diarization.json"), phase_cb=emit)


def _stage_finalize(work, audio, emit):
    from modular_transcriptor.finalize import finalize_run

    finalize_run(work, emit)


# Map stage name -> module attribute name. Resolved through the module
# namespace at dispatch time (not pre-bound) so tests can monkeypatch the
# individual _stage_* functions.
_STAGES = {
    "amplify": "_stage_amplify",
    "parakeet": "_stage_parakeet",
    "pyannote": "_stage_pyannote",
    "finalize": "_stage_finalize",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="modular_transcriptor.runner")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--audio-track", type=int, default=None)
    args = ap.parse_args(argv)
    fn_name = _STAGES.get(args.stage)
    fn = globals().get(fn_name) if fn_name is not None else None
    if fn is None:
        print(
            f"runner_error: unknown stage {args.stage!r}", file=sys.stderr, flush=True
        )
        return 2
    args.work.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "amplify":
            fn(args.work, args.audio, _emit_phase, track=args.audio_track)
        else:
            fn(args.work, args.audio, _emit_phase)
    except Exception as e:  # noqa: BLE001
        print(f"runner_error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
