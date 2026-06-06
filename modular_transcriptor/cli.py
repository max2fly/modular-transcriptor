from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Engine stages run in order; each is a subprocess so its GPU model is freed
# before the next loads. 'pyannote' is conditional; 'merge' always runs last.
_PRE_MERGE_STAGES = ["amplify", "parakeet", "mms", "whisper"]


def _run_stage(stage: str, work: Path, audio: Path) -> None:
    """Run one pipeline stage as a subprocess, streaming its {phase,pct} lines."""
    cmd = [
        sys.executable,
        "-m",
        "modular_transcriptor.runner",
        "--stage",
        stage,
        "--work",
        str(work),
        "--audio",
        str(audio),
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        try:
            evt = json.loads(line)
            pct = evt.get("pct")
            pct_s = f"{pct:5.1f}%" if isinstance(pct, (int, float)) else ""
            print(f"  [{stage}] {pct_s} {evt.get('phase', '')}", flush=True)
        except json.JSONDecodeError:
            print(f"  [{stage}] {line}", flush=True)
    if proc.wait() != 0:
        raise SystemExit(f"stage {stage!r} failed (see output above)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="modular-transcriptor",
        description="Transcribe a video/audio file (Parakeet + MMS + WhisperX merge).",
    )
    ap.add_argument("input", help="video or audio file")
    ap.add_argument(
        "--out",
        help="output transcript.json path (default: <input>.transcript.json)",
    )
    ap.add_argument("--work", help="work dir (default: a temp dir, deleted after)")
    ap.add_argument(
        "--no-diarization",
        action="store_true",
        help="skip speaker diarization (no HF token / pyannote model needed)",
    )
    ap.add_argument(
        "--keep-work", action="store_true", help="keep the intermediate work dir"
    )
    args = ap.parse_args(argv)

    inp = Path(args.input).expanduser().resolve()
    if not inp.exists():
        raise SystemExit(f"input not found: {inp}")

    if not args.no_diarization and not os.environ.get("HF_TOKEN"):
        raise SystemExit(
            "Diarization requires HF_TOKEN (and an accepted pyannote model). "
            "Set HF_TOKEN, or re-run with --no-diarization to skip speaker labels."
        )

    if args.work:
        base = Path(args.work).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        base = Path(tempfile.mkdtemp(prefix="modtx_"))
        cleanup = not args.keep_work
    work = base / "transcription_work"
    work.mkdir(parents=True, exist_ok=True)

    try:
        for stage in _PRE_MERGE_STAGES:
            _run_stage(stage, work, inp)

        if args.no_diarization:
            (work / "diarization.json").write_text("[]")
        else:
            _run_stage("pyannote", work, inp)
            diar = work / "diarization.json"
            turns = json.loads(diar.read_text()) if diar.exists() else []
            if not turns:
                raise SystemExit(
                    "Diarization produced no speakers — the pyannote model is likely "
                    "not cached/accepted. Accept it on HuggingFace and set HF_TOKEN, "
                    "or re-run with --no-diarization."
                )

        _run_stage("merge", work, inp)

        out_json = (
            Path(args.out).expanduser().resolve()
            if args.out
            else inp.with_name(inp.stem + ".transcript.json")
        )
        out_csv = out_json.with_suffix(".csv")
        shutil.copyfile(work.parent / "transcript.json", out_json)
        shutil.copyfile(work.parent / "transcript.csv", out_csv)
        print(f"wrote {out_json}", flush=True)
        print(f"wrote {out_csv}", flush=True)
    finally:
        if cleanup:
            shutil.rmtree(base, ignore_errors=True)
    return 0
