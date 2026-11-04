from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from modular_transcriptor._logging import get_logger

log = get_logger("cli")

# Engine stages run in order; each is a subprocess so its GPU model is freed
# before the next loads. 'pyannote' is conditional; 'finalize' always runs last.
_PRE_DIARIZE_STAGES = ["amplify", "parakeet"]


def _load_env() -> None:
    """Load `KEY=VALUE` pairs from a `.env` in the current directory (or the
    nearest ancestor that has one) into os.environ WITHOUT overriding vars that
    are already set. Lets users keep HF_TOKEN in a .env instead of exporting it;
    stage subprocesses inherit the populated environment. Minimal parser:
    supports `KEY=value` and `# comments`; ignores quoting/`export` syntax."""
    cwd = Path.cwd()
    for d in (cwd, *cwd.parents):
        env = d / ".env"
        if not env.exists():
            continue
        for raw in env.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()
        return


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
        description="Transcribe a video/audio file (Parakeet + pyannote diarization).",
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

    _load_env()

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
        for stage in _PRE_DIARIZE_STAGES:
            _run_stage(stage, work, inp)

        if args.no_diarization:
            (work / "diarization.json").write_text("[]")
        else:
            _run_stage("pyannote", work, inp)
            diar = work / "diarization.json"
            data = json.loads(diar.read_text()) if diar.exists() else []
            # diarization.json is the v2 dict {"schema":2,"turns":[...]} but may
            # still be a legacy bare list; parse turns inline to keep cli self-
            # contained (no import of finalize's helper).
            turns = data if isinstance(data, list) else data.get("turns", [])
            if not turns:
                # Don't abort: a failed/empty diarization must NOT skip finalize
                # and regress to no transcript. Warn and finalize non-diarized.
                log.warning(
                    "diarization_empty_proceeding",
                    detail="pyannote produced no speaker turns; finalizing a "
                    "non-diarized transcript (the model may not be "
                    "cached/accepted)",
                )

        _run_stage("finalize", work, inp)

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
