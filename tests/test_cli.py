from __future__ import annotations

import json
import os

import pytest

from modular_transcriptor import cli


def _fake_pipeline(monkeypatch, tmp_path, diar_turns):
    """Make _run_stage a RECORDING fake: it appends every stage name to a list
    (so tests can assert the exact sequence), writes diarization.json on
    'pyannote', and writes the finalize outputs on 'finalize'."""
    inp = tmp_path / "video.mp4"
    inp.write_bytes(b"x")
    calls: list[str] = []

    def fake_run_stage(stage, work, audio):
        calls.append(stage)
        if stage == "pyannote":
            (work / "diarization.json").write_text(json.dumps(diar_turns))
        if stage == "finalize":
            (work.parent / "transcript.json").write_text(
                json.dumps({"segments": [{"start": 1.0, "text": "hi", "speaker": "A"}]})
            )
            (work.parent / "transcript.csv").write_text(
                "time,speaker,text\n0:00:01,A,hi\n"
            )

    monkeypatch.setattr(cli, "_run_stage", fake_run_stage)
    # Neutralize .env loading so a real .env in the cwd can't inject HF_TOKEN and
    # invalidate the token-precondition tests.
    monkeypatch.setattr(cli, "_load_env", lambda: None)
    return inp, calls


def test_cli_drives_expected_stage_sequence(tmp_path, monkeypatch):
    inp, calls = _fake_pipeline(
        monkeypatch, tmp_path, diar_turns=[{"start": 0, "end": 1, "speaker": "A"}]
    )
    monkeypatch.setenv("HF_TOKEN", "x")
    out = tmp_path / "seq.json"
    rc = cli.main([str(inp), "--out", str(out)])
    assert rc == 0
    assert calls == ["amplify", "parakeet", "pyannote", "finalize"]


def test_cli_sequence_skips_pyannote_with_no_diarization(tmp_path, monkeypatch):
    inp, calls = _fake_pipeline(monkeypatch, tmp_path, diar_turns=[])
    out = tmp_path / "seq.json"
    rc = cli.main([str(inp), "--no-diarization", "--out", str(out)])
    assert rc == 0
    assert calls == ["amplify", "parakeet", "finalize"]


def test_cli_no_diarization_writes_outputs(tmp_path, monkeypatch):
    inp, _ = _fake_pipeline(monkeypatch, tmp_path, diar_turns=[])
    out = tmp_path / "result.json"
    rc = cli.main([str(inp), "--no-diarization", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert out.with_suffix(".csv").exists()


def test_cli_requires_hf_token_by_default(tmp_path, monkeypatch):
    inp, _ = _fake_pipeline(monkeypatch, tmp_path, diar_turns=[])
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit) as e:
        cli.main([str(inp)])
    assert "HF_TOKEN" in str(e.value)


def test_cli_errors_when_diarization_empty(tmp_path, monkeypatch):
    inp, _ = _fake_pipeline(
        monkeypatch, tmp_path, diar_turns=[]
    )  # model produced nothing
    monkeypatch.setenv("HF_TOKEN", "x")
    with pytest.raises(SystemExit) as e:
        cli.main([str(inp)])
    assert "no speakers" in str(e.value).lower()


def test_cli_diarization_success(tmp_path, monkeypatch):
    inp, _ = _fake_pipeline(
        monkeypatch, tmp_path, diar_turns=[{"start": 0, "end": 1, "speaker": "A"}]
    )
    monkeypatch.setenv("HF_TOKEN", "x")
    out = tmp_path / "ok.json"
    rc = cli.main([str(inp), "--out", str(out)])
    assert rc == 0 and out.exists()


def test_load_env_reads_file_without_overriding(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("HF_TOKEN=from_dotenv\n# comment\nOTHER=val\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("OTHER", "preexisting")  # already set -> must NOT override
    cli._load_env()
    assert os.environ["HF_TOKEN"] == "from_dotenv"
    assert os.environ["OTHER"] == "preexisting"
