from __future__ import annotations

import json

import pytest

from modular_transcriptor import cli


def _fake_pipeline(monkeypatch, tmp_path, diar_turns):
    """Make _run_stage a no-op that, on the 'merge' stage, writes the outputs
    that run_merge would, and on 'pyannote' writes diarization.json."""
    inp = tmp_path / "video.mp4"
    inp.write_bytes(b"x")

    def fake_run_stage(stage, work, audio):
        if stage == "pyannote":
            (work / "diarization.json").write_text(json.dumps(diar_turns))
        if stage == "merge":
            (work.parent / "transcript.json").write_text(
                json.dumps({"segments": [{"start": 1.0, "text": "hi", "speaker": "A"}]})
            )
            (work.parent / "transcript.csv").write_text(
                "time,speaker,text\n0:00:01,A,hi\n"
            )

    monkeypatch.setattr(cli, "_run_stage", fake_run_stage)
    return inp


def test_cli_no_diarization_writes_outputs(tmp_path, monkeypatch):
    inp = _fake_pipeline(monkeypatch, tmp_path, diar_turns=[])
    out = tmp_path / "result.json"
    rc = cli.main([str(inp), "--no-diarization", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert out.with_suffix(".csv").exists()


def test_cli_requires_hf_token_by_default(tmp_path, monkeypatch):
    inp = _fake_pipeline(monkeypatch, tmp_path, diar_turns=[])
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit) as e:
        cli.main([str(inp)])
    assert "HF_TOKEN" in str(e.value)


def test_cli_errors_when_diarization_empty(tmp_path, monkeypatch):
    inp = _fake_pipeline(monkeypatch, tmp_path, diar_turns=[])  # model produced nothing
    monkeypatch.setenv("HF_TOKEN", "x")
    with pytest.raises(SystemExit) as e:
        cli.main([str(inp)])
    assert "no speakers" in str(e.value).lower()


def test_cli_diarization_success(tmp_path, monkeypatch):
    inp = _fake_pipeline(
        monkeypatch, tmp_path, diar_turns=[{"start": 0, "end": 1, "speaker": "A"}]
    )
    monkeypatch.setenv("HF_TOKEN", "x")
    out = tmp_path / "ok.json"
    rc = cli.main([str(inp), "--out", str(out)])
    assert rc == 0 and out.exists()
