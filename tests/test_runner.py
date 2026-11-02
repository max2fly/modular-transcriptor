from __future__ import annotations

from modular_transcriptor import runner


def test_stages_are_parakeet_alone():
    assert set(runner._STAGES) == {"amplify", "parakeet", "pyannote", "finalize"}
    assert "whisper" not in runner._STAGES and "mms" not in runner._STAGES
    assert "merge" not in runner._STAGES


def test_finalize_stage_dispatches(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(
        runner, "_stage_finalize", lambda w, a, e: called.setdefault("ok", True)
    )
    assert (
        runner.main(
            ["--stage", "finalize", "--work", str(tmp_path), "--audio", "x"]
        )
        == 0
    )
    assert called["ok"]
