from __future__ import annotations

import modular_transcriptor.runner as r


def test_dispatch_calls_parakeet(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(
        r,
        "_stage_parakeet",
        lambda work, audio, emit: called.setdefault("p", (work, audio)),
    )
    rc = r.main(
        [
            "--stage",
            "parakeet",
            "--work",
            str(tmp_path),
            "--audio",
            str(tmp_path / "a.wav"),
        ]
    )
    assert rc == 0
    assert called["p"][0] == tmp_path


def test_unknown_stage_errors(tmp_path):
    rc = r.main(["--stage", "nope", "--work", str(tmp_path), "--audio", "x"])
    assert rc == 2
