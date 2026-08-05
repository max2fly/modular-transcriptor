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


def test_amplify_forwards_audio_track(monkeypatch, tmp_path):
    seen = {}

    def fake_amplify(src, dst, *, gain_db=6, track=None):
        seen["track"] = track

    monkeypatch.setattr(
        "modular_transcriptor.audio.amplify_to_wav", fake_amplify
    )
    rc = r.main(
        [
            "--stage",
            "amplify",
            "--work",
            str(tmp_path),
            "--audio",
            "x.mp4",
            "--audio-track",
            "3",
        ]
    )
    assert rc == 0
    assert seen["track"] == 3


def test_unknown_stage_errors(tmp_path):
    rc = r.main(["--stage", "nope", "--work", str(tmp_path), "--audio", "x"])
    assert rc == 2
