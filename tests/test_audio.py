from __future__ import annotations

from pathlib import Path

import modular_transcriptor.audio as audio


def test_amplify_builds_correct_ffmpeg_command(monkeypatch, tmp_path):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        Path(cmd[-1]).write_bytes(b"WAV")

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    src = tmp_path / "in.ts"
    src.write_bytes(b"x")
    dst = tmp_path / "out.wav"

    audio.amplify_to_wav(src, dst, gain_db=6)

    cmd = calls["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "volume=6dB,alimiter=limit=0.95" in cmd
    assert "16000" in cmd
    assert cmd[-1] == str(dst)
    assert calls["kwargs"]["check"] is True


def test_amplify_skips_if_dst_exists(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("should not run ffmpeg when dst exists")

    monkeypatch.setattr(audio.subprocess, "run", boom)
    dst = tmp_path / "out.wav"
    dst.write_bytes(b"existing")
    audio.amplify_to_wav(tmp_path / "in.ts", dst, gain_db=6)  # no raise
