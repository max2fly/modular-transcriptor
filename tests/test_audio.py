from __future__ import annotations

import json
import os
from pathlib import Path

import modular_transcriptor.audio as audio


def _recorder(tmp_bytes=b"WAV"):
    """A fake ffmpeg that records how many times it ran and writes the dst."""
    state = {"n": 0, "cmd": None, "kwargs": None}

    def fake_run(cmd, **kwargs):
        state["n"] += 1
        state["cmd"] = cmd
        state["kwargs"] = kwargs
        Path(cmd[-1]).write_bytes(tmp_bytes + str(state["n"]).encode())

        class R:
            returncode = 0

        return R()

    return state, fake_run


def test_amplify_builds_command_and_writes_sidecar(monkeypatch, tmp_path):
    state, fake_run = _recorder()
    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    src = tmp_path / "in.ts"
    src.write_bytes(b"video-content")
    dst = tmp_path / "out.wav"

    audio.amplify_to_wav(src, dst, gain_db=6)

    cmd = state["cmd"]
    assert cmd[0] == "ffmpeg"
    assert any("volume=6dB,alimiter=limit=0.95" in a for a in cmd)
    assert "16000" in cmd
    assert state["kwargs"]["check"] is True

    # sidecar records the exact source identity the WAV was made from
    sc = json.loads((dst.with_name(dst.name + ".src.json")).read_text())
    st = src.stat()
    assert sc == {"src_size": st.st_size, "src_mtime_ns": st.st_mtime_ns}


def test_amplify_reuses_when_source_unchanged(monkeypatch, tmp_path):
    state, fake_run = _recorder()
    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    src = tmp_path / "in.ts"
    src.write_bytes(b"video")
    dst = tmp_path / "out.wav"

    audio.amplify_to_wav(src, dst)  # extracts, writes sidecar
    audio.amplify_to_wav(src, dst)  # unchanged source -> reuse

    assert state["n"] == 1


def test_amplify_reextracts_on_same_length_different_content(monkeypatch, tmp_path):
    # The core case: video re-merged to the SAME byte length but different
    # content (re-cut seam / replaced segment). Duration is identical, so only
    # the file identity (mtime) reveals the change — and it must force a redo.
    state, fake_run = _recorder()
    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    src = tmp_path / "in.ts"
    src.write_bytes(b"AAAAAAAA")
    dst = tmp_path / "out.wav"

    audio.amplify_to_wav(src, dst)  # n == 1
    src.write_bytes(b"BBBBBBBB")  # same 8 bytes, different content
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    audio.amplify_to_wav(src, dst)  # must re-extract
    assert state["n"] == 2


def test_amplify_reextracts_on_size_change(monkeypatch, tmp_path):
    state, fake_run = _recorder()
    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    src = tmp_path / "in.ts"
    src.write_bytes(b"short")
    dst = tmp_path / "out.wav"

    audio.amplify_to_wav(src, dst)  # n == 1
    src.write_bytes(b"much-longer-video-content")  # size changed

    audio.amplify_to_wav(src, dst)  # must re-extract
    assert state["n"] == 2


def test_amplify_reextracts_when_no_sidecar(monkeypatch, tmp_path):
    # A legacy WAV extracted before the sidecar existed is unverifiable and
    # must be re-extracted (err toward caution).
    state, fake_run = _recorder(tmp_bytes=b"FRESH")
    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    src = tmp_path / "in.ts"
    src.write_bytes(b"video")
    dst = tmp_path / "out.wav"
    dst.write_bytes(b"LEGACY")  # exists, but no sidecar

    audio.amplify_to_wav(src, dst)
    assert state["n"] == 1
    assert dst.read_bytes().startswith(b"FRESH")
