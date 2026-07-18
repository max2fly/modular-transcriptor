from __future__ import annotations

import json

import pytest

import modular_transcriptor.pyannote_diarize as pdz
from modular_transcriptor import cache


def test_turns_from_dataframe_records():
    rows = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.5, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
    turns = pdz._turns_from_records(rows)
    assert turns == rows


def test_run_skips_when_output_fresh(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)  # proceed would RuntimeError
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "diar.json"
    out.write_text(json.dumps([]))
    cache.stamp(out, audio)
    before = out.read_text()
    assert pdz.run(audio, out) == out
    assert out.read_text() == before  # untouched -> genuinely skipped


def test_run_recomputes_when_source_changed(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "diar.json"
    out.write_text(json.dumps([]))
    cache.stamp(out, audio)
    audio.write_bytes(b"RE-EXTRACTED-DIFFERENT")  # WAV changed -> stale
    with pytest.raises(RuntimeError):  # proceeded past guard to the HF_TOKEN check
        pdz.run(audio, out)
