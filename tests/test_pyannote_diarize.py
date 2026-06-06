from __future__ import annotations

import json

import modular_transcriptor.pyannote_diarize as pdz


def test_turns_from_dataframe_records():
    rows = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.5, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
    turns = pdz._turns_from_records(rows)
    assert turns == rows


def test_run_idempotent(tmp_path):
    out = tmp_path / "diar.json"
    out.write_text(json.dumps([]))
    pdz.run(tmp_path / "audio.wav", out)  # returns without importing whisperx/torch
