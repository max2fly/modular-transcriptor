from __future__ import annotations

from modular_transcriptor.speakers import assign_speakers


def test_assigns_word_and_segment_speakers():
    segs = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "a b",
            "words": [
                {"start": 0.0, "end": 1.0, "word": "a"},
                {"start": 2.0, "end": 3.0, "word": "b"},
            ],
        },
    ]
    turns = [
        {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
        {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},
    ]
    out = assign_speakers(segs, turns)
    assert out[0]["words"][0]["speaker"] == "SPEAKER_00"
    assert out[0]["words"][1]["speaker"] == "SPEAKER_01"
    assert out[0]["speaker"] in ("SPEAKER_00", "SPEAKER_01")


def test_no_turns_leaves_speaker_none():
    segs = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "a",
            "words": [{"start": 0.0, "end": 1.0, "word": "a"}],
        }
    ]
    out = assign_speakers(segs, [])
    assert out[0]["speaker"] is None
    assert out[0]["words"][0]["speaker"] is None
