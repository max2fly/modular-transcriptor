from __future__ import annotations

import modular_transcriptor.transcript_merger as m
from modular_transcriptor.vad import derive_vad_intervals


def _subsegs(items):
    return [
        {"start": s, "end": e, "text": t, "words": [{"start": s, "end": e, "word": t}]}
        for s, e, t in items
    ]


def test_inplace_match_score_high_when_overlapping_match():
    subs = _subsegs([(20.0, 21.0, "with chopsticks ghost peppers")])
    subs_sorted = sorted(subs, key=lambda s: s["start"])
    sub_norm = [m.normalize_for_compare(s["text"]) for s in subs_sorted]
    ws_here = {"start": 20.0, "end": 21.0, "text": "with chopsticks ghost peppers"}
    ws_drift = {"start": 5.0, "end": 6.0, "text": "with chopsticks ghost peppers"}
    assert m._inplace_match_score(ws_here, subs_sorted, sub_norm) >= 80
    assert m._inplace_match_score(ws_drift, subs_sorted, sub_norm) < 80


def test_shift_segment_moves_seg_and_words():
    seg = {
        "start": 5.0,
        "end": 7.0,
        "words": [
            {"start": 5.0, "end": 5.5, "word": "a"},
            {"start": 6.0, "end": 7.0, "word": "b"},
        ],
    }
    m._shift_segment(seg, 15.0)
    assert seg["start"] == 20.0 and seg["end"] == 22.0
    assert seg["words"][0]["start"] == 20.0 and seg["words"][1]["end"] == 22.0


def _para(items):
    return {
        "segments": [
            {
                "start": s,
                "end": e,
                "text": t,
                "words": [{"start": s, "end": e, "word": t}],
            }
            for s, e, t in items
        ]
    }


def test_reanchor_shifts_drifted_block_forward():
    parakeet = _para([(20.0, 21.5, "with chopsticks maybe we will get ghost peppers")])
    whisper = {
        "segments": [
            {
                "start": 5.0,
                "end": 6.5,
                "text": "With chopsticks. Maybe we will get ghost peppers.",
                "words": [
                    {
                        "start": 5.0,
                        "end": 6.5,
                        "word": "With chopsticks maybe we will get ghost peppers",
                    }
                ],
            }
        ]
    }
    out = m.reanchor_whisper_segments(whisper, parakeet)
    seg = out["segments"][0]
    assert seg["start"] == 20.0
    assert seg["words"][0]["start"] == 20.0


def test_reanchor_leaves_hallucination_alone():
    parakeet = _para([(900.0, 901.0, "completely unrelated content here")])
    whisper = {
        "segments": [
            {
                "start": 5.0,
                "end": 5.6,
                "text": "Thank you.",
                "words": [{"start": 5.0, "end": 5.6, "word": "Thank you"}],
            }
        ]
    }
    out = m.reanchor_whisper_segments(whisper, parakeet)
    assert out["segments"][0]["start"] == 5.0


def test_reanchor_does_not_shift_backward():
    parakeet = _para([(10.0, 11.5, "with chopsticks maybe we will get ghost peppers")])
    whisper = {
        "segments": [
            {
                "start": 30.0,
                "end": 31.5,
                "text": "With chopsticks. Maybe we will get ghost peppers.",
                "words": [
                    {
                        "start": 30.0,
                        "end": 31.5,
                        "word": "With chopsticks maybe we will get ghost peppers",
                    }
                ],
            }
        ]
    }
    out = m.reanchor_whisper_segments(whisper, parakeet)
    assert out["segments"][0]["start"] == 30.0


def test_reanchored_whisper_drives_vad_at_true_location():
    parakeet = _para([(20.0, 21.5, "with chopsticks maybe we will get ghost peppers")])
    whisper = {
        "segments": [
            {
                "start": 5.0,
                "end": 6.5,
                "text": "With chopsticks. Maybe we will get ghost peppers.",
                "words": [
                    {
                        "start": 5.0,
                        "end": 6.5,
                        "word": "With chopsticks maybe we will get ghost peppers",
                    }
                ],
            }
        ]
    }
    corrected = m.reanchor_whisper_segments(whisper, parakeet)
    vad = derive_vad_intervals(corrected["segments"])
    # VAD now marks the TRUE location (~20s), not the noise location (5s).
    assert any(iv["start"] <= 20.0 <= iv["end"] for iv in vad)
    assert not any(iv["start"] <= 5.0 <= iv["end"] for iv in vad)
