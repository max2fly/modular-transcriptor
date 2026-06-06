from __future__ import annotations

import modular_transcriptor.transcript_merger as m
from modular_transcriptor.speakers import assign_speakers

LEAN_KEYS = {"start", "end", "text", "source", "kind", "words"}
DIAG_ONLY = {
    "parakeet_text",
    "mms_text",
    "parakeet_score",
    "wp_agreement",
    "word_swaps",
}


def test_lean_segments_have_only_consumer_fields():
    parakeet = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hi",
                "words": [{"start": 0.0, "end": 1.0, "word": "hi"}],
            }
        ]
    }
    whisper = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hi",
                "words": [{"word": "hi", "start": 0.0, "end": 1.0, "score": 0.9}],
            }
        ]
    }
    mms = [{"char": "h", "t": 0.0}, {"char": "i", "t": 0.1}]
    segs = m.TranscriptMerger(parakeet, mms, whisper, [], audio=None, sr=16000).run()
    assign_speakers(segs, [])
    for s in segs:
        assert set(s).issubset(LEAN_KEYS | {"speaker"})
        assert not (set(s) & DIAG_ONLY)
        assert "speaker" in s
        for w in s["words"]:
            assert set(w).issubset({"start", "end", "word", "speaker"})
