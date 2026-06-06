from __future__ import annotations

import modular_transcriptor.transcript_merger as m


def _run(parakeet, mms_chars, whisper, vad, monkeypatch, mms_text="zzz"):
    monkeypatch.setattr(m, "run_mms_focused", lambda *a, **k: mms_text)
    merger = m.TranscriptMerger(
        parakeet,
        mms_chars,
        whisper,
        vad,
        audio=object(),
        sr=16000,
        mms_proc=1,
        mms_model=1,
        mms_pad_id=0,
        mms_word_delim_id=1,
        mms_id_to_char={},
    )
    return merger.run()


def test_consensus_pw_keeps_parakeet_on_high_wp(monkeypatch):
    # Parakeet "happy wednesday", Whisper agrees, MMS sees junk -> flagged ->
    # whisper_arbiter: wp_agreement high -> consensus_pw, keep Parakeet text.
    parakeet = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "happy wednesday",
                "words": [
                    {"start": 0.0, "end": 1.0, "word": "happy"},
                    {"start": 1.1, "end": 2.0, "word": "wednesday"},
                ],
            }
        ]
    }
    mms_chars = [{"char": "x", "t": 0.1}]  # mismatch -> flag
    whisper = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "happy wednesday",
                "words": [
                    {"word": "happy", "start": 0.0, "end": 1.0, "score": 0.9},
                    {"word": "wednesday", "start": 1.1, "end": 2.0, "score": 0.9},
                ],
            }
        ]
    }
    segs = _run(parakeet, mms_chars, whisper, [], monkeypatch, mms_text="x")
    assert segs[0]["source"] == "consensus_pw"
    assert segs[0]["text"] == "happy wednesday"


def test_vocal_gate_low_distinct_mms(monkeypatch):
    # VAD-active gap, Whisper empty there, MMS sparse low-distinct -> [V] vocalization
    parakeet = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "intro",
                "words": [{"start": 0.0, "end": 2.0, "word": "intro"}],
            },
            {
                "start": 12.0,
                "end": 14.0,
                "text": "outro",
                "words": [{"start": 12.0, "end": 14.0, "word": "outro"}],
            },
        ]
    }
    whisper = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "intro",
                "words": [{"word": "intro", "start": 0.0, "end": 2.0, "score": 0.9}],
            },
            {
                "start": 12.0,
                "end": 14.0,
                "text": "outro",
                "words": [{"word": "outro", "start": 12.0, "end": 14.0, "score": 0.9}],
            },
        ]
    }
    vad = [{"start": 4.0, "end": 9.0}]  # speech in the gap
    segs = _run(
        parakeet,
        [{"char": "a", "t": 0.0}],
        whisper,
        vad,
        monkeypatch,
        mms_text="awawawawawa awa wa",
    )
    vocal = [s for s in segs if s["kind"] == "non_speech_vocal"]
    assert vocal and vocal[0]["source"] == "vad_gap_fill"


def test_whisper_gap_fill_speech(monkeypatch):
    # VAD-active gap, Whisper HAS unowned text, MMS not vocal -> [Wg] speech
    parakeet = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "intro",
                "words": [{"start": 0.0, "end": 2.0, "word": "intro"}],
            },
            {
                "start": 12.0,
                "end": 14.0,
                "text": "outro",
                "words": [{"start": 12.0, "end": 14.0, "word": "outro"}],
            },
        ]
    }
    whisper = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "intro",
                "words": [{"word": "intro", "start": 0.0, "end": 2.0, "score": 0.9}],
            },
            {
                # Gap-fill text must NOT smear-match either neighbor ("intro"/
                # "outro") above WHISPER_SMEAR_MATCH_THRESHOLD (55), or
                # assign_whisper_owners claims it for a neighbor subseg and it
                # never reaches whisper_text_for_gap. "but it isnt drowning"
                # scores partial_ratio 60 vs both -> owned -> no gap fill; this
                # phrase scores 40 -> stays unowned -> [Wg].
                "start": 5.0,
                "end": 6.0,
                "text": "welcome back everybody",
                "words": [
                    {"word": "welcome", "start": 5.0, "end": 5.4, "score": 0.9},
                    {"word": "back", "start": 5.5, "end": 5.7, "score": 0.9},
                ],
            },
            {
                "start": 12.0,
                "end": 14.0,
                "text": "outro",
                "words": [{"word": "outro", "start": 12.0, "end": 14.0, "score": 0.9}],
            },
        ]
    }
    vad = [{"start": 4.5, "end": 7.0}]
    segs = _run(
        parakeet,
        [{"char": "x", "t": 0.0}],
        whisper,
        vad,
        monkeypatch,
        mms_text="welcome back everybody",
    )
    wg = [s for s in segs if s["source"] == "whisper_gap_fill"]
    assert wg and wg[0]["kind"] == "speech"


def test_word_swap_screen_to_stream(monkeypatch):
    parakeet = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "the screen tomorrow",
                "words": [
                    {"start": 0.0, "end": 0.4, "word": "the"},
                    {"start": 0.5, "end": 1.0, "word": "screen"},
                    {"start": 1.1, "end": 2.0, "word": "tomorrow"},
                ],
            }
        ]
    }
    whisper = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "the stream tomorrow",
                "words": [
                    {"word": "the", "start": 0.0, "end": 0.4, "score": 0.9},
                    {"word": "stream", "start": 0.5, "end": 1.0, "score": 0.9},
                    {"word": "tomorrow", "start": 1.1, "end": 2.0, "score": 0.9},
                ],
            }
        ]
    }
    mms_chars = []
    tt = 0.0
    for ch in "the|stream|tomorrow":
        mms_chars.append({"char": ch, "t": tt})
        tt += 0.1
    segs = _run(parakeet, mms_chars, whisper, [], monkeypatch, mms_text="stream")
    assert "stream tomorrow" in segs[0]["text"]
    assert segs[0]["source"].endswith("+word_swap")
