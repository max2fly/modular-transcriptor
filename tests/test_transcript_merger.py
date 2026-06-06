from __future__ import annotations

import pytest

import modular_transcriptor.transcript_merger as m


def test_normalize_for_compare():
    assert m.normalize_for_compare("Hello, World!") == "hello world"
    assert m.normalize_for_compare("it's   FUN?") == "it's fun"


def test_binary_lb():
    arr = [{"t": 0.0}, {"t": 1.0}, {"t": 2.0}]
    assert m._binary_lb(arr, 1.5) == 2
    assert m._binary_lb(arr, 0.0) == 0


def test_build_mms_window_stops_at_hi():
    chars = [
        {"char": "h", "t": 1.0},
        {"char": "i", "t": 1.1},
        {"char": "|", "t": 1.2},
        {"char": "y", "t": 1.3},
    ]
    assert m.build_mms_window(chars, 0.5, 1.25) == "hi"


def test_whisper_word_reliable_filters():
    assert m._whisper_word_reliable({"score": 0.9, "start": 0.0, "end": 0.5}) is True
    assert m._whisper_word_reliable({"score": 0.0, "start": 0.0, "end": 0.5}) is False
    assert m._whisper_word_reliable({"score": 0.9, "start": 0.0, "end": 2.0}) is False


def test_neighbor_aware_window_caps_into_clean_gap():
    subs = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.2, "end": 2.0},
        {"start": 3.0, "end": 4.0},
    ]
    lo, hi = m.compute_neighbor_aware_window(subs, 1, pad_start=0.1, pad_end=0.4)
    assert lo == pytest.approx(1.1)
    assert hi == pytest.approx(2.4)


def test_pack_subseg():
    words = [
        {"start": 1.0, "end": 1.5, "word": "hi"},
        {"start": 1.6, "end": 2.0, "word": "you"},
    ]
    sub = m._pack_subseg(words)
    assert sub["start"] == 1.0 and sub["end"] == 2.0 and sub["text"] == "hi you"


def test_resegment_splits_on_silence():
    data = {
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "a b c",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": "a"},
                    {"start": 0.6, "end": 1.0, "word": "b"},  # gap 0.1 -> same
                    {"start": 3.0, "end": 3.5, "word": "c"},  # gap ~1.6 -> split
                ],
            }
        ]
    }
    subs = m.resegment_parakeet(data, silence_threshold_s=1.0, end_pad_s=0.4)
    assert len(subs) == 2
    assert subs[0]["text"] == "a b"
    assert subs[1]["text"] == "c"


def test_dedupe_drops_contained():
    subs = [
        {"start": 0.0, "end": 2.0, "text": "completion is to anyone", "words": [1]},
        {"start": 0.1, "end": 1.9, "text": "to anyone", "words": [1]},
    ]
    out = m.dedupe_overlapping_subsegs(subs)
    assert len(out) == 1
    assert out[0]["text"] == "completion is to anyone"


def test_assign_owners_single_overlap():
    subs = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "hello there",
            "words": [{"start": 0.0, "end": 1.0, "word": "hello"}],
        }
    ]
    wsegs = [{"start": 0.5, "end": 2.0, "text": "hello there", "words": []}]
    owners, multi = m.assign_whisper_owners(
        subs, wsegs, tolerance_s=10.0, content_threshold=55
    )
    assert owners[0] == 0
    assert multi == set()


def test_whisper_text_for_subseg_uses_owned():
    subs = [{"start": 0.0, "end": 3.0}]
    wsegs = [{"start": 0.5, "end": 2.0, "text": "hello there", "words": []}]
    txt = m.whisper_text_for_subseg(subs[0], 0, wsegs, [0], set())
    assert txt == "hello there"


def test_gap_absorbed_by_neighbor():
    subs = [{"start": 0.0, "end": 2.0}]
    wsegs = [{"start": 0.0, "end": 3.6, "text": "x"}]  # extends past gap_hi
    assert m.gap_absorbed_by_neighbor(subs, wsegs, [0], gap_lo=2.0, gap_hi=3.5) is True


def _ws(words):
    return [
        {
            "start": words[0][1],
            "end": words[-1][2],
            "text": " ".join(w[0] for w in words),
            "words": [
                {"word": w[0], "start": w[1], "end": w[2], "score": 0.9} for w in words
            ],
        }
    ]


def test_continuation_appends_tail():
    pk_words = [
        {"word": "bearing", "start": 5.0, "end": 5.3},
        {"word": "with", "start": 5.4, "end": 5.6},
    ]
    wsegs = _ws(
        [
            ("bearing", 5.0, 5.3),
            ("with", 5.4, 5.6),
            ("me", 5.7, 5.9),
            ("anyway", 6.0, 6.4),
        ]
    )
    cont = m.find_continuation_words(pk_words, wsegs, ceiling_s=99.0)
    assert [c["word"] for c in cont] == ["me", "anyway"]


def test_prefix_prepends_head():
    pk_words = [
        {"word": "cream", "start": 5.7, "end": 5.9},
        {"word": "cheese", "start": 6.0, "end": 6.3},
    ]
    wsegs = _ws(
        [
            ("homemade", 5.0, 5.4),
            ("a", 5.45, 5.5),
            ("cream", 5.7, 5.9),
            ("cheese", 6.0, 6.3),
        ]
    )
    pre = m.find_prefix_words(pk_words, wsegs, floor_s=0.0)
    assert [p["word"] for p in pre] == ["homemade", "a"]


def test_prefix_guard_blocks_single_extra_word():
    pk_words = [
        {"word": "and", "start": 5.0, "end": 5.2},
        {"word": "then", "start": 5.3, "end": 5.5},
        {"word": "she", "start": 5.6, "end": 5.8},
    ]
    wsegs = _ws([("then", 5.3, 5.5), ("she", 5.6, 5.8)])
    assert m.find_prefix_words(pk_words, wsegs, floor_s=0.0) == []


def test_find_vad_active_gaps():
    subs = [{"start": 0.0, "end": 2.0}, {"start": 6.0, "end": 8.0}]
    vad = [{"start": 3.0, "end": 4.0}]
    gaps = m.find_vad_active_gaps(subs, vad, threshold_s=1.0, prev_end_pad_s=0.4)
    assert gaps == [(2.4, 6.0)]


def test_no_gap_without_vad():
    subs = [{"start": 0.0, "end": 2.0}, {"start": 6.0, "end": 8.0}]
    assert m.find_vad_active_gaps(subs, [], threshold_s=1.0) == []


def test_per_word_refine_swaps_when_mms_sides_with_whisper(monkeypatch):
    # MMS phonetically says "cold" -> should swap Parakeet "good" -> "cold"
    monkeypatch.setattr(m, "run_mms_focused", lambda *a, **k: "cold")
    sub = {
        "start": 10.0,
        "end": 11.0,
        "text": "good saddle",
        "final_text": "good saddle",
        "final_source": "parakeet",
        "words": [
            {"start": 10.0, "end": 10.4, "word": "good"},
            {"start": 10.5, "end": 11.0, "word": "saddle"},
        ],
    }
    wsegs = [
        {
            "start": 10.0,
            "end": 11.0,
            "text": "cold saddle",
            "words": [
                {"word": "cold", "start": 10.0, "end": 10.4, "score": 0.9},
                {"word": "saddle", "start": 10.5, "end": 11.0, "score": 0.9},
            ],
        }
    ]
    swaps = m.per_word_refine(
        sub,
        wsegs,
        audio=None,
        sr=16000,
        mms_proc=None,
        mms_model=None,
        mms_pad_id=0,
        mms_word_delim_id=1,
        mms_id_to_char={},
    )
    assert len(swaps) == 1
    assert swaps[0]["p_word"] == "good" and swaps[0]["w_word"] == "cold"
    assert sub["final_source"].endswith("+word_swap")
    assert "cold saddle" in sub["final_text"]


def test_transfer_punctuation_copies_formatting():
    sub = {
        "start": 0.0,
        "end": 2.0,
        "words": [
            {"start": 0.0, "end": 0.5, "word": "they"},
            {"start": 0.6, "end": 1.0, "word": "exist"},
        ],
    }
    wsegs = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "They, exist.",
            "words": [
                {"word": "They,", "start": 0.0, "end": 0.5, "score": 0.9},
                {"word": "exist.", "start": 0.6, "end": 1.0, "score": 0.9},
            ],
        }
    ]
    n = m.transfer_whisper_punctuation(sub, 0, wsegs, [0], set())
    assert n == 2
    assert sub["words"][0]["word"] == "They,"
    assert sub["words"][1]["word"] == "exist."


def test_merger_all_agree_no_models_needed():
    parakeet = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello world",
                "words": [
                    {"start": 0.0, "end": 0.8, "word": "hello"},
                    {"start": 1.0, "end": 2.0, "word": "world"},
                ],
            }
        ]
    }
    mms_chars = []
    t = 0.0
    for ch in "hello|world":
        mms_chars.append({"char": ch, "t": t})
        t += 0.15
    whisper = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.8, "score": 0.9},
                    {"word": "world", "start": 1.0, "end": 2.0, "score": 0.9},
                ],
            }
        ]
    }
    merger = m.TranscriptMerger(
        parakeet, mms_chars, whisper, vad_intervals=[], audio=None, sr=16000
    )
    segs = merger.run()  # all engines agree, no gaps -> never touches the MMS model
    assert len(segs) == 1
    assert segs[0]["source"] == "parakeet"
    assert segs[0]["text"] == "hello world"
    assert "parakeet_text" not in segs[0]  # lean schema
    assert set(segs[0]["words"][0].keys()) == {"start", "end", "word"}


def test_merger_snaps_final_words_to_mms(monkeypatch):
    monkeypatch.setattr(m, "SNAP_WORDS_TO_MMS", True)
    parakeet = {
        "segments": [
            {
                "start": 1173.760,
                "end": 1174.080,
                "text": "Its",
                "words": [{"start": 1173.760, "end": 1174.080, "word": "Its"}],
            }
        ]
    }
    mms_chars = [
        {"char": "i", "t": 1173.937},
        {"char": "t", "t": 1173.957},
        {"char": "s", "t": 1174.037},
    ]
    whisper = {"segments": []}
    merger = m.TranscriptMerger(parakeet, mms_chars, whisper, [], audio=None)
    lean = merger.run()
    w0 = lean[0]["words"][0]
    assert w0["start"] == 1173.937 and w0["end"] == 1174.037


def test_smear_length_guard_blocks_tiny_word_claiming_long_segment():
    # A 2-char subseg ("hi") must NOT claim a long, non-overlapping Whisper
    # sentence via the partial_ratio substring trap ('hi' is inside 'chili').
    subs = [
        {
            "start": 100.0,
            "end": 100.3,
            "text": "hi",
            "words": [{"start": 100.0, "end": 100.3, "word": "hi"}],
        }
    ]
    wsegs = [
        {
            "start": 103.0,
            "end": 108.0,
            "text": "ooh chili how do i know when i am done",
            "words": [],
        }
    ]
    owners, multi = m.assign_whisper_owners(
        subs, wsegs, tolerance_s=10.0, content_threshold=55
    )
    assert owners[0] is None  # length guard blocks the smear


def test_smear_still_attributes_comparable_length_segment():
    # A genuine time-misplaced smear (comparable length, same utterance) is
    # still attributed.
    subs = [
        {
            "start": 100.0,
            "end": 102.0,
            "text": "now we got that out of the way",
            "words": [{"start": 100.0, "end": 102.0, "word": "now"}],
        }
    ]
    wsegs = [
        {
            "start": 105.0,
            "end": 107.0,
            "text": "Now we got that out of the way",
            "words": [],
        }
    ]
    owners, multi = m.assign_whisper_owners(
        subs, wsegs, tolerance_s=10.0, content_threshold=55
    )
    assert owners[0] == 0  # comparable length -> legit smear still works


def test_assign_owners_splits_short_leading_subseg():
    # "Yeah." (0.4s) + "It's happening." (0.72s); one Whisper seg spans both.
    # Overlap of the Whisper seg with "Yeah." is 0.267s (< 0.5 absolute) but
    # 67% of the 0.4s subseg, so it must still claim and distribute words.
    subs = [
        {
            "start": 1173.280,
            "end": 1173.680,
            "text": "Yeah.",
            "words": [{"start": 1173.280, "end": 1173.680, "word": "Yeah."}],
        },
        {
            "start": 1173.760,
            "end": 1174.480,
            "text": "It's happening.",
            "words": [
                {"start": 1173.760, "end": 1174.080, "word": "It's"},
                {"start": 1174.080, "end": 1174.480, "word": "happening."},
            ],
        },
    ]
    wsegs = [
        {
            "start": 1173.413,
            "end": 1174.554,
            "text": "Yeah, it's happening.",
            "words": [
                {"start": 1173.413, "end": 1173.653, "word": "Yeah,", "score": 0.9},
                {"start": 1173.914, "end": 1174.054, "word": "it's", "score": 0.9},
                {
                    "start": 1174.074,
                    "end": 1174.554,
                    "word": "happening.",
                    "score": 0.9,
                },
            ],
        }
    ]
    owners, multi = m.assign_whisper_owners(subs, wsegs, 10.0, 55)
    # ambiguous content (both substrings of the Whisper text) -> multi-overlap
    assert 0 in multi and owners[0] is None
    t0 = m.whisper_text_for_subseg(subs[0], 0, wsegs, owners, multi)
    t1 = m.whisper_text_for_subseg(subs[1], 1, wsegs, owners, multi)
    assert t0 == "Yeah,"
    assert t1 == "it's happening."


def _phrase_arb_inputs(global_mms_word: str):
    parakeet = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "the cat sat",
                "words": [
                    {"start": 0.0, "end": 0.6, "word": "the"},
                    {"start": 0.7, "end": 1.2, "word": "cat"},
                    {"start": 1.3, "end": 2.0, "word": "sat"},
                ],
            }
        ]
    }
    mms_chars = []
    t = 0.0
    for ch in f"the|cat|{global_mms_word}":
        mms_chars.append({"char": ch, "t": t})
        t += 0.15
    whisper = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "the cat ran",
                "words": [
                    {"word": "the", "start": 0.0, "end": 0.6, "score": 0.9},
                    {"word": "cat", "start": 0.7, "end": 1.2, "score": 0.9},
                    {"word": "ran", "start": 1.3, "end": 2.0, "score": 0.9},
                ],
            }
        ]
    }
    return parakeet, mms_chars, whisper


def _run_arb(parakeet, mms_chars, whisper, monkeypatch, focused: str):
    monkeypatch.setattr(
        m, "PHRASE_ARB_WP_MAX", 95
    )  # widen so the synthetic pair qualifies
    monkeypatch.setattr(m, "run_mms_focused", lambda *a, **k: focused)
    merger = m.TranscriptMerger(
        parakeet,
        mms_chars,
        whisper,
        [],
        audio=object(),
        sr=16000,
        mms_proc=1,
        mms_model=1,
        mms_pad_id=0,
        mms_word_delim_id=1,
        mms_id_to_char={},
    )
    return merger.run()


def test_phrase_arb_flips_when_both_mms_favor_whisper(monkeypatch):
    # global MMS ("the cat ran") AND focused both favor Whisper -> adopt Whisper.
    parakeet, mms_chars, whisper = _phrase_arb_inputs("ran")
    segs = _run_arb(parakeet, mms_chars, whisper, monkeypatch, focused="the cat ran")
    assert segs[0]["source"] == "whisper_phrase_arb"
    assert segs[0]["text"] == "the cat ran"
    assert [w["word"] for w in segs[0]["words"]] == ["the", "cat", "ran"]


def test_phrase_arb_keeps_parakeet_when_global_mms_disagrees(monkeypatch):
    # global MMS favors Parakeet ("the cat sat") even though focused leans Whisper
    # -> conservative gate (need BOTH) keeps Parakeet; no phrase-arb flip.
    parakeet, mms_chars, whisper = _phrase_arb_inputs("sat")
    segs = _run_arb(parakeet, mms_chars, whisper, monkeypatch, focused="the cat sat")
    assert segs[0]["source"] != "whisper_phrase_arb"


def test_snap_words_to_mms_snaps_to_frames():
    # "its" Parakeet boundary is loose; MMS frames pin it tighter.
    words = [{"start": 1173.760, "end": 1174.080, "word": "It's"}]
    chars = [
        {"char": "|", "t": 1173.897},
        {"char": "i", "t": 1173.937},
        {"char": "t", "t": 1173.957},
        {"char": "s", "t": 1174.037},
        {"char": "|", "t": 1174.117},
    ]
    out = m.snap_words_to_mms(words, chars, pad_s=0.1)
    assert out[0]["start"] == 1173.937  # first alpha frame
    assert out[0]["end"] == 1174.037  # last alpha frame


def test_snap_words_to_mms_falls_back_when_no_chars():
    # short word with no MMS coverage -> keep Parakeet timing unchanged
    words = [{"start": 5.000, "end": 5.300, "word": "Yeah."}]
    chars = [{"char": "x", "t": 100.0}]
    out = m.snap_words_to_mms(words, chars, pad_s=0.1)
    assert out[0]["start"] == 5.000 and out[0]["end"] == 5.300


def test_normalize_numbers_spells_digits():
    assert m._normalize_numbers("30") == "thirty"
    assert m._normalize_numbers("11") == "eleven"
    assert m._normalize_numbers("i have 2 cats") == "i have two cats"
    assert m._normalize_numbers("fuck it") == "fuck it"


def test_numbers_equivalent():
    assert m._numbers_equivalent("11", "eleven") is True
    assert m._numbers_equivalent("30", "thirty") is True
    assert m._numbers_equivalent("30", "dirty") is False
    assert m._numbers_equivalent("sorry kate", "fuck it") is False


def test_is_filler_run():
    assert m._is_filler_run("um") is True
    assert m._is_filler_run("um uh") is True
    assert m._is_filler_run("Uh,") is True
    assert m._is_filler_run("fuck it") is False
    assert m._is_filler_run("") is False


def test_base_source_strips_suffixes():
    assert m._base_source("parakeet") == "parakeet"
    assert m._base_source("parakeet+word_swap") == "parakeet"
    assert m._base_source("parakeet+run_arb") == "parakeet"
    assert m._base_source("parakeet+word_swap+run_arb") == "parakeet"


def test_align_word_runs_isolates_substitution():
    p_words = [
        {"start": 0.0, "end": 0.2, "word": "Sorry,"},
        {"start": 0.2, "end": 0.4, "word": "Kate,"},
        {"start": 0.4, "end": 0.6, "word": "I"},
        {"start": 0.6, "end": 0.8, "word": "don't"},
    ]
    w_words = [
        {"start": 0.0, "end": 0.25, "word": "Fuck"},
        {"start": 0.25, "end": 0.4, "word": "it"},
        {"start": 0.4, "end": 0.6, "word": "I"},
        {"start": 0.6, "end": 0.8, "word": "don't"},
    ]
    ops = m._align_word_runs(p_words, w_words)
    replaces = [(p, w) for tag, p, w in ops if tag == "replace"]
    assert len(replaces) == 1
    p_slice, w_slice = replaces[0]
    assert [x["word"] for x in p_slice] == ["Sorry,", "Kate,"]
    assert [x["word"] for x in w_slice] == ["Fuck", "it"]
    # The agreeing tail is an 'equal' op, never a replace.
    assert any(tag == "equal" for tag, _, _ in ops)


def test_decide_run_swap_accepts_sorry_kate():
    # MMS 'farki d' backs 'fuck it' (~50) far more than 'sorry kate' (~31).
    swap, sc = m.decide_run_swap("Sorry, Kate,", "Fuck it", "farki d", "farki d")
    assert swap is True
    assert sc["wf"] >= sc["pf"] + m.RUN_ARB_MARGIN


def test_decide_run_swap_rejects_number_word_thirty():
    # Parakeet '30' normalizes to 'thirty' and matches MMS 'thirty' perfectly;
    # Whisper hallucinated 'dirty'. Must NOT swap.
    swap, sc = m.decide_run_swap("30", "dirty", "thirty", "thirty")
    assert swap is False


def test_decide_run_swap_rejects_when_parakeet_also_matches():
    # Both sides match the acoustics well -> ambiguous -> keep Parakeet.
    swap, _ = m.decide_run_swap("screen", "stream", "screen", "screen")
    assert swap is False


def test_decide_run_swap_rejects_empty_mms():
    swap, _ = m.decide_run_swap("sorry kate", "fuck it", "", "")
    assert swap is False


def test_apply_run_swaps_preserves_reading_order_not_timestamp():
    # The swapped Whisper word 'X' carries a LATE timestamp (0.9) that would sort
    # after 'c' (0.4). The rebuild must keep alignment order [a, X, c], never the
    # timestamp-sorted [a, c, X] (which scrambled real '...sir' segments).
    ops = [
        (
            "equal",
            [{"start": 0.0, "end": 0.1, "word": "a"}],
            [{"start": 0.0, "end": 0.1, "word": "a"}],
        ),
        (
            "replace",
            [{"start": 0.2, "end": 0.3, "word": "b"}],
            [{"start": 0.9, "end": 1.0, "word": "X"}],
        ),
        (
            "equal",
            [{"start": 0.4, "end": 0.5, "word": "c"}],
            [{"start": 0.4, "end": 0.5, "word": "c"}],
        ),
    ]
    out = m._apply_run_swaps(ops, {1: ops[1][2]})
    assert [w["word"] for w in out] == ["a", "X", "c"]


def test_apply_run_swaps_drops_inserts_keeps_unswapped():
    ops = [
        (
            "equal",
            [{"start": 0.0, "end": 0.1, "word": "a"}],
            [{"start": 0.0, "end": 0.1, "word": "a"}],
        ),
        ("insert", [], [{"start": 0.1, "end": 0.2, "word": "ins"}]),
        (
            "replace",
            [{"start": 0.2, "end": 0.3, "word": "b"}],
            [{"start": 0.2, "end": 0.3, "word": "X"}],
        ),
    ]
    out = m._apply_run_swaps(ops, {})  # nothing approved for swap
    assert [w["word"] for w in out] == ["a", "b"]


def test_decide_run_swap_requires_focused_view_not_just_global():
    # Global slice strongly backs Whisper ('the little') but the focused decode
    # is garbage ('gaaa') -> a strong global slice alone must NOT trigger a swap.
    swap, sc = m.decide_run_swap("like", "the little", "gaaa", "the little")
    assert sc["wg"] >= m.RUN_ARB_WHISPER_FLOOR  # global view would have passed
    assert sc["wf"] < m.RUN_ARB_WHISPER_FLOOR  # but focused view does not
    assert swap is False


def _mk_ctc(chars_at):
    return [{"char": c, "t": t} for c, t in chars_at]


def test_arbitrate_substitution_runs_swaps_sorry_kate():
    sub = {
        "start": 0.0,
        "end": 1.4,
        "text": "Sorry, Kate, I don't",
        "final_text": "Sorry, Kate, I don't",
        "final_source": "parakeet",
        "kind": "speech",
        "wp_agreement": 82,
        "words": [
            {"start": 0.0, "end": 0.2, "word": "Sorry,"},
            {"start": 0.2, "end": 0.4, "word": "Kate,"},
            {"start": 0.6, "end": 0.8, "word": "I"},
            {"start": 0.8, "end": 1.0, "word": "don't"},
        ],
    }
    subsegs = [sub]
    whisper_segs = [
        {
            "start": 0.0,
            "end": 0.4,
            "text": "Fuck it",
            "words": [
                {"start": 0.0, "end": 0.25, "word": "Fuck", "score": 0.8},
                {"start": 0.25, "end": 0.4, "word": "it", "score": 0.8},
            ],
        },
        {
            "start": 0.6,
            "end": 1.0,
            "text": "I don't",
            "words": [
                {"start": 0.6, "end": 0.8, "word": "I", "score": 0.8},
                {"start": 0.8, "end": 1.0, "word": "don't", "score": 0.8},
            ],
        },
    ]
    single_owners = [0, 0]
    multi = set()
    # Global slice across the run span resolves to 'farki d'.
    ctc = _mk_ctc(
        [
            ("f", 0.05),
            ("a", 0.1),
            ("r", 0.15),
            ("k", 0.2),
            ("i", 0.25),
            ("|", 0.3),
            ("d", 0.35),
        ]
    )
    m.arbitrate_substitution_runs(
        subsegs,
        whisper_segs,
        single_owners,
        multi,
        ctc,
        decode_fn=lambda lo, hi: "farki d",
    )
    assert [w["word"] for w in sub["words"]] == ["Fuck", "it", "I", "don't"]
    assert sub["final_text"] == "Fuck it I don't"
    assert sub["final_source"].endswith("+run_arb")
    assert sub["run_swaps"] and sub["run_swaps"][0]["w_run"] == "Fuck it"


def test_arbitrate_substitution_runs_keeps_when_mms_backs_parakeet():
    sub = {
        "start": 0.0,
        "end": 0.4,
        "text": "30",
        "final_text": "30",
        "final_source": "parakeet",
        "kind": "speech",
        "wp_agreement": 90,
        "words": [{"start": 0.0, "end": 0.4, "word": "30"}],
    }
    whisper_segs = [
        {
            "start": 0.0,
            "end": 0.4,
            "text": "dirty",
            "words": [{"start": 0.0, "end": 0.4, "word": "dirty", "score": 0.8}],
        }
    ]
    ctc = _mk_ctc(
        [("t", 0.05), ("h", 0.1), ("i", 0.15), ("r", 0.2), ("t", 0.3), ("y", 0.35)]
    )
    m.arbitrate_substitution_runs(
        [sub], whisper_segs, [0], set(), ctc, decode_fn=lambda lo, hi: "thirty"
    )
    assert [w["word"] for w in sub["words"]] == ["30"]
    assert "run_arb" not in sub["final_source"]
    assert "run_swaps" not in sub


def test_arbitrate_substitution_runs_skips_outside_wp_band():
    sub = {
        "start": 0.0,
        "end": 0.4,
        "text": "Sorry,",
        "final_text": "Sorry,",
        "final_source": "parakeet",
        "kind": "speech",
        "wp_agreement": 60,
        "words": [{"start": 0.0, "end": 0.4, "word": "Sorry,"}],
    }
    whisper_segs = [
        {
            "start": 0.0,
            "end": 0.4,
            "text": "Fuck",
            "words": [{"start": 0.0, "end": 0.4, "word": "Fuck", "score": 0.8}],
        }
    ]
    called = []
    m.arbitrate_substitution_runs(
        [sub],
        whisper_segs,
        [0],
        set(),
        [],
        decode_fn=lambda lo, hi: called.append((lo, hi)) or "farki",
    )
    assert called == []  # wp < PHRASE_ARB_WP_MAX -> never reached decode
    assert "run_arb" not in sub["final_source"]


def test_transfer_punctuation_base_check_accepts_run_arb():
    # A run-arbitrated sub must still be eligible for punctuation transfer
    # (base resolves to 'parakeet').
    assert m._base_source("parakeet+run_arb") in m.PARAKEET_FINAL_SOURCES
    assert m._base_source("consensus_pw+word_swap+run_arb") in m.PARAKEET_FINAL_SOURCES


def test_render_debug_md_shows_run_swap():
    segs = [
        {
            "start": 503.04,
            "end": 505.76,
            "text": "Fuck it I don't care about the timer!",
            "source": "parakeet+run_arb",
            "kind": "speech",
            "parakeet_text": "Sorry, Kate, I don't care about the timer.",
            "mms_text": "farki d don't care about the time or",
            "whisper_text": "Fuck it! I don't care about the timer!",
            "parakeet_score": 80,
            "wp_agreement": 82,
            "words": [],
            "run_swaps": [
                {
                    "p_run": "Sorry, Kate,",
                    "w_run": "Fuck it",
                    "focused_mms": "farki d",
                    "global_mms": "farki d",
                    "pf": 31,
                    "wf": 50,
                    "pg": 31,
                    "wg": 50,
                },
            ],
        }
    ]
    md = m.render_debug_md(segs)
    assert "[P~]" in md
    assert "**Run swap:**" in md
    assert "`Sorry, Kate,` → `Fuck it`" in md
