from __future__ import annotations

import re

DISAGREEMENT_THRESHOLD = 65
SILENCE_SPLIT_S = 1.0
MMS_WINDOW_PAD_START_S = 0.1
MMS_WINDOW_PAD_END_S = 0.4
# A Parakeet subseg claims a Whisper segment when their overlap is >= 0.5s
# absolute OR >= this fraction of the shorter of the two spans. The absolute
# floor alone starves sub-1s subsegs (e.g. "Yeah.", 0.4s) which can never
# accumulate 0.5s of overlap, mis-routing their Whisper words to a neighbor.
CLAIM_OVERLAP_FRACTION = 0.6
# Forward-only rigid re-anchor of noise-anchored Whisper drift. Whisper's
# acoustic model anchors text to non-speech noise that PRECEDES the speech, so
# the error is always early -> search and shift FORWARD only. Fire only on a
# strong, long-enough match (the hallucination gate).
REANCHOR_WINDOW_S = 60.0
REANCHOR_MIN_SCORE = 80
REANCHOR_MIN_LEN = 12
REANCHOR_MAX_RUN_GAP_S = 2.0
PG_CONSENSUS_THRESHOLD = 70

GAP_FILL_THRESHOLD_S = 1.0
GAP_FILL_VOCAL_MIN_LEN = 10
GAP_FILL_VOCAL_MAX_DISTINCT = 4
GAP_FILL_SMEAR_THRESHOLD = 60
GAP_FILL_VOCAL_MIN_CHARS = 3
FOCUSED_MMS_PAD_START_S = 0.0
FOCUSED_MMS_PAD_END_S = 0.0

WORD_REFINE_AGREEMENT_THRESHOLD = 80
WORD_REFINE_SWAP_MARGIN = 10
WORD_REFINE_MMS_PAD_S = 0.15
WORD_REFINE_OVERLAP_FRACTION = 0.4

# Phrase-level arbitration: when Parakeet and Whisper disagree a lot on a subseg
# (low wp_agreement) yet it ended Parakeet-final (escaped flagging), pick the
# full text the MMS phrase supports — comparing FULL Parakeet vs FULL Whisper
# text against BOTH the global MMS slice AND a wide focused decode. Conservative
# (high precision): flip only when BOTH MMS views favor Whisper by the margin
# AND the better whisper ratio clears the floor. Per-word tight clips can't do
# this (they compress 'shadow'->'sad'); the phrase does.
PHRASE_ARB_WP_MAX = 70
PHRASE_ARB_PAD_S = 0.4
PHRASE_ARB_MARGIN = 5
PHRASE_ARB_FLOOR = 55

# Localized substitution-run arbitration. Operates only on the high-agreement
# band [PHRASE_ARB_WP_MAX, 100) that phrase_arb skips. MMS phonetic text is
# rough, so the decision is margin-driven (Whisper must beat Parakeet against
# BOTH the focused and global MMS views) plus a low Whisper floor and a
# Parakeet "clearly mismatches" ceiling — NOT a high absolute floor, which
# would reject genuine fixes (e.g. 'fuck it' only scores ~50 vs MMS 'farki d').
RUN_ARB_MARGIN = 12
RUN_ARB_WHISPER_FLOOR = 45
RUN_ARB_PARAKEET_CEIL = 50
RUN_ARB_PAD_S = 0.2
RUN_ARB_MIN_MMS_CHARS = 3
RUN_ARB_MAX_RUN_WORDS = 5
RUN_ARB_MIN_WRUN_ALPHA = 3
FILLER_WORDS = {
    "um",
    "umm",
    "uh",
    "uhh",
    "uhm",
    "mm",
    "mmm",
    "mhm",
    "hmm",
    "hm",
    "er",
    "err",
    "ah",
    "ahh",
    "oh",
    "eh",
}

_INFLECT = None

WHISPER_WORD_MIN_SCORE = 0.3
WHISPER_WORD_MAX_DURATION_S = 1.5

WHISPER_SMEAR_TOLERANCE_S = 10.0
WHISPER_SMEAR_MATCH_THRESHOLD = 55
# A time-misplaced Whisper segment and its true Parakeet subseg are the SAME
# utterance, hence comparable in length. Without this, partial_ratio of a long
# Whisper text against a tiny Parakeet word gives a false 100 ('hi' is a
# substring of 'chili'), letting a 2-char subseg claim an unrelated sentence.
WHISPER_SMEAR_LEN_RATIO = 0.4

# Snap final word boundaries to the frame-accurate MMS stream (Issue 3).
# Gated ON only after the golden harness shows it beats the Parakeet baseline.
SNAP_WORDS_TO_MMS = True

PARAKEET_FINAL_SOURCES = {
    "parakeet",
    "parakeet_kept",
    "parakeet_kept_after_disagreement",
    "parakeet_sample_skipped",
    "focused_mms_resolved",
    "consensus_pw",
}

MMS_MODEL_ID = "facebook/mms-1b-all"
MMS_LANG_ADAPTER = "eng"


def fmt_mmss(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def normalize_for_compare(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _word_normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _normalize_numbers(text: str) -> str:
    """Spell out pure-digit tokens so they compare against MMS/Whisper, which
    always write numbers as words ('30' -> 'thirty'). Other tokens pass
    through. Lowercased; hyphens/punctuation flattened to spaces."""
    global _INFLECT
    out: list[str] = []
    for tok in text.split():
        if tok.isdigit():
            if _INFLECT is None:
                import inflect

                _INFLECT = inflect.engine()
            words = _INFLECT.number_to_words(int(tok), andword="")
            out.append(re.sub(r"[^a-z0-9]+", " ", words.lower()))
        else:
            out.append(tok)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def _numbers_equivalent(a: str, b: str) -> bool:
    """True when two run strings are equal once digits are spelled out — guards
    the '11' vs 'eleven' normalization-only 'disagreement'."""
    return normalize_for_compare(_normalize_numbers(a)) == normalize_for_compare(
        _normalize_numbers(b)
    )


def _is_filler_run(text: str) -> bool:
    """True when every token of the run is a vocal filler (um/uh/...). These
    are low-value to correct and risky, so the run is skipped."""
    toks = normalize_for_compare(text).split()
    return bool(toks) and all(t in FILLER_WORDS for t in toks)


def _base_source(source: str) -> str:
    """Strip the additive merge-stage suffixes to recover the underlying source
    (so membership tests against PARAKEET_FINAL_SOURCES still work after a
    word swap and/or a run swap have been appended)."""
    return source.replace("+word_swap", "").replace("+run_arb", "")


def _align_word_runs(p_words: list[dict], w_words: list[dict]) -> list[tuple]:
    """Align the Parakeet and Whisper word lists on normalized tokens and
    return difflib opcodes carried as (tag, parakeet_word_slice,
    whisper_word_slice). 'replace' ops are the substitution runs we arbitrate;
    'equal' ops are the agreeing remainder (never touched)."""
    from difflib import SequenceMatcher

    pn = [_word_normalize(str(w.get("word", ""))) for w in p_words]
    wn = [_word_normalize(str(w.get("word", ""))) for w in w_words]
    sm = SequenceMatcher(a=pn, b=wn, autojunk=False)
    ops: list[tuple] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ops.append((tag, p_words[i1:i2], w_words[j1:j2]))
    return ops


def decide_run_swap(
    p_run: str,
    w_run: str,
    focused_mms: str,
    global_mms: str,
    margin: int = RUN_ARB_MARGIN,
    whisper_floor: int = RUN_ARB_WHISPER_FLOOR,
    parakeet_ceil: int = RUN_ARB_PARAKEET_CEIL,
) -> tuple[bool, dict]:
    """Decide whether to replace the Parakeet run with the Whisper run, judged
    against the focused (tight) and global MMS decodes of the run's audio span.
    Conservative: swap only when Whisper beats Parakeet by `margin` against
    BOTH MMS views, Whisper plausibly matches in BOTH views (each >=
    whisper_floor — a strong global slice alone is not enough; the focused
    decode must agree too), and Parakeet clearly does NOT match (<=
    parakeet_ceil). Digits are spelled out first so '30' compares as 'thirty'."""
    from rapidfuzz import fuzz

    pn = normalize_for_compare(_normalize_numbers(p_run))
    wn = normalize_for_compare(_normalize_numbers(w_run))
    fn = normalize_for_compare(focused_mms)
    gn = normalize_for_compare(global_mms)
    scores = {"pf": 0, "wf": 0, "pg": 0, "wg": 0}
    if not pn or not wn or not fn or not gn:
        return False, scores
    pf = int(fuzz.partial_ratio(pn, fn))
    wf = int(fuzz.partial_ratio(wn, fn))
    pg = int(fuzz.partial_ratio(pn, gn))
    wg = int(fuzz.partial_ratio(wn, gn))
    scores = {"pf": pf, "wf": wf, "pg": pg, "wg": wg}
    both_favor = (wf >= pf + margin) and (wg >= pg + margin)
    whisper_plausible = min(wf, wg) >= whisper_floor
    parakeet_mismatch = min(pf, pg) <= parakeet_ceil
    return (both_favor and whisper_plausible and parakeet_mismatch), scores


def _apply_run_swaps(ops: list[tuple], swapped: dict[int, list]) -> list[dict]:
    """Rebuild a word list from alignment ops, substituting the Whisper slice
    for each run index in `swapped`. Returns words in ALIGNMENT (reading) order.
    It must NOT re-sort by timestamp: earlier extension passes (prefix/suffix)
    can leave non-monotonic word timings, and sorting by start would scramble
    the sentence. Insertions (Whisper-only words) are dropped; deletions and
    un-swapped replaces keep the Parakeet words."""
    new_words: list[dict] = []
    for op_i, (tag, p_slice, w_slice) in enumerate(ops):
        if tag == "insert":
            continue
        if tag == "replace" and op_i in swapped:
            new_words.extend(swapped[op_i])
        else:
            new_words.extend(p_slice)
    return new_words


def arbitrate_substitution_runs(
    subsegs: list[dict],
    whisper_segs: list[dict],
    single_owners: list,
    multi_overlap: set,
    ctc_chars: list[dict],
    decode_fn,
    pad_s: float = RUN_ARB_PAD_S,
) -> None:
    """Correct localized wrong-word runs in Parakeet-final subsegs of the
    high-agreement band [PHRASE_ARB_WP_MAX, 100): align Parakeet vs owned
    Whisper words, and for each contiguous substitution run, swap to Whisper
    iff focused + global MMS agree it is the better acoustic match (see
    decide_run_swap). `decode_fn(lo, hi) -> str` returns the focused MMS decode
    for a span (injected so the GPU dependency stays out of unit tests).
    Mutates subsegs in place; records sub['run_swaps'] and appends
    '+run_arb' to final_source. Insertions are not handled (substitutions
    only); deletions keep the Parakeet word."""
    for sub_idx, sub in enumerate(subsegs):
        if sub.get("kind", "speech") != "speech":
            continue
        if _base_source(sub.get("final_source", "")) not in PARAKEET_FINAL_SOURCES:
            continue
        wp = sub.get("wp_agreement")
        if wp is None or wp < PHRASE_ARB_WP_MAX or wp >= 100:
            continue
        p_words = sub.get("words") or []
        if not p_words:
            continue
        w_words = owned_whisper_words(
            sub_idx,
            whisper_segs,
            single_owners,
            multi_overlap,
            float(sub["start"]),
            float(sub["end"]),
        )
        if not w_words:
            continue

        ops = _align_word_runs(p_words, w_words)
        if not any(tag == "replace" for tag, _, _ in ops):
            continue

        swapped: dict[int, list] = {}
        run_swaps: list[dict] = []
        for op_i, (tag, p_slice, w_slice) in enumerate(ops):
            if tag != "replace" or not p_slice or not w_slice:
                continue
            if (
                len(p_slice) > RUN_ARB_MAX_RUN_WORDS
                or len(w_slice) > RUN_ARB_MAX_RUN_WORDS
            ):
                continue
            p_run = " ".join(str(w.get("word", "")) for w in p_slice).strip()
            w_run = " ".join(str(w.get("word", "")) for w in w_slice).strip()
            if _is_filler_run(p_run) or _numbers_equivalent(p_run, w_run):
                continue
            if len(re.sub(r"[^a-z0-9]", "", w_run.lower())) < RUN_ARB_MIN_WRUN_ALPHA:
                continue
            span = p_slice + w_slice
            lo = min(float(x["start"]) for x in span) - pad_s
            hi = max(float(x["end"]) for x in span) + pad_s
            focused = decode_fn(lo, hi) or ""
            if len(re.sub(r"[^a-z0-9]", "", focused.lower())) < RUN_ARB_MIN_MMS_CHARS:
                continue
            global_mms = build_mms_window(ctc_chars, lo, hi)
            do_swap, scores = decide_run_swap(p_run, w_run, focused, global_mms)
            if do_swap:
                swapped[op_i] = w_slice
                run_swaps.append(
                    {
                        "p_run": p_run,
                        "w_run": w_run,
                        "focused_mms": focused,
                        "global_mms": global_mms,
                        **scores,
                    }
                )

        if not swapped:
            continue

        new_words = _apply_run_swaps(ops, swapped)
        sub["words"] = new_words
        sub["final_text"] = " ".join(str(w.get("word", "")) for w in new_words).strip()
        sub["run_swaps"] = run_swaps
        sub["final_source"] = sub.get("final_source", "parakeet") + "+run_arb"


def _inplace_match_score(ws: dict, sub_sorted: list[dict], sub_norm: list[str]) -> int:
    """Best partial_ratio of a Whisper segment's text against the Parakeet
    subsegs it currently time-overlaps. Low score => the Whisper text is
    anchored where its content does not exist (noise-anchored drift)."""
    from rapidfuzz import fuzz

    wn = normalize_for_compare(ws.get("text", "") or "")
    if not wn:
        return 100
    lo = float(ws["start"])
    hi = float(ws["end"])
    best = 0
    for i, s in enumerate(sub_sorted):
        if float(s["end"]) < lo or float(s["start"]) > hi:
            continue
        if sub_norm[i]:
            best = max(best, int(fuzz.partial_ratio(wn, sub_norm[i])))
    return best


def _shift_segment(seg: dict, offset: float) -> None:
    """Rigidly translate a Whisper segment (and its word timings) by `offset`
    seconds. Preserves internal word-to-word spacing — the block was correctly
    transcribed, only mis-anchored to noise."""
    seg["start"] = float(seg["start"]) + offset
    seg["end"] = float(seg["end"]) + offset
    for w in seg.get("words") or []:
        if w.get("start") is not None:
            w["start"] = float(w["start"]) + offset
        if w.get("end") is not None:
            w["end"] = float(w["end"]) + offset


def reanchor_whisper_segments(
    whisper_data: dict,
    parakeet_data: dict,
    window_s: float = REANCHOR_WINDOW_S,
    min_score: int = REANCHOR_MIN_SCORE,
    min_len: int = REANCHOR_MIN_LEN,
    max_run_gap_s: float = REANCHOR_MAX_RUN_GAP_S,
) -> dict:
    """Detect runs of Whisper segments anchored to noise (their text does not
    match the Parakeet content where they sit) and rigidly shift each run
    FORWARD onto the Parakeet span its text matches. Forward-only: noise
    precedes speech, so Whisper drift is always early. Mutates the input
    segment dicts in place (callers retaining a reference to whisper_data
    see the shifted timings) and returns the same dict with segments sorted;
    the caller re-derives VAD afterward so the noise location loses (and the
    true location gains) speech activity."""
    from rapidfuzz import fuzz

    segs = sorted(whisper_data.get("segments", []), key=lambda s: float(s["start"]))
    subsegs = resegment_parakeet(parakeet_data, SILENCE_SPLIT_S, MMS_WINDOW_PAD_END_S)
    sub_sorted = sorted(subsegs, key=lambda s: float(s["start"]))
    sub_starts = [float(s["start"]) for s in sub_sorted]
    sub_norm = [normalize_for_compare(s.get("text", "") or "") for s in sub_sorted]

    drifted = sorted(
        i
        for i, ws in enumerate(segs)
        if normalize_for_compare(ws.get("text", "") or "")
        and _inplace_match_score(ws, sub_sorted, sub_norm) < min_score
    )

    runs: list[list[int]] = []
    di = 0
    while di < len(drifted):
        run = [drifted[di]]
        di += 1
        while (
            di < len(drifted)
            and drifted[di] == run[-1] + 1
            and float(segs[drifted[di]]["start"]) - float(segs[run[-1]]["end"])
            <= max_run_gap_s
        ):
            run.append(drifted[di])
            di += 1
        runs.append(run)

    for run in runs:
        run_segs = [segs[k] for k in run]
        run_text = normalize_for_compare(
            " ".join(s.get("text", "") or "" for s in run_segs)
        )
        if len(run_text.replace(" ", "")) < min_len:
            continue
        run_start = float(run_segs[0]["start"])
        target = len(run_text)
        best_score = -1
        best_offset = None
        for i in range(len(sub_sorted)):
            ss = sub_starts[i]
            if ss <= run_start:
                continue
            if ss - run_start > window_s:
                break
            parts: list[str] = []
            total = 0
            j = i
            while j < len(sub_norm) and total < target * 1.5:
                if sub_norm[j]:
                    parts.append(sub_norm[j])
                    total += len(sub_norm[j])
                j += 1
            cand = " ".join(parts)
            if not cand:
                continue
            score = int(fuzz.partial_ratio(run_text, cand))
            if score > best_score:
                best_score = score
                best_offset = ss - run_start
        if best_score >= min_score and best_offset is not None and best_offset > 0:
            for s in run_segs:
                _shift_segment(s, best_offset)

    result = dict(whisper_data)
    result["segments"] = segs
    return result


def _whisper_word_reliable(w: dict) -> bool:
    """Filter out WhisperX word-level alignments that are clearly bad —
    score below threshold (alignment failed) or duration > 1.5s (a single
    word can't reasonably last that long; WhisperX stretches words across
    unscored regions)."""
    try:
        score = float(w.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    if score < WHISPER_WORD_MIN_SCORE:
        return False
    try:
        dur = float(w.get("end", 0.0)) - float(w.get("start", 0.0))
    except (TypeError, ValueError):
        dur = 0.0
    if dur > WHISPER_WORD_MAX_DURATION_S:
        return False
    return True


def _binary_lb(arr: list[dict], t: float) -> int:
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid]["t"] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def build_mms_window(ctc_chars: list[dict], lo_t: float, hi_t: float) -> str:
    parts: list[str] = []
    last_was_space = True
    lo_i = _binary_lb(ctc_chars, lo_t)
    for i in range(lo_i, len(ctc_chars)):
        c = ctc_chars[i]
        if c["t"] > hi_t:
            break
        ch = c["char"]
        if ch == "|":
            if not last_was_space and parts:
                parts.append(" ")
                last_was_space = True
        else:
            parts.append(ch.lower())
            last_was_space = False
    return "".join(parts).strip()


def compute_neighbor_aware_window(
    subsegs: list[dict], i: int, pad_start: float, pad_end: float
) -> tuple[float, float]:
    """Compute the MMS window [lo, hi] for subseg i with neighbor-aware pad
    capping. For clean gaps between adjacent subsegs, pads are capped so
    windows don't overlap into the neighbor's audio. For overlapping Parakeet
    subsegs, pads are NOT capped."""
    sub = subsegs[i]
    desired_lo = sub["start"] - pad_start
    desired_hi = sub["end"] + pad_end

    if i > 0:
        prev = subsegs[i - 1]
        if prev["end"] <= sub["start"]:
            lo = max(desired_lo, prev["end"])
        else:
            lo = desired_lo
    else:
        lo = max(0.0, desired_lo)

    if i + 1 < len(subsegs):
        nxt = subsegs[i + 1]
        if sub["end"] <= nxt["start"]:
            hi = min(desired_hi, nxt["start"])
        else:
            hi = desired_hi
    else:
        hi = desired_hi
    return lo, hi


def _pack_subseg(words: list[dict]) -> dict:
    return {
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "text": " ".join(w["word"] for w in words).strip(),
        "words": words,
    }


def _find_prev_subseg_idx(subsegs: list[dict], t: float) -> int | None:
    best: int | None = None
    best_end = float("-inf")
    for i, s in enumerate(subsegs):
        end = float(s.get("end", 0.0))
        if end <= t and end > best_end:
            best = i
            best_end = end
    return best


def _find_next_subseg_idx(subsegs: list[dict], t: float) -> int | None:
    best: int | None = None
    best_start = float("inf")
    for i, s in enumerate(subsegs):
        start = float(s.get("start", 0.0))
        if start >= t and start < best_start:
            best = i
            best_start = start
    return best


def run_mms_focused(
    audio,
    sr: int,
    lo_s: float,
    hi_s: float,
    processor,
    model,
    pad_id: int,
    word_delim_id: int,
    id_to_char: dict,
) -> str:
    """Run MMS-CTC on a tight audio span and return the decoded text. Used
    for per-subseg re-arbitration, gap-fill, and per-word refine. Tight bounds
    (no padding) decode better than the global 30s-chunk pass."""
    import torch

    a = max(0, int(lo_s * sr))
    b = min(audio.shape[0], int(hi_s * sr))
    clip = audio[a:b]
    if clip.shape[0] < 1600:
        return ""
    inputs = processor(clip, sampling_rate=sr, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    parts: list[str] = []
    prev = -1
    last_was_space = True
    for idx in ids:
        idx_int = int(idx)
        if idx_int == prev:
            continue
        prev = idx_int
        if idx_int == pad_id:
            continue
        ch = id_to_char.get(idx_int)
        if not ch:
            continue
        if idx_int == word_delim_id:
            if not last_was_space and parts:
                parts.append(" ")
                last_was_space = True
        else:
            parts.append(ch.lower())
            last_was_space = False
    return "".join(parts).strip()


def resegment_parakeet(
    parakeet_data: dict, silence_threshold_s: float, end_pad_s: float
) -> list[dict]:
    """Split Parakeet segments wherever the EFFECTIVE gap between consecutive
    words exceeds silence_threshold_s. Effective gap accounts for Parakeet's
    systematic truncation of word end-times — each word is treated as ending
    at (reported_end + end_pad_s) when measuring silence."""
    out: list[dict] = []
    for seg in parakeet_data["segments"]:
        words = seg.get("words") or []
        if not words:
            if seg.get("text", "").strip():
                out.append(
                    {
                        "start": float(seg["start"]),
                        "end": float(seg["end"]),
                        "text": seg["text"].strip(),
                        "words": [],
                    }
                )
            continue
        current: list[dict] = []
        for w in words:
            if current:
                effective_prev_end = float(current[-1]["end"]) + end_pad_s
                gap = float(w["start"]) - effective_prev_end
                if gap > silence_threshold_s:
                    out.append(_pack_subseg(current))
                    current = []
            current.append(w)
        if current:
            out.append(_pack_subseg(current))
    return out


def dedupe_overlapping_subsegs(subsegs: list[dict]) -> list[dict]:
    """Drop subsegs whose normalized text is contained inside an overlapping
    subseg's text (Parakeet chunk-overlap dedup occasionally leaves these).
    Keep the longer."""
    sorted_subs = sorted(subsegs, key=lambda s: (float(s["start"]), float(s["end"])))
    to_remove: set = set()
    for i in range(len(sorted_subs)):
        if i in to_remove:
            continue
        a = sorted_subs[i]
        a_hi = float(a["end"])
        a_norm = normalize_for_compare(a.get("text", ""))
        if not a_norm:
            continue
        for j in range(i + 1, len(sorted_subs)):
            if j in to_remove:
                continue
            b = sorted_subs[j]
            b_lo = float(b["start"])
            if b_lo >= a_hi:
                break
            b_norm = normalize_for_compare(b.get("text", ""))
            if not b_norm:
                continue
            if b_norm and b_norm in a_norm and len(b_norm) < len(a_norm):
                to_remove.add(j)
            elif a_norm in b_norm and len(a_norm) < len(b_norm):
                to_remove.add(i)
                break
    return [s for k, s in enumerate(sorted_subs) if k not in to_remove]


def assign_whisper_owners(
    subsegs: list[dict],
    whisper_segs: list[dict],
    tolerance_s: float,
    content_threshold: int,
) -> list:
    """Assign each Whisper segment to ONE owner Parakeet subseg index, by
    time-overlap (>= 0.5s absolute OR >= CLAIM_OVERLAP_FRACTION of the shorter
    span — so a sub-1s subseg can still claim; multi-overlap broken by content
    dominance) or, for non-overlapping segments, by content match within
    tolerance_s (smear)."""
    from rapidfuzz import fuzz

    p_subs = [(i, s) for i, s in enumerate(subsegs) if s.get("words")]
    single_owners: list = [None] * len(whisper_segs)
    multi_overlap_segs: set = set()
    if not p_subs:
        return single_owners, multi_overlap_segs

    min_overlap_s = 0.5
    content_dominance_margin = 30
    for wi, ws in enumerate(whisper_segs):
        ws_lo = float(ws["start"])
        ws_hi = float(ws["end"])
        ws_mid = (ws_lo + ws_hi) / 2
        ws_text = (ws.get("text") or "").strip()
        w_norm = normalize_for_compare(ws_text)
        if not w_norm:
            continue

        time_claimants = []
        for pi, ps in p_subs:
            ps_lo = float(ps["start"])
            ps_hi = float(ps["end"])
            overlap = max(0.0, min(ps_hi, ws_hi) - max(ps_lo, ws_lo))
            ps_dur = max(1e-6, ps_hi - ps_lo)
            ws_dur = max(1e-6, ws_hi - ws_lo)
            frac_bar = CLAIM_OVERLAP_FRACTION * min(ps_dur, ws_dur)
            if overlap >= min_overlap_s or overlap >= frac_bar:
                p_norm_c = normalize_for_compare(ps.get("text", "") or "")
                content = int(fuzz.partial_ratio(w_norm, p_norm_c)) if p_norm_c else 0
                time_claimants.append((pi, overlap, content))

        if len(time_claimants) == 1:
            single_owners[wi] = time_claimants[0][0]
            continue
        if len(time_claimants) >= 2:
            time_claimants.sort(key=lambda x: x[2], reverse=True)
            top_pi, _ot, top_content = time_claimants[0]
            _pi2, _o2, second_content = time_claimants[1]
            if top_content - second_content >= content_dominance_margin:
                single_owners[wi] = top_pi
            else:
                multi_overlap_segs.add(wi)
            continue

        best_pi = None
        best_score = 0
        for pi, ps in p_subs:
            ps_mid = (float(ps["start"]) + float(ps["end"])) / 2
            if abs(ws_mid - ps_mid) > tolerance_s:
                continue
            p_norm = normalize_for_compare(ps.get("text", "") or "")
            if not p_norm:
                continue
            lp, lw = len(p_norm), len(w_norm)
            if min(lp, lw) / max(lp, lw) < WHISPER_SMEAR_LEN_RATIO:
                continue
            score = int(fuzz.partial_ratio(w_norm, p_norm))
            if score > best_score:
                best_score = score
                best_pi = pi

        if best_pi is None or best_score < content_threshold:
            continue
        single_owners[wi] = best_pi

        sub = subsegs[best_pi]
        offset = float(sub["start"]) - ws_lo
        shifted: list[dict] = []
        for ww in ws.get("words", []) or []:
            if not _whisper_word_reliable(ww):
                continue
            try:
                w_lo = float(ww.get("start", ws_lo))
                w_hi = float(ww.get("end", ws_hi))
            except (TypeError, ValueError):
                continue
            word_text = str(ww.get("word", "")).strip()
            if not word_text:
                continue
            shifted.append(
                {
                    "start": w_lo + offset,
                    "end": w_hi + offset,
                    "word": word_text,
                }
            )
        smears = sub.setdefault("whisper_smear_attributions", [])
        smears.append(
            {
                "from_start": ws_lo,
                "from_end": ws_hi,
                "text": ws_text,
                "content_score": best_score,
                "offset_s": offset,
                "shifted_words": shifted,
            }
        )

    return single_owners, multi_overlap_segs


def whisper_text_for_subseg(
    sub: dict,
    subseg_idx: int,
    whisper_segs: list[dict],
    single_owners: list,
    multi_overlap_segs: set,
) -> str:
    sub_lo = float(sub["start"])
    sub_hi = float(sub["end"])
    parts: list[str] = []
    for wi, ws in enumerate(whisper_segs):
        if single_owners[wi] == subseg_idx:
            txt = (ws.get("text") or "").strip()
            if txt:
                parts.append(txt)
            continue
        if wi in multi_overlap_segs:
            ws_lo = float(ws["start"])
            ws_hi = float(ws["end"])
            words = ws.get("words") or []
            if words:
                inside: list[str] = []
                for w in words:
                    if not _whisper_word_reliable(w):
                        continue
                    w_lo = float(w.get("start", ws_lo))
                    w_hi = float(w.get("end", ws_hi))
                    if w_hi > sub_lo and w_lo < sub_hi:
                        word_text = str(w.get("word", "")).strip()
                        if word_text:
                            inside.append(word_text)
                if inside:
                    parts.append(" ".join(inside))
    return " ".join(parts).strip()


def owned_whisper_words(
    subseg_idx: int,
    whisper_segs: list[dict],
    single_owners: list,
    multi_overlap_segs: set,
    sub_lo: float,
    sub_hi: float,
) -> list[dict]:
    """Reliable Whisper words this subseg owns (mirror of whisper_text_for_subseg
    at the word level). Used when phrase arbitration adopts Whisper's full text,
    so the segment also carries Whisper's word timings instead of Parakeet's."""
    out: list[dict] = []
    for wi, ws in enumerate(whisper_segs):
        owns_full = single_owners[wi] == subseg_idx
        is_multi = wi in multi_overlap_segs
        if not owns_full and not is_multi:
            continue
        for w in ws.get("words") or []:
            if not _whisper_word_reliable(w):
                continue
            w_lo = float(w.get("start", ws["start"]))
            w_hi = float(w.get("end", ws["end"]))
            if is_multi and not (w_hi > sub_lo and w_lo < sub_hi):
                continue
            word_text = str(w.get("word", "")).strip()
            if word_text:
                out.append({"start": w_lo, "end": w_hi, "word": word_text})
    out.sort(key=lambda w: w["start"])
    return out


def gap_absorbed_by_neighbor(
    subsegs: list[dict],
    whisper_segs: list[dict],
    single_owners: list,
    gap_lo: float,
    gap_hi: float,
) -> bool:
    """A gap immediately following a Parakeet subseg is 'absorbed' when one of
    that subseg's owned Whisper segments extends past the subseg's end into the
    gap — the gap audio is the trailing tail of the prior utterance."""
    prev_idx = _find_prev_subseg_idx(subsegs, gap_lo)
    if prev_idx is None:
        return False
    for wi, ws in enumerate(whisper_segs):
        if single_owners[wi] != prev_idx:
            continue
        ws_hi = float(ws["end"])
        if ws_hi >= gap_hi - 0.05:
            return True
    return False


def whisper_text_for_gap(
    gap_lo: float,
    gap_hi: float,
    whisper_segs: list[dict],
    single_owners: list,
    multi_overlap_segs: set,
) -> str:
    parts: list[str] = []
    for wi, ws in enumerate(whisper_segs):
        if single_owners[wi] is not None or wi in multi_overlap_segs:
            continue
        ws_lo = float(ws["start"])
        ws_hi = float(ws["end"])
        if ws_hi < gap_lo or ws_lo > gap_hi:
            continue
        words = ws.get("words") or []
        if words:
            inside: list[str] = []
            for w in words:
                if not _whisper_word_reliable(w):
                    continue
                w_lo = float(w.get("start", ws_lo))
                w_hi = float(w.get("end", ws_hi))
                if w_hi > gap_lo and w_lo < gap_hi:
                    word_text = str(w.get("word", "")).strip()
                    if word_text:
                        inside.append(word_text)
            if inside:
                parts.append(" ".join(inside))
        else:
            txt = (ws.get("text") or "").strip()
            if txt:
                parts.append(txt)
    return " ".join(parts).strip()


def find_continuation_words(
    parakeet_words: list[dict],
    owned_whisper_segs: list[dict],
    ceiling_s: float,
    min_extra_words: int = 2,
    min_alignment_score: int = 70,
) -> list[dict]:
    """Find Whisper words that continue past the last Parakeet word (the tail
    Parakeet cut off). Score parakeet_text against whisper_text[:E] for each
    end position E; everything after the best E is the continuation. Guards:
    min_extra_words and min_alignment_score prevent the head/tail duplication
    bug. Word end is clamped to ceiling_s."""
    from rapidfuzz import fuzz

    all_w_words: list[dict] = []
    for ws in owned_whisper_segs:
        for ww in ws.get("words", []) or []:
            text = str(ww.get("word", "")).strip()
            if not text:
                continue
            try:
                w_lo = float(ww.get("start", ws.get("start", 0.0)))
                w_hi = float(ww.get("end", ws.get("end", 0.0)))
            except (TypeError, ValueError):
                continue
            all_w_words.append({"start": w_lo, "end": w_hi, "word": text})
    if not all_w_words:
        return []

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    p_norm_full = [_norm(str(w.get("word", ""))) for w in parakeet_words]
    p_norm = [w for w in p_norm_full if w]
    w_norm = [_norm(ww["word"]) for ww in all_w_words]
    if not p_norm or not w_norm:
        return []

    p_str = " ".join(p_norm)
    best_score = -1
    best_w_end = len(w_norm)
    for w_end in range(1, len(w_norm) + 1):
        w_substr = " ".join(w_norm[:w_end])
        if not w_substr:
            continue
        score = int(fuzz.ratio(p_str, w_substr))
        if score > best_score:
            best_score = score
            best_w_end = w_end

    extra_words = len(w_norm) - best_w_end
    if best_score < min_alignment_score or extra_words < min_extra_words:
        return []

    trailing = all_w_words[best_w_end:]
    clamped: list[dict] = []
    for tw in trailing:
        if tw["start"] >= ceiling_s:
            break
        end_clamped = min(tw["end"], ceiling_s)
        if end_clamped <= tw["start"]:
            end_clamped = tw["start"] + 0.01
        clamped.append(
            {
                "start": tw["start"],
                "end": end_clamped,
                "word": tw["word"],
            }
        )
    return clamped


def find_prefix_words(
    parakeet_words: list[dict],
    owned_whisper_segs: list[dict],
    floor_s: float,
    min_extra_words: int = 2,
    min_alignment_score: int = 70,
) -> list[dict]:
    """Mirror of find_continuation_words for the START of an utterance. Score
    Parakeet's full text against whisper_text[w_start:] for each start; the
    best position marks where Parakeet begins, everything before is the prefix.
    Guards prevent the leading-word duplication bug. Word start clamped to
    floor_s."""
    from rapidfuzz import fuzz

    all_w_words: list[dict] = []
    for ws in owned_whisper_segs:
        for ww in ws.get("words", []) or []:
            text = str(ww.get("word", "")).strip()
            if not text:
                continue
            try:
                w_lo = float(ww.get("start", ws.get("start", 0.0)))
                w_hi = float(ww.get("end", ws.get("end", 0.0)))
            except (TypeError, ValueError):
                continue
            all_w_words.append({"start": w_lo, "end": w_hi, "word": text})
    if not all_w_words:
        return []

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    p_norm_full = [_norm(str(w.get("word", ""))) for w in parakeet_words]
    p_norm = [w for w in p_norm_full if w]
    w_norm = [_norm(ww["word"]) for ww in all_w_words]
    if not p_norm or not w_norm:
        return []

    p_str = " ".join(p_norm)
    best_score = -1
    best_w_start = 0
    for w_start in range(len(w_norm)):
        w_substr = " ".join(w_norm[w_start:])
        if not w_substr:
            continue
        score = int(fuzz.ratio(p_str, w_substr))
        if score > best_score:
            best_score = score
            best_w_start = w_start

    if best_score < min_alignment_score or best_w_start < min_extra_words:
        return []

    preceding = all_w_words[:best_w_start]
    clamped: list[dict] = []
    for tw in preceding:
        if tw["end"] <= floor_s:
            continue
        start_clamped = max(tw["start"], floor_s)
        if tw["end"] <= start_clamped:
            continue
        clamped.append(
            {
                "start": start_clamped,
                "end": tw["end"],
                "word": tw["word"],
            }
        )
    return clamped


def find_vad_active_gaps(
    subsegs: list[dict],
    vad_intervals: list[dict],
    threshold_s: float,
    prev_end_pad_s: float = 0.0,
) -> list[tuple[float, float]]:
    """Find gaps between adjacent subsegs that exceed threshold_s AND overlap a
    VAD speech-active interval. `prev_end_pad_s` is applied to the previous
    subseg's reported end before measuring the gap (Parakeet truncates word-end
    timestamps; this avoids pulling the trailing audio of the previous word into
    the gap as a spurious vocalization)."""
    if len(subsegs) < 2 or not vad_intervals:
        return []
    sorted_subs = sorted(subsegs, key=lambda s: s["start"])
    gaps: list[tuple[float, float]] = []
    prev_end = sorted_subs[0]["end"] + prev_end_pad_s
    for sub in sorted_subs[1:]:
        gap_lo = prev_end
        gap_hi = sub["start"]
        if gap_hi - gap_lo > threshold_s:
            for v in vad_intervals:
                if v["end"] > gap_lo and v["start"] < gap_hi:
                    gaps.append((gap_lo, gap_hi))
                    break
        prev_end = max(prev_end, sub["end"] + prev_end_pad_s)
    return gaps


def per_word_refine(
    sub: dict,
    whisper_segs: list[dict],
    audio,
    sr: int,
    mms_proc,
    mms_model,
    mms_pad_id: int,
    mms_word_delim_id: int,
    mms_id_to_char: dict,
) -> list[dict]:
    """Word-level refinement: when Parakeet's text was kept but Whisper
    disagrees on one or more single words (e.g. 'screen' vs 'stream', 'Good'
    vs 'Cold'), use focused MMS on each disagreeing word's audio span to
    decide. If MMS phonetically sides with Whisper by WORD_REFINE_SWAP_MARGIN,
    swap that word. Returns the list of swaps performed."""
    from rapidfuzz import fuzz

    words = sub.get("words") or []
    if not words:
        return []
    sub_lo = float(sub["start"])
    sub_hi = float(sub["end"])

    w_words: list[dict] = []
    for ws in whisper_segs:
        if ws["end"] < sub_lo or ws["start"] > sub_hi:
            continue
        for ww in ws.get("words", []):
            if not _whisper_word_reliable(ww):
                continue
            ww_lo = float(ww.get("start", ws["start"]))
            ww_hi = float(ww.get("end", ws["end"]))
            if ww_hi > sub_lo and ww_lo < sub_hi:
                w_words.append(
                    {
                        "start": ww_lo,
                        "end": ww_hi,
                        "word": str(ww.get("word", "")).strip(),
                    }
                )
    for smear in sub.get("whisper_smear_attributions") or []:
        for sw in smear.get("shifted_words", []):
            sw_lo = float(sw.get("start", sub_lo))
            sw_hi = float(sw.get("end", sub_hi))
            if sw_hi > sub_lo and sw_lo < sub_hi:
                w_words.append(
                    {
                        "start": sw_lo,
                        "end": sw_hi,
                        "word": str(sw.get("word", "")).strip(),
                    }
                )
    if not w_words:
        return []

    swaps: list[dict] = []
    new_word_strs: list[str] = []
    for i, pw in enumerate(words):
        pw_lo = float(pw["start"])
        pw_hi = float(pw["end"])
        pw_text = str(pw.get("word", "")).strip()
        new_word_strs.append(pw_text)

        candidates: list[tuple[dict, float]] = []
        for w in w_words:
            overlap = max(0.0, min(w["end"], pw_hi) - max(w["start"], pw_lo))
            if overlap > 0.0:
                candidates.append((w, overlap))
        if not candidates:
            continue
        p_norm = _word_normalize(pw_text)
        if any(_word_normalize(c[0]["word"]) == p_norm for c in candidates):
            continue
        candidates.sort(key=lambda c: c[1], reverse=True)
        best_w, best_overlap = candidates[0]
        bw_lo = float(best_w["start"])
        bw_hi = float(best_w["end"])
        pw_dur = max(1e-6, pw_hi - pw_lo)
        bw_dur = max(1e-6, bw_hi - bw_lo)
        if (
            best_overlap / pw_dur < WORD_REFINE_OVERLAP_FRACTION
            or best_overlap / bw_dur < WORD_REFINE_OVERLAP_FRACTION
        ):
            continue
        pw_mid = (pw_lo + pw_hi) / 2
        bw_mid = (bw_lo + bw_hi) / 2
        if not (bw_lo <= pw_mid <= bw_hi):
            continue
        if not (pw_lo <= bw_mid <= pw_hi):
            continue
        parakeet_overlap_count = sum(
            1
            for pw2 in words
            if (min(float(pw2["end"]), bw_hi) - max(float(pw2["start"]), bw_lo)) > 0.0
        )
        if parakeet_overlap_count > 1:
            continue
        wh_text = best_w["word"]
        w_norm = _word_normalize(wh_text)
        if not p_norm or not w_norm or p_norm == w_norm:
            continue
        if len(w_norm) < max(2, len(p_norm) // 2):
            continue
        prev_norm = (
            _word_normalize(str(words[i - 1].get("word", "")).strip()) if i > 0 else ""
        )
        next_norm = (
            _word_normalize(str(words[i + 1].get("word", "")).strip())
            if i + 1 < len(words)
            else ""
        )
        if prev_norm and prev_norm in w_norm:
            continue
        if next_norm and next_norm in w_norm:
            continue
        if int(fuzz.ratio(p_norm, w_norm)) >= WORD_REFINE_AGREEMENT_THRESHOLD:
            continue

        mms_text = run_mms_focused(
            audio,
            sr,
            pw_lo - WORD_REFINE_MMS_PAD_S,
            pw_hi + WORD_REFINE_MMS_PAD_S,
            mms_proc,
            mms_model,
            mms_pad_id,
            mms_word_delim_id,
            mms_id_to_char,
        )
        m_norm = _word_normalize(mms_text)
        if not m_norm:
            continue
        mp = int(fuzz.partial_ratio(p_norm, m_norm))
        mw = int(fuzz.partial_ratio(w_norm, m_norm))
        if mw >= mp + WORD_REFINE_SWAP_MARGIN:
            new_word_strs[-1] = wh_text
            pw["word"] = wh_text
            swaps.append(
                {
                    "start": pw_lo,
                    "end": pw_hi,
                    "p_word": pw_text,
                    "w_word": wh_text,
                    "mms": mms_text,
                    "mp_score": mp,
                    "mw_score": mw,
                }
            )

    if swaps:
        new_text = " ".join(new_word_strs).strip()
        sub["text_after_word_swap"] = new_text
        if sub.get("final_text") == sub.get("text"):
            sub["final_text"] = new_text
        sub["word_swaps"] = swaps
        if not sub.get("final_source", "").endswith("+word_swap"):
            sub["final_source"] = sub.get("final_source", "parakeet") + "+word_swap"
    return swaps


def transfer_whisper_punctuation(
    sub: dict,
    subseg_idx: int,
    whisper_segs: list[dict],
    single_owners: list,
    multi_overlap_segs: set,
) -> int:
    """Copy Whisper's punctuation + capitalisation onto Parakeet words.
    Parakeet TDT emits raw word strings with no commas/periods; Whisper
    preserves them. For each Parakeet word in sub['words'] whose normalized
    text matches an overlapping Whisper word from the same subseg's
    owned/shared segments, replace the Parakeet word with the Whisper word's
    formatted version. Modifies sub['words'][i]['word'] in place; returns the
    count updated."""
    sub_lo = float(sub["start"])
    sub_hi = float(sub["end"])

    candidates: list[dict] = []
    for wi, ws in enumerate(whisper_segs):
        if single_owners[wi] != subseg_idx and wi not in multi_overlap_segs:
            continue
        for ww in ws.get("words", []) or []:
            if not _whisper_word_reliable(ww):
                continue
            try:
                ww_lo = float(ww.get("start", ws.get("start", sub_lo)))
                ww_hi = float(ww.get("end", ws.get("end", sub_hi)))
            except (TypeError, ValueError):
                continue
            text = str(ww.get("word", "")).strip()
            if not text:
                continue
            candidates.append({"start": ww_lo, "end": ww_hi, "word": text})

    if not candidates:
        return 0

    updated = 0
    for pw in sub.get("words") or []:
        pw_lo = float(pw.get("start", sub_lo))
        pw_hi = float(pw.get("end", sub_hi))
        pw_text = str(pw.get("word", "")).strip()
        pw_norm = _word_normalize(pw_text)
        if not pw_norm:
            continue
        for ww in candidates:
            if ww["end"] <= pw_lo or ww["start"] >= pw_hi:
                continue
            if _word_normalize(ww["word"]) != pw_norm:
                continue
            if ww["word"] != pw_text:
                pw["word"] = ww["word"]
                updated += 1
            break
    return updated


def snap_words_to_mms(
    words: list[dict], ctc_chars: list[dict], pad_s: float = 0.1
) -> list[dict]:
    """Refine each word's [start, end] to the frame-accurate MMS char stream:
    set start/end to the first/last alphabetic MMS frame within the word's span
    (+/- pad_s). Falls back to the original (Parakeet) timing where MMS has no
    coverage — MMS is sparse/empty on short words, so Parakeet stays the
    baseline there. ctc_chars must be sorted by 't'."""
    if not words or not ctc_chars:
        return words
    out: list[dict] = []
    for w in words:
        lo = float(w["start"])
        hi = float(w["end"])
        i = _binary_lb(ctc_chars, lo - pad_s)
        first = None
        last = None
        j = i
        while j < len(ctc_chars) and ctc_chars[j]["t"] <= hi + pad_s:
            ch = ctc_chars[j]["char"]
            if ch and ch.isalpha():
                t = float(ctc_chars[j]["t"])
                if first is None:
                    first = t
                last = t
            j += 1
        nw = dict(w)
        if first is not None and last is not None and last > first:
            nw["start"] = first
            nw["end"] = last
        out.append(nw)
    return out


class TranscriptMerger:
    def __init__(
        self,
        parakeet_data,
        mms_chars,
        whisper_data,
        vad_intervals,
        audio,
        sr=16000,
        mms_proc=None,
        mms_model=None,
        mms_pad_id=None,
        mms_word_delim_id=None,
        mms_id_to_char=None,
    ):
        self._parakeet = parakeet_data
        self._mms_chars = sorted(mms_chars, key=lambda c: c["t"])
        self._whisper_segs = sorted(
            whisper_data.get("segments", []), key=lambda s: float(s["start"])
        )
        self._vad = vad_intervals
        self._audio = audio
        self._sr = sr
        self._mms_proc = mms_proc
        self._mms_model = mms_model
        self._mms_pad_id = mms_pad_id
        self._mms_word_delim_id = mms_word_delim_id
        self._mms_id_to_char = mms_id_to_char
        self._diagnostic_segments: list[dict] = []

    def diagnostic_segments(self) -> list[dict]:
        return self._diagnostic_segments

    def run(self, progress_cb=None) -> list[dict]:
        from rapidfuzz import fuzz

        def _p(label: str, frac: float) -> None:
            """Report merge sub-progress. `frac` is 0..1 within the merge stage."""
            if progress_cb is not None:
                progress_cb(label, frac)

        parakeet_data = self._parakeet
        ctc_chars = self._mms_chars
        whisper_segs = self._whisper_segs

        _p("resegmenting", 0.0)
        subsegs = resegment_parakeet(
            parakeet_data, SILENCE_SPLIT_S, MMS_WINDOW_PAD_END_S
        )

        subsegs = dedupe_overlapping_subsegs(subsegs)

        _p("assigning whisper owners", 0.03)
        whisper_single_owners: list = [None] * len(whisper_segs)
        whisper_multi_overlap: set = set()
        if whisper_segs:
            whisper_single_owners, whisper_multi_overlap = assign_whisper_owners(
                subsegs,
                whisper_segs,
                WHISPER_SMEAR_TOLERANCE_S,
                WHISPER_SMEAR_MATCH_THRESHOLD,
            )

        _p("scoring vs MMS", 0.08)
        flagged_idxs: list[int] = []
        for i, sub in enumerate(subsegs):
            lo, hi = compute_neighbor_aware_window(
                subsegs, i, MMS_WINDOW_PAD_START_S, MMS_WINDOW_PAD_END_S
            )
            mms_text = build_mms_window(ctc_chars, lo, hi)
            sub["mms_text"] = mms_text
            sub["mms_window"] = [lo, hi]
            whisper_text = whisper_text_for_subseg(
                sub, i, whisper_segs, whisper_single_owners, whisper_multi_overlap
            )
            sub["whisper_text"] = whisper_text
            p_norm = normalize_for_compare(sub["text"])
            m_norm = normalize_for_compare(mms_text)
            w_norm = normalize_for_compare(whisper_text)
            sub["wp_agreement"] = (
                int(fuzz.ratio(p_norm, w_norm)) if p_norm and w_norm else 0
            )
            if not p_norm and not m_norm:
                sub["parakeet_score"] = 100
                continue
            if not p_norm or not m_norm:
                sub["parakeet_score"] = 0
                flagged_idxs.append(i)
                continue
            sub["parakeet_score"] = int(fuzz.ratio(p_norm, m_norm))
            if sub["parakeet_score"] < DISAGREEMENT_THRESHOLD:
                flagged_idxs.append(i)

        vad_intervals = self._vad

        gaps = find_vad_active_gaps(
            subsegs,
            vad_intervals,
            GAP_FILL_THRESHOLD_S,
            prev_end_pad_s=MMS_WINDOW_PAD_END_S,
        )

        need_audio = bool(flagged_idxs) or bool(gaps)
        if need_audio and self._audio is None:
            raise RuntimeError(
                "merge needs audio + MMS model for flagged/gap subsegs "
                "but none provided"
            )
        audio = self._audio
        sr = self._sr
        mms_proc = self._mms_proc
        mms_model = self._mms_model
        mms_pad_id = self._mms_pad_id
        mms_word_delim_id = self._mms_word_delim_id
        mms_id_to_char = self._mms_id_to_char

        still_flagged_idxs: list[int] = []
        if flagged_idxs:
            n_flagged = len(flagged_idxs)
            _p(f"focused-MMS arbitration (0/{n_flagged})", 0.12)
            for _fi, idx in enumerate(flagged_idxs):
                if _fi and _fi % 25 == 0:
                    _p(
                        f"focused-MMS arbitration ({_fi}/{n_flagged})",
                        0.12 + 0.18 * _fi / n_flagged,
                    )
                sub = subsegs[idx]
                f_lo, f_hi = compute_neighbor_aware_window(
                    subsegs, idx, FOCUSED_MMS_PAD_START_S, FOCUSED_MMS_PAD_END_S
                )
                focused_text = run_mms_focused(
                    audio,
                    sr,
                    f_lo,
                    f_hi,
                    mms_proc,
                    mms_model,
                    mms_pad_id,
                    mms_word_delim_id,
                    mms_id_to_char,
                )
                sub["focused_mms_text"] = focused_text
                sub["focused_mms_window"] = [f_lo, f_hi]
                p_norm = normalize_for_compare(sub["text"])
                f_norm = normalize_for_compare(focused_text)
                if p_norm and f_norm:
                    f_score = int(fuzz.ratio(p_norm, f_norm))
                else:
                    f_score = 0
                sub["focused_mms_score"] = f_score
                if f_score >= DISAGREEMENT_THRESHOLD:
                    sub["final_text"] = sub["text"]
                    sub["final_source"] = "focused_mms_resolved"
                else:
                    still_flagged_idxs.append(idx)

        if still_flagged_idxs:
            _p("whisper arbiter", 0.32)
            for idx in still_flagged_idxs:
                sub = subsegs[idx]
                whisper_text = sub.get("whisper_text", "")
                wp_agreement = int(sub.get("wp_agreement") or 0)

                focused_text = sub.get("focused_mms_text", "")
                reference_mms = focused_text if focused_text else sub["mms_text"]
                m_norm = normalize_for_compare(reference_mms)
                w_norm = normalize_for_compare(whisper_text)
                wm_score = int(fuzz.ratio(w_norm, m_norm)) if w_norm and m_norm else 0
                sub["whisper_mms_score"] = wm_score

                p_text_clean = sub["text"].strip()
                if wp_agreement >= PG_CONSENSUS_THRESHOLD:
                    sub["final_text"] = sub["text"]
                    sub["final_source"] = "consensus_pw"
                elif wm_score >= DISAGREEMENT_THRESHOLD:
                    sub["final_text"] = whisper_text
                    sub["final_source"] = "whisper_corrected"
                elif len(p_text_clean) >= 3:
                    sub["final_text"] = sub["text"]
                    sub["final_source"] = "parakeet_kept_after_disagreement"
                elif (
                    reference_mms
                    and len(reference_mms.replace(" ", "").strip())
                    >= GAP_FILL_VOCAL_MIN_CHARS
                ):
                    sub["final_text"] = f"[vocalization] {reference_mms}"
                    sub["final_source"] = "mms_ctc_fallback"
                    sub["kind"] = "non_speech_vocal"
                else:
                    sub["final_text"] = sub["text"]
                    sub["final_source"] = "parakeet_kept"

        for sub in subsegs:
            if "final_text" not in sub:
                sub["final_text"] = sub["text"]
                sub["final_source"] = "parakeet"

        if whisper_segs:
            _p("extending (suffix)", 0.36)
            owned_by_sub: dict[int, list[int]] = {}
            for wi, owner in enumerate(whisper_single_owners):
                if owner is not None:
                    owned_by_sub.setdefault(owner, []).append(wi)
            sorted_starts = sorted(
                (float(s["start"]), j) for j, s in enumerate(subsegs) if s.get("words")
            )
            for sub_idx, sub in enumerate(subsegs):
                base = sub.get("final_source", "").replace("+word_swap", "")
                if base not in PARAKEET_FINAL_SOURCES:
                    continue
                seg_indices = owned_by_sub.get(sub_idx, [])
                if not seg_indices:
                    continue
                sub_end = float(sub["end"])
                next_start = float("inf")
                for s_start, j in sorted_starts:
                    if j != sub_idx and s_start >= sub_end:
                        next_start = s_start
                        break
                owned_segs = [whisper_segs[wi] for wi in seg_indices]
                cont_words = find_continuation_words(
                    sub.get("words") or [],
                    owned_segs,
                    ceiling_s=next_start,
                )
                if not cont_words:
                    continue
                new_end = sub_end
                for cw in cont_words:
                    new_end = max(new_end, cw["end"])
                sub["words"] = list(sub.get("words") or []) + cont_words
                sub["end"] = new_end
                continuation_text = " ".join(cw["word"] for cw in cont_words)
                current_final = sub.get("final_text", sub["text"])
                sub["final_text"] = (
                    current_final.rstrip() + " " + continuation_text
                ).strip()
                sub["whisper_continuation"] = continuation_text

        if whisper_segs:
            _p("extending (prefix)", 0.40)
            owned_by_sub_pref: dict[int, list[int]] = {}
            for wi, owner in enumerate(whisper_single_owners):
                if owner is not None:
                    owned_by_sub_pref.setdefault(owner, []).append(wi)
            sorted_ends = sorted(
                (float(s["end"]), j) for j, s in enumerate(subsegs) if s.get("words")
            )
            for sub_idx, sub in enumerate(subsegs):
                base = sub.get("final_source", "").replace("+word_swap", "")
                if base not in PARAKEET_FINAL_SOURCES:
                    continue
                seg_indices = owned_by_sub_pref.get(sub_idx, [])
                if not seg_indices:
                    continue
                sub_start = float(sub["start"])
                prev_end = 0.0
                for s_end, j in sorted_ends:
                    if j != sub_idx and s_end <= sub_start and s_end > prev_end:
                        prev_end = s_end
                owned_segs = [whisper_segs[wi] for wi in seg_indices]
                pre_words = find_prefix_words(
                    sub.get("words") or [],
                    owned_segs,
                    floor_s=prev_end,
                )
                if not pre_words:
                    continue
                new_start = sub_start
                for pw in pre_words:
                    new_start = min(new_start, pw["start"])
                sub["words"] = list(pre_words) + list(sub.get("words") or [])
                sub["start"] = new_start
                prefix_text = " ".join(pw["word"] for pw in pre_words)
                current_final = sub.get("final_text", sub["text"])
                sub["final_text"] = (prefix_text + " " + current_final.lstrip()).strip()
                sub["whisper_prefix"] = prefix_text

        if vad_intervals:
            gaps = find_vad_active_gaps(
                subsegs,
                vad_intervals,
                GAP_FILL_THRESHOLD_S,
                prev_end_pad_s=MMS_WINDOW_PAD_END_S,
            )

        if gaps:
            n_gaps = len(gaps)
            _p(f"gap fill (0/{n_gaps})", 0.55)
            for _gi, (gap_lo, gap_hi) in enumerate(gaps):
                if _gi and _gi % 25 == 0:
                    _p(f"gap fill ({_gi}/{n_gaps})", 0.55 + 0.10 * _gi / n_gaps)
                if gap_absorbed_by_neighbor(
                    subsegs,
                    whisper_segs,
                    whisper_single_owners,
                    gap_lo,
                    gap_hi,
                ):
                    continue
                gap_whisper = whisper_text_for_gap(
                    gap_lo,
                    gap_hi,
                    whisper_segs,
                    whisper_single_owners,
                    whisper_multi_overlap,
                )
                gap_mms = run_mms_focused(
                    audio,
                    sr,
                    gap_lo,
                    gap_hi,
                    mms_proc,
                    mms_model,
                    mms_pad_id,
                    mms_word_delim_id,
                    mms_id_to_char,
                )
                mms_no_space = gap_mms.replace(" ", "")
                distinct = len({c for c in mms_no_space if c.isalpha()})
                looks_vocal = (
                    len(mms_no_space) >= GAP_FILL_VOCAL_MIN_LEN
                    and distinct <= GAP_FILL_VOCAL_MAX_DISTINCT
                )
                whisper_smear = False
                if gap_whisper:
                    w_norm_gap = normalize_for_compare(gap_whisper)
                    if w_norm_gap:
                        for neighbor_idx in (
                            _find_prev_subseg_idx(subsegs, gap_lo),
                            _find_next_subseg_idx(subsegs, gap_hi),
                        ):
                            if neighbor_idx is None:
                                continue
                            neighbor = subsegs[neighbor_idx]
                            for field in ("text", "whisper_text"):
                                n_norm = normalize_for_compare(
                                    neighbor.get(field, "") or ""
                                )
                                if not n_norm:
                                    continue
                                if (
                                    int(fuzz.partial_ratio(w_norm_gap, n_norm))
                                    >= GAP_FILL_SMEAR_THRESHOLD
                                ):
                                    whisper_smear = True
                                    break
                            if whisper_smear:
                                break
                if looks_vocal:
                    subsegs.append(
                        {
                            "start": float(gap_lo),
                            "end": float(gap_hi),
                            "text": "",
                            "words": [],
                            "mms_text": gap_mms,
                            "mms_window": [gap_lo, gap_hi],
                            "whisper_text": gap_whisper,
                            "final_text": f"[vocalization] {gap_mms}",
                            "final_source": "vad_gap_fill",
                            "kind": "non_speech_vocal",
                        }
                    )
                elif (
                    gap_whisper and len(gap_whisper.strip()) >= 3 and not whisper_smear
                ):
                    subsegs.append(
                        {
                            "start": float(gap_lo),
                            "end": float(gap_hi),
                            "text": "",
                            "words": [],
                            "mms_text": gap_mms,
                            "whisper_text": gap_whisper,
                            "final_text": gap_whisper,
                            "final_source": "whisper_gap_fill",
                            "kind": "speech",
                        }
                    )
                elif (
                    gap_mms
                    and len(gap_mms.replace(" ", "").strip())
                    >= GAP_FILL_VOCAL_MIN_CHARS
                ):
                    subsegs.append(
                        {
                            "start": float(gap_lo),
                            "end": float(gap_hi),
                            "text": "",
                            "words": [],
                            "mms_text": gap_mms,
                            "mms_window": [gap_lo, gap_hi],
                            "whisper_text": gap_whisper,
                            "final_text": f"[vocalization] {gap_mms}",
                            "final_source": "vad_gap_fill",
                            "kind": "non_speech_vocal",
                        }
                    )

        for sub in subsegs:
            if "final_text" not in sub:
                sub["final_text"] = sub["text"]
                sub["final_source"] = "parakeet"

        if mms_model is not None and whisper_segs:
            n_sub = len(subsegs)
            _p(f"per-word refine (0/{n_sub})", 0.66)
            for _si, sub in enumerate(subsegs):
                if _si and _si % 50 == 0:
                    _p(f"per-word refine ({_si}/{n_sub})", 0.66 + 0.24 * _si / n_sub)
                base_source = sub.get("final_source", "")
                if base_source not in PARAKEET_FINAL_SOURCES:
                    continue
                if (sub.get("wp_agreement") or 0) >= 100:
                    continue
                per_word_refine(
                    sub,
                    whisper_segs,
                    audio,
                    sr,
                    mms_proc,
                    mms_model,
                    mms_pad_id,
                    mms_word_delim_id,
                    mms_id_to_char,
                )

        # Phrase-level arbitration (conservative): for low-agreement subsegs that
        # ended Parakeet-final, adopt the FULL Whisper text iff BOTH the global
        # MMS slice and a wide focused MMS decode favor it (per-word tight clips
        # can't recover e.g. 'shadow' from a 'sad'-shaped clip; the phrase can).
        if mms_model is not None and whisper_segs:
            _p("phrase arbitration", 0.90)
            for sub_idx, sub in enumerate(subsegs):
                base = sub.get("final_source", "").replace("+word_swap", "")
                if base not in PARAKEET_FINAL_SOURCES:
                    continue
                wp = sub.get("wp_agreement")
                if wp is None or wp >= PHRASE_ARB_WP_MAX:
                    continue
                p_norm = normalize_for_compare(sub.get("text", "") or "")
                w_text = sub.get("whisper_text", "") or ""
                w_norm = normalize_for_compare(w_text)
                if not p_norm or not w_norm or p_norm == w_norm:
                    continue
                g_norm = normalize_for_compare(sub.get("mms_text", "") or "")
                focused = run_mms_focused(
                    audio,
                    sr,
                    float(sub["start"]) - PHRASE_ARB_PAD_S,
                    float(sub["end"]) + PHRASE_ARB_PAD_S,
                    mms_proc,
                    mms_model,
                    mms_pad_id,
                    mms_word_delim_id,
                    mms_id_to_char,
                )
                f_norm = normalize_for_compare(focused)
                if not g_norm or not f_norm:
                    continue
                pg = int(fuzz.ratio(p_norm, g_norm))
                wg = int(fuzz.ratio(w_norm, g_norm))
                pf = int(fuzz.ratio(p_norm, f_norm))
                wf = int(fuzz.ratio(w_norm, f_norm))
                both_favor = (wg >= pg + PHRASE_ARB_MARGIN) and (
                    wf >= pf + PHRASE_ARB_MARGIN
                )
                if not (both_favor and max(wg, wf) >= PHRASE_ARB_FLOOR):
                    continue
                sub["final_text"] = w_text
                sub["final_source"] = "whisper_phrase_arb"
                sub["focused_mms_text"] = focused
                new_words = owned_whisper_words(
                    sub_idx,
                    whisper_segs,
                    whisper_single_owners,
                    whisper_multi_overlap,
                    float(sub["start"]),
                    float(sub["end"]),
                )
                if new_words:
                    sub["words"] = new_words

        if mms_model is not None and whisper_segs:
            _p("substitution-run arbitration", 0.91)

            def _run_arb_decode(lo: float, hi: float) -> str:
                return run_mms_focused(
                    audio,
                    sr,
                    lo,
                    hi,
                    mms_proc,
                    mms_model,
                    mms_pad_id,
                    mms_word_delim_id,
                    mms_id_to_char,
                )

            arbitrate_substitution_runs(
                subsegs,
                whisper_segs,
                whisper_single_owners,
                whisper_multi_overlap,
                ctc_chars,
                _run_arb_decode,
            )

        if whisper_segs:
            _p("transferring punctuation", 0.92)
            for sub_idx, sub in enumerate(subsegs):
                base = _base_source(sub.get("final_source", ""))
                if base not in PARAKEET_FINAL_SOURCES:
                    continue
                n = transfer_whisper_punctuation(
                    sub,
                    sub_idx,
                    whisper_segs,
                    whisper_single_owners,
                    whisper_multi_overlap,
                )
                if n:
                    rebuilt = " ".join(
                        w.get("word", "") for w in (sub.get("words") or [])
                    ).strip()
                    if rebuilt:
                        sub["final_text"] = rebuilt

        if SNAP_WORDS_TO_MMS and ctc_chars:
            for sub in subsegs:
                if sub.get("words"):
                    sub["words"] = snap_words_to_mms(sub["words"], ctc_chars)

        _p("finalizing", 0.98)
        subsegs.sort(key=lambda s: s["start"])

        diagnostic_segments = [
            {
                "start": float(sub["start"]),
                "end": float(sub["end"]),
                "text": sub["final_text"],
                "source": sub["final_source"],
                "kind": sub.get("kind", "speech"),
                "parakeet_text": sub.get("text"),
                "mms_text": sub.get("mms_text"),
                "focused_mms_text": sub.get("focused_mms_text"),
                "whisper_text": sub.get("whisper_text"),
                "parakeet_score": sub.get("parakeet_score"),
                "focused_mms_score": sub.get("focused_mms_score"),
                "whisper_mms_score": sub.get("whisper_mms_score"),
                "wp_agreement": sub.get("wp_agreement"),
                "word_swaps": sub.get("word_swaps"),
                "run_swaps": sub.get("run_swaps"),
                "words": [
                    {
                        "start": float(w.get("start", sub["start"])),
                        "end": float(w.get("end", sub["end"])),
                        "word": str(w.get("word", "")),
                    }
                    for w in (sub.get("words") or [])
                ],
                "whisper_continuation": sub.get("whisper_continuation"),
            }
            for sub in subsegs
        ]

        self._diagnostic_segments = diagnostic_segments
        lean = [
            {
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "source": s["source"],
                "kind": s["kind"],
                "words": [
                    {"start": w["start"], "end": w["end"], "word": w["word"]}
                    for w in s["words"]
                ],
            }
            for s in diagnostic_segments
        ]
        return lean


def render_debug_md(diagnostic_segments: list[dict]) -> str:
    md: list[str] = [
        "# Transcript v3 — Diagnostic view " "(Parakeet + MMS-CTC + Whisper)\n"
    ]
    md.append(
        f"Architecture: noise-anchored Whisper drift is rigid-shifted FORWARD "
        f"onto matching Parakeet content (pre-VAD) → Parakeet (re-segmented on "
        f"inter-word silence > {SILENCE_SPLIT_S}s) → score vs neighbor-aware "
        f"MMS slice → focused per-subseg MMS re-arbitration on flagged "
        f"(score < {DISAGREEMENT_THRESHOLD}) → Whisper-segment arbiter on "
        "still-flagged → default-to-Parakeet on full disagreement. "
        f"After arbitration, VAD-active Parakeet-coverage gaps "
        f"> {GAP_FILL_THRESHOLD_S}s are filled with Whisper text where "
        "available, else focused MMS as [vocalization], else dropped "
        "as silence. Final word boundaries are snapped to the frame-accurate "
        "MMS stream where it has coverage (Parakeet timing kept otherwise).\n"
    )
    md.append("\nLegend:\n")
    md.append("- `[P]`  Parakeet kept (initial MMS-CTC score ≥ threshold)\n")
    md.append(
        "- `[F]`  Focused MMS resolved (was flagged, focused per-subseg "
        "MMS agreed with Parakeet)\n"
    )
    md.append("- `[C]`  Parakeet & Whisper consensus (whisper supported parakeet)\n")
    md.append("- `[W]`  Whisper corrected (Parakeet failed, Whisper matched MMS)\n")
    md.append("- `[K]`  Parakeet kept after full disagreement (default fallback)\n")
    md.append(
        "- `[V]`  Vocalization — either S4 fallback (engines disagree, "
        "MMS phonetic) or VAD-gap-filled vocal region (awawa-style)\n"
    )
    md.append(
        "- `[Wg]` Whisper gap-fill — VAD-active gap where Whisper "
        "had speech but Parakeet didn't\n\n"
    )
    md.append(
        "- `[P~]` Parakeet kept but a localized wrong-word run was swapped to "
        "Whisper (run arbitration, MMS-backed)\n"
    )
    md.append("Every segment shows all signals for debugging:\n")
    md.append("- **Final** — what the pipeline chose\n")
    md.append("- **Parakeet** — what Parakeet emitted in this time window\n")
    md.append("- **MMS-CTC** — phonetic acoustic transcription (global-pass slice)\n")
    md.append("- **Focused MMS** — tight per-subseg MMS pass (only when flagged)\n")
    md.append("- **Whisper** — Whisper text in this time window\n\n")
    md.append("---\n")

    for seg in diagnostic_segments:
        source = seg.get("source", "parakeet")
        word_swapped = "+word_swap" in source
        run_swapped = "+run_arb" in source
        base = _base_source(source)
        if base == "focused_mms_resolved":
            tag = "[F]"
        elif base == "consensus_pw":
            tag = "[C]"
        elif base == "whisper_corrected":
            tag = "[W]"
        elif base == "parakeet_kept_after_disagreement":
            tag = "[K]"
        elif base == "whisper_gap_fill":
            tag = "[Wg]"
        elif base == "whisper_phrase_arb":
            tag = "[Wp]"
        elif base in ("mms_ctc_fallback", "vad_gap_fill"):
            tag = "[V]"
        else:
            tag = "[P]"
        if word_swapped:
            tag = tag[:-1] + "*]"
        if run_swapped:
            tag = tag[:-1] + "~]"

        ps = seg.get("parakeet_score")
        fs = seg.get("focused_mms_score")
        ws = seg.get("whisper_mms_score")
        wp = seg.get("wp_agreement")
        score_line_parts = []
        if ps is not None:
            score_line_parts.append(f"parakeet_score={ps}")
        if fs is not None:
            score_line_parts.append(f"focused_mms_score={fs}")
        if ws is not None:
            score_line_parts.append(f"whisper_mms_score={ws}")
        if wp is not None:
            score_line_parts.append(f"wp_agreement={wp}")
        score_line = "  ".join(score_line_parts)

        md.append(
            f"\n### `{fmt_mmss(seg['start'])}-{fmt_mmss(seg['end'])}`  "
            f"{tag} `{seg['text']}`\n"
        )
        if score_line:
            md.append(f"_{score_line}_\n")
        md.append(f"- **Parakeet:** `{seg.get('parakeet_text', '')}`\n")
        md.append(f"- **MMS-CTC:**  `{seg.get('mms_text', '')}`\n")
        if seg.get("focused_mms_text"):
            md.append(f"- **Focused MMS:**  `{seg.get('focused_mms_text')}`\n")
        if seg.get("whisper_text"):
            md.append(f"- **Whisper:**  `{seg.get('whisper_text')}`\n")
        for sw in seg.get("word_swaps") or []:
            md.append(
                f"- **Word swap:**  `{sw['p_word']}` → `{sw['w_word']}`  "
                f"(mms=`{sw.get('mms', '')}`, "
                f"mp={sw.get('mp_score')}, mw={sw.get('mw_score')})\n"
            )
        for rs in seg.get("run_swaps") or []:
            md.append(
                f"- **Run swap:**  `{rs['p_run']}` → `{rs['w_run']}`  "
                f"(focused=`{rs.get('focused_mms', '')}`, "
                f"pf={rs.get('pf')}, wf={rs.get('wf')}, "
                f"pg={rs.get('pg')}, wg={rs.get('wg')})\n"
            )

    return "\n".join(md)
