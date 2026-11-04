from __future__ import annotations

import json
from pathlib import Path

from modular_transcriptor.formats import write_transcript_csv
from modular_transcriptor.speakers import assign_speakers
from modular_transcriptor.types import PhaseCb

MODEL_NAME = "parakeet-tdt-0.6b-v3"
PRODUCED_BY_VERSION = "0.4.0"

# Caption segmentation. Some chunks (fast/sung/overlapping audio) decode into one
# giant run-on segment with no sentence punctuation, so Parakeet's own segment
# boundaries can't be trusted for captions. Any segment longer than
# RESEG_MAX_S is re-split on its word timestamps at natural pauses (and at
# sentence punctuation when present), capped so no caption runs longer than
# RESEG_MAX_S. Normal short segments pass through untouched.
RESEG_MAX_S = 12.0
RESEG_MIN_S = 1.2  # don't cut a caption shorter than this at a pause
RESEG_GAP_S = 0.5  # a word-gap this long is a natural break
_SENT_END = (".", "!", "?")


def _split_words_into_captions(words: list[dict]) -> list[list[dict]]:
    """Group word-timestamp dicts into caption-sized runs.

    This only ever runs on a run-on (a Parakeet segment whose PnC collapsed into
    one >RESEG_MAX_S unpunctuated block), so there are few real boundaries to go
    on. For each caption we scan ahead and cut at, in order of preference: the
    first sentence punctuation past RESEG_MIN_S (a true boundary); else, when the
    RESEG_MAX_S cap forces a cut, the LARGEST qualifying word-gap in the window
    rather than the first — the old code broke at the first >=RESEG_GAP_S breath,
    which chopped a sentence at an incidental pause (lowercase continuation) when
    a bigger, more sentence-like silence sat a little further on. The trailing
    sub-cap remainder is kept whole (fewer, better-placed breaks)."""
    caps: list[list[dict]] = []
    i = 0
    n = len(words)
    while i < n:
        start_t = words[i]["start"]
        sentence_cut: int | None = None
        hardcap_cut: int | None = None
        best_gap = -1.0
        best_gap_cut: int | None = None
        j = i + 1
        while j < n:
            prev = words[j - 1]
            dur = prev["end"] - start_t
            if dur >= RESEG_MAX_S:
                hardcap_cut = j
                break
            if dur >= RESEG_MIN_S:
                if prev["word"].rstrip()[-1:] in _SENT_END:
                    sentence_cut = j
                    break
                gap = words[j]["start"] - prev["end"]
                if gap >= RESEG_GAP_S and gap > best_gap:
                    best_gap, best_gap_cut = gap, j
            j += 1
        if sentence_cut is not None:
            cut = sentence_cut
        elif hardcap_cut is not None:
            # forced by the cap: prefer the largest natural pause in the window
            # over a mid-utterance chop at the cap.
            cut = best_gap_cut if best_gap_cut is not None else hardcap_cut
        else:
            cut = n  # reached the end: keep the remainder whole
        caps.append(words[i:cut])
        i = cut
    return caps


def _lean_segments(parakeet: dict) -> list[dict]:
    """Project each raw Parakeet segment onto the lean transcript schema,
    dropping per-word `language` and tagging source + kind. Per-word `score`
    (model confidence) IS carried through so the transcript keeps a per-word
    low-confidence signal for debugging. Segments longer than RESEG_MAX_S are
    re-segmented on their word timestamps into caption-sized pieces (Parakeet
    occasionally emits a whole chunk as one unpunctuated run-on)."""
    lean: list[dict] = []
    for seg in parakeet.get("segments") or []:
        words = [
            {
                "start": float(w["start"]),
                "end": float(w["end"]),
                "word": w["word"],
                "score": w.get("score"),
            }
            for w in (seg.get("words") or [])
        ]
        duration = float(seg["end"]) - float(seg["start"])
        if words and duration > RESEG_MAX_S:
            # re-segment: each piece's text is rebuilt from its own words
            for piece in _split_words_into_captions(words):
                lean.append(
                    {
                        "start": piece[0]["start"],
                        "end": piece[-1]["end"],
                        "text": " ".join(w["word"] for w in piece).strip(),
                        "source": "parakeet",
                        "kind": "speech",
                        "words": piece,
                    }
                )
        else:
            # short segment: keep Parakeet's own text (and casing/punctuation)
            lean.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "text": seg.get("text", ""),
                    "source": "parakeet",
                    "kind": "speech",
                    "words": words,
                }
            )
    return lean


def _render_debug_md(segments: list[dict]) -> str:
    """One line per segment: [start-end] SPEAKER: text."""
    lines = ["# Transcript — parakeet-alone finalize\n"]
    for seg in segments:
        spk = seg.get("speaker") or "-"
        text = (seg.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{seg['start']:.2f}-{seg['end']:.2f}] {spk}: {text}")
    return "\n".join(lines) + "\n"


# Title/abbreviation words that end in a period but are NOT sentence ends, so a
# caption must not break after them ("...surgeon, Dr." | "Hun!").
_ABBREVS = frozenset(
    {
        "dr", "mr", "mrs", "ms", "st", "jr", "sr", "prof", "rev", "gen", "sgt",
        "lt", "col", "capt", "cpl", "maj", "fr", "vs", "no", "etc", "inc", "ltd",
    }
)


def _ends_in_abbrev(text: str) -> bool:
    words = (text or "").strip().split()
    if not words or not words[-1].endswith("."):
        return False
    return words[-1][:-1].strip(".").lower() in _ABBREVS


def _merge_abbrev_splits(segments: list[dict]) -> list[dict]:
    """Merge a caption into the next when it ends in an abbreviation (Dr./Mr./
    etc.) — Parakeet treats the trailing period as a sentence end and splits
    there, which reads wrong ("...Dr." | "Hun!")."""
    out: list[dict] = []
    for seg in segments:
        if out and _ends_in_abbrev(out[-1]["text"]):
            prev = out[-1]
            prev["text"] = (prev["text"].rstrip() + " " + seg["text"].lstrip()).strip()
            prev["end"] = seg["end"]
            prev["words"] = (prev.get("words") or []) + (seg.get("words") or [])
        else:
            out.append(dict(seg))
    return out


def _load_turns(diar_json: list | dict) -> list[dict]:
    """Extract diarization turns from either schema. v1 is a bare list of
    `{start,end,speaker}`; Task 1's v2 is a dict `{"schema":2,"turns":[...],
    "embeddings":{...},...}`. finalize consumes only the turns (the clipper
    reads the embeddings), so the two shapes collapse to one list here."""
    if isinstance(diar_json, list):
        return diar_json
    return diar_json.get("turns", [])


def finalize_run(work: Path, emit: PhaseCb | None = None) -> None:
    """Turn `work/parakeet.json` (+ optional `work/diarization.json`) into the
    final `transcript.json` and its sidecars in `work.parent`.

    Reads ONLY parakeet.json and diarization.json; any stale `whisper.json` /
    `mms_ctc.json` left in the work dir is ignored. Idempotent: returns
    immediately if `work.parent/transcript.json` already exists.
    """
    work = Path(work)
    out = work.parent
    if (out / "transcript.json").exists():
        return  # idempotent

    def _emit(label: str, pct: float | None = None) -> None:
        if emit is not None:
            emit(label, pct)

    _emit("finalizing transcript", 5.0)
    parakeet = json.loads((work / "parakeet.json").read_text())
    lean = _merge_abbrev_splits(_lean_segments(parakeet))

    diar_path = work / "diarization.json"
    turns = _load_turns(json.loads(diar_path.read_text())) if diar_path.exists() else []

    _emit("assigning speakers", 80.0)
    assign_speakers(lean, turns)

    # Prefer Parakeet's true decoded media length; a VOD ending in silence or
    # music runs past its last segment end, so max-seg-end under-reports it.
    pk_duration = parakeet.get("duration_s")
    duration_s = (
        float(pk_duration)
        if pk_duration is not None
        else max((seg["end"] for seg in lean), default=0.0)
    )
    diarized = bool(turns)

    _emit("writing outputs", 95.0)
    # Write the sidecars FIRST, then transcript.json LAST and atomically. The
    # idempotency guard keys on transcript.json's existence, so it must only
    # appear once every sidecar is on disk — otherwise a crash mid-write leaves
    # the guard believing the run finished while sidecars are missing.
    write_transcript_csv(out / "transcript.csv", lean)
    (out / "transcript.debug.json").write_text(
        json.dumps({"segments": lean}, ensure_ascii=False)
    )
    (out / "transcript.debug.md").write_text(_render_debug_md(lean))

    tmp = out / "transcript.json.partial"
    tmp.write_text(
        json.dumps(
            {
                "language": "en",
                "model": MODEL_NAME,
                "compute_type": "float16",
                "diarized": diarized,
                "duration_s": duration_s,
                "produced_by_version": PRODUCED_BY_VERSION,
                "segments": lean,
            },
            ensure_ascii=False,
        )
    )
    tmp.replace(out / "transcript.json")
