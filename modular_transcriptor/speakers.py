from __future__ import annotations

from collections import Counter


def _speaker_for_span(lo: float, hi: float, turns: list[dict]) -> str | None:
    best_spk = None
    best_overlap = 0.0
    for t in turns:
        ov = min(hi, float(t["end"])) - max(lo, float(t["start"]))
        if ov > best_overlap:
            best_overlap = ov
            best_spk = str(t["speaker"])
    return best_spk


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Annotate each word with the speaker of its max-overlap diarization
    turn; each segment's speaker is its words' majority (None if no turns)."""
    for seg in segments:
        words = seg.get("words") or []
        spk_counts: Counter = Counter()
        for w in words:
            spk = (
                _speaker_for_span(float(w["start"]), float(w["end"]), turns)
                if turns
                else None
            )
            w["speaker"] = spk
            if spk:
                spk_counts[spk] += 1
        if spk_counts:
            seg["speaker"] = spk_counts.most_common(1)[0][0]
        elif turns:
            seg["speaker"] = _speaker_for_span(
                float(seg["start"]), float(seg["end"]), turns
            )
        else:
            seg["speaker"] = None
    return segments
