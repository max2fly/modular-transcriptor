from __future__ import annotations

PAD_S = 0.05
MERGE_GAP_S = 0.15


def derive_vad_intervals(whisper_segments: list[dict]) -> list[dict]:
    """Derive speech intervals from Whisper segment boundaries — the same
    speech regions WhisperX's internal pyannote VAD produced. Pads each
    segment by PAD_S and merges intervals within MERGE_GAP_S."""
    raw = [
        (max(0.0, float(s["start"]) - PAD_S), float(s["end"]) + PAD_S)
        for s in whisper_segments
    ]
    raw.sort(key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    for lo, hi in raw:
        if merged and lo - merged[-1][1] <= MERGE_GAP_S:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return [{"start": float(lo), "end": float(hi)} for lo, hi in merged]
