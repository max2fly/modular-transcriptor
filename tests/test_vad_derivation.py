from __future__ import annotations

from modular_transcriptor.vad import derive_vad_intervals


def test_merges_close_segments():
    segs = [{"start": 1.0, "end": 2.0}, {"start": 2.1, "end": 3.0}]  # gap 0.1 < 0.15
    out = derive_vad_intervals(segs)
    assert len(out) == 1
    assert out[0]["start"] == 0.95  # 1.0 - 0.05 pad
    assert out[0]["end"] == 3.05


def test_keeps_far_segments_separate():
    segs = [{"start": 1.0, "end": 2.0}, {"start": 5.0, "end": 6.0}]
    out = derive_vad_intervals(segs)
    assert len(out) == 2


def test_empty():
    assert derive_vad_intervals([]) == []
