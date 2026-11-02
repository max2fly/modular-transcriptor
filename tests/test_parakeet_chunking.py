from __future__ import annotations

import numpy as np

from modular_transcriptor.parakeet_transcriber import (
    CHUNK_HARD_MAX_S,
    CHUNK_OVERLAP_S,
    MIN_CHUNK_S,
    parakeet_cache_extra,
    speech_aware_bounds,
)


def _audio_with_pauses(sr: int, total_s: int, pause_centers_s: list[float]) -> np.ndarray:
    """Loud 'speech' noise everywhere, with 1s silent gaps centered at the given
    seconds — so the quietest point in any window is a real pause."""
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(total_s * sr).astype(np.float32) * 0.3
    for c in pause_centers_s:
        audio[int((c - 0.5) * sr) : int((c + 0.5) * sr)] = 0.0
    return audio


def test_speech_aware_bounds_cuts_at_pauses_with_overlap():
    sr = 16000
    # pauses spaced ~40s apart (the target), where cuts should land
    pauses = [40.0, 80.0, 120.0, 160.0, 200.0]
    audio = _audio_with_pauses(sr, 240, pauses)
    bounds = speech_aware_bounds(audio, sr)
    assert bounds[0][0] == 0 and bounds[-1][1] == len(audio)
    # chunks overlap by ~CHUNK_OVERLAP_S: next start = prev cut - overlap
    for (_s0, e0), (s1, _e1) in zip(bounds, bounds[1:]):
        assert abs((e0 - s1) - CHUNK_OVERLAP_S * sr) <= sr  # ~overlap, not a gap
    # every internal cut lands inside one of the silent gaps (±0.5s)
    for _s, e in bounds[:-1]:
        assert any(abs(e / sr - c) <= 0.5 for c in pauses), e / sr
    # chunk lengths respect the bounds
    for s, e in bounds[:-1]:
        assert MIN_CHUNK_S * sr <= (e - s) <= CHUNK_HARD_MAX_S * sr


def test_speech_aware_bounds_short_audio_single_chunk():
    sr = 16000
    audio = np.zeros(30 * sr, dtype=np.float32)
    assert speech_aware_bounds(audio, sr) == [(0, 30 * sr)]


def test_speech_aware_bounds_caps_at_max_without_pause():
    sr = 16000
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(200 * sr).astype(np.float32) * 0.3  # no pauses
    bounds = speech_aware_bounds(audio, sr)
    assert all((e - s) <= CHUNK_HARD_MAX_S * sr for s, e in bounds)
    assert bounds[0][0] == 0 and bounds[-1][1] == len(audio)
    # must make progress (no infinite loop); contiguous (s1 == e0 at overlap 0)
    for (_s0, e0), (s1, _e1) in zip(bounds, bounds[1:]):
        assert s1 <= e0 and s1 > _s0


def test_parakeet_cache_extra_changes_with_boost(tmp_path):
    base = parakeet_cache_extra(None, 0.0)
    assert base["target_chunk_s"] == 40 and base["hard_max_chunk_s"] == 300
    assert base["boost_sha"] is None
    f = tmp_path / "boost.txt"
    f.write_text("Shondo\nMitacle\n")
    a = parakeet_cache_extra(f, 4.0)
    assert a["boost_sha"] is not None and a["boost_alpha"] == 4.0
    f.write_text("Shondo\nMitacle\nLucy\n")
    assert parakeet_cache_extra(f, 4.0)["boost_sha"] != a["boost_sha"]


def test_cache_extra_alpha_inert_without_boost_file():
    assert parakeet_cache_extra(None, 4.0) == parakeet_cache_extra(None, 0.0)
