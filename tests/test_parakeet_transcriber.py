from __future__ import annotations

import json
import sys
import types

import pytest

import modular_transcriptor.parakeet_transcriber as pk
from modular_transcriptor import cache


def test_dedup_keeps_longer_text_on_overlap():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "to anyone"},
        {"start": 0.1, "end": 2.1, "text": "completion is to anyone"},
    ]
    out = pk._dedup_overlapping(segs)
    assert len(out) == 1
    assert out[0]["text"] == "completion is to anyone"


def test_dedup_keeps_disjoint_segments():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "hello"},
        {"start": 5.0, "end": 7.0, "text": "world"},
    ]
    assert len(pk._dedup_overlapping(segs)) == 2


def test_convert_to_canonical_offsets_times():
    class Hyp:
        timestamp = {
            "segment": [{"start": 1.0, "end": 2.0, "segment": "hi there"}],
            "word": [
                {"word": "hi", "start": 1.0, "end": 1.4, "confidence": 0.9},
                {"word": "there", "start": 1.5, "end": 2.0, "confidence": 0.8},
            ],
        }

    segs = pk._convert_to_canonical(Hyp(), time_offset=100.0)
    assert segs[0]["start"] == 101.0
    assert segs[0]["text"] == "hi there"
    assert segs[0]["words"][0]["word"] == "hi"
    assert segs[0]["words"][0]["start"] == 101.0
    assert segs[0]["language"] == "en"


def test_run_skips_when_output_fresh(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "parakeet.json"
    out.write_text(json.dumps({"engine": "x", "segments": []}))
    cache.stamp(out, audio)
    # If it wrongly proceeds, this poisoned import fails fast (no model load).
    monkeypatch.setitem(sys.modules, "nemo", types.ModuleType("nemo"))
    before = out.read_text()
    assert pk.run(audio, out) == out
    assert out.read_text() == before  # untouched -> genuinely skipped


def test_run_recomputes_when_source_changed(tmp_path, monkeypatch):
    # A parakeet.json stamped against an older WAV must NOT be reused after the
    # WAV is re-extracted (the stale-timeline bug that duplicated subtitles).
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "parakeet.json"
    out.write_text(json.dumps({"segments": []}))
    cache.stamp(out, audio)
    audio.write_bytes(b"RE-EXTRACTED-DIFFERENT")  # WAV changed -> stale
    monkeypatch.setitem(sys.modules, "nemo", types.ModuleType("nemo"))
    with pytest.raises(ModuleNotFoundError):  # proceeded past the guard
        pk.run(audio, out)
