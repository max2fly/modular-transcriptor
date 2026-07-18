from __future__ import annotations

import json
import sys
import types

import pytest

import modular_transcriptor.mms_ctc_pass as mms
from modular_transcriptor import cache


def test_decode_frames_drops_pad_and_dups():
    id_to_char = {0: "<pad>", 1: "|", 2: "h", 3: "i"}
    ids = [0, 2, 2, 3, 1, 0]  # pad, h, h(dup), i, |, pad
    chars = mms._decode_frames(
        ids,
        frame_dur_s=0.1,
        chunk_start_s=10.0,
        core_start=10.0,
        core_end=11.0,
        pad_id=0,
        word_delim_id=1,
        id_to_char=id_to_char,
    )
    texts = [c["char"] for c in chars]
    assert texts == ["h", "i", "|"]
    assert chars[0]["t"] == 10.1  # frame index 1 * 0.1 + 10.0


def test_decode_respects_core_window():
    # Distinct ids per frame so the CTC dedup (which tracks the last *seen* id
    # across the whole frame sequence, per the verbatim debug behavior) doesn't
    # interact with the window test: only the in-window frame survives.
    id_to_char = {0: "<pad>", 2: "h", 3: "i", 4: "j"}
    ids = [2, 3, 4]  # frames at t=0, 1, 2
    chars = mms._decode_frames(
        ids,
        frame_dur_s=1.0,
        chunk_start_s=0.0,
        core_start=2.0,
        core_end=5.0,
        pad_id=0,
        word_delim_id=1,
        id_to_char=id_to_char,
    )
    # only t=2 ('j') is within [2.0, 5.0)
    assert [c["char"] for c in chars] == ["j"]


def test_run_skips_when_output_fresh(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "mms.json"
    out.write_text(json.dumps([]))
    cache.stamp(out, audio)
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    before = out.read_text()
    assert mms.run(audio, out) == out
    assert out.read_text() == before  # untouched -> genuinely skipped


def test_run_recomputes_when_source_changed(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "mms.json"
    out.write_text(json.dumps([]))
    cache.stamp(out, audio)
    audio.write_bytes(b"RE-EXTRACTED-DIFFERENT")  # WAV changed -> stale
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    with pytest.raises(ImportError):  # proceeded past the guard into heavy imports
        mms.run(audio, out)
