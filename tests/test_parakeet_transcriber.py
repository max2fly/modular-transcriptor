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


def test_dedup_keeps_sequential_segments_sharing_a_bridge_phrase():
    # Two ADJACENT chunks segment this region differently, producing two
    # genuinely-sequential sentences that share only the bridge "and she's
    # sitting" (0.80s / 32% partial overlap). The old length-only rule treated
    # them as duplicates and deleted "So it's an anime girl" entirely.
    segs = [
        {
            "start": 5907.04,
            "end": 5909.52,
            "text": "So it's an anime girl, and she's sitting.",
        },
        {
            "start": 5908.72,
            "end": 5914.56,
            "text": "And she's sitting, she is she has opened herself "
            "up to the viewer, to you.",
        },
    ]
    out = pk._dedup_overlapping(segs)
    joined = " ".join(s["text"] for s in out)
    assert "anime girl" in joined  # distinct content must survive
    assert len(out) == 2


def test_dedup_merges_divergent_seam_duplicate_keeping_complete():
    # A 300s chunk boundary splits one utterance: chunk A (central) gets the
    # complete sentence, chunk B (boundary-truncated) re-transcribes the shared
    # tail and garbles its last word. They overlap ~97% in time but the texts
    # DIVERGE, so the containment gate would miss them. Heavy time-overlap must
    # collapse them to the longer, complete one.
    segs = [
        {
            "start": 17279.44,
            "end": 17282.88,
            "text": "They're very tiny but very cute, and you can build your own Shamdo.",
        },
        {
            "start": 17280.48,
            "end": 17282.96,
            "text": "but very cute and you can build your own Sheptur.",
        },
    ]
    out = pk._dedup_overlapping(segs)
    assert len(out) == 1
    assert out[0]["text"].startswith("They're very tiny")  # complete one kept


def test_dedup_keeps_stutter_repeat_of_identical_sentence():
    # A streamer stutters the same sentence back-to-back: identical text but
    # DISJOINT time. These are two real utterances, not a re-transcription, so
    # both must be kept.
    segs = [
        {"start": 10.0, "end": 12.0, "text": "I can't believe it."},
        {"start": 12.2, "end": 14.2, "text": "I can't believe it."},
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


def test_chunk_constants():
    # Speech-aware chunking targets ~40 s (the model's PnC training max; longer
    # chunks collapse punctuation into one run-on), cut at pauses with a small
    # overlap so no boundary word is dropped.
    assert pk.TARGET_CHUNK_S < pk.CHUNK_HARD_MAX_S
    assert pk.MIN_CHUNK_S < pk.TARGET_CHUNK_S
    assert pk.TARGET_CHUNK_S == 40
    assert pk.CHUNK_OVERLAP_S == 0  # cut in the pause midpoint -> contiguous


def test_run_uses_full_context_attention(monkeypatch, tmp_path):
    # Drive run() end-to-end against fakes (no GPU / no model download) and
    # assert it NEVER switches the model to local attention: the 300 s chunks
    # decode with the model's native full-context attention. A 3s clip yields a
    # single chunk; the existing n_chunks formula already covers the tail.
    import torch
    import torchaudio

    class FakeModel:
        def eval(self):
            return self

        def to(self, d):
            return self

        def change_attention_model(self, *a, **k):
            raise AssertionError(
                "run() must not switch attention — full-context decode only"
            )

        def transcribe(self, clips, timestamps, verbose):
            class H:
                timestamp = {"segment": [], "word": []}

            return [H()]

    class FakeASRModel:
        @staticmethod
        def from_pretrained(model_id):
            return FakeModel()

    nemo = types.ModuleType("nemo")
    nemo_collections = types.ModuleType("nemo.collections")
    nemo_asr = types.ModuleType("nemo.collections.asr")
    nemo.collections = nemo_collections
    nemo_collections.asr = nemo_asr
    nemo_asr.models = types.SimpleNamespace(ASRModel=FakeASRModel)
    monkeypatch.setitem(sys.modules, "nemo", nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", nemo_collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", nemo_asr)

    monkeypatch.setattr(
        torchaudio, "load", lambda p: (torch.ones(1, 16000 * 3), 16000)
    )

    out = tmp_path / "out.json"
    # The FakeModel raises if change_attention_model is ever called, so a clean
    # run() is itself the proof that full-context decode is used.
    pk.run(tmp_path / "a.wav", out)
    written = json.loads(out.read_text())
    assert written["duration_s"] == pytest.approx(3.0)


def test_run_skips_when_output_fresh(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "parakeet.json"
    out.write_text(json.dumps({"engine": "x", "segments": []}))
    # Stamp with the SAME keyed extra run() computes for its default (no-boost)
    # path: parakeet_cache_extra normalizes alpha to 0.0 when boost_file is None,
    # so run()'s is_fresh(..., extra=parakeet_cache_extra(None, 4.0)) must hit.
    cache.stamp(out, audio, extra=pk.parakeet_cache_extra(None, 4.0))
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
