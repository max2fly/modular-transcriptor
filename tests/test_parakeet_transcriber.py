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
    from omegaconf import OmegaConf

    applied: dict = {}

    class FakeModel:
        # run() reads model.cfg.decoding and re-installs it (to enable
        # confidence, + boosting when requested); give it a minimal OmegaConf.
        cfg = OmegaConf.create({"decoding": {"strategy": "greedy_batch", "greedy": {}}})

        def eval(self):
            return self

        def to(self, d):
            return self

        def change_attention_model(self, *a, **k):
            raise AssertionError(
                "run() must not switch attention — full-context decode only"
            )

        def change_decoding_strategy(self, cfg):
            applied["cfg"] = cfg

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
        torchaudio, "load", lambda p: (torch.zeros(1, 16000 * 3), 16000)
    )

    out = tmp_path / "out.json"
    # The FakeModel raises if change_attention_model is ever called, so a clean
    # run() is itself the proof that full-context decode is used.
    pk.run(tmp_path / "a.wav", out)
    written = json.loads(out.read_text())
    assert written["duration_s"] == pytest.approx(3.0)
    # per-word confidence is always enabled (debug signal), even with no boost
    assert applied["cfg"].confidence_cfg.preserve_word_confidence is True


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


def test_convert_to_canonical_reads_word_confidence():
    # Per-word confidence is a separate index-aligned list on the hypothesis;
    # _convert_to_canonical must land it on each word's `score` and average it
    # onto the segment (with the time offset applied to all timestamps).
    class H:
        timestamp = {
            "segment": [{"start": 0.0, "end": 2.0, "segment": "hi there"}],
            "word": [
                {"word": "hi", "start": 0.0, "end": 0.5},
                {"word": "there", "start": 0.6, "end": 1.2},
            ],
        }
        word_confidence = [0.9, 0.1]

    segs = pk._convert_to_canonical(H(), time_offset=10.0)
    assert len(segs) == 1
    s = segs[0]
    assert s["start"] == 10.0 and s["end"] == 12.0
    assert s["words"][0]["start"] == 10.0  # offset applied
    assert [w["score"] for w in s["words"]] == [0.9, 0.1]
    assert s["avg_logprob"] == pytest.approx(0.5)  # mean(0.9, 0.1)


def test_convert_to_canonical_defaults_score_without_confidence():
    # No word_confidence (confidence not requested) -> score falls back to 1.0.
    class H:
        timestamp = {
            "segment": [{"start": 0.0, "end": 1.0, "segment": "hi"}],
            "word": [{"word": "hi", "start": 0.0, "end": 0.5}],
        }

    segs = pk._convert_to_canonical(H())
    assert segs[0]["words"][0]["score"] == 1.0
    assert segs[0]["avg_logprob"] == pytest.approx(1.0)


def test_run_falls_back_when_word_confidence_crashes(monkeypatch, tmp_path):
    # NeMo's word-confidence aggregation hard-raises on some chunks. run() must
    # re-decode that chunk WITHOUT confidence instead of dropping it, then re-arm
    # confidence for the next chunk — never losing real speech to a debug signal.
    import torch
    import torchaudio
    from omegaconf import OmegaConf

    calls = {"transcribe": 0, "strategies": []}

    class FakeModel:
        cfg = OmegaConf.create({"decoding": {"strategy": "greedy_batch", "greedy": {}}})

        def eval(self):
            return self

        def to(self, d):
            return self

        def change_decoding_strategy(self, cfg):
            calls["strategies"].append("confidence_cfg" in cfg)
            # Real NeMo UPDATES the live decoding cfg on install; the fallback
            # must rebuild from a pristine snapshot, not this mutated cfg, or the
            # confidence-off install would re-inherit confidence and crash again.
            type(self).cfg = cfg

        def transcribe(self, clips, timestamps, verbose):
            calls["transcribe"] += 1
            if calls["transcribe"] == 1:
                raise RuntimeError(
                    "Something went wrong with word-level confidence aggregation."
                    " len(words): 3, len(word_confidence): 4"
                )

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
    monkeypatch.setattr(torchaudio, "load", lambda p: (torch.zeros(1, 16000 * 3), 16000))

    pk.run(tmp_path / "a.wav", tmp_path / "out.json")
    assert calls["transcribe"] == 2  # crashed once, retried the chunk once
    # confidence ON, then OFF for the fallback decode, then re-armed ON
    assert calls["strategies"] == [True, False, True]


def test_run_reraises_non_confidence_runtime_errors(monkeypatch, tmp_path):
    # A RuntimeError unrelated to confidence must NOT be swallowed by the
    # fallback — only the confidence path is recovered.
    import torch
    import torchaudio
    from omegaconf import OmegaConf

    class FakeModel:
        cfg = OmegaConf.create({"decoding": {"strategy": "greedy_batch", "greedy": {}}})

        def eval(self):
            return self

        def to(self, d):
            return self

        def change_decoding_strategy(self, cfg):
            pass

        def transcribe(self, clips, timestamps, verbose):
            raise RuntimeError("CUDA error: illegal memory access")

    nemo = types.ModuleType("nemo")
    nemo_collections = types.ModuleType("nemo.collections")
    nemo_asr = types.ModuleType("nemo.collections.asr")
    nemo.collections = nemo_collections
    nemo_collections.asr = nemo_asr
    nemo_asr.models = types.SimpleNamespace(
        ASRModel=types.SimpleNamespace(from_pretrained=lambda m: FakeModel())
    )
    monkeypatch.setitem(sys.modules, "nemo", nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", nemo_collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", nemo_asr)
    monkeypatch.setattr(torchaudio, "load", lambda p: (torch.zeros(1, 16000 * 3), 16000))

    with pytest.raises(RuntimeError, match="illegal memory access"):
        pk.run(tmp_path / "a.wav", tmp_path / "out.json")


def _fake_hyp(words):
    # words: list of (start, end, text) in clip-relative seconds
    class H:
        timestamp = {
            "segment": ([{"start": words[0][0], "end": words[-1][1],
                          "segment": " ".join(w[2] for w in words)}] if words else []),
            "word": [{"word": w[2], "start": w[0], "end": w[1]} for w in words],
        }
        word_confidence = [1.0] * len(words)
    return H()


def test_repair_holes_recovers_dropped_speech():
    import numpy as np
    sr = 16000
    audio = (np.random.default_rng(0).standard_normal(60 * sr) * 0.3).astype(np.float32)
    # words cover [0,20] and [40,60]; [20,40] is a speech-energy HOLE
    segs = [
        {"start": 0.0, "end": 20.0, "text": "a", "words": [{"start": 0.0, "end": 20.0, "word": "a"}]},
        {"start": 40.0, "end": 60.0, "text": "b", "words": [{"start": 40.0, "end": 60.0, "word": "b"}]},
    ]
    calls = {"n": 0}

    def fake(clip):
        calls["n"] += 1
        # hole is [20,40], window t0=8 -> relative 17s == abs 25s (in the hole)
        return _fake_hyp([(17.0, 17.5, "recovered"), (17.5, 18.0, "speech")])

    out = pk._repair_holes(audio, sr, segs, fake)
    assert calls["n"] == 1  # exactly one hole re-decoded
    texts = " ".join(s["text"] for s in out)
    assert "recovered speech" in texts
    # merged + sorted, recovered segment sits between the two originals
    starts = [s["start"] for s in out]
    assert starts == sorted(starts)


def test_repair_holes_self_gates_on_silence():
    import numpy as np
    sr = 16000
    # near-silent hole region -> not even detected as a hole
    audio = (np.random.default_rng(1).standard_normal(60 * sr) * 0.3).astype(np.float32)
    audio[20 * sr:40 * sr] = 0.0  # silence where the gap is
    segs = [
        {"start": 0.0, "end": 20.0, "text": "a", "words": [{"start": 0.0, "end": 20.0, "word": "a"}]},
        {"start": 40.0, "end": 60.0, "text": "b", "words": [{"start": 40.0, "end": 60.0, "word": "b"}]},
    ]
    calls = {"n": 0}

    def fake(clip):
        calls["n"] += 1
        return _fake_hyp([(17.0, 17.5, "hallucinated")])

    out = pk._repair_holes(audio, sr, segs, fake)
    assert calls["n"] == 0  # silent gap is not a candidate hole -> no re-decode
    assert len(out) == 2
