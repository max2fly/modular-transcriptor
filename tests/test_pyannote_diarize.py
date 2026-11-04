from __future__ import annotations

import json
import types

import numpy as np
import pytest
import torch

import modular_transcriptor.pyannote_diarize as pdz
from conftest import fake_modules, make_module
from modular_transcriptor import cache


class _FakeSegment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeAnnotation:
    """Mimics a pyannote Annotation: labels() + itertracks(yield_label=True)."""

    def __init__(self, tracks, labels):
        self._tracks = tracks  # list of (segment, track_id, label)
        self._labels = labels

    def labels(self):
        return list(self._labels)

    def itertracks(self, yield_label=False):
        for seg, track, label in self._tracks:
            if yield_label:
                yield seg, track, label
            else:
                yield seg, track


class _FakeDiarizeOutput:
    def __init__(self, annotation, embeddings):
        self.speaker_diarization = annotation
        self.speaker_embeddings = embeddings


class _FakePipeline:
    def __init__(self, output=None, raise_on_apply=False, dim=256):
        self._output = output
        self._raise = raise_on_apply
        self._embedding = types.SimpleNamespace(dimension=dim)

    def to(self, device):  # pyannote Pipeline.to returns self
        return self

    def apply(self, file):
        if self._raise:
            raise RuntimeError("fake diarization failure")
        return self._output


def _pyannote_module(pipeline):
    class Pipeline:
        @staticmethod
        def from_pretrained(model_id, token=None):
            return pipeline

    return make_module("pyannote.audio", Pipeline=Pipeline)


def _fake_heavy_modules(pipeline):
    """sys.modules stubs for the lazy imports inside run(): pyannote + torchaudio.
    torch and numpy stay real."""
    torchaudio = make_module(
        "torchaudio", load=lambda path: (torch.zeros(1, 16000), 16000)
    )
    return {
        "pyannote.audio": _pyannote_module(pipeline),
        "torchaudio": torchaudio,
    }


def _make_output(labels, embeddings):
    tracks = [
        (_FakeSegment(0.0, 2.0), "A", labels[0]),
        (_FakeSegment(2.5, 4.0), "B", labels[-1]),
    ]
    return _FakeDiarizeOutput(_FakeAnnotation(tracks, labels), embeddings)


def _write_parakeet(path, word_spans):
    """Write a minimal parakeet.json whose flattened words are `word_spans`
    (list of (start, end)), one word per segment so coalescing is exercised."""
    segments = [
        {
            "start": s,
            "end": e,
            "text": "w",
            "words": [{"word": "w", "start": s, "end": e, "score": 1.0}],
        }
        for (s, e) in word_spans
    ]
    path.write_text(json.dumps({"segments": segments}))


def _gapfill_centroids():
    # Label A = unit along axis 0, Label B = unit along axis 1 (both non-zero).
    labels = ["SPEAKER_00", "SPEAKER_01"]
    emb = np.zeros((2, 256), dtype=np.float32)
    emb[0, 0] = 1.0  # A
    emb[1, 1] = 1.0  # B
    return labels, emb


def test_gapfill_assigns_long_confident_span_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "diar.json"
    words = tmp_path / "parakeet.json"

    labels, emb = _gapfill_centroids()
    # Diarization covers (0,2)->A and (2.5,4)->B. Words: one covered word, a
    # LONG uncovered gap (3 contiguous words = 1.5s >= 1.0s), and a SHORT
    # uncovered gap (0.4s < 1.0s). Coalescing must merge the 5.0-6.5 run.
    _write_parakeet(
        words,
        [(0.5, 1.0), (5.0, 5.5), (5.5, 6.0), (6.0, 6.5), (8.0, 8.4)],
    )

    def fake_embed(pipeline, waveform, sr, start, end):
        v = np.zeros(256, dtype=np.float64)
        v[0] = 1.0  # cosine 1.0 with label A, 0.0 with label B
        return v

    monkeypatch.setattr(pdz, "_embed_span", fake_embed)
    pipeline = _FakePipeline(output=_make_output(labels, emb))
    with fake_modules(**_fake_heavy_modules(pipeline)):
        assert pdz.run(audio, out, device="cpu", words_path=words) == out

    data = json.loads(out.read_text())
    gaps = [t for t in data["turns"] if t.get("source") == "gapfill"]
    assert len(gaps) == 1
    g = gaps[0]
    assert g["speaker"] == "SPEAKER_00"
    assert g["start"] == 5.0
    assert g["end"] == 6.5


def test_gapfill_skips_low_confidence(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "diar.json"
    words = tmp_path / "parakeet.json"

    labels, emb = _gapfill_centroids()
    _write_parakeet(words, [(5.0, 5.5), (5.5, 6.0), (6.0, 6.5)])  # 1.5s gap

    def fake_embed(pipeline, waveform, sr, start, end):
        v = np.zeros(256, dtype=np.float64)
        v[5] = 1.0  # orthogonal to every centroid -> cosine 0 < 0.80
        return v

    monkeypatch.setattr(pdz, "_embed_span", fake_embed)
    pipeline = _FakePipeline(output=_make_output(labels, emb))
    with fake_modules(**_fake_heavy_modules(pipeline)):
        assert pdz.run(audio, out, device="cpu", words_path=words) == out

    data = json.loads(out.read_text())
    assert not [t for t in data["turns"] if t.get("source") == "gapfill"]


def test_run_skips_when_output_fresh(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)  # proceed would RuntimeError
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "diar.json"
    out.write_text(json.dumps(pdz._empty_v2()))
    cache.stamp(out, audio, {"diar_schema": pdz.DIAR_SCHEMA_VERSION})
    before = out.read_text()
    assert pdz.run(audio, out) == out
    assert out.read_text() == before  # untouched -> genuinely skipped


def test_run_recomputes_when_source_changed(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "diar.json"
    out.write_text(json.dumps([]))
    cache.stamp(out, audio)
    audio.write_bytes(b"RE-EXTRACTED-DIFFERENT")  # WAV changed -> stale
    with pytest.raises(RuntimeError):  # proceeded past guard to the HF_TOKEN check
        pdz.run(audio, out)


def test_writes_v2_schema_with_centroids(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "diar.json"

    labels = ["SPEAKER_00", "SPEAKER_01"]
    emb = np.zeros((2, 256), dtype=np.float32)
    emb[0, :] = 1.0  # non-zero -> kept
    # emb[1] stays all-zero -> must be skipped (under-cluster padded row)
    pipeline = _FakePipeline(output=_make_output(labels, emb))

    with fake_modules(**_fake_heavy_modules(pipeline)):
        assert pdz.run(audio, out, device="cpu") == out

    data = json.loads(out.read_text())
    assert data["schema"] == 2
    assert data["embedding_model"] == "pyannote-community-1"
    assert data["embedding_dim"] == 256
    assert data["turns"]  # non-empty
    assert data["turns"][0] == {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}
    # only the non-zero label survives
    assert list(data["embeddings"].keys()) == ["SPEAKER_00"]
    assert len(data["embeddings"]["SPEAKER_00"]) == 256
    assert data["embeddings"]["SPEAKER_00"][0] == 1.0


def test_soft_degrade_writes_empty_v2(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "diar.json"

    pipeline = _FakePipeline(raise_on_apply=True)
    with fake_modules(**_fake_heavy_modules(pipeline)):
        assert pdz.run(audio, out, device="cpu") == out  # no exception

    data = json.loads(out.read_text())
    assert data == {
        "schema": 2,
        "turns": [],
        "embeddings": {},
        "embedding_model": "pyannote-community-1",
        "embedding_dim": 0,
    }


def test_cleanup_error_preserves_payload(tmp_path, monkeypatch):
    # A CUDA/GC cleanup failure (empty_cache raising) must NOT fall through to
    # the outer soft-degrade and discard the good diarization — that would force
    # an expensive full re-transcribe downstream.
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "diar.json"

    def boom():
        raise RuntimeError("cuda cleanup blew up")

    monkeypatch.setattr(torch.cuda, "empty_cache", boom)

    labels = ["SPEAKER_00", "SPEAKER_01"]
    emb = np.zeros((2, 256), dtype=np.float32)
    emb[0, :] = 1.0
    emb[1, :] = 2.0
    pipeline = _FakePipeline(output=_make_output(labels, emb))

    with fake_modules(**_fake_heavy_modules(pipeline)):
        assert pdz.run(audio, out, device="cpu") == out  # no exception

    data = json.loads(out.read_text())
    # Real turns + embeddings survived — NOT the empty-v2 soft-degrade payload.
    assert data["schema"] == 2
    assert data["turns"]
    assert data["turns"][0] == {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}
    assert data["embeddings"]
    assert data["embedding_dim"] == 256


def test_cache_version_bump_regenerates_v1(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"AUDIO")
    out = tmp_path / "diar.json"
    # pre-existing v1 artifact: a bare list, stamped with the OLD (no-extra) key
    out.write_text(json.dumps([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]))
    cache.stamp(out, audio)  # old-schema provenance, lacks diar_schema

    labels = ["SPEAKER_00"]
    emb = np.ones((1, 256), dtype=np.float32)
    pipeline = _FakePipeline(output=_make_output(labels, emb))
    with fake_modules(**_fake_heavy_modules(pipeline)):
        pdz.run(audio, out, device="cpu")

    data = json.loads(out.read_text())
    assert isinstance(data, dict)  # regenerated, not the old bare list
    assert data["schema"] == 2
