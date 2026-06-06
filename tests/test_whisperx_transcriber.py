from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_fake_whisperx(monkeypatch, *, align_raises: Exception | None = None):
    """Install a fake `whisperx` module so the production transcriber can be
    exercised without real GPU / model weights. The transcriber now does ASR +
    alignment only (no diarization)."""
    fake = types.ModuleType("whisperx")
    fake.load_audio = MagicMock(return_value="AUDIO_NDARRAY")
    asr = MagicMock()
    asr.transcribe = MagicMock(
        return_value={
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": " hi"}],
        }
    )
    fake.load_model = MagicMock(return_value=asr)
    align_model = MagicMock()
    fake.load_align_model = MagicMock(return_value=(align_model, {"meta": "x"}))
    if align_raises is not None:
        fake.align = MagicMock(side_effect=align_raises)
    else:
        fake.align = MagicMock(
            return_value={
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": " hi",
                        "words": [
                            {"word": "hi", "start": 0.0, "end": 0.5, "score": 0.9}
                        ],
                    }
                ]
            }
        )
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    return fake


def _install_fake_torch(monkeypatch):
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        empty_cache=MagicMock(),
        is_available=MagicMock(return_value=True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def test_whisperx_transcriber_produces_expected_shape(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    fake_wx = _install_fake_whisperx(monkeypatch)
    fake_torch = _install_fake_torch(monkeypatch)

    from modular_transcriptor.whisperx_transcriber import WhisperXTranscriber

    audio = tmp_path / "video.ts"
    audio.write_bytes(b"\x00")

    t = WhisperXTranscriber(
        model_size="large-v3",
        compute_type="float16",
        language="en",
        batch_size=32,
        device="cuda",
    )
    result = t.transcribe(audio)

    assert result["language"] == "en"
    # Whisper is text-authority only now: it never diarizes.
    assert result["diarized"] is False
    assert result["segments"][0]["text"] == " hi"
    assert result["segments"][0]["words"][0]["word"] == "hi"
    assert "speaker" not in result["segments"][0]
    assert isinstance(result["duration_s"], float)
    # Models loaded once and unloaded.
    assert fake_wx.load_model.call_count == 1
    assert fake_wx.load_align_model.call_count == 1
    assert not hasattr(fake_wx, "DiarizationPipeline")
    assert fake_torch.cuda.empty_cache.called


def test_whisperx_transcriber_unloads_on_exception(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    _install_fake_whisperx(monkeypatch, align_raises=RuntimeError("align_boom"))
    fake_torch = _install_fake_torch(monkeypatch)

    from modular_transcriptor.whisperx_transcriber import WhisperXTranscriber

    audio = tmp_path / "video.ts"
    audio.write_bytes(b"\x00")

    t = WhisperXTranscriber(
        model_size="large-v3",
        compute_type="float16",
        language="en",
        batch_size=32,
        device="cuda",
    )
    with pytest.raises(RuntimeError, match="align_boom"):
        t.transcribe(audio)
    # The finally block must still free GPU memory on the error path.
    assert fake_torch.cuda.empty_cache.called


def test_whisperx_transcriber_fast_fails_without_hf_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _install_fake_whisperx(monkeypatch)
    _install_fake_torch(monkeypatch)

    from modular_transcriptor.whisperx_transcriber import WhisperXTranscriber

    audio = tmp_path / "video.ts"
    audio.write_bytes(b"\x00")

    t = WhisperXTranscriber(
        model_size="large-v3",
        compute_type="float16",
        language="en",
        batch_size=32,
        device="cuda",
    )
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        t.transcribe(audio)


def test_per_phase_timing_logs_emitted(monkeypatch, tmp_path):
    """Confirm that each remaining phase (load, asr, align) emits a
    'transcribe_phase_done' log line with a wall_ms field."""
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    _install_fake_whisperx(monkeypatch)
    _install_fake_torch(monkeypatch)

    from structlog.testing import capture_logs

    from modular_transcriptor.whisperx_transcriber import WhisperXTranscriber

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 16)

    t = WhisperXTranscriber(
        model_size="tiny",
        compute_type="float16",
        language="en",
        batch_size=32,
        device="cpu",
    )

    with capture_logs() as cap:
        t.transcribe(audio)

    phase_done_events = [e for e in cap if e.get("event") == "transcribe_phase_done"]
    phases = {e.get("phase") for e in phase_done_events}
    assert {"load", "asr", "align"} <= phases, phases
    # Diarization phases are gone.
    assert "diarize_load" not in phases
    assert "diarize_run" not in phases
    for e in phase_done_events:
        assert isinstance(e.get("wall_ms"), int), e


def test_predecode_false_feeds_input_directly_without_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    fake_wx = _install_fake_whisperx(monkeypatch)
    _install_fake_torch(monkeypatch)

    from modular_transcriptor.whisperx_transcriber import WhisperXTranscriber

    audio = tmp_path / "audio.amplified.wav"
    audio.write_bytes(b"\x00")

    t = WhisperXTranscriber(predecode=False)
    calls: list = []
    monkeypatch.setattr(t, "_predecode_audio", lambda p: calls.append(p) or p)
    monkeypatch.setattr(t, "_probe_duration", lambda p: 1.0)

    t.transcribe(audio)

    # predecode skipped entirely; whisperx reads the supplied WAV directly.
    assert calls == []
    fake_wx.load_audio.assert_called_with(str(audio))
