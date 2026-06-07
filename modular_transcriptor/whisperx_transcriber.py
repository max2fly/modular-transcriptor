from __future__ import annotations

import gc
import os
import subprocess
import time
from pathlib import Path

from modular_transcriptor._logging import get_logger
from modular_transcriptor.types import PhaseCb, TranscriptResult

log = get_logger("whisperx_transcriber")


class WhisperXTranscriber:
    """Production Whisper engine: ASR + word alignment only. Diarization is
    NOT done here — in the multi-engine pipeline Whisper is the text-authority
    engine and speaker labels are assigned later from a dedicated pyannote
    diarization pass. All torch / whisperx imports are LAZY (inside
    `_run_blocking`) so this module is importable in test environments without
    those packages installed; tests inject fakes via `sys.modules`."""

    def __init__(
        self,
        *,
        model_size: str = "large-v3",
        compute_type: str = "float16",
        language: str = "en",
        batch_size: int = 32,
        device: str = "cuda",
        predecode: bool = True,
    ) -> None:
        self._model_size = model_size
        self._compute_type = compute_type
        self._language = language
        self._batch_size = batch_size
        self._device = device
        # Predecode ffmpeg-converts the input to 16 kHz mono WAV. Skip it when
        # the caller already supplies that format (the pipeline's amplify stage
        # does) to avoid a redundant ffmpeg pass + a ~430 MB temp file.
        self._predecode = predecode

    def transcribe(
        self, audio_path: Path, phase_cb: PhaseCb | None = None
    ) -> TranscriptResult:
        return self._run_blocking(audio_path, phase_cb)

    def _run_blocking(
        self, audio_path: Path, phase_cb: PhaseCb | None = None
    ) -> TranscriptResult:
        def emit(label: str, pct: float | None = None) -> None:
            if phase_cb is None:
                return
            try:
                phase_cb(label, pct)
            except Exception as e:  # noqa: BLE001
                log.warning("phase_cb_failed", err=str(e))

        decoded = self._predecode_audio(audio_path) if self._predecode else audio_path
        try:
            import whisperx  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]

            asr = align_model = align_meta = None
            try:
                emit(f"loading whisper {self._model_size}", 2.0)
                log.info(
                    "models_loading",
                    model=self._model_size,
                    compute_type=self._compute_type,
                )
                load_start = self._phase_timer()
                asr = whisperx.load_model(
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                )
                log.info(
                    "transcribe_phase_done",
                    phase="load",
                    wall_ms=int((self._phase_timer() - load_start) * 1000),
                )
                emit("loading align model (may download wav2vec2)", 12.0)
                align_model, align_meta = whisperx.load_align_model(
                    language_code=self._language, device=self._device
                )

                emit("loading audio", 18.0)
                audio = whisperx.load_audio(str(decoded))
                duration_s = self._probe_duration(audio_path)

                phase_start = self._phase_timer()
                emit("running ASR", 22.0)
                log.info("asr_run", path=str(audio_path))
                asr_out = asr.transcribe(
                    audio, batch_size=self._batch_size, language=self._language
                )
                log.info(
                    "transcribe_phase_done",
                    phase="asr",
                    wall_ms=int((self._phase_timer() - phase_start) * 1000),
                )

                emit("aligning words", 60.0)
                log.info("align_run")
                phase_start = self._phase_timer()
                aligned = whisperx.align(
                    asr_out["segments"],
                    align_model,
                    align_meta,
                    audio,
                    self._device,
                    return_char_alignments=False,
                )
                log.info(
                    "transcribe_phase_done",
                    phase="align",
                    wall_ms=int((self._phase_timer() - phase_start) * 1000),
                )

                return {
                    "language": asr_out.get("language", self._language),
                    "duration_s": duration_s,
                    "diarized": False,
                    "segments": aligned.get("segments", []),
                }
            finally:
                del asr, align_model, align_meta
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception as e:  # noqa: BLE001
                    log.warning("cuda_empty_cache_failed", err=str(e))
        finally:
            if decoded != audio_path:
                try:
                    decoded.unlink()
                except OSError:
                    pass

    @staticmethod
    def _phase_timer() -> float:
        return time.monotonic()

    @staticmethod
    def _predecode_audio(audio_path: Path) -> Path:
        """ffmpeg-decode source to 16 kHz mono WAV. Skips librosa's slow
        internal decode path. Returns the new tmp file path; caller must
        clean up. If ffmpeg is unavailable or fails, falls back to the
        original path."""
        out = audio_path.with_suffix(".whisperx-pre.wav")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(audio_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(out),
                ],
                check=True,
                timeout=600,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            log.warning("predecode_failed_falling_back_to_raw", err=str(e))
            return audio_path
        return out

    @staticmethod
    def _probe_duration(audio_path: Path) -> float:
        # ffprobe is part of the ffmpeg install; WhisperX already requires
        # ffmpeg via its loader, so this adds no new dependency.
        try:
            out = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nw=1:nk=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            return float(out.stdout.strip() or "0")
        except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e:
            log.warning("ffprobe_duration_failed", err=str(e))
            return 0.0
