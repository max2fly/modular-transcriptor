from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, TypedDict

PhaseCb = Callable[[str, float | None], None]


class TranscriptResult(TypedDict):
    """Subset of WhisperX's natural output, plus a `duration_s` field we
    compute (audio duration in seconds) so the caller can write the on-disk
    wrapper without re-probing the file."""

    language: str
    duration_s: float
    segments: list[dict[str, Any]]
    diarized: bool


class Transcriber(Protocol):
    def transcribe(
        self, audio_path: Path, phase_cb: PhaseCb | None = None
    ) -> TranscriptResult:
        """Read audio at `audio_path`, return a TranscriptResult.

        If `phase_cb(label, pct)` is supplied, the implementation should call
        it with a short human label (e.g. "running ASR") at each major step.
        `pct` is a coarse 0-100 hint or None when no measurable progress is
        available. The callback must be tolerant of being invoked from worker
        threads.
        """
        ...


class MergedWord(TypedDict, total=False):
    start: float
    end: float
    word: str
    speaker: str | None


class MergedSegment(TypedDict, total=False):
    start: float
    end: float
    text: str
    speaker: str | None
    kind: str  # "speech" | "non_speech_vocal"
    source: str
    words: list[MergedWord]


class DiarizationTurn(TypedDict):
    start: float
    end: float
    speaker: str


class MmsChar(TypedDict):
    char: str
    t: float
