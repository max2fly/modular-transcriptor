from __future__ import annotations

import json
from pathlib import Path

import pytest

from modular_transcriptor.finalize import (
    MODEL_NAME,
    PRODUCED_BY_VERSION,
    finalize_run,
)


def _write_parakeet(work: Path, segments: list[dict]) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "parakeet.json").write_text(
        json.dumps({"engine": "parakeet", "segments": segments})
    )


def _one_segment() -> list[dict]:
    return [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "hello world",
            "language": "en",
            "avg_logprob": -0.1,
            "words": [
                {"word": "hello", "start": 0.0, "end": 1.0, "score": 0.9},
                {"word": "world", "start": 1.0, "end": 2.0, "score": 0.8},
            ],
        }
    ]


def _run_finalize(tmp_path: Path, with_diar: bool = False) -> dict:
    work = tmp_path / "work"
    _write_parakeet(work, _one_segment())
    if with_diar:
        (work / "diarization.json").write_text(
            json.dumps([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}])
        )
    finalize_run(work)
    return json.loads((tmp_path / "transcript.json").read_text())


def test_lean_schema_and_source(tmp_path):
    out = _run_finalize(tmp_path)
    seg = out["segments"][0]
    assert set(seg) >= {"start", "end", "text", "speaker", "kind", "source", "words"}
    assert seg["source"] == "parakeet" and seg["kind"] == "speech"
    assert seg["speaker"] is None and out["diarized"] is False
    assert out["model"] == MODEL_NAME
    assert out["model"] == "parakeet-tdt-0.6b-v3 (local-attn)"
    assert out["produced_by_version"] == PRODUCED_BY_VERSION == "0.4.0"
    assert out["language"] == "en" and out["compute_type"] == "float16"
    assert out["duration_s"] == pytest.approx(seg["end"])
    word = seg["words"][0]
    assert set(word) >= {"start", "end", "word"}
    assert "score" not in word


def test_duration_s_uses_parakeet_value_when_present(tmp_path):
    # Parakeet records true decoded media length; a VOD ending in silence/music
    # has media longer than its last segment end, and finalize must honor it.
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "parakeet.json").write_text(
        json.dumps(
            {"engine": "parakeet", "duration_s": 42.5, "segments": _one_segment()}
        )
    )
    finalize_run(work)
    out = json.loads((tmp_path / "transcript.json").read_text())
    assert out["duration_s"] == pytest.approx(42.5)  # NOT the 2.0 max-seg-end


def test_duration_s_falls_back_to_max_seg_end_when_absent(tmp_path):
    out = _run_finalize(tmp_path)  # parakeet.json carries no duration_s key
    assert out["duration_s"] == pytest.approx(2.0)


def test_assigns_speaker_when_diarization_present(tmp_path):
    out = _run_finalize(tmp_path, with_diar=True)
    assert out["segments"][0]["speaker"] == "SPEAKER_00"
    assert out["segments"][0]["words"][0]["speaker"] == "SPEAKER_00"
    assert out["diarized"] is True


def test_idempotent_when_transcript_exists(tmp_path):
    work = tmp_path / "work"
    _write_parakeet(work, _one_segment())
    sentinel = {"segments": [], "sentinel": True}
    (tmp_path / "transcript.json").write_text(json.dumps(sentinel))
    finalize_run(work)
    assert json.loads((tmp_path / "transcript.json").read_text()) == sentinel


def test_empty_parakeet_yields_empty_segments(tmp_path):
    work = tmp_path / "work"
    _write_parakeet(work, [])
    finalize_run(work)
    out = json.loads((tmp_path / "transcript.json").read_text())
    assert out["segments"] == []
    assert out["duration_s"] == 0.0
    assert out["diarized"] is False
    assert (tmp_path / "transcript.csv").exists()
    assert (tmp_path / "transcript.debug.json").exists()
    assert (tmp_path / "transcript.debug.md").exists()


def test_ignores_stale_whisper_and_mms(tmp_path):
    work = tmp_path / "work"
    _write_parakeet(work, _one_segment())
    (work / "whisper.json").write_text("{not even valid json")
    (work / "mms_ctc.json").write_text("garbage-too")
    finalize_run(work)
    out = json.loads((tmp_path / "transcript.json").read_text())
    assert out["segments"][0]["text"] == "hello world"
    assert out["segments"][0]["source"] == "parakeet"


def test_writes_all_sidecar_files(tmp_path):
    out_dict = _run_finalize(tmp_path)
    work_parent = tmp_path
    debug = json.loads((work_parent / "transcript.debug.json").read_text())
    assert debug["segments"] == out_dict["segments"]
    md = (work_parent / "transcript.debug.md").read_text()
    assert md.count("\n") >= 1
