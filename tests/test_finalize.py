from __future__ import annotations

import json
from pathlib import Path

import pytest

from modular_transcriptor.finalize import (
    MODEL_NAME,
    PRODUCED_BY_VERSION,
    _split_words_into_captions,
    finalize_run,
)


def test_split_prefers_largest_gap_when_cap_forces_a_break():
    # A run-on (>RESEG_MAX_S, no punctuation) with a SMALL gap early and a
    # LARGER gap later. The forced break must land at the largest gap (the more
    # sentence-like pause), not the first qualifying breath.
    words = []
    t = 0.0
    for k in range(30):
        words.append({"word": f"w{k}", "start": t, "end": t + 0.5, "score": 1.0})
        gap = 0.1
        if k == 5:
            gap = 0.6  # modest pause, qualifies but is not the largest
        elif k == 12:
            gap = 1.5  # the largest pause -> the preferred break
        t += 0.5 + gap
    caps = _split_words_into_captions(words)
    assert len(caps) >= 2
    assert caps[0][-1]["word"] == "w12"  # cut at the largest gap, not w5


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
    assert out["model"] == "parakeet-tdt-0.6b-v3"  # stale "(local-attn)" dropped
    assert out["produced_by_version"] == PRODUCED_BY_VERSION == "0.4.0"
    assert out["language"] == "en" and out["compute_type"] == "float16"
    assert out["duration_s"] == pytest.approx(seg["end"])
    word = seg["words"][0]
    assert set(word) >= {"start", "end", "word", "score"}
    assert word["score"] == 0.9  # per-word confidence carried through for debug


def test_long_unpunctuated_segment_is_resegmented_at_pauses(tmp_path):
    # A chunk that collapsed into one 40s unpunctuated run-on. Words carry
    # timestamps with clear pauses; finalize must split it into caption-sized
    # pieces (<= RESEG_MAX_S) broken at those pauses.
    words = []
    t = 0.0
    for i in range(40):
        words.append({"word": f"w{i}", "start": t, "end": t + 0.4, "score": 1.0})
        # insert a 0.8s pause every 8 words
        t += 0.4 + (0.8 if (i + 1) % 8 == 0 else 0.1)
    seg = [
        {
            "start": 0.0,
            "end": words[-1]["end"],
            "text": " ".join(w["word"] for w in words),  # no punctuation
            "words": words,
        }
    ]
    work = tmp_path / "work"
    _write_parakeet(work, seg)
    finalize_run(work)
    out = json.loads((tmp_path / "transcript.json").read_text())
    segs = out["segments"]
    assert len(segs) > 1  # the run-on got split
    assert all((s["end"] - s["start"]) <= 12.0 + 1e-6 for s in segs)
    # coverage preserved: first start and last end unchanged
    assert segs[0]["start"] == 0.0
    assert segs[-1]["end"] == pytest.approx(words[-1]["end"])


def test_abbreviation_split_is_merged(tmp_path):
    # Parakeet splits after "Dr." (trailing period); captions must re-merge it.
    segs = [
        {"start": 0.0, "end": 2.0, "text": "I am a surgeon, Dr.",
         "words": [{"word": "Dr.", "start": 1.8, "end": 2.0, "score": 1.0}]},
        {"start": 2.1, "end": 2.6, "text": "Hun!",
         "words": [{"word": "Hun!", "start": 2.1, "end": 2.6, "score": 1.0}]},
        {"start": 3.0, "end": 4.0, "text": "Okay.",
         "words": [{"word": "Okay.", "start": 3.0, "end": 4.0, "score": 1.0}]},
    ]
    work = tmp_path / "work"
    _write_parakeet(work, segs)
    finalize_run(work)
    out = json.loads((tmp_path / "transcript.json").read_text())["segments"]
    assert out[0]["text"] == "I am a surgeon, Dr. Hun!"  # merged
    assert out[0]["end"] == pytest.approx(2.6)
    assert out[1]["text"] == "Okay."  # non-abbrev segment untouched


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


def test_reads_v2_diarization_turns(tmp_path):
    # Task 1's v2 dict: turns live under "turns"; embeddings etc. are ignored by
    # finalize. Segments must still get speakers and diarized must be True.
    work = tmp_path / "work"
    _write_parakeet(work, _one_segment())
    (work / "diarization.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "turns": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
                "embeddings": {"SPEAKER_00": [0.0] * 256},
                "embedding_model": "pyannote/embedding",
                "embedding_dim": 256,
            }
        )
    )
    finalize_run(work)
    out = json.loads((tmp_path / "transcript.json").read_text())
    assert out["segments"][0]["speaker"] == "SPEAKER_00"
    assert out["segments"][0]["words"][0]["speaker"] == "SPEAKER_00"
    assert out["diarized"] is True


def test_reads_legacy_list_diarization(tmp_path):
    # v1 bare-list form must keep working (back-compat for older runs).
    out = _run_finalize(tmp_path, with_diar=True)
    assert out["segments"][0]["speaker"] == "SPEAKER_00"
    assert out["diarized"] is True


def test_empty_v2_is_not_diarized(tmp_path):
    # A v2 dict with no turns -> not diarized, speakers all None.
    work = tmp_path / "work"
    _write_parakeet(work, _one_segment())
    (work / "diarization.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "turns": [],
                "embeddings": {},
                "embedding_model": "pyannote/embedding",
                "embedding_dim": 256,
            }
        )
    )
    finalize_run(work)
    out = json.loads((tmp_path / "transcript.json").read_text())
    assert out["diarized"] is False
    assert all(s["speaker"] is None for s in out["segments"])
