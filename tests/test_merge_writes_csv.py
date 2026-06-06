from __future__ import annotations

import json

import modular_transcriptor.merge_runner as mr


def test_run_merge_writes_csv(tmp_path, monkeypatch):
    work = tmp_path / "transcription_work"
    work.mkdir()
    (work / "parakeet.json").write_text(json.dumps({"segments": []}))
    (work / "mms_ctc.json").write_text(json.dumps([]))
    (work / "whisper.json").write_text(json.dumps({"segments": [], "duration_s": 0}))
    (work / "diarization.json").write_text("[]")

    lean = [{"start": 1.0, "end": 2.0, "text": "hello", "speaker": "A", "words": []}]

    # Stub the heavy pieces: reanchor passthrough, VAD empty, MMS/audio load no-op,
    # and a fake merger that returns our lean segments.
    monkeypatch.setattr(mr.tm, "reanchor_whisper_segments", lambda w, p: w)
    monkeypatch.setattr(mr, "derive_vad_intervals", lambda segs: [])
    monkeypatch.setattr(
        mr, "_load_mms_and_audio", lambda work: (None, 16000, None, None, 0, 0, {})
    )

    class _FakeMerger:
        def __init__(self, *a, **k):
            pass

        def run(self, progress_cb=None):
            return lean

        def diagnostic_segments(self):
            return lean

    monkeypatch.setattr(mr.tm, "TranscriptMerger", _FakeMerger)
    monkeypatch.setattr(mr, "assign_speakers", lambda lean, diar: None)

    mr.run_merge(work, lambda label, pct=None: None)

    out = work.parent
    assert (out / "transcript.json").exists()
    csv_body = (out / "transcript.csv").read_text(encoding="utf-8")
    assert csv_body.splitlines()[0] == "time,speaker,text"
    assert "0:00:01,A,hello" in csv_body
