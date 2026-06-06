from __future__ import annotations

from modular_transcriptor import formats


def test_csv_rows_format():
    segs = [
        {"start": 0.0, "speaker": "SPEAKER_00", "text": "Hello there"},
        {"start": 3725.4, "speaker": "", "text": "line one\nline two"},
        {"start": 65.0, "text": "no speaker key"},
    ]
    rows = formats.transcript_csv_rows(segs)
    assert rows[0] == ["time", "speaker", "text"]
    assert rows[1] == ["0:00:00", "SPEAKER_00", "Hello there"]
    # 3725.4s -> 1:02:05 ; newline collapsed to space
    assert rows[2] == ["1:02:05", "", "line one line two"]
    # missing speaker key -> empty string
    assert rows[3] == ["0:01:05", "", "no speaker key"]


def test_write_transcript_csv(tmp_path):
    out = tmp_path / "transcript.csv"
    formats.write_transcript_csv(out, [{"start": 1.0, "speaker": "A", "text": "hi"}])
    body = out.read_text(encoding="utf-8")
    assert body.splitlines()[0] == "time,speaker,text"
    assert "0:00:01,A,hi" in body
    assert not (tmp_path / "transcript.csv.partial").exists()
