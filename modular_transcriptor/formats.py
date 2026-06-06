from __future__ import annotations

import csv as _csv
from io import StringIO
from pathlib import Path

CSV_HEADER = ["time", "speaker", "text"]


def _fmt_time(start) -> str:
    """Whole-second H:MM:SS (hours not zero-padded) — matches the webgui export."""
    s = int(round(float(start or 0.0)))
    return f"{s // 3600:d}:{(s // 60) % 60:02d}:{s % 60:02d}"


def transcript_csv_rows(segments) -> list[list[str]]:
    """[header, *rows] for a transcript's segments. Each row is
    [time, speaker, text]; text is stripped with newlines collapsed to spaces;
    a missing/None speaker becomes ''."""
    rows: list[list[str]] = [list(CSV_HEADER)]
    for seg in segments or []:
        text = (seg.get("text") or "").strip().replace("\n", " ")
        rows.append([_fmt_time(seg.get("start")), seg.get("speaker") or "", text])
    return rows


def transcript_csv_string(segments) -> str:
    buf = StringIO()
    _csv.writer(buf).writerows(transcript_csv_rows(segments))
    return buf.getvalue()


def write_transcript_csv(out_path, segments) -> Path:
    """Atomically write transcript.csv (via a .partial temp then replace)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    with tmp.open("w", newline="", encoding="utf-8") as fp:
        _csv.writer(fp).writerows(transcript_csv_rows(segments))
    tmp.replace(out_path)
    return out_path
