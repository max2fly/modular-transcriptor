from __future__ import annotations

import modular_transcriptor.transcript_merger as m


def test_run_emits_monotonic_progress():
    parakeet = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hi",
                "words": [{"start": 0.0, "end": 1.0, "word": "hi"}],
            }
        ]
    }
    whisper = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hi",
                "words": [{"word": "hi", "start": 0.0, "end": 1.0, "score": 0.9}],
            }
        ]
    }
    mms = [{"char": "h", "t": 0.0}, {"char": "i", "t": 0.1}]

    calls: list = []
    merger = m.TranscriptMerger(parakeet, mms, whisper, [], audio=None, sr=16000)
    merger.run(progress_cb=lambda label, frac: calls.append((label, frac)))

    assert calls, "progress_cb was never invoked"
    fracs = [f for _, f in calls]
    assert all(0.0 <= f <= 1.0 for f in fracs)
    assert fracs == sorted(fracs)  # non-decreasing
    assert fracs[0] == 0.0
    assert all(isinstance(label, str) and label for label, _ in calls)
