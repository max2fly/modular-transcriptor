from __future__ import annotations

import json
from pathlib import Path

from modular_transcriptor import transcript_merger as tm
from modular_transcriptor.formats import write_transcript_csv
from modular_transcriptor.speakers import assign_speakers
from modular_transcriptor.vad import derive_vad_intervals


def _load_mms_and_audio(work: Path):
    import numpy as np
    import torchaudio
    from transformers import AutoModelForCTC, AutoProcessor

    wf, sr = torchaudio.load(str(work / "audio.amplified.wav"))
    if wf.shape[0] > 1:
        wf = wf.mean(dim=0, keepdim=True)
    audio = wf.squeeze(0).numpy().astype(np.float32)
    proc = AutoProcessor.from_pretrained(tm.MMS_MODEL_ID)
    model = AutoModelForCTC.from_pretrained(tm.MMS_MODEL_ID).to("cuda").eval()
    proc.tokenizer.set_target_lang(tm.MMS_LANG_ADAPTER)
    model.load_adapter(tm.MMS_LANG_ADAPTER)
    pad_id = proc.tokenizer.pad_token_id
    delim = proc.tokenizer.encoder.get("|")
    id_to_char = {v: k for k, v in proc.tokenizer.encoder.items()}
    return audio, sr, proc, model, pad_id, delim, id_to_char


def run_merge(work: Path, emit) -> None:
    if (work.parent / "transcript.json").exists():
        return  # idempotent
    parakeet = json.loads((work / "parakeet.json").read_text())
    mms_chars = json.loads((work / "mms_ctc.json").read_text())
    whisper = json.loads((work / "whisper.json").read_text())
    diar = json.loads((work / "diarization.json").read_text())
    # Forward-only rigid re-anchor of noise-anchored Whisper drift BEFORE VAD
    # derivation, so the noise location loses speech activity (no phantom
    # gap-fill) and the true location gains it.
    whisper = tm.reanchor_whisper_segments(whisper, parakeet)
    vad = derive_vad_intervals(whisper.get("segments", []))
    (work / "vad.json").write_text(json.dumps(vad))

    emit("loading MMS + audio", 5.0)
    audio, sr, proc, model, pad_id, delim, id_to_char = _load_mms_and_audio(work)

    # Map the merge's internal 0..1 progress into this stage's 30..99% band so
    # the infobox shows which merge sub-stage is running instead of a static
    # "merging" for the whole (potentially minutes-long) pass.
    def _merge_prog(label: str, frac: float) -> None:
        # The worker prefixes the stage name ("merge: ..."), so emit only the
        # sub-stage label here.
        emit(label, 30.0 + max(0.0, min(frac, 1.0)) * 69.0)

    merger = tm.TranscriptMerger(
        parakeet,
        mms_chars,
        whisper,
        vad,
        audio,
        sr,
        proc,
        model,
        pad_id,
        delim,
        id_to_char,
    )
    lean = merger.run(progress_cb=_merge_prog)
    emit("assigning speakers", 99.0)
    assign_speakers(lean, diar)

    duration_s = float(whisper.get("duration_s", 0.0))
    diarized = bool(diar)
    out = work.parent
    (out / "transcript.json").write_text(
        json.dumps(
            {
                "language": "en",
                "model": "parakeet-tdt-0.6b-v3 + mms-1b-all + whisperx-large-v3 (merge)",
                "compute_type": "float16",
                "diarized": diarized,
                "duration_s": duration_s,
                "produced_by_version": "0.3.0",
                "segments": lean,
            },
            ensure_ascii=False,
        )
    )
    write_transcript_csv(out / "transcript.csv", lean)
    (out / "transcript.debug.json").write_text(
        json.dumps({"segments": merger.diagnostic_segments()}, ensure_ascii=False)
    )
    (out / "transcript.debug.md").write_text(
        tm.render_debug_md(merger.diagnostic_segments())
    )
