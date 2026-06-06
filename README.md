# modular-transcriptor

Combines several speech-to-text models into one high-accuracy transcript. It runs
three engines over the same audio and merges them so each covers the others'
weaknesses:

- **NVIDIA Parakeet** (`parakeet-tdt-0.6b-v3`) — fast, accurate word timing; the base text.
- **Meta MMS-CTC** (`facebook/mms-1b-all`, eng) — a phonetic acoustic transcription used as the **referee** when the engines disagree.
- **WhisperX** (`large-v3`) — punctuation, casing, and correction of mis-heard runs.
- **pyannote** speaker diarization (optional) — labels each segment with a speaker.

The merge keeps Parakeet's words where the acoustics agree, and only overrides
them when WhisperX **and** MMS agree against it (so a localized mis-hear like
`Sorry, Kate` → `Fuck it` gets corrected without disturbing the rest of the line).

Give it a video or audio file; you get `transcript.json` + `transcript.csv`.

## Requirements

- **Python 3.11**, a **CUDA GPU**, and **`ffmpeg`** on `PATH`.
- Heavy ML deps (installed automatically): `torch`, `torchaudio`, `whisperx`,
  `faster-whisper`, `nemo_toolkit[asr]`, `pyannote.audio`. First run downloads the
  model weights from HuggingFace (several GB).
- **For speaker diarization only:** a HuggingFace token and one-time acceptance of
  the gated pyannote model (see *Diarization* below). Not needed if you pass
  `--no-diarization`.

## Install

```bash
git clone https://github.com/max2fly/modular-transcriptor.git
cd modular-transcriptor
pip install -e .
```

(A fresh virtualenv or conda env on Python 3.11 is recommended.)

## Quick start

```bash
# No speaker labels — no HuggingFace setup needed:
modular-transcriptor input.mp4 --no-diarization

# Full pipeline including speaker labels:
echo "HF_TOKEN=hf_xxx" > .env          # see Diarization
modular-transcriptor input.mp4
```

Outputs (next to the input by default): `input.transcript.json` and
`input.transcript.csv`. Any ffmpeg-readable video or audio file works — the audio
is extracted automatically.

## Diarization (speaker labels)

Diarization is **on by default** and **fails loudly** if it can't run, so you
never get a silently speaker-less transcript by accident:

1. Create a HuggingFace token at <https://huggingface.co/settings/tokens>.
2. Accept the gated model terms for the pyannote speaker-diarization model on its
   HuggingFace page (one time).
3. Provide the token via a `.env` file in the working directory (auto-loaded):

   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxx
   ```

   (`.env.example` shows the format. `.env` is gitignored — never commit your
   token. An exported `HF_TOKEN` environment variable also works and takes
   precedence over `.env`.)

If the token is missing, or the model isn't available/accepted, the run stops
with a clear message. To transcribe **without** speakers, pass
`--no-diarization` — it skips diarization entirely and needs no HuggingFace setup.

## CLI options

```
modular-transcriptor <input> [options]
```

| Option | Description |
|---|---|
| `<input>` | Video or audio file (anything ffmpeg can read). |
| `--out PATH` | Output `transcript.json` path. Default: `<input>.transcript.json`. The CSV is written next to it as `…transcript.csv`. |
| `--no-diarization` | Skip speaker diarization (no HuggingFace token/model needed). |
| `--work DIR` | Use `DIR` for intermediate files instead of a temp dir (kept after the run; lets a re-run resume from completed stages). |
| `--keep-work` | Keep the temp work directory instead of deleting it. |

### Tuning (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `large-v3` | WhisperX model size. |
| `WHISPER_COMPUTE_TYPE` | `float16` | WhisperX compute precision. |
| `WHISPER_BATCH_SIZE` | `32` | WhisperX batch size (lower if GPU memory is tight). |
| `HF_TOKEN` | — | HuggingFace token for diarization (or put it in `.env`). |

## Output

**`transcript.csv`** — one row per segment, easy to skim or import:

```csv
time,speaker,text
0:00:08,SPEAKER_00,Fuck it! I don't care about the timer!
0:00:11,SPEAKER_00,I don't give a shit!
```

`time` is `H:MM:SS` from the segment start; `speaker` is empty when diarization is off.

**`transcript.json`** — structured, with per-word timings:

```json
{
  "language": "en",
  "model": "parakeet-tdt-0.6b-v3 + mms-1b-all + whisperx-large-v3 (merge)",
  "diarized": true,
  "duration_s": 90.1,
  "segments": [
    {
      "start": 8.23, "end": 10.76,
      "text": "Fuck it! I don't care about the timer!",
      "speaker": "SPEAKER_00",
      "kind": "speech",
      "source": "parakeet+run_arb",
      "words": [{"start": 8.23, "end": 8.46, "word": "Fuck"}]
    }
  ]
}
```

`source` records how each segment was decided (e.g. `parakeet`, `whisper_corrected`,
`parakeet+run_arb`) — useful for debugging. `kind` is `speech` or `non_speech_vocal`.

## How it runs

The pipeline runs as a sequence of stages, each in its own subprocess so its GPU
model is freed before the next loads: **amplify** (16 kHz mono + gentle gain) →
**parakeet** → **mms** → **whisper** → **pyannote** (unless `--no-diarization`) →
**merge**. Each stage is idempotent (skips if its output already exists), so a run
pointed at an existing `--work` dir resumes where it left off.

## Disclaimer

This is most likely overkill for 99% of the use cases, as just running base whisper gives 95% accuracy, and the final percents take exponential amount of compute.

This project is specifically useful where you need near perfect transcription without human supervision.

There is definitely still room for improvement due to so many moving parts and it is definitely not in the final state with many bugs still being discovered though the continued use.
