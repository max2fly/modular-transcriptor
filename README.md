# modular-transcriptor

Combines several speech-to-text models (NVIDIA Parakeet + Meta MMS-CTC +
WhisperX) into one high-accuracy transcript, with optional pyannote speaker
diarization.

## Requirements

- Python 3.11, a CUDA GPU, and `ffmpeg` on PATH.
- For diarization: a HuggingFace token (`HF_TOKEN`) and acceptance of the gated
  `pyannote/speaker-diarization` model. Or skip it with `--no-diarization`.

## Install

```bash
pip install -e .        # heavy deps: torch, whisperx, nemo, pyannote.audio
```

## Usage

```bash
# Full pipeline incl. speaker labels (needs HF_TOKEN + accepted pyannote model):
export HF_TOKEN=hf_xxx
modular-transcriptor input.mp4

# No diarization (no HF setup needed):
modular-transcriptor input.mp4 --no-diarization

# Custom output path / keep intermediates:
modular-transcriptor input.mp4 --out out.json --keep-work
```

Outputs `<input>.transcript.json` and `<input>.transcript.csv` (or `--out` path).
Any ffmpeg-readable video or audio file works.
