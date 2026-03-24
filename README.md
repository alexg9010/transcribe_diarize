# transcribe_diarize

Transcribe audio files with speaker labels using [Whisper](https://github.com/openai/whisper) (speech-to-text) and [pyannote.audio](https://github.com/pyannote/pyannote-audio) (speaker diarization). Runs entirely locally.

**Supports:** `.m4a`, `.mp3`, `.wav`, `.flac`, `.ogg`, `.webm`, and most other common audio formats.

---

## Output

```
SPEAKER_00  [0:12 → 0:45]
Hi everyone, thanks for joining today's call.

SPEAKER_01  [0:46 → 1:03]
Happy to be here. Let's get started.

SPEAKER_00  [1:04 → 1:30]
Great. So the first item on the agenda...
```

Or use `--output result.json` for structured JSON with per-segment timestamps.

---

## Setup

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip / brew / winget — see https://docs.astral.sh/uv/getting-started/installation/
```

No need to manually create a virtualenv or install packages — uv reads the inline dependency metadata at the top of the script and handles everything automatically on first run.

> **GPU users (optional):** If you have a CUDA-capable GPU, uv will install CPU-only PyTorch by default. To get the CUDA build, install it manually into uv's managed env or use a `uv.lock`-based project. The script detects and uses CUDA automatically if available.

### 2. Accept the pyannote model terms

pyannote.audio uses gated models on Hugging Face. You need to accept the terms for both (free, one-time):

- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### 3. Create a Hugging Face access token

Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a token. Read-only access is sufficient.

Set it as an environment variable so you don't have to pass it every time:

```bash
# macOS / Linux
export HF_TOKEN=your_token_here

# Windows (Command Prompt)
set HF_TOKEN=your_token_here

# Windows (PowerShell)
$env:HF_TOKEN="your_token_here"
```

---

## Usage

```bash
uv run transcribe_diarize.py <audio_file> [options]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--hf_token` | `$HF_TOKEN` | Hugging Face access token |
| `--num_speakers` | auto | Number of speakers if known — improves diarization accuracy |
| `--output` | `<name>_transcript.txt` | Output path. Use `.json` extension for structured output |

### Examples

```bash
# Basic usage (reads HF_TOKEN from env)
uv run transcribe_diarize.py interview.m4a

# Higher accuracy model
uv run transcribe_diarize.py interview.m4a --model medium

# Known speaker count
uv run transcribe_diarize.py interview.m4a --num_speakers 2

# Save as JSON
uv run transcribe_diarize.py interview.m4a --output interview.json

# All options
uv run transcribe_diarize.py interview.m4a --model large --num_speakers 3 --output result.txt
```

---

## Model sizes

Larger models are more accurate but slower and require more memory.

| Model | ~Size | Relative speed | Best for |
|---|---|---|---|
| `tiny` | 75 MB | Fastest | Quick drafts, testing |
| `base` | 145 MB | Fast | Most use cases (default) |
| `small` | 465 MB | Moderate | Better accuracy |
| `medium` | 1.5 GB | Slow | High accuracy |
| `large` | 3 GB | Slowest | Best possible accuracy |

---

## JSON output format

When using `--output result.json`, each segment looks like:

```json
[
  {
    "start": 12.4,
    "end": 45.1,
    "speaker": "SPEAKER_00",
    "text": "Hi everyone, thanks for joining today's call."
  },
  ...
]
```

---

## Troubleshooting

**`401 Unauthorized` from Hugging Face**
You haven't accepted the model terms, or your token is invalid. Re-check steps 2 and 3 above.

**`ffmpeg not found`**
Whisper requires ffmpeg to decode audio. Install it:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

**Out of memory**
Try a smaller `--model`, or ensure no other large processes are using your GPU/RAM.

**Poor speaker separation**
Pass `--num_speakers N` if you know the count. Also try `--model small` or larger — better transcription helps alignment.

---

## How it works

1. **Whisper** transcribes the audio into text segments with timestamps.
2. **pyannote.audio** independently segments the audio by speaker, assigning each time range a speaker label.
3. The script overlaps the two outputs, assigning each transcript segment to whichever speaker dominated that time window.

---

## License

Whisper is released under the [MIT License](https://github.com/openai/whisper/blob/main/LICENSE). pyannote.audio is released under the [MIT License](https://github.com/pyannote/pyannote-audio/blob/develop/LICENSE.txt). Check each project's terms for your use case.
