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

### 1. Run the installer

```bash
./install.sh
```

The installer currently supports:

- macOS with Homebrew
- Ubuntu/Debian with `apt`

It will:

- install required dependencies if missing: `uv`, `ffmpeg`
- run `uv sync` to create the managed Python environment
- detect optional `ollama` support and, in interactive terminals, ask whether to install it
- check whether `HF_TOKEN` is already set and print the next steps if not

On unsupported systems, it prints the manual commands you need and exits.

> **GPU users (optional):** If you have a CUDA-capable GPU, uv will install CPU-only PyTorch by default. To get the CUDA build, install it manually into uv's managed env. The script detects and uses CUDA/MPS automatically if available.

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

The installer checks whether `HF_TOKEN` is set, but it does not write `.env` files or modify your shell profile for you.

---

## Usage

```bash
uv run transcribe_diarize.py <audio_file> [options]
```

To see all available options:

```bash
uv run transcribe_diarize.py --help
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--hf_token` | `$HF_TOKEN` | Hugging Face access token |
| `--num_speakers` | auto | Number of speakers if known. Must be a positive integer |
| `--output-prefix` | `<audio_stem>` | Prefix used for generated files like `_transcript.txt`, `_transcript.json`, `_summary.md` |
| `--json` | off | Also save the transcript as JSON |
| `--only-json` | off | Save only the JSON transcript |
| `--summarize` | off | Summarize the transcript using Ollama |
| `--ollama-model` | `llama3.2` | Ollama model for summarization |
| `--save-summary [path]` | stdout only | Optionally save the summary. If no path is given, uses `<output_prefix>_summary.md` |
| `--speakers` | — | Speaker names: `'Alex,Ahmed'` (by order) or `'SPEAKER_00=Alex,SPEAKER_01=Ahmed'` |
| `--force` | off | Re-run transcription even if output file already exists |

### Examples

```bash
# Basic usage (reads HF_TOKEN from env)
uv run transcribe_diarize.py interview.m4a

# Higher accuracy model
uv run transcribe_diarize.py interview.m4a --model medium

# Known speaker count
uv run transcribe_diarize.py interview.m4a --num_speakers 2

# Save both text and JSON transcripts
uv run transcribe_diarize.py interview.m4a --json

# Save only JSON transcript
uv run transcribe_diarize.py interview.m4a --only-json

# Summarize with Ollama (requires ollama to be installed)
uv run transcribe_diarize.py interview.m4a --summarize

# Summarize and save to the default summary path
uv run transcribe_diarize.py interview.m4a --summarize --save-summary

# Summarize and save to a custom path
uv run transcribe_diarize.py interview.m4a --summarize --save-summary notes/interview.md

# Summarize with speaker names (in order of appearance)
uv run transcribe_diarize.py interview.m4a --summarize --speakers 'Alex,Ahmed'

# Use a different Ollama model
uv run transcribe_diarize.py interview.m4a --summarize --ollama-model mistral

# Custom output prefix
uv run transcribe_diarize.py interview.m4a --output-prefix outputs/interview --json --summarize --save-summary
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

When using `--json` or `--only-json`, the generated `_transcript.json` file looks like:

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

## Summarization

Summarization uses [Ollama](https://ollama.com) to run a local LLM. If `ollama` is missing, `./install.sh` offers to install it in interactive terminals. You can also install it manually and pull a model before using `--summarize`:

```bash
# Install Ollama (see https://ollama.com for other platforms)
brew install ollama

# Pull a model
ollama pull llama3.2
```

The `--summarize` flag sends the transcript to Ollama with a prompt template (`prompt_summarize.txt`) that:

1. **Infers speaker identities** from context clues (introductions, names mentioned)
2. **Produces a structured summary** with overview, key points, and action items

Summaries are printed to stdout by default. If you want to keep one, pass `--save-summary` to use the default path, or `--save-summary path/to/summary.md` for a custom location.

If the LLM can't identify speakers, use `--speakers` to provide names manually:

```bash
uv run transcribe_diarize.py meeting.m4a --summarize --speakers 'SPEAKER_00=Alex,SPEAKER_01=Ahmed'
```

You can customize `prompt_summarize.txt` to change the summary format, or use a different model with `--ollama-model`.

---

## How it works

1. **Whisper** transcribes the audio into text segments with timestamps.
2. **pyannote.audio** independently segments the audio by speaker, assigning each time range a speaker label.
3. The script overlaps the two outputs, assigning each transcript segment to whichever speaker dominated that time window.

---

## License

This project is released under the [MIT License](LICENSE).

Whisper is released under the [MIT License](https://github.com/openai/whisper/blob/main/LICENSE). pyannote.audio is released under the [MIT License](https://github.com/pyannote/pyannote-audio/blob/develop/LICENSE.txt). Check each project's terms for your use case.
