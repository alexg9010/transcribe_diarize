# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "openai-whisper",
#   "pyannote.audio==4.0.0",
#   "torch==2.8.0",
#   "torchaudio==2.8.0",
#   "python-dotenv",
# ]
# ///
"""
Transcribe an audio file with speaker diarization using
Whisper (transcription) and pyannote.audio 4.x (speaker segmentation).

Uses the pyannote/speaker-diarization-community-1 pipeline.
See README.md for full setup instructions and usage examples.
"""

import argparse
import contextlib
import functools
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def configure_warning_filters():
    """Silence known non-fatal runtime warnings from third-party audio deps."""
    warnings.filterwarnings(
        "ignore",
        message="FP16 is not supported on CPU; using FP32 instead",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"pyannote\.audio\.core\.io",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"speechbrain\.utils\.torch_audio_backend",
    )


@functools.lru_cache(maxsize=None)
def require_dependency(import_name: str, package_name: str):
    """Import a runtime dependency with a useful install hint on failure."""
    try:
        module = __import__(import_name)
        for part in import_name.split(".")[1:]:
            module = getattr(module, part)
    except ImportError as exc:
        print(
            f"Error: missing dependency '{package_name}' ({exc}). "
            f"Run this script via `uv run transcribe_diarize.py ...` or install {package_name}.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return module


def choose_device(torch_module) -> str:
    """Prefer CUDA, then MPS, otherwise CPU."""
    if torch_module.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def load_audio_for_pyannote(audio_path: str):
    """Preload audio with Whisper/ffmpeg to avoid pyannote's torchcodec decoder path."""
    torch = require_dependency("torch", "torch")
    whisper = require_dependency("whisper", "openai-whisper")
    audio = whisper.load_audio(audio_path)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": 16000}


@contextlib.contextmanager
def pyannote_torch_load_compat(torch_module):
    """Restore pre-2.6 torch.load behavior for trusted pyannote checkpoints."""
    original_load = torch_module.load

    def compat_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch_module.load = compat_load
    try:
        yield
    finally:
        torch_module.load = original_load


def ensure_ffmpeg_available():
    if shutil.which("ffmpeg") is None:
        print(
            "Error: ffmpeg not found in PATH. Install ffmpeg before running transcription.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def transcribe(audio_path: str, model_size: str = "base") -> list[dict]:
    """Run Whisper and return list of {start, end, text} segments."""
    whisper = require_dependency("whisper", "openai-whisper")
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path, word_timestamps=False)
    return result["segments"]


def diarize(audio_path: str, hf_token: str, num_speakers: int | None = None) -> object:
    """Run pyannote diarization and return annotation object."""
    torch = require_dependency("torch", "torch")
    pyannote_audio = require_dependency("pyannote.audio", "pyannote.audio")
    Pipeline = pyannote_audio.Pipeline

    print("Running speaker diarization...")
    with pyannote_torch_load_compat(torch):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )
    device = choose_device(torch)
    if device != "cpu":
        print(f"Using {device.upper()} for diarization")
        pipeline.to(torch.device(device))
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    return pipeline(load_audio_for_pyannote(audio_path), **kwargs)


def get_annotation_from_diarization(diarization) -> object:
    """Support both legacy Annotation output and pyannote 4 DiarizeOutput."""
    if hasattr(diarization, "exclusive_speaker_diarization"):
        return diarization.exclusive_speaker_diarization
    if hasattr(diarization, "speaker_diarization"):
        return diarization.speaker_diarization
    return diarization


def assign_speakers(segments: list[dict], diarization) -> list[dict]:
    """Match each Whisper segment to the dominant diarized speaker."""
    annotation = get_annotation_from_diarization(diarization)
    tracks = list(annotation.itertracks(yield_label=True))
    results = []
    for seg in segments:
        seg_start, seg_end = seg["start"], seg["end"]
        speaker_time: dict[str, float] = {}
        for turn, _, speaker in tracks:
            if turn.end <= seg_start:
                continue
            if turn.start >= seg_end:
                break
            overlap = min(seg_end, turn.end) - max(seg_start, turn.start)
            if overlap > 0:
                speaker_time[speaker] = speaker_time.get(speaker, 0) + overlap
        dominant = max(speaker_time, key=speaker_time.get) if speaker_time else "UNKNOWN"
        results.append({
            "start": round(seg_start, 2),
            "end": round(seg_end, 2),
            "speaker": dominant,
            "text": seg["text"].strip(),
        })
    return results


def format_output(segments: list[dict]) -> str:
    """Format as readable transcript with speaker labels."""
    lines = []
    current_speaker = None
    buffer = []

    def flush(speaker, segs):
        if segs:
            ts = f"[{fmt_time(segs[0]['start'])} → {fmt_time(segs[-1]['end'])}]"
            lines.append(f"\n{speaker}  {ts}")
            lines.append(" ".join(s["text"] for s in segs))

    for seg in segments:
        if seg["speaker"] != current_speaker:
            flush(current_speaker, buffer)
            current_speaker = seg["speaker"]
            buffer = [seg]
        else:
            buffer.append(seg)
    flush(current_speaker, buffer)
    return "\n".join(lines).strip()


def apply_speaker_names(text: str, speakers: dict[str, str]) -> str:
    """Replace SPEAKER_XX labels with provided names."""
    for label, name in speakers.items():
        text = text.replace(label, name)
    return text


def parse_speakers(speakers_str: str) -> dict[str, str]:
    """Parse speaker mappings.

    Accepts either format:
      - Ordered list:  'Alex,Ahmed'        → SPEAKER_00=Alex, SPEAKER_01=Ahmed
      - Explicit map:  'SPEAKER_00=Alex,SPEAKER_01=Ahmed'
    """
    parts = [p.strip() for p in speakers_str.split(",")]
    if "=" not in speakers_str:
        return {f"SPEAKER_{i:02d}": name for i, name in enumerate(parts)}
    mapping = {}
    for pair in parts:
        if "=" not in pair:
            print(f"Warning: ignoring invalid speaker mapping '{pair}' (expected KEY=VALUE)", file=sys.stderr)
            continue
        label, name = pair.split("=", 1)
        mapping[label.strip()] = name.strip()
    return mapping


def load_prompt_template() -> str:
    """Load the summarization prompt template from prompt_summarize.txt."""
    prompt_path = Path(__file__).parent / "prompt_summarize.txt"
    if not prompt_path.exists():
        print(f"Error: prompt template not found: {prompt_path}", file=sys.stderr)
        raise SystemExit(1)
    return prompt_path.read_text(encoding="utf-8")


def summarize(transcript: str, ollama_model: str = "llama3.2") -> str:
    """Summarize a transcript using Ollama."""
    if shutil.which("ollama") is None:
        print("Error: ollama not found in PATH. Install from https://ollama.com", file=sys.stderr)
        raise SystemExit(1)

    prompt = load_prompt_template() + transcript

    print(f"Summarizing with {ollama_model}...")
    result = subprocess.run(
        ["ollama", "run", ollama_model],
        input=prompt,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error: ollama failed: {result.stderr.strip()}", file=sys.stderr)
        raise SystemExit(1)
    return result.stdout.strip()


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def main():
    configure_warning_filters()
    parser = argparse.ArgumentParser(description="Transcribe + diarize an audio file.")
    parser.add_argument("audio", help="Path to audio file (mp3, m4a, wav, etc.)")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base). Larger = more accurate but slower.")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token (or set HF_TOKEN env var).")
    parser.add_argument("--num_speakers", type=int, default=None,
                        help="Number of speakers if known (optional, improves accuracy).")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: <audio_name>_transcript.txt). Use .json for raw JSON.")
    parser.add_argument("--summarize", action="store_true",
                        help="Summarize the transcript using Ollama (requires ollama to be installed).")
    parser.add_argument("--ollama-model", default="llama3.2",
                        help="Ollama model to use for summarization (default: llama3.2).")
    parser.add_argument("--speakers", default=None,
                        help="Speaker names: 'Alex,Ahmed' (by order) or 'SPEAKER_00=Alex,SPEAKER_01=Ahmed'.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run transcription even if output file already exists.")
    args = parser.parse_args()

    audio_path = args.audio
    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    stem = Path(audio_path).stem
    out_path = args.output or f"{stem}_transcript.txt"
    out_file = Path(out_path)
    if out_file.parent != Path("."):
        out_file.parent.mkdir(parents=True, exist_ok=True)

    speaker_map = parse_speakers(args.speakers) if args.speakers else {}

    if out_file.exists() and not args.force:
        print(f"Transcript already exists: {out_file}")
        if args.summarize:
            print("Reusing existing transcript for summarization...")
            if out_path.endswith(".json"):
                with open(out_file, encoding="utf-8") as f:
                    data = json.load(f)
                segments = data["segments"] if isinstance(data, dict) else data
                transcript = format_output(segments)
            else:
                with open(out_file, encoding="utf-8") as f:
                    transcript = f.read()
            if speaker_map:
                transcript = apply_speaker_names(transcript, speaker_map)
            summary = summarize(transcript, ollama_model=args.ollama_model)
            print(f"\n{summary}\n")
        else:
            print("Use --force to re-run, or --summarize to summarize the existing transcript.")
        return

    if not args.hf_token:
        print("Error: Hugging Face token required. Pass --hf_token or set HF_TOKEN env var.")
        print("Get a token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    ensure_ffmpeg_available()

    try:
        segments = transcribe(audio_path, model_size=args.model)
        diarization = diarize(audio_path, hf_token=args.hf_token, num_speakers=args.num_speakers)
        labeled = assign_speakers(segments, diarization)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)

    transcript = format_output(labeled)
    if speaker_map:
        transcript = apply_speaker_names(transcript, speaker_map)

    if out_path.endswith(".json"):
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(labeled, f, indent=2, ensure_ascii=False)
    else:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(transcript)
    print(f"Saved transcript to {out_file}")

    if args.summarize:
        summary = summarize(transcript, ollama_model=args.ollama_model)
        print(f"\n{summary}\n")
        summary_file = out_file.with_suffix(".summary.md")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary + "\n")
        print(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    main()
