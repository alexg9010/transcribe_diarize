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
transcribe_diarize.py
Transcribes an audio file with speaker diarization using
Whisper (transcription) + pyannote.audio (speaker segmentation).

SETUP
-----
1. Install uv if you haven't already:
   https://docs.astral.sh/uv/getting-started/installation/

2. Accept the pyannote community diarization model terms on Hugging Face:
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   (Create a free account and accept the model card.)

3. Create a Hugging Face access token:
   https://huggingface.co/settings/tokens
   (Read-only is enough.)

4. Run (uv handles the venv and installs dependencies automatically):
   uv run transcribe_diarize.py audio1705062906.m4a --hf_token YOUR_TOKEN_HERE

   Or set the env var so you don't have to type it each time:
   export HF_TOKEN=YOUR_TOKEN_HERE
   uv run transcribe_diarize.py audio1705062906.m4a
"""

import argparse
import contextlib
import functools
import inspect
import json
import os
import shutil
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
        module = __import__(import_name, fromlist=["*"])
    except ImportError as exc:
        print(
            f"Error: missing dependency '{package_name}'. "
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


def patch_hf_download_compat(pyannote_audio_module):
    """Bridge older pyannote auth kwargs to newer huggingface_hub versions."""
    huggingface_hub = require_dependency("huggingface_hub", "huggingface_hub")
    hf_signature = inspect.signature(huggingface_hub.hf_hub_download)
    if "use_auth_token" in hf_signature.parameters:
        return

    pipeline_module = pyannote_audio_module.core.pipeline
    if not hasattr(pipeline_module, "hf_hub_download"):
        return
    original_download = pipeline_module.hf_hub_download

    def compat_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return original_download(*args, **kwargs)

    pipeline_module.hf_hub_download = compat_hf_hub_download


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
    patch_hf_download_compat(pyannote_audio)
    Pipeline = pyannote_audio.Pipeline

    print("Running speaker diarization...")
    pipeline_signature = inspect.signature(Pipeline.from_pretrained)
    auth_kwarg = "token" if "token" in pipeline_signature.parameters else "use_auth_token"
    with pyannote_torch_load_compat(torch):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            **{auth_kwarg: hf_token},
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
    results = []
    for seg in segments:
        seg_start, seg_end = seg["start"], seg["end"]
        speaker_time: dict[str, float] = {}
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            overlap_start = max(seg_start, turn.start)
            overlap_end = min(seg_end, turn.end)
            overlap = max(0.0, overlap_end - overlap_start)
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

    def flush():
        if buffer:
            ts = f"[{fmt_time(buffer[0]['start'])} → {fmt_time(buffer[-1]['end'])}]"
            lines.append(f"\n{current_speaker}  {ts}")
            lines.append(" ".join(s["text"] for s in buffer))

    for seg in segments:
        if seg["speaker"] != current_speaker:
            flush()
            current_speaker = seg["speaker"]
            buffer = [seg]
        else:
            buffer.append(seg)
    flush()
    return "\n".join(lines).strip()


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
    args = parser.parse_args()

    if not args.hf_token:
        print("Error: Hugging Face token required. Pass --hf_token or set HF_TOKEN env var.")
        print("Get a token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    audio_path = args.audio
    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    ensure_ffmpeg_available()

    try:
        segments = transcribe(audio_path, model_size=args.model)
        diarization = diarize(audio_path, hf_token=args.hf_token, num_speakers=args.num_speakers)
        labeled = assign_speakers(segments, diarization)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)

    stem = Path(audio_path).stem
    out_path = args.output or f"{stem}_transcript.txt"
    out_file = Path(out_path)
    if out_file.parent != Path("."):
        out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_path.endswith(".json"):
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(labeled, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON to {out_file}")
    else:
        transcript = format_output(labeled)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"Saved transcript to {out_file}")


if __name__ == "__main__":
    main()
