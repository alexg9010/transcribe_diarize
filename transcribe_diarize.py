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
"""Compatibility wrapper for running the CLI directly via uv."""

import os
from pathlib import Path

os.environ.setdefault("TRANSCRIBE_DIARIZE_REPO_ROOT", str(Path(__file__).resolve().parent))

from transcribe_diarize_pkg.cli import main


if __name__ == "__main__":
    main()
