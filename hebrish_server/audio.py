"""Any audio or video file to 16 kHz mono float32, through ffmpeg."""

import subprocess
import tempfile

import numpy as np

from .segment import SAMPLE_RATE


def decode(data: bytes) -> np.ndarray:
    # A temp file, not stdin: MP4/M4A files written with the index (moov) at the end need a
    # seekable input, and from a pipe ffmpeg returned 0 samples with exit code 0 (2026-10-30)
    with tempfile.NamedTemporaryFile() as f:
        f.write(data)
        f.flush()
        p = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", f.name,
                            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
                           capture_output=True)
    if p.returncode != 0:
        raise ValueError("ffmpeg could not read the file: " + p.stderr.decode(errors="replace").strip()[-300:])
    if not p.stdout:
        raise ValueError("no audio in the file: " + (p.stderr.decode(errors="replace").strip()[-300:] or "ffmpeg decoded nothing"))
    return np.frombuffer(p.stdout, np.float32).copy()
