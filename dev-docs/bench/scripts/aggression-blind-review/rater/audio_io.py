"""Decode + loudness-match audio for the aggression rater.

Everything goes through ffmpeg: decode any container, downmix to mono,
resample to 48 kHz (CLAP's rate), and normalise to -23 LUFS (EBU R128) so
that loudness cannot proxy for aggression. The rater only ever sees a
level-matched waveform -- no filenames, tags, or metadata.
"""
from __future__ import annotations

import subprocess
import tempfile
import os
import numpy as np
import soundfile as sf

TARGET_SR = 48000
TARGET_LUFS = -23.0


def decode_normalized(path: str, sr: int = TARGET_SR) -> np.ndarray:
    """Return a mono float32 waveform at `sr`, loudness-matched to -23 LUFS."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-i", path,
            "-af", f"loudnorm=I={TARGET_LUFS}:TP=-1.0:LRA=11",
            "-ar", str(sr), "-ac", "1", "-c:a", "pcm_f32le",
            out, "-loglevel", "error",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        data, got_sr = sf.read(out, dtype="float32")
        if got_sr != sr:
            raise RuntimeError(f"expected {sr} Hz, got {got_sr}")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.ascontiguousarray(data, dtype=np.float32)
    finally:
        if os.path.exists(out):
            os.remove(out)


def rms_dbfs(wave: np.ndarray) -> float:
    """RMS level in dBFS -- used only to flag near-silent / insufficient clips."""
    if wave.size == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(np.square(wave, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 0 else -np.inf
