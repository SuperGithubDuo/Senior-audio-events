"""WAV loading: mono, float32 in [-1, 1], resampled to 16 kHz."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .config import SAMPLE_RATE


def _to_float(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return (x.astype(np.float32) - 128.0) / 128.0
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.float32) / float(np.iinfo(x.dtype).max + 1)
    return x.astype(np.float32)


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Return (samples, sample_rate); samples is float32, shape (n,) or (n, ch)."""
    path = str(path)
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        return data, int(sr)
    except ImportError:
        from scipy.io import wavfile

        sr, data = wavfile.read(path)
        return _to_float(data), int(sr)


def to_mono_16k(data: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(data, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, sr)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return np.clip(x, -1.0, 1.0)


def load_wav(path: str | Path) -> np.ndarray:
    data, sr = read_audio(path)
    if data.size == 0:
        raise ValueError(f"{path}: empty audio")
    return to_mono_16k(data, sr)


def write_wav(path: str | Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    from scipy.io import wavfile

    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, pcm)
