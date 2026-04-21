"""Decoded-audio sample source. A `WaveSource` owns the full decoded PCM
buffer and a read cursor; the phase vocoder and resample fallback pull
samples out through it."""
from __future__ import annotations

import numpy as np


class WaveSource:
    """Decoded audio + a read pointer. `generate(n, ch)` returns the next
    `n` frames of float32 samples, interleaved for `ch` channels.

    The pointer is expressed as float samples so that fractional playback
    positions from upstream resample/stretch don't cause drift when the
    engine seeks."""
    def __init__(self, samples: np.ndarray, sr: int):
        # samples: shape (n,) mono or (n, ch) stereo, int16 or float32.
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32) / 32768.0
        if samples.ndim == 1:
            samples = samples[:, None]
        self.samples = np.ascontiguousarray(samples)   # (n, ch)
        self.sr = int(sr)
        self.pos = 0         # integer frame index
        self.n_frames = samples.shape[0]
        self.src_channels = samples.shape[1]

    @property
    def duration(self) -> float:
        return self.n_frames / self.sr

    def seek(self, frame: int) -> None:
        self.pos = max(0, min(int(frame), self.n_frames))

    def read(self, n: int) -> np.ndarray:
        """Return up to `n` frames of native-channel float32 samples,
        advancing the read pointer. Returns fewer (possibly zero) frames
        at end-of-file."""
        end = min(self.pos + n, self.n_frames)
        out = self.samples[self.pos:end]
        self.pos = end
        return out
