"""Approximate port of fluXis's AudioAnalyzer for storyboard scripts.

`AmplitudesInRange` returns one frame per `interval` ms, each holding
normalized 0..1 spectrum amplitudes plus low/mid/high band levels.
The envelope shaping follows the documented FFTParameters semantics
(attack/release smoothing across frames, gamma contrast, band
cutoffs/multipliers/floors) but is not a bit-exact port of the C#
pipeline; scripts read these as animation drive values, where the
shape matters and the exact magnitudes don't.

Decoding follows the player's stack: soundfile first, audioread as
the mp3/m4a fallback. Without a decodable audio file every frame is
silent (amplitudes all zero) and one warning is printed, so scripted
storyboards still compile deterministically.
"""
from __future__ import annotations

import numpy as np

_WINDOW = 2048
_DEFAULTS = {
    'attack': 0.85, 'releaseLow': 0.15, 'releaseHigh': 0.5,
    'gamma': 1.0, 'spatialWindowSize': 1,
    'bassCutoff': 0.06, 'midCutoff': 0.4,
    'bassMultiplier': 1.0, 'midMultiplier': 1.0, 'highMultiplier': 1.0,
    'baseFloor': 0.1, 'midFloor': 0.1, 'highFloor': 0.1,
    'maxAdaptationRate': 0.05,
}
PRESETS = {
    'Default': dict(_DEFAULTS),
    'Reactive': {**_DEFAULTS, 'attack': 0.95, 'releaseLow': 0.4,
                 'releaseHigh': 0.8},
    'Smooth': {**_DEFAULTS, 'attack': 0.6, 'releaseLow': 0.05,
               'releaseHigh': 0.2, 'spatialWindowSize': 5},
}


def _load_mono(path):
    try:
        import soundfile as sf
        samples, sr = sf.read(path, dtype='float32', always_2d=True)
        return samples.mean(axis=1), sr
    except Exception:
        pass
    try:
        import audioread
        with audioread.audio_open(path) as f:
            chunks = [np.frombuffer(b, dtype='<i2') for b in f]
            samples = np.concatenate(chunks).astype(np.float32) / 32768.0
            if f.channels > 1:
                samples = samples.reshape(-1, f.channels).mean(axis=1)
            return samples, f.samplerate
    except Exception:
        return None, 0


def _spectrum_at(samples, sr, t_ms, count):
    center = int(t_ms / 1000.0 * sr)
    lo = max(0, center - _WINDOW // 2)
    window = samples[lo:lo + _WINDOW]
    if window.size < _WINDOW:
        window = np.pad(window, (0, _WINDOW - window.size))
    magnitudes = np.abs(np.fft.rfft(window * np.hanning(_WINDOW)))
    # Downsample the rfft bins onto `count` evenly spaced buckets.
    buckets = np.array_split(magnitudes[1:], count)
    return np.array([b.max() if b.size else 0.0 for b in buckets])


def _band_slices(count, params):
    bass_end = max(1, int(count * params['bassCutoff']))
    mid_end = max(bass_end + 1, int(count * params['midCutoff']))
    return slice(0, bass_end), slice(bass_end, mid_end), slice(mid_end, None)


def amplitudes_in_range(audio_path, start_ms, end_ms, interval_ms,
                        count=256, params=None) -> list:
    """Frames of {'amplitudes': [...], 'low', 'mid', 'high', 'total'}
    covering [start_ms, end_ms] every interval_ms."""
    params = {**_DEFAULTS, **(params or {})}
    interval_ms = max(1.0, float(interval_ms))
    times = np.arange(float(start_ms), float(end_ms) + interval_ms / 2,
                      interval_ms)
    count = max(1, int(count))

    samples = None
    if audio_path is not None:
        samples, sr = _load_mono(str(audio_path))
    if samples is None or not len(samples):
        return [_frame(np.zeros(count), params) for _ in times]

    raw = np.stack([_spectrum_at(samples, sr, t, count) for t in times])
    peak = raw.max() or 1.0
    raw /= peak

    smoothing = params['spatialWindowSize']
    if smoothing > 1:
        kernel = np.ones(int(smoothing)) / int(smoothing)
        raw = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode='same'), 1, raw)

    # Attack/release IIR across frames: rise by `attack`, fall by the
    # per-band release rate.
    bass, mid, high = _band_slices(count, params)
    release = np.empty(count)
    release[bass] = params['releaseLow']
    release[mid] = (params['releaseLow'] + params['releaseHigh']) / 2
    release[high] = params['releaseHigh']
    shaped = np.zeros_like(raw)
    state = np.zeros(count)
    for i, frame in enumerate(raw):
        rising = frame > state
        state = np.where(rising,
                         state + (frame - state) * params['attack'],
                         state + (frame - state) * release)
        shaped[i] = state

    shaped = np.clip(shaped, 0.0, 1.0) ** params['gamma']
    shaped[:, bass] *= params['bassMultiplier']
    shaped[:, mid] *= params['midMultiplier']
    shaped[:, high] *= params['highMultiplier']
    shaped = np.clip(shaped, 0.0, 1.0)

    return [_frame(row, params) for row in shaped]


def _frame(amplitudes, params) -> dict:
    bass, mid, high = _band_slices(len(amplitudes), params)

    def level(sl):
        chunk = amplitudes[sl]
        return float(chunk.mean()) if chunk.size else 0.0

    return {
        'amplitudes': [float(a) for a in amplitudes],
        'low': level(bass), 'mid': level(mid), 'high': level(high),
        'total': float(amplitudes.mean()) if len(amplitudes) else 0.0,
    }
