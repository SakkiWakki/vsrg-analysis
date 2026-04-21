"""Rate-aware streaming audio engine for the replay player.

Design (modeled on the IMS course's PyAudio setup — pull-based `generate`
chain feeding an output stream):

    WaveSource  →  StreamingPhaseVocoder  →  sounddevice callback

The callback runs on the audio thread at ~5ms cadence and pulls N frames
from the top of the chain. Because every effect works per-buffer, rate
changes and pitch-correct toggles are instant — no pre-building, no UI
freeze. `set_state(t, rate, playing)` keeps the same public API so the
PlayerTab doesn't need to know anything about the new internals.

The phase vocoder carries its own STFT state across callbacks: on every
request it reads `ceil(N / hop_s)` hops of analysis frames from the source,
accumulates phase, and emits exactly N output frames."""
from __future__ import annotations
import math
import os
import threading
import time

import numpy as np

# sounddevice is the PortAudio binding we drive the output stream with.
# Imported lazily so the module still imports in test environments that
# don't have an audio device.
try:
    import sounddevice as _sd
except Exception:
    _sd = None

# pygame is still used elsewhere for chart rendering; we load its Sound
# reader to decode the audio file without a second decoder dependency.
try:
    import pygame
except Exception:
    pygame = None


# ===== source: decoded wav / mp3 / ogg served one chunk at a time =====

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


# ===== streaming phase vocoder =====

class StreamingPhaseVocoder:
    """Phase-vocoder time-stretch that operates on the source one
    window-worth at a time, keeping inter-buffer state so the output is
    continuous across callbacks.

    The trick: we process one analysis frame per call to `_step()` and
    overlap-add it into a small ring buffer. `generate(n)` pulls from that
    ring, calling `_step()` as many times as needed to fill `n` frames.
    That means rate/pitch-correct changes take effect within one hop."""
    N_FFT = 2048
    HOP = 512          # synthesis hop at rate=1; analysis hop scales by rate

    def __init__(self, source: WaveSource, rate: float = 1.0,
                 pitch_correct: bool = True):
        self.source = source
        self.rate = float(rate)
        self.pitch_correct = bool(pitch_correct)
        self.channels = source.src_channels

        self._win = np.hanning(self.N_FFT).astype(np.float32)
        self._win_sq = self._win * self._win
        # Bin center freqs in rad/sample. The expected phase advance across
        # an analysis hop of `hop_a` samples is `bin_freqs * hop_a`; across
        # a synthesis hop of `hop_s` samples it's `bin_freqs * hop_s`. These
        # two are what makes the vocoder pitch-preserving.
        self._bin_freqs = 2.0 * np.pi * np.arange(self.N_FFT // 2 + 1,
                                                   dtype=np.float32) / self.N_FFT
        # Per-channel PV state. `prev_phase` is the last analysis frame's
        # phase; `out_phase` is the accumulated synthesis phase. Both are
        # per-bin arrays of size n_bins.
        n_bins = self.N_FFT // 2 + 1
        self._prev_phase = [np.zeros(n_bins, dtype=np.float32)
                            for _ in range(self.channels)]
        self._out_phase = [np.zeros(n_bins, dtype=np.float32)
                           for _ in range(self.channels)]
        # Output ring: overlap-add target. Must be at least N_FFT + HOP long
        # so we have room to add a full IFFT frame past the current read
        # cursor. We use a simple grow-as-needed linear buffer; since HOP is
        # small and we drain it every callback, it never gets big.
        self._ola = np.zeros((self.N_FFT * 2, self.channels), dtype=np.float32)
        self._ola_norm = np.zeros(self.N_FFT * 2, dtype=np.float32)
        self._ola_read = 0
        self._ola_write = 0     # next output-frame index that will be written
        # Analysis-frame read position in the source (fractional frames).
        self._src_pos = 0.0
        # Source buffer for the current analysis frame (N_FFT samples).
        # We keep it around so we don't re-read overlapping slices every step.
        self._src_buf = np.zeros((self.N_FFT, self.channels), dtype=np.float32)
        self._src_buf_start = -self.N_FFT   # invalid / not yet filled
        self._first_frame = True
        self._ended = False

    # --- public controls ---
    def set_rate(self, rate: float) -> None:
        self.rate = max(0.05, float(rate))

    def set_pitch_correct(self, on: bool) -> None:
        self.pitch_correct = bool(on)

    def seek(self, chart_time: float) -> None:
        """Jump playback to `chart_time` seconds. We seek the source to the
        underlying sample, reset PV state, and clear the OLA buffer — small
        audible transient is acceptable."""
        self.source.seek(int(round(chart_time * self.source.sr)))
        self._src_pos = float(self.source.pos)
        self._ola.fill(0.0)
        self._ola_norm.fill(0.0)
        self._ola_read = 0
        self._ola_write = 0
        for p in self._prev_phase: p.fill(0.0)
        for p in self._out_phase:  p.fill(0.0)
        self._src_buf_start = -self.N_FFT
        self._first_frame = True
        self._ended = False

    def generate(self, n_frames: int) -> tuple[np.ndarray, bool]:
        """Return (samples (n_frames, channels) float32, continue_flag).

        continue_flag goes False once the source has run out AND we've
        drained the residual OLA tail."""
        # Fast path when rate==1 and pitch correction is on (nothing to do).
        # Also when pitch correction is off we do a simple linear-interp
        # resample — pitch tracks rate, same as the old engine's fallback.
        if not self.pitch_correct:
            return self._resample_read(n_frames)

        if abs(self.rate - 1.0) < 1e-3:
            data = self.source.read(n_frames)
            if len(data) < n_frames:
                # pad with silence and report end-of-stream
                pad = np.zeros((n_frames - len(data), self.channels),
                               dtype=np.float32)
                return (np.concatenate([data, pad], axis=0),
                        len(data) > 0)
            return data, True

        # Pitch-corrected path: pump the PV until we have n_frames of output.
        while (self._ola_write - self._ola_read) < n_frames and not self._ended:
            self._step()

        out = self._drain(n_frames)
        cont = not (self._ended and (self._ola_write - self._ola_read) <= 0)
        return out, cont

    # --- PV internals ---
    def _step(self) -> None:
        """Process one analysis frame: read N_FFT source samples at the
        current rate-dependent position, FFT, phase-propagate, IFFT, and
        overlap-add into the output ring."""
        hop_a = self.HOP * self.rate   # analysis hop (float)
        hop_s = self.HOP               # synthesis hop (fixed)

        start = int(math.floor(self._src_pos))
        # Read N_FFT samples from source starting at `start`. If we've
        # already got them from the previous step (overlapping), reuse.
        needed_end = start + self.N_FFT
        if self._src_buf_start != start:
            # Re-read — cheaper than sliding, since N_FFT is small.
            self.source.seek(start)
            raw = self.source.read(self.N_FFT)
            if len(raw) < self.N_FFT:
                pad = np.zeros((self.N_FFT - len(raw), self.channels),
                               dtype=np.float32)
                raw = np.concatenate([raw, pad], axis=0)
                self._ended_after_this = True
            else:
                self._ended_after_this = False
            self._src_buf = raw
            self._src_buf_start = start

        # FFT per channel; phase-propagate; IFFT; OLA.
        frame = self._src_buf * self._win[:, None]      # (N_FFT, ch)
        spec = np.fft.rfft(frame, axis=0)               # (n_bins, ch)
        n_bins = spec.shape[0]

        time_frame = np.empty((self.N_FFT, self.channels), dtype=np.float32)
        # Expected phase advance of each bin across `hop_a` source samples.
        omega_a = self._bin_freqs * hop_a
        # And the ratio we scale the true instantaneous frequency by when
        # moving to the synthesis timeline.
        hop_ratio = hop_s / hop_a

        for c in range(self.channels):
            mag = np.abs(spec[:, c])
            phase = np.angle(spec[:, c]).astype(np.float32)
            if self._first_frame:
                self._out_phase[c] = phase.copy()
                self._prev_phase[c] = phase.copy()
            else:
                dphi = phase - self._prev_phase[c] - omega_a
                dphi -= 2.0 * np.pi * np.floor(dphi / (2.0 * np.pi) + 0.5)
                true_freq = omega_a + dphi        # rad per analysis hop
                # Peak-locked phase update (Laroche & Dolson 1999). The
                # standard PV lets every bin evolve its phase independently,
                # which destroys the vertical (across-bin) phase coherence
                # that music relies on — the result is the muffled/"phasy"
                # sound. Peak-locking fixes this by:
                #   1. finding local magnitude peaks in this frame,
                #   2. advancing each peak's phase by its own true_freq,
                #   3. locking neighboring bins to the nearest peak — their
                #      phases become peak_phase plus the original input
                #      offset from the peak.
                # That preserves transients and formants way better.
                new_peak_phase = (self._out_phase[c]
                                  + true_freq * hop_ratio)
                peaks = _find_peaks(mag)
                if peaks.size:
                    # Assign each bin to the nearest peak.
                    boundaries = (peaks[:-1] + peaks[1:] + 1) // 2
                    assign = np.searchsorted(boundaries, np.arange(n_bins))
                    nearest = peaks[assign]
                    # out_phase = peak's propagated phase + (input bin
                    # phase - input peak phase). Near-peak bins ride the
                    # peak's phase trajectory together.
                    self._out_phase[c] = (new_peak_phase[nearest]
                                          + phase - phase[nearest])
                else:
                    self._out_phase[c] = new_peak_phase
                self._prev_phase[c] = phase
            out_spec = mag * np.exp(1j * self._out_phase[c])
            time_frame[:, c] = np.fft.irfft(out_spec, n=self.N_FFT).astype(
                np.float32)
        self._first_frame = False

        # Write into OLA buffer at position self._ola_write.
        self._ensure_ola_capacity(self._ola_write + self.N_FFT)
        self._ola[self._ola_write:self._ola_write + self.N_FFT] += (
            time_frame * self._win[:, None])
        self._ola_norm[self._ola_write:self._ola_write + self.N_FFT] += (
            self._win_sq)
        self._ola_write += hop_s      # advance by synthesis hop

        # Advance source position by analysis hop.
        self._src_pos += hop_a
        if self._src_pos >= self.source.n_frames:
            self._ended = True

    def _ensure_ola_capacity(self, need: int) -> None:
        if need <= self._ola.shape[0]:
            return
        new_size = max(need + self.N_FFT, self._ola.shape[0] * 2)
        new_ola = np.zeros((new_size, self.channels), dtype=np.float32)
        new_norm = np.zeros(new_size, dtype=np.float32)
        new_ola[:self._ola.shape[0]] = self._ola
        new_norm[:self._ola_norm.shape[0]] = self._ola_norm
        self._ola = new_ola
        self._ola_norm = new_norm

    def _drain(self, n: int) -> np.ndarray:
        """Take up to `n` frames out of the OLA ring, normalize by window^2
        sum, and shift the ring so the read cursor stays near zero."""
        avail = self._ola_write - self._ola_read
        take = min(n, max(0, avail))
        out = self._ola[self._ola_read:self._ola_read + take].copy()
        norm = self._ola_norm[self._ola_read:self._ola_read + take]
        np.maximum(norm, 1e-8, out=norm)
        out /= norm[:, None]
        # Pad with silence if we're at end-of-stream.
        if take < n:
            pad = np.zeros((n - take, self.channels), dtype=np.float32)
            out = np.concatenate([out, pad], axis=0)
        self._ola_read += take
        # Compact the ring periodically so it doesn't grow unbounded.
        if self._ola_read > self.N_FFT * 4:
            shift = self._ola_read
            self._ola[:-shift] = self._ola[shift:]
            self._ola[-shift:] = 0.0
            self._ola_norm[:-shift] = self._ola_norm[shift:]
            self._ola_norm[-shift:] = 0.0
            self._ola_read -= shift
            self._ola_write -= shift
        return out

    # --- cheap fallback when pitch correction is off ---
    def _resample_read(self, n_frames: int) -> tuple[np.ndarray, bool]:
        """Pitch-shifting resample: read `n*rate` source frames and
        linearly interpolate down to `n`. Same feel as osu!/Etterna's
        built-in rate mods."""
        rate = max(0.05, self.rate)
        if abs(rate - 1.0) < 1e-3:
            data = self.source.read(n_frames)
            if len(data) < n_frames:
                pad = np.zeros((n_frames - len(data), self.channels),
                               dtype=np.float32)
                return np.concatenate([data, pad], axis=0), len(data) > 0
            return data, True
        need = int(math.ceil(n_frames * rate)) + 2
        src = self.source.read(need)
        if len(src) == 0:
            return (np.zeros((n_frames, self.channels), dtype=np.float32),
                    False)
        # Linear interp per channel. Any underrun at end gets zero-padded.
        in_idx = np.arange(len(src), dtype=np.float32)
        out_idx = np.arange(n_frames, dtype=np.float32) * rate
        ok = out_idx < (len(src) - 1)
        out = np.zeros((n_frames, self.channels), dtype=np.float32)
        valid = out_idx[ok]
        for c in range(self.channels):
            out[ok, c] = np.interp(valid, in_idx, src[:, c])
        # Rewind the source by the amount we over-read, so the next call
        # picks up cleanly. We read `need` frames but consumed only
        # n_frames * rate of them.
        used = int(math.ceil(n_frames * rate))
        self.source.pos -= max(0, need - used)
        continuing = ok.all() and self.source.pos < self.source.n_frames
        return out, continuing


# ===== public engine, same API as before =====

class AudioEngine:
    RESYNC_THRESHOLD_S = 0.15

    def __init__(self, audio_path: str | None, volume: float = 0.5,
                 pitch_correct: bool = True):
        self.ready = False
        self._volume = float(volume)
        self._pitch_correct = bool(pitch_correct)
        self._sr = 44100
        self._base_duration = 0.0
        self._ended = False
        self._playing = False
        # Engine state kept under a lock because the audio callback (on the
        # PortAudio thread) and set_state/seek (on the GUI thread) both
        # touch it.
        self._lock = threading.Lock()
        self._source: WaveSource | None = None
        self._pv: StreamingPhaseVocoder | None = None
        self._stream = None
        self._chart_time = 0.0       # last known chart time at set_state
        self._rate = 1.0

        if not audio_path or not os.path.exists(audio_path):
            return
        if _sd is None:
            print('audio: sounddevice not installed — no audio playback')
            return

        samples, sr = self._decode(audio_path)
        if samples is None:
            return
        self._sr = int(sr)
        self._source = WaveSource(samples, sr)
        self._base_duration = self._source.duration
        self._pv = StreamingPhaseVocoder(self._source, rate=1.0,
                                          pitch_correct=self._pitch_correct)
        try:
            self._stream = _sd.OutputStream(
                samplerate=self._sr,
                channels=self._source.src_channels,
                dtype='float32',
                blocksize=512,
                callback=self._callback,
            )
            self._stream.start()
            self.ready = True
        except Exception as e:
            print(f'audio: failed to open output stream: {e}')
            self._stream = None

    # --- decoding: use pygame's loader so we don't add another dep ---
    def _decode(self, path: str):
        if pygame is None:
            print('audio: pygame not available for decoding')
            return None, 0
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            raw = pygame.mixer.Sound(path)
            arr = pygame.sndarray.array(raw)
            init = pygame.mixer.get_init()
            sr = int(init[0]) if init else 44100
            return arr, sr
        except Exception as e:
            print(f'audio: decode failed: {e}')
            return None, 0

    # --- the PortAudio callback ---
    def _callback(self, outdata, frames, time_info, status):
        # Runs on the audio thread. Must be real-time-safe-ish: no I/O, no
        # allocations beyond what the PV does internally.
        with self._lock:
            pv = self._pv
            playing = self._playing
            volume = self._volume
        if pv is None or not playing:
            outdata.fill(0.0)
            if pv is None or not playing:
                # report ended when we're past the end of the source
                return
            return
        try:
            samples, cont = pv.generate(frames)
        except Exception as e:
            outdata.fill(0.0)
            print(f'audio callback: {e}')
            return
        if samples.shape[0] < frames:
            pad = np.zeros((frames - samples.shape[0], samples.shape[1]),
                           dtype=np.float32)
            samples = np.concatenate([samples, pad], axis=0)
        np.multiply(samples, volume, out=outdata)
        if not cont:
            with self._lock:
                self._ended = True

    # --- public API (matches the old engine) ---
    def set_volume(self, v: float) -> None:
        with self._lock:
            self._volume = float(v)

    def set_pitch_correct(self, on: bool) -> None:
        with self._lock:
            self._pitch_correct = bool(on)
            if self._pv is not None:
                self._pv.set_pitch_correct(on)

    @property
    def _pitch_correct_public(self) -> bool:
        return self._pitch_correct

    def set_state(self, t: float, rate: float, playing: bool) -> None:
        if not self.ready:
            return
        rate = max(0.05, float(rate))
        with self._lock:
            source_pos_s = (self._source.pos / self._source.sr
                            if self._source else 0.0)
            # Rate change: update PV rate, no reseek needed.
            if abs(rate - self._rate) > 1e-3:
                self._rate = rate
                if self._pv is not None:
                    self._pv.set_rate(rate)
            # Seek if chart time disagrees with where we think we are.
            want_seek = False
            if not playing:
                # Pause: stop producing but keep the current position so
                # resume picks up cleanly.
                self._playing = False
                self._chart_time = t
                self._ended = False
                return
            if t >= self._base_duration:
                self._playing = False
                self._ended = True
                return
            # If we were paused OR drifted, seek to chart time.
            if not self._playing or abs(source_pos_s - t) > self.RESYNC_THRESHOLD_S:
                want_seek = True
            if want_seek and self._pv is not None:
                self._pv.seek(t)
            self._playing = True
            self._chart_time = t
            self._ended = False

    def stop(self) -> None:
        with self._lock:
            self._playing = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None

    def prewarm_rates(self, rates) -> None:
        """No-op — the streaming PV has no per-rate precompute."""
        return


def _monotonic() -> float:
    return time.monotonic()


def _find_peaks(mag: np.ndarray) -> np.ndarray:
    """Return indices of local magnitude peaks in `mag`.

    A bin is a peak if its magnitude is strictly greater than its two
    neighbors on each side (Laroche & Dolson recommend a ±2-bin window
    for phase locking — wider windows over-lock, narrower ones leak).
    Below a low-magnitude threshold bins are ignored so background noise
    doesn't spawn spurious peaks."""
    if mag.size < 5:
        return np.empty(0, dtype=np.int64)
    # Relative floor: bins below 1/1000 of the frame's max contribute
    # nothing audible and shouldn't act as peaks.
    thresh = mag.max() * 1e-3
    k = np.arange(2, mag.size - 2)
    is_peak = ((mag[k] > mag[k - 1]) & (mag[k] > mag[k + 1])
               & (mag[k] > mag[k - 2]) & (mag[k] > mag[k + 2])
               & (mag[k] > thresh))
    return k[is_peak]
