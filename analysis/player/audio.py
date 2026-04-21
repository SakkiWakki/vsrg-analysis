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
import heapq
import json
import math
import os
import shutil
import subprocess
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

# Decoding stack: soundfile (libsndfile — wav/flac/ogg/opus) as the primary,
# audioread (ffmpeg/gstreamer wrapper — mp3/m4a/anything) as the fallback for
# formats libsndfile doesn't handle. Both are pure decoders that hand us
# float samples directly, no mixer roundtrip.
try:
    import soundfile as _sf
except Exception:
    _sf = None

try:
    import audioread as _audioread
except Exception:
    _audioread = None


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
    """Streaming phase-vocoder time stretch with phase-gradient heap
    integration.

    This follows the practical shape of Průša & Holighaus' "Phase Vocoder
    Done Right" demo rather than the magnitude-only PhaseReT RTPGHI path:
    we already have the complex analysis STFT, so the phase gradients come
    from the analysis phase. The time gradient is the classical
    Flanagan-Portnoff per-bin true-frequency estimate, scaled to the fixed
    synthesis hop. The frequency gradient is the wrapped bin-to-bin phase
    derivative, scaled by the stretch factor, and a max-heap integrates both
    gradients through the current frame.

    The STFT is deliberately computed in a zero-phase convention
    (`ifftshift` before FFT, `fftshift` after IFFT). LTFAT's real-time DGT
    does the same circular shift internally. Without this, a Hann-windowed
    sinusoid carries an artificial adjacent-bin pi flip from the window
    being centered halfway through the FFT buffer; magnitude-only PGHI then
    appears to "collapse" phases and the normal IFFT cancels the main lobe.

    All phase state is float64 — float32 overflows bin*omega_a products at
    high bins after a few seconds and manifests as high-frequency buzz."""
    N_FFT = 2048
    HOP = 512          # synthesis hop at rate=1; analysis hop scales by rate
    # Relative magnitude tolerance below which bins skip heap integration.
    # LTFAT's reference uses 1e-6. We still keep those bins phase-continuous
    # with a time-only fallback instead of randomizing them, because this is
    # a playback vocoder and we have a valid phase history.
    REL_TOL = 1e-6

    def __init__(self, source: WaveSource, rate: float = 1.0,
                 pitch_correct: bool = True):
        self.source = source
        self.rate = float(rate)
        self.pitch_correct = bool(pitch_correct)
        self.channels = source.src_channels

        # Hann window; float64 to keep OLA and win^2 sums precise. We cast
        # to float32 only at the very end when writing output.
        self._win = np.hanning(self.N_FFT).astype(np.float64)
        self._win_sq = self._win * self._win
        # Analytic COLA divisor: the steady-state value of
        # sum_k w(n - k*hop)^2 equals sum(w^2) / hop. For Hann N=2048,
        # hop=512 this evaluates to 1.5 — verified numerically in the
        # test suite.
        self._cola_norm = float(np.sum(self._win_sq) / self.HOP)
        # Bin center freqs in rad/sample, float64; used for the
        # heterodyned true-frequency estimate.
        n_bins = self.N_FFT // 2 + 1
        self._bin_freqs = (2.0 * np.pi
                           * np.arange(n_bins, dtype=np.float64)
                           / self.N_FFT)
        # Scalar 2π used throughout.
        self._two_pi = 2.0 * np.pi
        # Per-channel state. All phase arrays are float64.
        self._prev_phase = [np.zeros(n_bins, dtype=np.float64)
                            for _ in range(self.channels)]
        self._prev_mag = [np.zeros(n_bins, dtype=np.float64)
                          for _ in range(self.channels)]
        self._prev_tgrad = [np.zeros(n_bins, dtype=np.float64)
                            for _ in range(self.channels)]
        self._prev_synth_phase = [np.zeros(n_bins, dtype=np.float64)
                                   for _ in range(self.channels)]
        # Back-compat alias for tests/diagnostics that still use the old
        # classical-PV field name.
        self._out_phase = self._prev_synth_phase
        self._have_prev_tgrad = [False for _ in range(self.channels)]
        self._prev_hop_a = float(self.HOP)
        # Output ring: a true circular OLA buffer. Sized as a multiple of
        # N_FFT so that a just-written window's tail can never be
        # overwritten before it's drained. `_ola_read` and `_ola_write`
        # are absolute (non-wrapping) sample counters; we take `% ring`
        # when indexing. Samples are zeroed on drain so the ring is always
        # clean for the next wrap.
        self._ola_ring = self.N_FFT * 4
        self._ola = np.zeros((self._ola_ring, self.channels), dtype=np.float64)
        self._ola_read = 0
        self._ola_write = 0     # next output-frame index that will be written
        # Analysis-frame read position in the source (fractional frames).
        self._src_pos = 0.0
        self._first_frame = True
        self._ended = False
        self._flux_baseline = 0.0
        self._last_transient = False

    @property
    def source_time(self) -> float:
        """Current source position in seconds, including PV-driven reads."""
        frame = min(max(self._src_pos, 0.0), float(self.source.n_frames))
        return frame / max(1, self.source.sr)

    def _sync_source_pos(self) -> None:
        """Mirror the PV cursor to WaveSource for external drift checks."""
        self.source.pos = max(
            0,
            min(int(round(self._src_pos)), self.source.n_frames),
        )

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
        # Reset both counters together; they share the same wrap-phase.
        self._ola_read = 0
        self._ola_write = 0
        for p in self._prev_phase:      p.fill(0.0)
        for p in self._prev_mag:        p.fill(0.0)
        for p in self._prev_tgrad:      p.fill(0.0)
        for p in self._prev_synth_phase: p.fill(0.0)
        self._have_prev_tgrad = [False for _ in range(self.channels)]
        self._prev_hop_a = float(self.HOP)
        self._first_frame = True
        self._ended = False
        self._flux_baseline = 0.0
        self._last_transient = False

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
            # Keep the PV's `_src_pos` in lockstep with source.pos so when
            # the user bumps rate back off 1.0 the next _step() starts from
            # a coherent position instead of wherever the last PV pass left
            # _src_pos.
            data = self.source.read(n_frames)
            self._src_pos = float(self.source.pos)
            if len(data) < n_frames:
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
    def _read_frame(self) -> np.ndarray:
        """Return N_FFT samples at (fractional) position self._src_pos,
        linearly interpolated between integer source frames. Zero-pads past
        the end of the source and flags `self._ended` when we reach it."""
        n_src = self.source.n_frames
        src = self.source.samples
        base = self._src_pos
        # Integer frames we need: floor(base) .. floor(base) + N_FFT.
        start = int(math.floor(base))
        end = start + self.N_FFT + 1  # +1 for the interpolation neighbor
        frac = base - start
        lo = max(0, start)
        hi = min(n_src, end)
        if lo >= n_src:
            self._ended = True
            return np.zeros((self.N_FFT, self.channels), dtype=np.float64)
        chunk = src[lo:hi]
        # Pre-pad if start < 0 (can happen right after seek to t<0).
        if start < 0:
            pad = np.zeros((-start, self.channels), dtype=np.float32)
            chunk = np.concatenate([pad, chunk], axis=0)
        # Post-pad to N_FFT + 1.
        if chunk.shape[0] < self.N_FFT + 1:
            pad = np.zeros((self.N_FFT + 1 - chunk.shape[0], self.channels),
                           dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
            if end >= n_src:
                self._ended_after_this = True
        # Linear interpolate: out[i] = (1-frac)*chunk[i] + frac*chunk[i+1].
        a = chunk[:self.N_FFT]
        b = chunk[1:self.N_FFT + 1]
        return ((1.0 - frac) * a + frac * b).astype(np.float64)

    def _principal_arg(self, phase: np.ndarray) -> np.ndarray:
        """Wrap phase to [-pi, pi] using LTFAT's round-to-nearest form."""
        return phase - self._two_pi * np.round(phase / self._two_pi)

    def _compute_tgrad_from_phase(self, phase: np.ndarray,
                                  prev_phase: np.ndarray,
                                  analysis_hop: float,
                                  hop_s: float) -> np.ndarray:
        """Per-bin synthesis-hop phase advance from analysis phase.

        This is the classical true-frequency estimate, but used as the
        time-gradient input to heap integration rather than as the final
        independent bin phase. `analysis_hop` is the actual source-frame
        distance between `prev_phase` and `phase`, which matters when the
        playback rate changes between callbacks.
        """
        omega_a = self._bin_freqs * analysis_hop
        dphi = self._principal_arg(phase - prev_phase - omega_a)
        return (omega_a + dphi) * (hop_s / max(1e-9, analysis_hop))

    def _compute_fgrad_from_phase(self, phase: np.ndarray,
                                  stretch: float) -> np.ndarray:
        """Frequency derivative of phase for the current zero-phase frame.

        `stretch` is output-time/input-time (hop_s / hop_a). Scaling the
        group-delay-like frequency gradient keeps transient locations in
        the stretched synthesis timeline.
        """
        fgrad = np.zeros_like(phase)
        if phase.size >= 3:
            fgrad[1:-1] = 0.5 * (
                self._principal_arg(phase[2:] - phase[1:-1])
                + self._principal_arg(phase[1:-1] - phase[:-2])
            ) * stretch
        return fgrad

    def _rtpghi(self, mag: np.ndarray,
                phase: np.ndarray, prev_phase: np.ndarray,
                prev_mag: np.ndarray, prev_tgrad: np.ndarray,
                prev_synth: np.ndarray, analysis_hop: float, hop_a: float,
                hop_s: float, have_prev_tgrad: bool
                ) -> tuple[np.ndarray, np.ndarray]:
        """Real-Time Phase Gradient Heap Integration for one frame.

        This mirrors LTFAT's heap update structure: every significant bin
        in the target frame enters the heap as a time-edge candidate keyed
        by previous-frame magnitude, then successful time propagation pushes
        the current-frame node keyed by current magnitude for frequency
        propagation.
        """
        n_bins = mag.size
        stretch = hop_s / max(1e-9, hop_a)

        tgrad_cur = self._compute_tgrad_from_phase(
            phase, prev_phase, analysis_hop, hop_s)
        fgrad_cur = self._compute_fgrad_from_phase(phase, stretch)

        abstol = self.REL_TOL * max(float(prev_mag.max()),
                                     float(mag.max()),
                                     1e-12)
        in_I = mag > abstol
        # Start every bin with a temporally coherent fallback. Significant
        # bins are overwritten by heap-integrated phase below. This avoids
        # frame-random phase on low-level bins, which can become audible as
        # noise-floor shimmer or pumping in dense music.
        synth = prev_synth + tgrad_cur

        heap: list = []
        counter = 0
        for m in np.flatnonzero(in_I):
            heapq.heappush(heap, (-float(prev_mag[m]), counter, int(m), 0))
            counter += 1

        while heap and in_I.any():
            _, _, m, tag = heapq.heappop(heap)
            if tag == 0:
                if in_I[m]:
                    # Time step: phi_s(m,n) = phi_s(m,n-1)
                    #   + 0.5 * (tgrad[m,n-1] + tgrad[m,n]).
                    # On the first propagated frame we only have the right
                    # endpoint, so use an Euler step to avoid a zero-gradient
                    # bias.
                    if have_prev_tgrad:
                        step = 0.5 * (prev_tgrad[m] + tgrad_cur[m])
                    else:
                        step = tgrad_cur[m]
                    synth[m] = prev_synth[m] + step
                    in_I[m] = False
                    heapq.heappush(heap, (-float(mag[m]), counter, m, 1))
                    counter += 1
            else:
                if m + 1 < n_bins and in_I[m + 1]:
                    synth[m + 1] = (synth[m]
                                    + 0.5 * (fgrad_cur[m]
                                             + fgrad_cur[m + 1]))
                    in_I[m + 1] = False
                    heapq.heappush(heap,
                                    (-float(mag[m + 1]), counter, m + 1, 1))
                    counter += 1
                if m - 1 >= 0 and in_I[m - 1]:
                    synth[m - 1] = (synth[m]
                                    - 0.5 * (fgrad_cur[m]
                                             + fgrad_cur[m - 1]))
                    in_I[m - 1] = False
                    heapq.heappush(heap,
                                    (-float(mag[m - 1]), counter, m - 1, 1))
                    counter += 1

        if in_I.any():
            # Should be rare because every target bin is seeded, but keep a
            # deterministic fallback for numerical edge cases.
            fallback = in_I.copy()
            synth[fallback] = prev_synth[fallback] + tgrad_cur[fallback]

        return synth, tgrad_cur

    def _update_transient_diagnostic(self, mags: np.ndarray) -> None:
        """Track spectral-flux onsets for diagnostics/tests.

        The phase-gradient path does not need explicit transient resets, but
        keeping this state is useful to verify that seeks clear analysis
        history and that stationary tones are not being mistaken for onsets.
        """
        if self._first_frame:
            self._last_transient = False
            return

        pos_flux = 0.0
        energy = 0.0
        for c in range(self.channels):
            cur = mags[:, c]
            pos_flux += float(np.maximum(cur - self._prev_mag[c], 0.0).sum())
            energy += float(cur.sum())

        flux = pos_flux / max(energy, 1e-12)
        threshold = max(0.03, 6.0 * self._flux_baseline)
        self._last_transient = bool(energy > 1e-6 and flux > threshold)
        if not self._last_transient:
            self._flux_baseline = 0.98 * self._flux_baseline + 0.02 * flux

    def _step(self) -> None:
        """Process one analysis frame: interpolate N_FFT source samples at
        the current fractional position, FFT, run RTPGHI to propagate
        phase, IFFT, OLA into the output ring."""
        hop_a = self.HOP * self.rate   # analysis hop (fractional samples)
        hop_s = self.HOP               # synthesis hop (fixed)

        src_frame = self._read_frame()                   # (N_FFT, ch) f64
        frame = np.fft.ifftshift(src_frame * self._win[:, None], axes=0)
        spec = np.fft.rfft(frame, axis=0)                # (n_bins, ch) c128
        mags = np.abs(spec)
        self._update_transient_diagnostic(mags)
        time_frame = np.empty((self.N_FFT, self.channels), dtype=np.float64)
        analysis_hop = self._prev_hop_a

        for c in range(self.channels):
            mag = mags[:, c]
            phase = np.angle(spec[:, c])                 # f64

            if self._first_frame:
                # First frame: emit source phases verbatim. Seed history
                # from the zero-phase analysis frame. The first propagated
                # frame uses an Euler time step because no left-endpoint
                # gradient exists yet.
                self._prev_phase[c] = phase.copy()
                self._prev_mag[c] = mag.copy()
                self._prev_tgrad[c].fill(0.0)
                self._prev_synth_phase[c] = phase.copy()
                out_spec = mag * np.exp(1j * phase)
                time_frame[:, c] = np.fft.fftshift(
                    np.fft.irfft(out_spec, n=self.N_FFT))
                continue

            synth, tgrad_cur = self._rtpghi(
                mag,
                phase, self._prev_phase[c],
                self._prev_mag[c], self._prev_tgrad[c],
                self._prev_synth_phase[c], analysis_hop, hop_a, hop_s,
                self._have_prev_tgrad[c])

            self._prev_phase[c] = phase.copy()
            self._prev_mag[c] = mag.copy()
            self._prev_tgrad[c] = tgrad_cur
            self._prev_synth_phase[c] = synth
            self._have_prev_tgrad[c] = True

            out_spec = mag * np.exp(1j * synth)
            time_frame[:, c] = np.fft.fftshift(
                np.fft.irfft(out_spec, n=self.N_FFT))

        self._first_frame = False

        # OLA into the circular output ring. Writes wrap modulo the ring
        # size; because the ring is N_FFT*4 and we only ever write N_FFT
        # at a time while draining at least hop_s ahead, an active
        # window's tail cannot be overwritten before it's drained (caller
        # always drains before pumping more steps).
        self._ola_write_frame(time_frame * self._win[:, None])
        self._ola_write += hop_s

        # Advance fractional source position.
        self._src_pos += hop_a
        self._prev_hop_a = hop_a
        self._sync_source_pos()
        if self._src_pos >= self.source.n_frames:
            self._ended = True

    def _ola_write_frame(self, frame: np.ndarray) -> None:
        """Add an N_FFT-sample frame into the ring starting at
        _ola_write, wrapping around the ring buffer."""
        ring = self._ola_ring
        start = self._ola_write % ring
        end = start + self.N_FFT
        if end <= ring:
            self._ola[start:end] += frame
        else:
            first = ring - start
            self._ola[start:] += frame[:first]
            self._ola[:end - ring] += frame[first:]

    def _drain(self, n: int) -> np.ndarray:
        """Take up to `n` frames out of the circular OLA ring, normalize
        by the analytic COLA constant, and zero the drained cells so they
        start clean the next time the write pointer wraps through."""
        avail = self._ola_write - self._ola_read
        take = min(n, max(0, avail))
        ring = self._ola_ring
        out = np.empty((take, self.channels), dtype=np.float32)
        norm = 1.0 / max(1e-9, self._cola_norm)
        if take > 0:
            start = self._ola_read % ring
            end = start + take
            if end <= ring:
                out[:] = (self._ola[start:end] * norm).astype(np.float32)
                self._ola[start:end] = 0.0
            else:
                first = ring - start
                out[:first] = (self._ola[start:] * norm).astype(np.float32)
                out[first:] = (self._ola[:end - ring] * norm).astype(np.float32)
                self._ola[start:] = 0.0
                self._ola[:end - ring] = 0.0
        if take < n:
            pad = np.zeros((n - take, self.channels), dtype=np.float32)
            out = np.concatenate([out, pad], axis=0)
        self._ola_read += take
        return out

    # --- cheap fallback when pitch correction is off ---
    def _resample_read(self, n_frames: int) -> tuple[np.ndarray, bool]:
        """Pitch-shifting resample: read `n*rate` source frames and
        linearly interpolate down to `n`. Same feel as osu!/Etterna's
        built-in rate mods."""
        rate = max(0.05, self.rate)
        if abs(rate - 1.0) < 1e-3:
            data = self.source.read(n_frames)
            self._src_pos = float(self.source.pos)
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
        self._src_pos = float(self.source.pos)
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

    # --- decoding: soundfile first (fast, float-native for wav/flac/ogg),
    # audioread fallback (ffmpeg, handles mp3/m4a/webm/...). Both return
    # float32 samples ready to feed the phase vocoder without a mixer
    # roundtrip.
    def _decode(self, path: str):
        if _sf is not None:
            try:
                arr, sr = _sf.read(path, dtype='float32', always_2d=True)
                return arr, int(sr)
            except Exception as e:
                # Fall through to audioread — libsndfile doesn't do mp3/m4a
                # on every build.
                sf_err = e
        else:
            sf_err = None

        audioread_err = None
        if _audioread is not None:
            try:
                with _audioread.audio_open(path) as f:
                    sr = int(f.samplerate)
                    channels = int(f.channels)
                    chunks = []
                    for buf in f:
                        # audioread yields little-endian int16 byte buffers.
                        chunks.append(np.frombuffer(buf, dtype='<i2'))
                    if not chunks:
                        return None, 0
                    pcm = np.concatenate(chunks)
                    arr = pcm.astype(np.float32) / 32768.0
                    if channels > 1:
                        arr = arr.reshape(-1, channels)
                    else:
                        arr = arr.reshape(-1, 1)
                    return arr, sr
            except Exception as e:
                audioread_err = e

        ffmpeg_err = None
        ffmpeg = shutil.which('ffmpeg')
        ffprobe = shutil.which('ffprobe')
        if ffmpeg and ffprobe:
            try:
                probe = subprocess.run(
                    [ffprobe, '-v', 'error', '-select_streams', 'a:0',
                     '-show_entries', 'stream=sample_rate,channels',
                     '-of', 'json', path],
                    check=True, capture_output=True, text=True)
                stream = json.loads(probe.stdout)['streams'][0]
                sr = int(stream['sample_rate'])
                channels = int(stream['channels'])
                raw = subprocess.run(
                    [ffmpeg, '-v', 'error', '-i', path, '-f', 'f32le',
                     '-acodec', 'pcm_f32le', '-'],
                    check=True, capture_output=True).stdout
                arr = np.frombuffer(raw, dtype='<f4')
                if channels > 1:
                    arr = arr.reshape(-1, channels)
                else:
                    arr = arr.reshape(-1, 1)
                return arr.copy(), sr
            except Exception as e:
                ffmpeg_err = e

        print('audio: no decoder available '
              f'(soundfile={sf_err!r}, audioread={audioread_err!r}, '
              f'ffmpeg={ffmpeg_err!r})')
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
            if self._pv is not None:
                source_pos_s = self._pv.source_time
            else:
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
