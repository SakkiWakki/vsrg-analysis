"""Phase-vocoder correctness tests.

The PV is easy to break subtly ; a wrong COLA constant, float32 phase
drift, or a botched first-frame init all produce the same "sounds muffled"
symptom. These tests pin down specific numerical invariants so regressions
show up as failing assertions instead of "something sounds off."

Signals are synthesized so we can assert RMS, spectral peaks, and
round-trip equality without needing fixtures. No audio device required ;
we instantiate StreamingPhaseVocoder directly against a WaveSource built
from an ndarray.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from analysis.player.audio import (
    StreamingPhaseVocoder,
    WaveSource,
)


SR = 44100


def _sine(freq_hz: float, dur_s: float, amp: float = 0.3, sr: int = SR,
          channels: int = 1) -> np.ndarray:
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    s = amp * np.sin(2.0 * np.pi * freq_hz * t)
    return (np.stack([s] * channels, axis=1) if channels > 1
            else s[:, None]).astype(np.float32)


def _multitone(freqs, dur_s: float, amp: float = 0.2, sr: int = SR):
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    s = amp * sum(np.sin(2.0 * np.pi * f * t) for f in freqs)
    return s.astype(np.float32)[:, None]


def _pump_all(pv: StreamingPhaseVocoder, out_samples: int,
              block: int = 2048) -> np.ndarray:
    out = []
    while len(out) * block < out_samples:
        chunk, cont = pv.generate(block)
        out.append(chunk)
        if not cont:
            break
    y = np.concatenate(out, axis=0) if out else np.zeros((0, pv.channels),
                                                          dtype=np.float32)
    return y[:out_samples]


# ---- COLA / gain -----------------------------------------------------------


def test_cola_constant_matches_empirical_sum_of_squared_windows():
    """The analytic COLA divisor must match a numerical computation of
    sum_k w(n - k*hop)^2 at a steady-state output position. If this drifts
    we get a wrong gain and a broadband tilt."""
    pv = StreamingPhaseVocoder(WaveSource(np.zeros(SR, dtype=np.float32), SR))
    N = pv.N_FFT
    HOP = pv.HOP
    w = np.hanning(N)
    tot = np.zeros(3 * N)
    for k in range(2 * (N // HOP) + 1):
        tot[k * HOP:k * HOP + N] += w * w
    empirical = float(np.mean(tot[N:2 * N]))
    assert pv._cola_norm == pytest.approx(empirical, rel=1e-3)


def test_rate_one_passthrough_preserves_rms():
    """Rate≈1 takes a fast path that bypasses the PV. Output must equal
    input RMS exactly (no gain loss)."""
    sig = _sine(440, 1.0, amp=0.3)
    src = WaveSource(sig, SR)
    pv = StreamingPhaseVocoder(src, rate=1.0)
    out = _pump_all(pv, SR)
    assert float(np.sqrt(np.mean(out ** 2))) == pytest.approx(0.3 / math.sqrt(2),
                                                              rel=0.01)


def test_stretched_rms_within_leakage_tolerance():
    """At rate 1.25 the analytic gain should put the output within a few
    percent of input RMS; larger deltas indicate a COLA bug."""
    sig = _sine(440, 2.0, amp=0.3)
    pv = StreamingPhaseVocoder(WaveSource(sig, SR), rate=1.25)
    out = _pump_all(pv, int(SR * 2 * 1.25))[2048:]  # skip OLA ramp
    rms = float(np.sqrt(np.mean(out ** 2)))
    expected = 0.3 / math.sqrt(2)
    # 20% tolerance ; windowing + finite-frame bias eats a bit of energy.
    assert rms == pytest.approx(expected, rel=0.2)


# ---- HF preservation -------------------------------------------------------


@pytest.mark.parametrize('rate', [0.85, 1.15, 1.5])
def test_all_frequencies_preserved_when_stretching(rate):
    """Stretching at modest rates should not attenuate high-frequency tones
    relative to low-frequency ones. This is the "muffled" regression
    guard ; a float32-overflowed phase accumulator silently kills HF
    content."""
    freqs = (100, 1000, 5000, 12000)
    sig = _multitone(freqs, 3.0, amp=0.2)
    pv = StreamingPhaseVocoder(WaveSource(sig, SR), rate=rate)
    out = _pump_all(pv, int(SR * 3 * rate))[:, 0]
    # Analyze the middle second to avoid OLA ramp-in/out.
    mid = out[SR // 2:SR + SR // 2]
    spec = np.abs(np.fft.rfft(mid))
    axis = np.fft.rfftfreq(len(mid), 1.0 / SR)
    peaks = []
    for f in freqs:
        band = (axis > f * 0.9) & (axis < f * 1.1)
        peaks.append(float(spec[band].max()))
    # Pitch-preserving stretch should give roughly flat spectral balance
    # for an equal-amplitude multitone. Allow 40% slack for window leakage
    # variance across frequencies.
    assert min(peaks) > 0.6 * max(peaks), (
        f'HF attenuation detected at rate {rate}: peaks={peaks}')


def test_pitch_preserved_under_stretch():
    """The whole point of pitch correction ; a 440Hz input at rate 1.25
    should still peak at 440Hz, not at 440*1.25=550Hz. That would mean the
    PV is accidentally resampling instead of time-stretching."""
    sig = _sine(440, 2.0, amp=0.3)
    pv = StreamingPhaseVocoder(WaveSource(sig, SR), rate=1.25)
    out = _pump_all(pv, int(SR * 2.5))[2048:, 0]
    spec = np.abs(np.fft.rfft(out))
    axis = np.fft.rfftfreq(len(out), 1.0 / SR)
    peak_hz = axis[int(np.argmax(spec))]
    # Bin resolution at this FFT size is ~1Hz; 5Hz tolerance is generous.
    assert abs(peak_hz - 440.0) < 5.0, f'peak at {peak_hz}Hz, expected 440'


# ---- Phase-state numerics --------------------------------------------------


def test_phase_state_is_float64():
    """Float32 phase accumulators drift for high bins over many hops
    because omega_a * (N_FFT/2) exceeds float32's mantissa precision. All
    phase-carrying state must be float64."""
    pv = StreamingPhaseVocoder(
        WaveSource(np.zeros(SR, dtype=np.float32), SR))
    assert pv._bin_freqs.dtype == np.float64
    assert pv._prev_phase[0].dtype == np.float64
    assert pv._out_phase[0].dtype == np.float64
    assert pv._win.dtype == np.float64


def test_first_frame_does_not_produce_phase_transient():
    """The first frame must emit source phases as-is. If it runs the
    propagation step with zeroed `_prev_phase`, the second-frame dphi is
    bogus and the output gets a spurious click. We check by asserting the
    first hop's output isn't wildly larger than the steady-state RMS."""
    sig = _sine(440, 1.0, amp=0.3)
    pv = StreamingPhaseVocoder(WaveSource(sig, SR), rate=1.2)
    out = _pump_all(pv, SR)[:, 0]
    head = float(np.abs(out[:pv.HOP]).max())
    body = float(np.sqrt(np.mean(out[4096:8192] ** 2)))
    # The first hop shouldn't be more than ~4x the steady-state peak.
    assert head < 4.0 * (body * math.sqrt(2)), (
        f'first-frame transient: head peak {head}, body rms {body}')


# ---- Fractional source reads ----------------------------------------------


def test_fractional_src_pos_does_not_snap_to_integer():
    """At rate=1.3, hop_a = 512 * 1.3 = 665.6 → _src_pos accumulates a
    non-zero fractional part. If the analysis read snaps to floor(_src_pos)
    without interpolating, we get dithered timing jitter. Assert the read
    returns a sample that's actually between the two neighbors."""
    n = 4096
    t = np.arange(n) / SR
    # Simple ramp so neighbor-interpolation is easy to verify.
    sig = np.linspace(-1, 1, n, dtype=np.float32)[:, None]
    pv = StreamingPhaseVocoder(WaveSource(sig, SR), rate=1.3)
    pv._src_pos = 100.25  # known fractional offset
    frame = pv._read_frame()
    expected_0 = 0.75 * sig[100, 0] + 0.25 * sig[101, 0]
    assert frame[0] == pytest.approx(expected_0, rel=1e-5)
    expected_100 = 0.75 * sig[200, 0] + 0.25 * sig[201, 0]
    assert frame[100] == pytest.approx(expected_100, rel=1e-5)


def test_rate_one_fast_path_keeps_src_pos_in_sync():
    """After a rate-1 fast-path generate, `_src_pos` must match the
    source's read cursor. Otherwise toggling rate back to a non-1 value
    starts the PV from a stale position and you get a discontinuity."""
    sig = _sine(220, 2.0)
    pv = StreamingPhaseVocoder(WaveSource(sig, SR), rate=1.0)
    pv.generate(4096)
    assert pv._src_pos == pytest.approx(pv.source.pos, abs=1.0)


# ---- End-to-end round-trip ------------------------------------------------


def test_stretched_length_matches_rate():
    """Output length should scale inversely with rate: at rate=1.25 a 1-
    second input should produce ~0.8 seconds of output. We pump until the
    source is exhausted and check the total."""
    sig = _sine(440, 1.0, amp=0.2)
    pv = StreamingPhaseVocoder(WaveSource(sig, SR), rate=1.25)
    total = 0
    for _ in range(200):
        chunk, cont = pv.generate(2048)
        total += chunk.shape[0]
        if not cont:
            break
    # Expected ≈ sr / rate = 35280 frames. Allow ±2 hops of tail wobble.
    expected = SR / 1.25
    assert abs(total - expected) < 4 * pv.HOP


# ---- Source-position bookkeeping (sync-drift regression) ------------------
#
# AudioEngine.set_state() checks `abs(source_pos_s - t) > RESYNC_THRESHOLD_S`
# every tick to decide whether to force a seek. If the cursor it reads is
# stale, playback will re-seek itself every ~150 ms and throb audibly. The
# invariants below pin down the two halves of the fix:
#   - PV.source_time reflects the PV's own fractional read cursor.
#   - WaveSource.pos is kept mirrored to that cursor after every _step().


def test_source_time_advances_during_stretched_playback():
    """`source_time` must progress when the PV is pumping frames. A stuck
    value is what triggered the set_state() drift-check to force a resync."""
    sig = _sine(440, 2.0, amp=0.3)
    pv = StreamingPhaseVocoder(WaveSource(sig, SR), rate=1.25)
    t0 = pv.source_time
    assert t0 == pytest.approx(0.0, abs=1e-6)
    _ = _pump_all(pv, int(SR * 0.5))
    t1 = pv.source_time
    # Expected source advance ≈ pumped_out_seconds * rate.
    # Allow generous slack for OLA ramp / partial final hop.
    assert t1 > 0.3, f'source_time did not advance: {t1}'
    assert t1 < 1.5, f'source_time ran past the source: {t1}'


def test_wave_source_pos_mirrors_pv_cursor_after_step():
    """Each `_step()` must mirror the fractional PV cursor back into
    `WaveSource.pos` so external drift checks reading that field see a
    live position, not the stale value from the last explicit seek."""
    sig = _sine(440, 2.0, amp=0.3)
    src = WaveSource(sig, SR)
    pv = StreamingPhaseVocoder(src, rate=1.3)
    assert src.pos == 0
    pv._step()
    assert src.pos > 0, 'WaveSource.pos was not advanced by _step()'
    # After the step, pos must equal the rounded PV cursor.
    assert src.pos == pytest.approx(int(round(pv._src_pos)), abs=1)
    # Further steps keep them in sync.
    for _ in range(20):
        pv._step()
    assert src.pos == pytest.approx(int(round(pv._src_pos)), abs=1)


def test_source_time_matches_source_pos_while_streaming():
    """`source_time` is what AudioEngine.set_state() reads; `source.pos` is
    the fallback path. Both must agree to within one frame so a mismatch
    between the two code paths can't re-introduce drift."""
    sig = _sine(440, 3.0, amp=0.3)
    pv = StreamingPhaseVocoder(WaveSource(sig, SR), rate=1.25)
    _ = _pump_all(pv, int(SR * 1.0))
    via_prop = pv.source_time
    via_src = pv.source.pos / pv.source.sr
    assert via_prop == pytest.approx(via_src, abs=1.0 / SR)


def test_seek_realigns_both_cursors():
    """After `seek()` both cursors must reflect the new chart time; the
    PV and AudioEngine paths would otherwise disagree on where we are."""
    sig = _sine(440, 3.0, amp=0.3)
    pv = StreamingPhaseVocoder(WaveSource(sig, SR), rate=1.2)
    # Advance somewhere, then jump.
    _ = _pump_all(pv, int(SR * 0.4))
    pv.seek(1.5)
    assert pv.source_time == pytest.approx(1.5, abs=1.0 / SR)
    assert pv.source.pos == pytest.approx(int(round(1.5 * SR)), abs=1)


def test_source_time_does_not_exceed_source_length():
    """Clamp check: once we've consumed the whole source, `source_time`
    caps at the source duration. A drift check comparing against chart
    time would otherwise race past the end of the file."""
    sig = _sine(440, 0.5, amp=0.3)
    pv = StreamingPhaseVocoder(WaveSource(sig, SR), rate=1.25)
    # Pump far past the source to guarantee exhaustion.
    for _ in range(200):
        _, cont = pv.generate(2048)
        if not cont:
            break
    assert pv.source_time <= pv.source.duration + 1e-6


def test_sync_source_pos_handles_out_of_range_cursor():
    """`_sync_source_pos` must clamp; otherwise an over-advanced cursor
    (end-of-file drain) would write a negative or out-of-bounds index
    into `WaveSource.pos` and break the set_state() fallback path."""
    sig = _sine(440, 0.5, amp=0.3)
    pv = StreamingPhaseVocoder(WaveSource(sig, SR), rate=1.0)
    pv._src_pos = -50.0
    pv._sync_source_pos()
    assert pv.source.pos == 0
    pv._src_pos = float(pv.source.n_frames + 10_000)
    pv._sync_source_pos()
    assert pv.source.pos == pv.source.n_frames


def test_seek_resets_phase_and_flux_state():
    """After seek() the PV must be bit-identical to a freshly-constructed
    one (modulo the source position). Otherwise seeking injects stale
    phase accumulator contents into the first output frame."""
    sig = _sine(440, 2.0, amp=0.3)
    pv = StreamingPhaseVocoder(WaveSource(sig, SR), rate=1.2)
    # Advance a bit.
    for _ in range(30):
        pv._step()
    pv.seek(0.5)
    assert pv._first_frame is True
    assert float(np.abs(pv._out_phase[0]).max()) == 0.0
    assert float(np.abs(pv._prev_mag[0]).max()) == 0.0
