"""Cull-space predictor (CullSpacePredictor) tests.

Predictor contract:

  Anchored on `(t, C(t), r(t^+), wall)` after every audio callback or
  discontinuity. Between anchors, returns
      C_anchor + r_anchor * (wall_now - wall_anchor) * play_rate
  except where the predicted chart-time would cross a breakpoint, in
  which case the anchor is rolled forward to the breakpoint and
  extrapolation continues from `(bp_t, C(bp_t), r(bp_t^+))`.

  Equivalence claim: on a single constant-rate segment, the predictor's
  output equals `engine.cumulative_at(chart_t_pred)` where
  `chart_t_pred = anchor_t + wall_dt * play_rate`. At breakpoint
  crossings the two paths agree term-for-term.

These tests use a stubbed clock (replaces `time.monotonic`) so the
wall-time advance is fully deterministic.
"""
import numpy as np
import pytest

from analysis.player.playback import cull_predictor as cp_mod
from analysis.player.playback.cull_predictor import CullSpacePredictor
from analysis.player.sv.engine import (BeatSpaceSVEngine, IdentitySVEngine,
                                        TimeSpaceSVEngine)


@pytest.fixture
def fake_clock(monkeypatch):
    """Patch `time.monotonic` inside the predictor module with a
    controllable clock so wall-time advances are deterministic."""
    state = {'t': 1000.0}

    def fake_monotonic():
        return state['t']

    monkeypatch.setattr(cp_mod.time, 'monotonic', fake_monotonic)
    return state


# ---------------------------------------------------------------------------
# Constant-rate equivalence
# ---------------------------------------------------------------------------


def test_constant_rate_segment_matches_cumulative_at(fake_clock):
    """On a single time-space segment with no boundary, the predictor's
    output must equal cumulative_at(raw_t) for any (raw_t, wall_now)
    pair that's consistent with the anchor."""
    engine = TimeSpaceSVEngine([(0.0, 2.0)])
    pred = CullSpacePredictor(engine, breakpoints=[])

    # Anchor at raw_t=0
    fake_clock['t'] = 1000.0
    pred.reset(0.0)

    # Advance wall by 0.1s, raw_t advances by 0.1 (play_rate=1).
    fake_clock['t'] = 1000.1
    raw_t = 0.1
    out = pred.cumulative_now(raw_t, play_rate=1.0)
    assert out == pytest.approx(engine.cumulative_at(raw_t),
                                 abs=1e-12)


def test_constant_rate_handles_play_rate_below_unity(fake_clock):
    """At play_rate=0.5, chart-time advances at half wall-time."""
    engine = TimeSpaceSVEngine([(0.0, 3.0)])
    pred = CullSpacePredictor(engine, breakpoints=[])

    fake_clock['t'] = 0.0
    pred.reset(0.0)
    pred._last_play_rate = 0.5  # avoid the rate-change re-anchor on first call

    # 0.2s of wall-time at 0.5x = 0.1s of chart-time.
    fake_clock['t'] = 0.2
    raw_t = 0.1
    out = pred.cumulative_now(raw_t, play_rate=0.5)
    # Cumulative at chart_t=0.1 with multiplier=3 => 0.3
    assert out == pytest.approx(0.3, abs=1e-12)


# ---------------------------------------------------------------------------
# Breakpoint crossings
# ---------------------------------------------------------------------------


def test_extrapolation_handles_single_breakpoint_crossing(fake_clock):
    """Two-segment chart: 1.0x on [0, 5), 2.0x on [5, inf). When the
    predictor extrapolates past t=5, it must roll the anchor forward to
    t=5 and apply the new rate. Result equals cumulative_at(t_pred)."""
    engine = TimeSpaceSVEngine([(0.0, 1.0), (5.0, 2.0)])
    breakpoints = engine.breakpoints() if hasattr(engine, 'breakpoints') \
        else np.array([0.0, 5.0])
    pred = CullSpacePredictor(engine, breakpoints=breakpoints)

    fake_clock['t'] = 0.0
    pred.reset(0.0)

    # Advance to chart_t=7. Crosses breakpoint at t=5.
    # Expected: cumulative_at(7) = 5*1.0 + 2*2.0 = 9.0.
    fake_clock['t'] = 7.0
    raw_t = 7.0
    out = pred.cumulative_now(raw_t, play_rate=1.0)
    assert out == pytest.approx(engine.cumulative_at(raw_t),
                                 abs=1e-12)
    assert out == pytest.approx(9.0, abs=1e-12)


def test_extrapolation_handles_multiple_breakpoints_in_one_frame(fake_clock):
    """Chart with three rapid SV changes. A single frame's extrapolation
    advances past all three -- the loop must handle this iteratively."""
    sections = [(0.0, 1.0), (1.0, 4.0), (2.0, 0.25), (3.0, 1.0)]
    engine = TimeSpaceSVEngine(sections)
    breakpoints = engine.breakpoints()
    pred = CullSpacePredictor(engine, breakpoints=breakpoints)

    fake_clock['t'] = 0.0
    pred.reset(0.0)

    # Advance past all three boundaries to t=4. Expected:
    # 0..1 at 1.0 = 1.0
    # 1..2 at 4.0 = 4.0
    # 2..3 at 0.25 = 0.25
    # 3..4 at 1.0 = 1.0
    # total = 6.25
    fake_clock['t'] = 4.0
    out = pred.cumulative_now(4.0, play_rate=1.0)
    assert out == pytest.approx(6.25, abs=1e-12)
    assert out == pytest.approx(engine.cumulative_at(4.0), abs=1e-12)


def test_extrapolation_at_bpm_change_in_beat_space(fake_clock):
    """Beat-space engine with a BPM change. The predictor's breakpoint
    list includes BPM-change times; extrapolation across them must use
    the new BPM-derived rate post-change."""
    engine = BeatSpaceSVEngine(
        scrolls=[(0.0, 1.0)],
        speeds=[],
        bpms=[(0.0, 120.0), (4.0, 240.0)],   # BPM doubles at beat 4
        sm_offset=0.0,
    )
    # Beat 4 @ 120 BPM = 2.0 seconds. After that, dB/dt doubles.
    # cumulative_at(3.0) should equal: 2*1.0 + 1*2.0 = 4.0 (in displayed-
    # beat seconds at base BPM=120 -> sec_per_base_beat=0.5).
    # Actually let me just rely on engine.cumulative_at as ground truth.
    breakpoints = np.array(sorted(set(
        list(engine._timing._time_enter)
        + list(engine._timing._time_exit)
    )))
    pred = CullSpacePredictor(engine, breakpoints=breakpoints)

    fake_clock['t'] = 0.0
    pred.reset(0.0)

    fake_clock['t'] = 3.0    # past the BPM change at chart_t=2
    out = pred.cumulative_now(3.0, play_rate=1.0)
    assert out == pytest.approx(engine.cumulative_at(3.0), abs=1e-9)


def test_extrapolation_across_warp_atom(fake_clock):
    """Warps add a Dirac-mass jump to dB. Predictor must include the
    atom contribution exactly when extrapolating past the warp's
    chart-time."""
    engine = BeatSpaceSVEngine(
        scrolls=[(0.0, 1.0)],
        speeds=[],
        bpms=[(0.0, 120.0)],
        sm_offset=0.0,
        warps=[(4.0, 4.0)],     # 4-beat warp at beat 4 (chart_t = 2.0s)
    )
    breakpoints = np.array(sorted(set(
        list(engine._timing._time_enter)
        + list(engine._timing._time_exit)
    )))
    pred = CullSpacePredictor(engine, breakpoints=breakpoints)

    fake_clock['t'] = 0.0
    pred.reset(0.0)

    # Just past the warp.
    fake_clock['t'] = 2.5
    out = pred.cumulative_now(2.5, play_rate=1.0)
    assert out == pytest.approx(engine.cumulative_at(2.5), abs=1e-9)


# ---------------------------------------------------------------------------
# Re-anchoring on raw_t jumps and rate changes
# ---------------------------------------------------------------------------


def test_seek_backwards_snaps(fake_clock):
    """A backward jump in raw_t (user seeks back) must immediately
    re-anchor at the new raw_t."""
    engine = TimeSpaceSVEngine([(0.0, 1.0)])
    pred = CullSpacePredictor(engine, breakpoints=[])

    fake_clock['t'] = 0.0
    pred.reset(5.0)

    # User seeks back to 1.0 -- raw_t jumps backward.
    fake_clock['t'] = 0.001
    out = pred.cumulative_now(1.0, play_rate=1.0)
    assert out == pytest.approx(engine.cumulative_at(1.0), abs=1e-12)


def test_audio_callback_re_anchors(fake_clock):
    """When raw_t jumps forward more than the predicted chart-time
    advance (i.e. an audio callback brought new exact data), the
    predictor must re-anchor at the new raw_t."""
    engine = TimeSpaceSVEngine([(0.0, 1.0)])
    pred = CullSpacePredictor(engine, breakpoints=[])

    fake_clock['t'] = 0.0
    pred.reset(0.0)

    # Wall barely advances; raw_t jumps far ahead -- this is what an
    # audio callback looks like (the engine processed a new block and
    # the DAC anchor moved ahead).
    fake_clock['t'] = 0.001
    out = pred.cumulative_now(0.5, play_rate=1.0)
    # Re-anchor expected; output is cumulative_at(0.5) exactly.
    assert out == pytest.approx(engine.cumulative_at(0.5), abs=1e-12)


def test_rate_change_re_anchors(fake_clock):
    """A play_rate change without a raw_t discontinuity re-anchors so
    future extrapolation uses the new rate."""
    engine = TimeSpaceSVEngine([(0.0, 1.0)])
    pred = CullSpacePredictor(engine, breakpoints=[])

    fake_clock['t'] = 0.0
    pred.reset(0.0)
    pred._last_play_rate = 1.0

    # Half a second of wall-time at 1.0x.
    fake_clock['t'] = 0.5
    pred.cumulative_now(0.5, play_rate=1.0)

    # Now rate changes to 2.0x; raw_t advances at the new rate from here.
    # 0.1 wall * 2.0 = 0.2 chart-time advance.
    fake_clock['t'] = 0.6
    raw_t = 0.7   # 0.5 + 0.2
    out = pred.cumulative_now(raw_t, play_rate=2.0)
    # The predictor re-anchored on the rate change at raw_t=0.5 (last
    # call); first cumulative_now after rate change returns _anchor_C =
    # cumulative_at(raw_t) at that moment.
    assert out == pytest.approx(engine.cumulative_at(raw_t),
                                 abs=1e-12)


# ---------------------------------------------------------------------------
# Identity-engine fallback
# ---------------------------------------------------------------------------


def test_identity_engine_returns_raw_t(fake_clock):
    """When the engine is identity (or disabled), the predictor short-
    circuits to raw_t."""
    pred = CullSpacePredictor(IdentitySVEngine(), breakpoints=[])
    out = pred.cumulative_now(7.5, play_rate=1.0)
    assert out == pytest.approx(7.5, abs=0)


def test_no_engine_returns_raw_t(fake_clock):
    pred = CullSpacePredictor(None, breakpoints=[])
    out = pred.cumulative_now(7.5, play_rate=1.0)
    assert out == pytest.approx(7.5, abs=0)


# ---------------------------------------------------------------------------
# Equivalence on the real chart fixture
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Monotonicity guard
# ---------------------------------------------------------------------------


def test_output_is_monotone_under_forward_playback(fake_clock):
    """Under forward playback, cumulative_now must be monotone non-
    decreasing in wall-time. Any sub-ms backward output is noise (stream-
    clock jitter, sub-tolerance audio backsteps) and must be clamped to
    the previous high-water mark.

    The cumulative function itself is monotone non-decreasing (Theorem,
    DESIGN.tex sec.4: dC/dt = v*w >= 0 a.e., warp atoms add positive
    mass), so this is the predictor honoring its target's contract."""
    engine = TimeSpaceSVEngine([(0.0, 1.0), (2.5, 1.5), (4.0, 0.5)])
    pred = CullSpacePredictor(engine, breakpoints=engine.breakpoints())

    fake_clock['t'] = 0.0
    pred.reset(0.0)

    # Simulate raw_t arriving with 0.5 ms RMS Gaussian jitter (typical
    # PortAudio stream.time noise) on a clean monotone chart-time stream.
    rng = np.random.default_rng(seed=0xC07107)
    chart_t = 0.0
    prev = 0.0
    for _ in range(2000):
        chart_t += 0.008      # ~125Hz frame cadence
        fake_clock['t'] = chart_t
        noisy_raw = chart_t + rng.normal(0.0, 0.0005)
        out = pred.cumulative_now(noisy_raw, play_rate=1.0)
        assert out >= prev - 1e-12, \
            f'predictor went backward: prev={prev}, out={out}'
        prev = out


def test_clamp_does_not_corrupt_correctness_on_clean_stream(fake_clock):
    """The clamp must not push the predictor away from cumulative_at on
    a clean stream. After settling past any initial transient, the
    predictor + clamp should still hit cumulative_at within float
    epsilon."""
    engine = TimeSpaceSVEngine([(0.0, 1.0), (2.5, 1.5), (4.0, 0.5)])
    pred = CullSpacePredictor(engine, breakpoints=engine.breakpoints())

    fake_clock['t'] = 0.0
    pred.reset(0.0)

    chart_t = 0.0
    while chart_t < 6.0:
        chart_t += 0.05
        fake_clock['t'] = chart_t
        out = pred.cumulative_now(chart_t, play_rate=1.0)
        assert out == pytest.approx(engine.cumulative_at(chart_t),
                                     abs=1e-9)


def test_reset_clears_the_clamp(fake_clock):
    """An explicit reset (seek, rate change, engine swap) is a
    legitimate backward move; the clamp must release."""
    engine = TimeSpaceSVEngine([(0.0, 1.0)])
    pred = CullSpacePredictor(engine, breakpoints=[])

    # Drive the clamp's high-water mark up to ~5.0 by anchoring there.
    fake_clock['t'] = 0.0
    pred.reset(5.0)

    # User seeks backward via reset(). New raw_t = 1.0 should be the
    # output, not the clamped 5.0.
    pred.reset(1.0)
    out = pred.cumulative_now(1.0, play_rate=1.0)
    assert out == pytest.approx(1.0, abs=1e-12)


def test_predictor_matches_cumulative_at_on_dense_sample(fake_clock):
    """Dense parity sweep: at every chart-time the predictor would see,
    its output must equal cumulative_at(raw_t) when the wall-time
    advance is consistent with raw_t. This is the core mathematical
    claim: the predictor is the same function as cumulative_at, just
    sampled differently in time."""
    engine = TimeSpaceSVEngine([(0.0, 1.0), (2.5, 1.5), (4.0, 0.5),
                                (6.0, 2.0)])
    pred = CullSpacePredictor(engine, breakpoints=engine.breakpoints())

    fake_clock['t'] = 0.0
    pred.reset(0.0)

    # Walk forward in 50ms wall-time steps; raw_t mirrors wall-time.
    chart_t = 0.0
    while chart_t < 8.0:
        chart_t += 0.05
        fake_clock['t'] = chart_t
        out = pred.cumulative_now(chart_t, play_rate=1.0)
        assert out == pytest.approx(engine.cumulative_at(chart_t),
                                     abs=1e-9), \
            f'mismatch at t={chart_t}: pred={out}, ' \
            f'cum={engine.cumulative_at(chart_t)}'
