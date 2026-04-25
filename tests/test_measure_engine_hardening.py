"""Hardening tests for the measure-based beat-space engine.

These cover the edge cases that were historically the source of bugs in
the reference engine:

  - Cumulative *inside* a STOP window (chart-beat is frozen, AC density
    is zero, integrator must return the pre-stop value).
  - WARPS with non-trivial SCROLLS active at the warp landing (the atom
    contribution must use v(tau_w^+), not v(tau_w^-)).
  - Stop+warp on the same beat (Etterna's same-row precedence).
  - SPEEDS lerp during a transition window.
  - Inverse_cumulative_at on a chart with a scroll<=0 plateau.

For each, the new measure-based engine must agree with the reference
engine to numerical precision.
"""
import numpy as np
import pytest

from analysis.player.sv.engine import BeatSpaceSVEngine
from analysis.player.sv.measure_engine import beat_space_engine


def _pair(**kw):
    """Build a (reference, new) pair for parameterized assertion."""
    return BeatSpaceSVEngine(**kw), beat_space_engine(**kw)


_CONST_BPM = [(0.0, 120.0)]   # 0.5 sec/beat


# ---------------------------------------------------------------------------
# Stops -- cumulative inside the stop window
# ---------------------------------------------------------------------------


def test_cumulative_inside_stop_matches_reference():
    # Stop at beat 4 (= t=2.0 under 120 BPM) for 0.5s. Cumulative on
    # [2.0, 2.5] must equal cumulative_at(2.0) -- chart-beat is frozen.
    ref, new = _pair(scrolls=[(0.0, 1.0)], speeds=[], bpms=_CONST_BPM,
                     sm_offset=0.0, stops=[(4.0, 0.5)])
    interior = np.linspace(2.0, 2.5, 25)
    for t in interior:
        assert new.cumulative_at(float(t)) == pytest.approx(
            ref.cumulative_at(float(t)), abs=1e-9), \
            f"stop interior mismatch at t={t}"


def test_distance_across_stop():
    ref, new = _pair(scrolls=[(0.0, 1.0)], speeds=[], bpms=_CONST_BPM,
                     sm_offset=0.0, stops=[(4.0, 0.5)])
    # Across the stop boundary: from t=1.5 (before) to t=3.0 (after).
    # Render distance must skip the stop.
    assert new.distance(1.5, 3.0) == pytest.approx(
        ref.distance(1.5, 3.0), abs=1e-9)


# ---------------------------------------------------------------------------
# Delays -- separate event type from stops, but same AC-density behavior
# ---------------------------------------------------------------------------


def test_cumulative_inside_delay_matches_reference():
    ref, new = _pair(scrolls=[(0.0, 1.0)], speeds=[], bpms=_CONST_BPM,
                     sm_offset=0.0, delays=[(4.0, 0.25)])
    samples = np.linspace(0.0, 4.0, 50)
    for t in samples:
        assert new.cumulative_at(float(t)) == pytest.approx(
            ref.cumulative_at(float(t)), abs=1e-9), \
            f"delay interior mismatch at t={t}"


# ---------------------------------------------------------------------------
# Warps -- atom mass uses v(tau_w^+), not v(tau_w^-)
# ---------------------------------------------------------------------------


def test_warp_with_scroll_change_at_landing():
    # Warp at beat 4 of length 4 beats. SCROLLS row at beat 8 with ratio 2.0.
    # Inside the warp [beat_4, beat_8), ratio is 1.0 (the post-prepend
    # default). At the warp's landing, ratio steps to 2.0. The atom mass
    # is Delta_b * v(tau_w^+) -- v at the post-warp beat. Verify parity.
    ref, new = _pair(
        scrolls=[(0.0, 1.0), (8.0, 2.0)],
        speeds=[],
        bpms=_CONST_BPM,
        sm_offset=0.0,
        warps=[(4.0, 4.0)],
    )
    samples = np.linspace(0.0, 6.0, 100)
    for t in samples:
        assert new.cumulative_at(float(t)) == pytest.approx(
            ref.cumulative_at(float(t)), abs=1e-7, rel=1e-9), \
            f"warp+scroll mismatch at t={t}"


def test_warp_at_beat_zero():
    # Warp at beat 0 -- no chart-time elapses before it. The atom should
    # still produce the right cumulative jump.
    ref, new = _pair(
        scrolls=[(0.0, 1.0)],
        speeds=[],
        bpms=_CONST_BPM,
        sm_offset=0.0,
        warps=[(0.0, 2.0)],
    )
    samples = np.linspace(0.0, 4.0, 100)
    for t in samples:
        assert new.cumulative_at(float(t)) == pytest.approx(
            ref.cumulative_at(float(t)), abs=1e-7, rel=1e-9), \
            f"beat-0 warp mismatch at t={t}"


# ---------------------------------------------------------------------------
# Stop + warp on the same beat (Etterna same-row precedence)
# ---------------------------------------------------------------------------


def test_stop_and_warp_same_beat():
    # Same row has both a STOP and a WARP. Etterna's precedence is
    # STOP-then-WARP after the marker; _TimingMap encodes this. Verify
    # both engines agree on cumulative across the event.
    ref, new = _pair(
        scrolls=[(0.0, 1.0)],
        speeds=[],
        bpms=_CONST_BPM,
        sm_offset=0.0,
        stops=[(4.0, 0.25)],
        warps=[(4.0, 2.0)],
    )
    samples = np.linspace(0.0, 5.0, 100)
    for t in samples:
        assert new.cumulative_at(float(t)) == pytest.approx(
            ref.cumulative_at(float(t)), abs=1e-7, rel=1e-9), \
            f"stop+warp same-row mismatch at t={t}"


# ---------------------------------------------------------------------------
# Multiple BPM changes interspersed with scroll
# ---------------------------------------------------------------------------


def test_bpm_changes_with_scroll():
    bpms = [(0.0, 120.0), (4.0, 240.0), (12.0, 60.0)]
    ref, new = _pair(
        scrolls=[(0.0, 1.0), (8.0, 1.5), (16.0, 0.5)],
        speeds=[],
        bpms=bpms,
        sm_offset=0.0,
    )
    samples = np.linspace(0.0, 30.0, 200)
    for t in samples:
        assert new.cumulative_at(float(t)) == pytest.approx(
            ref.cumulative_at(float(t)), abs=1e-7, rel=1e-9), \
            f"multi-BPM mismatch at t={t}"


# ---------------------------------------------------------------------------
# SPEEDS lerp window
# ---------------------------------------------------------------------------


def test_speeds_lerp_render_multiplier():
    # Two SPEEDS rows: (beat=0, ratio=1.0, delay=0, unit=0) and
    # (beat=4, ratio=2.0, delay=2.0, unit=0) -- a 2-beat lerp from 1x to 2x
    # starting at beat 4. render_multiplier_at must lerp during the window.
    speeds = [(0.0, 1.0, 0.0, 0), (4.0, 2.0, 2.0, 0)]
    ref, new = _pair(scrolls=[(0.0, 1.0)], speeds=speeds, bpms=_CONST_BPM,
                     sm_offset=0.0)
    samples = np.linspace(0.0, 5.0, 100)
    for t in samples:
        assert new.render_multiplier_at(float(t)) == pytest.approx(
            ref.render_multiplier_at(float(t)), abs=1e-9), \
            f"SPEEDS lerp mismatch at t={t}"


def test_distance_with_speeds_active():
    # Distance applies z(a) at the playhead -- non-trivial during a SPEEDS
    # transition.
    speeds = [(0.0, 1.0, 0.0, 0), (4.0, 2.0, 2.0, 0)]
    ref, new = _pair(scrolls=[(0.0, 1.0)], speeds=speeds, bpms=_CONST_BPM,
                     sm_offset=0.0)
    pairs = [(0.0, 5.0), (1.5, 4.0), (3.0, 3.5), (4.5, 6.0)]
    for a, b in pairs:
        assert new.distance(a, b) == pytest.approx(
            ref.distance(a, b), abs=1e-7, rel=1e-9), \
            f"SPEEDS distance mismatch at ({a},{b})"


# ---------------------------------------------------------------------------
# Inverse cumulative -- including scroll<=0 plateau
# ---------------------------------------------------------------------------


def test_inverse_with_zero_scroll_segment():
    # Scroll = 0 between beats 4 and 8 -- displayed-beat plateaus. The
    # inverse must collapse the plateau to the segment start (matches
    # reference engine's ScrollsCache plateau handling).
    ref, new = _pair(
        scrolls=[(0.0, 1.0), (4.0, 0.0), (8.0, 1.0)],
        speeds=[], bpms=_CONST_BPM, sm_offset=0.0,
    )
    # Sample the SV-space domain, invert through both engines.
    sv_samples = np.linspace(0.0, 6.0, 50)
    for sv in sv_samples:
        assert new.inverse_cumulative_at(float(sv)) == pytest.approx(
            ref.inverse_cumulative_at(float(sv)), abs=1e-9), \
            f"inverse mismatch at sv={sv}"


def test_inverse_round_trip_with_no_scrolls():
    # Regression: when a chart has no #SCROLLS at all, both engines used
    # to short-circuit `inverse_cumulative_at(sv)` to `float(sv)` (ref) /
    # `float(sv) / sec_per_base_beat` (measure), leaking the cumulative
    # value back as if it were chart-time. With no #SCROLLS, displayed
    # beat is the identity so the inverse must be beat_to_time(sv / spb).
    ref, new = _pair(scrolls=[], speeds=[], bpms=_CONST_BPM, sm_offset=0.0)
    for t in np.linspace(0.5, 30.0, 60):
        cn_ref = ref.cumulative_at(float(t))
        cn_new = new.cumulative_at(float(t))
        assert ref.inverse_cumulative_at(cn_ref) == pytest.approx(t, abs=1e-9)
        assert new.inverse_cumulative_at(cn_new) == pytest.approx(t, abs=1e-9)


def test_inverse_strictly_increasing_chart():
    # Sanity: on a chart with monotone-increasing cumulative, the inverse
    # is exact and should match.
    ref, new = _pair(
        scrolls=[(0.0, 1.0), (4.0, 1.5), (12.0, 0.5)],
        speeds=[], bpms=_CONST_BPM, sm_offset=0.0,
    )
    sv_samples = np.linspace(0.0, 10.0, 50)
    for sv in sv_samples:
        assert new.inverse_cumulative_at(float(sv)) == pytest.approx(
            ref.inverse_cumulative_at(float(sv)), abs=1e-9), \
            f"inverse mismatch at sv={sv}"


# ---------------------------------------------------------------------------
# Project_times stress test (vectorized path)
# ---------------------------------------------------------------------------


def test_project_beats_matches_reference():
    # Beat-space engines expose project_beats; chart-stream sprites in
    # negative-BPM warp aliases use it to position from chart-beat directly.
    ref, new = _pair(
        scrolls=[(0.0, 1.0), (4.0, 0.5), (8.0, 1.5)],
        speeds=[], bpms=_CONST_BPM, sm_offset=0.0,
    )
    beats = np.linspace(0.0, 16.0, 100)
    np.testing.assert_allclose(new.project_beats(beats),
                                ref.project_beats(beats),
                                atol=1e-9)


def test_time_space_engine_has_no_project_beats():
    # render.py uses hasattr() to dispatch; time-space engines must not
    # expose project_beats so the time-based fallback fires.
    from analysis.player.sv.measure_engine import time_space_engine
    eng = time_space_engine([(0.0, 1.0), (5.0, 2.0)])
    assert not hasattr(eng, 'project_beats')


def test_project_times_with_warp_and_stop():
    ref, new = _pair(
        scrolls=[(0.0, 1.0), (12.0, 0.5)],
        speeds=[],
        bpms=[(0.0, 120.0), (8.0, 240.0)],
        sm_offset=0.05,
        stops=[(4.0, 0.3)],
        warps=[(16.0, 4.0)],
    )
    # Dense sample including stop and warp boundaries.
    samples = np.linspace(-0.5, 25.0, 500)
    np.testing.assert_allclose(new.project_times(samples),
                                ref.project_times(samples),
                                atol=1e-7, rtol=1e-9)
