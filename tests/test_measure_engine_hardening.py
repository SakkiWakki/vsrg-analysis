"""Hardening tests for the measure-based beat-space engine.

These cover the edge cases that have historically been bug magnets in
beat-space SV integration:

  - Cumulative *inside* a STOP window (chart-beat is frozen, AC density
    is zero, integrator must return the pre-stop value).
  - WARPS with non-trivial SCROLLS active at the warp landing (the atom
    contribution must use v(tau_w^+), not v(tau_w^-)).
  - Stop+warp on the same beat (Etterna's same-row precedence).
  - SPEEDS lerp during a transition window.
  - inverse_cumulative_at on a chart with a scroll<=0 plateau.

Each case asserts against precomputed expected values: pytest.approx with
absolute tolerance, no comparison engine. If a future refactor shifts the
integrator's output, these will catch it.
"""
import numpy as np
import pytest

from analysis.player.sv.measure_engine import beat_space_engine, time_space_engine


_CONST_BPM = [(0.0, 120.0)]   # 0.5 sec/beat


# ---------------------------------------------------------------------------
# Stops -- cumulative inside the stop window
# ---------------------------------------------------------------------------


def test_cumulative_inside_stop_is_frozen():
    # Stop at beat 4 (= t=2.0 under 120 BPM) for 0.5s. Cumulative on
    # [2.0, 2.5] must equal cumulative_at(2.0) -- chart-beat is frozen.
    eng = beat_space_engine(scrolls=[(0.0, 1.0)], speeds=[], bpms=_CONST_BPM,
                            sm_offset=0.0, stops=[(4.0, 0.5)])
    pre_stop = eng.cumulative_at(2.0)
    interior = np.linspace(2.0, 2.5, 25)
    for t in interior:
        assert eng.cumulative_at(float(t)) == pytest.approx(
            pre_stop, abs=1e-9), \
            f"stop interior should be flat at t={t}"


def test_distance_across_stop_skips_frozen_window():
    eng = beat_space_engine(scrolls=[(0.0, 1.0)], speeds=[], bpms=_CONST_BPM,
                            sm_offset=0.0, stops=[(4.0, 0.5)])
    # 1.5s (beat 3) -> 3.0s (beat 5, since the 0.5s stop ate half a second).
    # Render distance under scrolls=1, base BPM=120: 2 displayed-beats *
    # 0.5 sec/beat = 1.0 cum-units (the stop interval contributes zero).
    assert eng.distance(1.5, 3.0) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Delays -- separate event type from stops, but same AC-density behavior
# ---------------------------------------------------------------------------


def test_cumulative_inside_delay_is_frozen():
    eng = beat_space_engine(scrolls=[(0.0, 1.0)], speeds=[], bpms=_CONST_BPM,
                            sm_offset=0.0, delays=[(4.0, 0.25)])
    # Delay starts at beat 4 (t=2.0) for 0.25s; chart-beat doesn't advance
    # during the delay.
    pre_delay = eng.cumulative_at(2.0)
    interior = np.linspace(2.0, 2.25, 12)
    for t in interior:
        assert eng.cumulative_at(float(t)) == pytest.approx(
            pre_delay, abs=1e-9), f"delay interior should be flat at t={t}"


# ---------------------------------------------------------------------------
# Warps -- atom mass uses v(tau_w^+), not v(tau_w^-)
# ---------------------------------------------------------------------------


def test_warp_with_scroll_change_at_landing():
    # Warp at beat 4 of length 4 beats -> displayed-beat 8 lives at
    # chart-time 2.0s. SCROLLS row at beat 8 with ratio 2.0. The atom
    # mass must use v at the post-warp beat (1.0 inside the warp's run-up,
    # since the new ratio doesn't kick in until beat 8 - which is also the
    # warp's landing beat).
    eng = beat_space_engine(
        scrolls=[(0.0, 1.0), (8.0, 2.0)],
        speeds=[],
        bpms=_CONST_BPM,
        sm_offset=0.0,
        warps=[(4.0, 4.0)],
    )
    # At t=2.0s (beat 4), we've integrated 4 displayed-beats * 0.5 = 2.0.
    # The warp atom adds 4 more beats at v(tau_w^+) = 1.0 (still under the
    # first scrolls segment) -> +2.0 cum-units. So cum_at(2.0) = 4.0.
    assert eng.cumulative_at(2.0) == pytest.approx(4.0, abs=1e-9)
    # Past the warp + new scrolls: t=3.0s = beat 6 in real beats, but
    # displayed-beat has skipped from 8 to 8 + (6-4)*1.0 ... wait, the warp
    # consumed 4 chart-beats so beat 6 chart = displayed_beat 6+4=10? No --
    # the warp jumps the displayed-beat counter forward at tau_w. The
    # post-warp playback is what matters: at t=3.0 we're 1.0s past the
    # warp landing, that's 2 real beats at 120 BPM, displayed-beat
    # advances by 2 * 2.0 (new scrolls=2.0) = 4 displayed-beats from the
    # post-warp anchor of 8. Total displayed-beat = 12, cum = 12 * 0.5 = 6.0.
    assert eng.cumulative_at(3.0) == pytest.approx(6.0, abs=1e-9)


def test_warp_at_beat_zero_produces_atom_only():
    # Warp at beat 0 of length 2 beats. With sm_offset=0, beat 0 = chart-time
    # 0.0; the warp adds an atom of length 2 beats at the start. After the
    # warp, displayed-beat is 2 (consumed by the atom), and chart-time keeps
    # advancing normally.
    eng = beat_space_engine(
        scrolls=[(0.0, 1.0)],
        speeds=[],
        bpms=_CONST_BPM,
        sm_offset=0.0,
        warps=[(0.0, 2.0)],
    )
    # cum_at(0) includes the atom: 2 displayed-beats * 0.5 sec/beat = 1.0.
    assert eng.cumulative_at(0.0) == pytest.approx(1.0, abs=1e-9)
    # cum_at(1.0): 1.0s past the warp = 2 real beats; displayed beat
    # increases by 2*1.0 = 2 -> total 4 displayed-beats, cum = 2.0.
    assert eng.cumulative_at(1.0) == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Stop + warp on the same beat (Etterna same-row precedence)
# ---------------------------------------------------------------------------


def test_stop_and_warp_same_beat_marker_precedence():
    # Same row has STOP=0.25s and WARP=2 beats. The marker fires before
    # the stop; chart-time freezes through the stop, then the warp atom
    # adds its mass. Verify that cumulative_at across the event is
    # continuous + monotone.
    eng = beat_space_engine(
        scrolls=[(0.0, 1.0)],
        speeds=[],
        bpms=_CONST_BPM,
        sm_offset=0.0,
        stops=[(4.0, 0.25)],
        warps=[(4.0, 2.0)],
    )
    samples = np.linspace(0.0, 5.0, 100)
    cums = [eng.cumulative_at(float(t)) for t in samples]
    diffs = np.diff(cums)
    # Monotone non-decreasing within float epsilon.
    assert np.all(diffs >= -1e-12), \
        f"cumulative dipped at index {np.argmin(diffs)}"


# ---------------------------------------------------------------------------
# Multiple BPM changes interspersed with scroll
# ---------------------------------------------------------------------------


def test_bpm_changes_with_scroll_keeps_cumulative_continuous():
    bpms = [(0.0, 120.0), (4.0, 240.0), (12.0, 60.0)]
    eng = beat_space_engine(
        scrolls=[(0.0, 1.0), (8.0, 1.5), (16.0, 0.5)],
        speeds=[], bpms=bpms, sm_offset=0.0,
    )
    samples = np.linspace(0.0, 30.0, 200)
    cums = [eng.cumulative_at(float(t)) for t in samples]
    # No discontinuities: |Delta cum| at consecutive sample pairs is
    # bounded by the local rate * Delta t. With max scrolls=1.5 and max
    # BPM=240 (4 beats/sec), max rate = 4 * 1.5 * 0.5 = 3.0 cum-units/sec.
    # Delta t between samples here is ~0.15s, so |Delta cum| <= 0.45.
    assert np.max(np.abs(np.diff(cums))) <= 0.5
    # And monotone non-decreasing.
    assert np.all(np.diff(cums) >= -1e-12)


# ---------------------------------------------------------------------------
# SPEEDS lerp window
# ---------------------------------------------------------------------------


def test_speeds_lerp_render_multiplier_endpoints():
    # SPEEDS rows: (beat=0, ratio=1.0, delay=0, unit=0) and
    # (beat=4, ratio=2.0, delay=2.0, unit=0) -- a 2-beat lerp from 1x to 2x
    # starting at beat 4 (t=2.0s under 120 BPM, ending at beat 6 = t=3.0s).
    speeds = [(0.0, 1.0, 0.0, 0), (4.0, 2.0, 2.0, 0)]
    eng = beat_space_engine(scrolls=[(0.0, 1.0)], speeds=speeds,
                            bpms=_CONST_BPM, sm_offset=0.0)
    # Pre-lerp window: render_multiplier == 1.0.
    assert eng.render_multiplier_at(1.0) == pytest.approx(1.0, abs=1e-9)
    # Mid-lerp at t=2.5s = beat 5 (1 beat into 2-beat ramp): halfway,
    # so multiplier is (1.0 + 2.0) / 2 = 1.5.
    assert eng.render_multiplier_at(2.5) == pytest.approx(1.5, abs=1e-9)
    # Post-lerp: settled at 2.0.
    assert eng.render_multiplier_at(4.0) == pytest.approx(2.0, abs=1e-9)


def test_speeds_lerp_render_multiplier_is_monotone_in_window():
    speeds = [(0.0, 1.0, 0.0, 0), (4.0, 2.0, 2.0, 0)]
    eng = beat_space_engine(scrolls=[(0.0, 1.0)], speeds=speeds,
                            bpms=_CONST_BPM, sm_offset=0.0)
    samples = np.linspace(2.0, 3.0, 50)   # the lerp window
    mults = np.array([eng.render_multiplier_at(float(t)) for t in samples])
    assert np.all(np.diff(mults) >= -1e-12)
    assert mults[0] == pytest.approx(1.0, abs=1e-9)
    assert mults[-1] == pytest.approx(2.0, abs=1e-9)


def test_distance_with_speeds_active_applies_zoom_at_a():
    # distance(a, b) = (cum(b) - cum(a)) * z(a). Pick a inside the lerp
    # window; verify that distance equals (raw cum diff) * mult_at_a.
    speeds = [(0.0, 1.0, 0.0, 0), (4.0, 2.0, 2.0, 0)]
    eng = beat_space_engine(scrolls=[(0.0, 1.0)], speeds=speeds,
                            bpms=_CONST_BPM, sm_offset=0.0)
    a, b = 2.5, 4.0   # a is mid-lerp (mult=1.5)
    raw = eng.cumulative_at(b) - eng.cumulative_at(a)
    expected = raw * eng.render_multiplier_at(a)
    assert eng.distance(a, b) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Inverse cumulative -- including scroll<=0 plateau
# ---------------------------------------------------------------------------


def test_inverse_lands_on_plateau_in_scroll_zero_region():
    # SCROLLS=0 between beats 4 and 8. Displayed-beat plateaus at the value
    # it had at beat 4 (= t=2.0); inversion is multi-valued over the
    # plateau. The contract is "return *some* chart-time whose cumulative
    # equals the queried sv" -- which point on the plateau is impl-
    # defined (the cache returns the segment-end chart-time).
    eng = beat_space_engine(
        scrolls=[(0.0, 1.0), (4.0, 0.0), (8.0, 1.0)],
        speeds=[], bpms=_CONST_BPM, sm_offset=0.0,
    )
    plateau_cum = eng.cumulative_at(2.0)
    t_back = eng.inverse_cumulative_at(plateau_cum)
    # Must round-trip: cumulative_at(inverse(c)) == c, even if the
    # specific t is somewhere on the plateau [t=2.0, t=4.0].
    assert eng.cumulative_at(t_back) == pytest.approx(
        plateau_cum, abs=1e-9)
    assert 2.0 - 1e-9 <= t_back <= 4.0 + 1e-9


def test_inverse_round_trip_with_no_scrolls():
    # Regression: when a chart has no #SCROLLS at all, the inverse used to
    # short-circuit to `float(sv) / sec_per_base_beat` (the displayed beat),
    # leaking a beat as if it were chart-time. With no #SCROLLS the
    # displayed_beat function is identity, so the inverse must round-trip
    # through beat_to_time.
    eng = beat_space_engine(scrolls=[], speeds=[], bpms=_CONST_BPM,
                            sm_offset=0.0)
    for t in np.linspace(0.5, 30.0, 60):
        cn = eng.cumulative_at(float(t))
        assert eng.inverse_cumulative_at(cn) == pytest.approx(t, abs=1e-9)


def test_inverse_round_trip_strictly_increasing_chart():
    # Sanity: on a chart with monotone-increasing cumulative, inverse is
    # exact and round-trip is the identity.
    eng = beat_space_engine(
        scrolls=[(0.0, 1.0), (4.0, 1.5), (12.0, 0.5)],
        speeds=[], bpms=_CONST_BPM, sm_offset=0.0,
    )
    for t in np.linspace(0.5, 20.0, 50):
        cn = eng.cumulative_at(float(t))
        assert eng.inverse_cumulative_at(cn) == pytest.approx(t, abs=1e-9)


# ---------------------------------------------------------------------------
# Project_times stress test (vectorized path)
# ---------------------------------------------------------------------------


def test_project_beats_matches_displayed_beat_table():
    # Beat-space engines expose project_beats; chart-stream sprites in
    # negative-BPM warp aliases use it to position from chart-beat directly.
    # With no STOPs/WARPs the relationship is straightforward:
    #   project_beats(b) == displayed_beat(b) * sec_per_base_beat.
    # For scrolls=1 throughout, displayed_beat == real beat, sec_per_base
    # = 60/120 = 0.5, so project_beats(b) == 0.5 * b.
    eng = beat_space_engine(scrolls=[(0.0, 1.0)], speeds=[],
                            bpms=_CONST_BPM, sm_offset=0.0)
    beats = np.linspace(0.0, 16.0, 100)
    np.testing.assert_allclose(eng.project_beats(beats), 0.5 * beats,
                                atol=1e-9)


def test_time_space_engine_has_no_project_beats():
    # render.py uses hasattr() to dispatch; time-space engines must not
    # expose project_beats so the time-based fallback fires.
    eng = time_space_engine([(0.0, 1.0), (5.0, 2.0)])
    assert not hasattr(eng, 'project_beats')


def test_project_times_vector_matches_scalar_cumulative():
    eng = beat_space_engine(
        scrolls=[(0.0, 1.0), (12.0, 0.5)],
        speeds=[],
        bpms=[(0.0, 120.0), (8.0, 240.0)],
        sm_offset=0.05,
        stops=[(4.0, 0.3)],
        warps=[(16.0, 4.0)],
    )
    samples = np.linspace(-0.5, 25.0, 500)
    vec = eng.project_times(samples)
    scalar = np.array([eng.cumulative_at(float(t)) for t in samples])
    np.testing.assert_allclose(vec, scalar, atol=1e-9)
