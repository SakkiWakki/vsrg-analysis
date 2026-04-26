"""Regression tests for the small in-engine SV implementations.

Identity and Quaver live here as dedicated classes; the time-space and
beat-space integrators are built via `measure_engine` factories and are
covered by `test_measure_engine_hardening.py`.
"""
import numpy as np
import pytest

from analysis.player.sv.engine import IdentitySVEngine


# ---------------------------------------------------------------------------
# Identity engine
# ---------------------------------------------------------------------------

def test_identity_engine_distance():
    e = IdentitySVEngine()
    assert e.distance(1.0, 5.0) == 4.0
    assert e.distance(5.0, 1.0) == -4.0
    assert e.distance(0.0, 0.0) == 0.0


def test_identity_engine_project_is_noop():
    e = IdentitySVEngine()
    arr = np.array([0.0, 1.5, 3.0])
    out = e.project_times(arr)
    assert np.array_equal(out, arr)


def test_identity_engine_enabled_is_false():
    assert IdentitySVEngine().enabled is False


def test_max_visible_is_infinity_for_non_etterna_engines():
    """Time-space and identity engines don't impose a beat-based cap."""
    from analysis.player.sv.engine import IdentitySVEngine
    from analysis.player.sv.measure_engine import time_space_engine
    assert IdentitySVEngine().max_visible_t_from(0.0) == float('inf')
    assert time_space_engine([(0.0, 1.0)]).max_visible_t_from(0.0) == float('inf')


# ---------------------------------------------------------------------------
# QuaverSVEngine
#
# Reference for the formulas tested below:
# Quaver/Shared/Screens/Gameplay/Rulesets/Keys/HitObjects/
#   ScrollGroupControllerKeys.cs::GetPositionFromTime + InitializePositionMarkers
# Position table is built in TrackRounding-100 units; we work in the unscaled
# (multiplier * dt) space here -- matches what _note_sv_cum stores.
# ---------------------------------------------------------------------------


def _quaver():
    from analysis.player.sv.engine import QuaverSVEngine
    return QuaverSVEngine


def test_quaver_empty_with_unit_initial_is_identity_like():
    e = _quaver()([], initial_velocity=1.0)
    assert e.enabled is False  # no-op vs identity
    assert e.cumulative_at(5.0) == pytest.approx(5.0)


def test_quaver_initial_velocity_only_pre_first_section():
    # initial_velocity=2.0; section starts at t=10 with multiplier 1.0.
    # For t<10: cum = t*2. At t=10: cum=20 (matches index==0 branch).
    # For t>10: cum = 20 + (t-10)*1.0.
    e = _quaver()([(10.0, 1.0)], initial_velocity=2.0)
    assert e.cumulative_at(0.0) == pytest.approx(0.0)
    assert e.cumulative_at(5.0) == pytest.approx(10.0)
    assert e.cumulative_at(10.0) == pytest.approx(20.0)
    assert e.cumulative_at(15.0) == pytest.approx(25.0)


def test_quaver_initial_velocity_unused_after_first_section():
    # Once we're past the first section start, initial_velocity must not
    # show up in the integral. Two engines with different initial_velocity
    # but the same sections must agree on cumulative_at after t>=first_t.
    e1 = _quaver()([(0.0, 1.0), (5.0, 2.0)], initial_velocity=1.0)
    e2 = _quaver()([(0.0, 1.0), (5.0, 2.0)], initial_velocity=42.0)
    for t in (0.0, 1.0, 5.0, 10.0):
        assert e1.cumulative_at(t) == pytest.approx(e2.cumulative_at(t))


def test_quaver_negative_multiplier_makes_cumulative_decrease():
    # Negative SV: cumulative goes backward. distance(a,b) signed.
    e = _quaver()([(0.0, 1.0), (10.0, -1.0)], initial_velocity=1.0)
    assert e.cumulative_at(5.0) == pytest.approx(5.0)
    assert e.cumulative_at(10.0) == pytest.approx(10.0)
    assert e.cumulative_at(15.0) == pytest.approx(5.0)
    assert e.distance(5.0, 15.0) == pytest.approx(0.0)


def test_quaver_cumulative_monotonic_flag_is_false():
    # The flag tells callers (culling) to fall back to time-domain bisect.
    assert _quaver()([(0.0, 1.0)]).cumulative_monotonic is False


def test_quaver_nan_multiplier_coerces_to_zero():
    # Quaver: `if (float.IsNaN(multiplier)) multiplier = 0`.
    e = _quaver()([(0.0, 1.0), (5.0, float('nan')), (10.0, 1.0)])
    # 0-5: gain 5; 5-10: NaN->0, no gain; 10+: gain 1/sec
    assert e.cumulative_at(7.5) == pytest.approx(5.0)
    assert e.cumulative_at(10.0) == pytest.approx(5.0)
    assert e.cumulative_at(15.0) == pytest.approx(10.0)


def test_quaver_project_times_matches_cumulative_at():
    e = _quaver()([(0.0, 1.0), (5.0, -2.0), (10.0, 0.5)],
                  initial_velocity=1.0)
    times = np.array([-1.0, 0.0, 2.5, 5.0, 7.5, 10.0, 15.0])
    proj = e.project_times(times)
    for t, p in zip(times, proj):
        assert e.cumulative_at(float(t)) == pytest.approx(float(p))


def test_quaver_breakpoints_are_section_starts():
    e = _quaver()([(1.0, 1.0), (4.0, 2.0), (9.0, -1.0)])
    bps = e.breakpoints()
    assert np.array_equal(bps, np.array([1.0, 4.0, 9.0]))


def test_quaver_distance_equals_cumulative_difference():
    # Holds even with sign flips; distance is a straight signed integral.
    e = _quaver()([(0.0, 2.0), (5.0, -1.0), (12.0, 3.0)])
    for a, b in [(0.0, 5.0), (3.0, 8.0), (10.0, 15.0), (-2.0, 1.0)]:
        assert e.distance(a, b) == pytest.approx(
            e.cumulative_at(b) - e.cumulative_at(a))
