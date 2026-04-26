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


# ---------------------------------------------------------------------------
# QuaverSVEngine: per-group dispatch (Quaver TimingGroups)
# ---------------------------------------------------------------------------


def _two_group_engine():
    """Engine with two streams: `$Default` is identity-like and `SG_FAST`
    runs at multiplier 5 from t=0. Lets us tell which stream a query
    landed on by inspecting the cumulative value."""
    Q = _quaver()
    groups = {
        '$Default': {'sections': [(0.0, 1.0)], 'initial_velocity': 1.0},
        'SG_FAST': {'sections': [(0.0, 5.0)], 'initial_velocity': 1.0},
    }
    return Q([], initial_velocity=1.0, groups=groups)


def test_quaver_groups_default_used_when_groups_arg_missing():
    e = _two_group_engine()
    times = np.array([0.0, 1.0, 2.0])
    # No groups -> default stream (multiplier 1) -> cum equals t for t>=0.
    assert np.allclose(e.project_times(times), times)


def test_quaver_groups_per_note_dispatch_picks_right_stream():
    e = _two_group_engine()
    times = np.array([1.0, 1.0, 2.0])
    groups = ['$Default', 'SG_FAST', 'SG_FAST']
    cum = e.project_times(times, groups=groups)
    # Default at t=1: 1.0 ; SG_FAST at t=1: 5.0 ; SG_FAST at t=2: 10.0.
    assert cum[0] == pytest.approx(1.0)
    assert cum[1] == pytest.approx(5.0)
    assert cum[2] == pytest.approx(10.0)


def test_quaver_groups_unknown_id_falls_back_to_default():
    e = _two_group_engine()
    cum = e.project_times(np.array([2.0]), groups=['SG_NONEXISTENT'])
    # Falls back to default stream's cum at t=2 (multiplier 1).
    assert cum[0] == pytest.approx(2.0)


def test_quaver_groups_single_time_methods_use_default_stream():
    # cumulative_at / cumulative_velocity_at / inverse / render_multiplier
    # are scalar APIs ; they must always read the default stream because
    # the playhead is group-agnostic.
    e = _two_group_engine()
    assert e.cumulative_at(2.0) == pytest.approx(2.0)
    assert e.cumulative_velocity_at(2.0) == pytest.approx(1.0)


def test_quaver_groups_missing_default_id_errors():
    Q = _quaver()
    with pytest.raises(ValueError, match='default'):
        Q([], groups={'SG_0': {'sections': [], 'initial_velocity': 1.0}})


def test_quaver_groups_breakpoints_union_across_streams():
    # Each stream contributes its section boundaries; the engine reports
    # the union so the predictor splits extrapolation correctly across
    # any group's reversal.
    Q = _quaver()
    groups = {
        '$Default': {'sections': [(1.0, 1.0)], 'initial_velocity': 1.0},
        'SG_A':     {'sections': [(2.0, 1.0), (3.0, -1.0)],
                     'initial_velocity': 1.0},
    }
    e = Q([], groups=groups)
    bps = e.breakpoints()
    assert set(bps.tolist()) == {1.0, 2.0, 3.0}


def test_quaver_cumulative_at_groups_buckets_per_group():
    e = _two_group_engine()
    cum = e.cumulative_at_groups(2.0, ['$Default', 'SG_FAST', '$Default'])
    assert cum[0] == pytest.approx(2.0)
    assert cum[1] == pytest.approx(10.0)
    assert cum[2] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# QuaverSVEngine: IsSVNegative (mirrors Quaver's check)
# ---------------------------------------------------------------------------


def test_quaver_is_sv_negative_uses_initial_when_pre_first_section():
    Q = _quaver()
    e = Q([(10.0, 1.0)], initial_velocity=-1.0)
    assert e.is_sv_negative_at(5.0) is True


def test_quaver_is_sv_negative_picks_last_nonzero_multiplier():
    # Section table: 1.0 -> 0.0 -> -2.0. Querying inside the zero
    # segment should report the negative state (matches Quaver: walk back
    # past zero multipliers).
    Q = _quaver()
    e = Q([(0.0, 1.0), (5.0, -2.0), (10.0, 0.0)])
    assert e.is_sv_negative_at(2.5) is False
    assert e.is_sv_negative_at(7.5) is True
    assert e.is_sv_negative_at(15.0) is True  # zero segment falls back to -2


def test_quaver_is_sv_negative_per_group():
    # Default forward, SG_BACK runs negative. Per-group dispatch picks
    # the right stream regardless of which one the playhead clock uses.
    Q = _quaver()
    groups = {
        '$Default': {'sections': [(0.0, 1.0)], 'initial_velocity': 1.0},
        'SG_BACK':  {'sections': [(0.0, -1.0)], 'initial_velocity': 1.0},
    }
    e = Q([], groups=groups)
    assert e.is_sv_negative_at(2.0, group_id='$Default') is False
    assert e.is_sv_negative_at(2.0, group_id='SG_BACK') is True


# ---------------------------------------------------------------------------
# QuaverSVEngine: body_extent (LN body sizing under SV reversal)
# ---------------------------------------------------------------------------


def test_quaver_body_extent_monotonic_is_just_endpoints():
    Q = _quaver()
    e = Q([(0.0, 1.0)])
    lo, hi = e.body_extent(2.0, 5.0)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(5.0)


def test_quaver_body_extent_finds_interior_max_under_reversal():
    # Stream goes +1 from t=0, -1 from t=10. LN spans [5, 15]: head=5,
    # tail=5 (same cum because of the reversal symmetry), interior peak
    # at t=10 -> cum=10. The hull must capture that 10.
    Q = _quaver()
    e = Q([(0.0, 1.0), (10.0, -1.0)])
    lo, hi = e.body_extent(5.0, 15.0)
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(10.0)


def test_quaver_body_extent_handles_double_reversal():
    # +1 [0,5], -2 [5,10], +1 [10,15]. LN [3, 13]:
    #   C(3)=3, C(5)=5, C(10)=5-2*5=-5, C(13)=-5+1*3=-2.
    # Hull = [-5, 5].
    Q = _quaver()
    e = Q([(0.0, 1.0), (5.0, -2.0), (10.0, 1.0)])
    lo, hi = e.body_extent(3.0, 13.0)
    assert lo == pytest.approx(-5.0)
    assert hi == pytest.approx(5.0)


def test_quaver_body_extent_per_group():
    Q = _quaver()
    groups = {
        '$Default': {'sections': [(0.0, 1.0)], 'initial_velocity': 1.0},
        'SG_REV':   {'sections': [(0.0, 1.0), (10.0, -1.0)],
                     'initial_velocity': 1.0},
    }
    e = Q([], groups=groups)
    # Default: monotonic, hull is [5, 15].
    lo_d, hi_d = e.body_extent(5.0, 15.0, group_id='$Default')
    assert lo_d == pytest.approx(5.0)
    assert hi_d == pytest.approx(15.0)
    # SG_REV: hull peaks at t=10 with cum 10.
    lo_r, hi_r = e.body_extent(5.0, 15.0, group_id='SG_REV')
    assert lo_r == pytest.approx(5.0)
    assert hi_r == pytest.approx(10.0)


def test_quaver_body_waypoints_returns_endpoints_and_changes():
    Q = _quaver()
    e = Q([(0.0, 1.0), (10.0, -1.0)])
    head_cum, tail_cum, ts, cs = e.body_waypoints(5.0, 15.0)
    # head_cum: at t=5 cum = 5 ; tail_cum: t=15 with reversal at 10 -> 5.
    assert head_cum == pytest.approx(5.0)
    assert tail_cum == pytest.approx(5.0)
    # One sign change at t=10 (positive -> negative), cum value 10.
    assert ts.tolist() == [10.0]
    assert cs.tolist() == pytest.approx([10.0])


def test_quaver_body_waypoints_empty_when_monotonic():
    Q = _quaver()
    e = Q([(0.0, 1.0)])
    head_cum, tail_cum, ts, cs = e.body_waypoints(2.0, 8.0)
    assert head_cum == pytest.approx(2.0)
    assert tail_cum == pytest.approx(8.0)
    assert ts.size == 0
    assert cs.size == 0


# ---------------------------------------------------------------------------
# QuaverSVEngine: ScrollSpeedFactor (render_multiplier_at)
# ---------------------------------------------------------------------------


def test_quaver_ssf_default_is_one():
    e = _quaver()([(0.0, 1.0)])
    assert e.render_multiplier_at(0.0) == pytest.approx(1.0)
    assert e.render_multiplier_at(100.0) == pytest.approx(1.0)


def test_quaver_ssf_lerps_between_keyframes():
    # SSF: (0s, 1.0) -> (10s, 3.0). Midpoint at t=5s should land on 2.0.
    Q = _quaver()
    groups = {
        '$Default': {
            'sections': [(0.0, 1.0)],
            'initial_velocity': 1.0,
            'ssf': [(0.0, 1.0), (10.0, 3.0)],
        },
    }
    e = Q([], groups=groups)
    assert e.render_multiplier_at(0.0) == pytest.approx(1.0)
    assert e.render_multiplier_at(5.0) == pytest.approx(2.0)
    assert e.render_multiplier_at(10.0) == pytest.approx(3.0)
    # Past the last keyframe, hold flat (matches Quaver's branch where
    # `sfIndex == Count - 1` returns the last multiplier directly).
    assert e.render_multiplier_at(15.0) == pytest.approx(3.0)


def test_quaver_ssf_per_group_via_render_multiplier_at_groups():
    Q = _quaver()
    groups = {
        '$Default': {
            'sections': [(0.0, 1.0)],
            'initial_velocity': 1.0,
            'ssf': [(0.0, 1.0)],
        },
        'SG_ZOOM': {
            'sections': [(0.0, 1.0)],
            'initial_velocity': 1.0,
            'ssf': [(0.0, 4.0)],
        },
    }
    e = Q([], groups=groups)
    out = e.render_multiplier_at_groups(0.0, ['$Default', 'SG_ZOOM',
                                              '$Default'])
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(4.0)
    assert out[2] == pytest.approx(1.0)


def test_quaver_ssf_pre_first_keyframe_clamps_to_first_value():
    Q = _quaver()
    groups = {
        '$Default': {
            'sections': [(0.0, 1.0)],
            'initial_velocity': 1.0,
            'ssf': [(5.0, 2.0), (10.0, 4.0)],
        },
    }
    e = Q([], groups=groups)
    assert e.render_multiplier_at(0.0) == pytest.approx(2.0)
    assert e.render_multiplier_at(5.0) == pytest.approx(2.0)


def test_quaver_engine_enabled_when_only_ssf():
    # A chart with no SV but a non-trivial SSF still needs the engine
    # active so render_multiplier_at_groups is used in batch_time_to_y.
    Q = _quaver()
    groups = {
        '$Default': {'sections': [], 'initial_velocity': 1.0,
                     'ssf': [(0.0, 2.0)]},
    }
    e = Q([], groups=groups)
    assert e.enabled is True


# ---------------------------------------------------------------------------
# .qua chart parser: TimingGroups + SSF round-trip
# ---------------------------------------------------------------------------


def test_qua_parser_extracts_timing_groups():
    from analysis.games.quaver.qua_chart import (parse_qua_file,
                                                  DEFAULT_GROUP_ID)
    import textwrap, tempfile, os
    src = textwrap.dedent("""\
        Mode: Keys4
        BPMDoesNotAffectScrollVelocity: true
        InitialScrollVelocity: 1
        TimingPoints:
        - StartTime: 0
          Bpm: 120
        SliderVelocities: []
        HitObjects:
        - StartTime: 1000
          Lane: 1
          TimingGroup: SG_A
        - StartTime: 1000
          Lane: 2
        TimingGroups:
          SG_A: !ScrollGroup
            InitialScrollVelocity: 2
            ScrollVelocities:
            - StartTime: 0
              Multiplier: 3
            ScrollSpeedFactors:
            - StartTime: 500
              Multiplier: 1.5
        """)
    with tempfile.NamedTemporaryFile(suffix='.qua', mode='w',
                                     delete=False) as f:
        f.write(src)
        path = f.name
    try:
        c = parse_qua_file(path)
    finally:
        os.unlink(path)
    # Default group + SG_A should both be present.
    assert DEFAULT_GROUP_ID in c['groups']
    assert 'SG_A' in c['groups']
    sg_a = c['groups']['SG_A']
    assert sg_a['initial_velocity'] == pytest.approx(2.0)
    assert sg_a['sections'] == [(0.0, 3.0)]
    assert sg_a['ssf'] == [(0.5, 1.5)]
    # Hit objects carry their group id ; the second one defaults.
    by_lane = {h['column']: h['group'] for h in c['hitobjects']}
    assert by_lane[0] == 'SG_A'
    assert by_lane[1] == DEFAULT_GROUP_ID
