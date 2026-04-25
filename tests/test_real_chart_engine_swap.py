"""Engine swap tests on a real Etterna chart with pathological gimmicks.

TSUSURVIVORGAMUSH (JenovaSephiroth) is a 13-old-gimmicks chart used as
the gold-standard regression target: it has BPM=500000 teleport rows,
negative-BPM warp aliases, BPM=10000 instant scrolls, and STOPs of
varying duration. If our engine swap survives this map, it survives.

These tests verify that a chart-load + engine-swap cycle leaves all
cumulative caches consistent with the active engine. A common bug is to
rebuild the primary `_note_sv_cum` array but forget the ghost-note caches
(miss-hold press/release, mines, lifts, fakes), producing a silent
offset where ghost sprites drift relative to live notes.
"""
from pathlib import Path

import numpy as np
import pytest

from analysis.games.etterna.sm_chart import parse_sm

_SM = Path(__file__).parent / 'fixtures' / 'charts' / 'tsusurvivorgamush' \
      / 'tsusurvivorgamush.sm'


@pytest.fixture(scope='module')
def chart_data():
    """Parse the chart once and return (sm_offset, bpms, scrolls, speeds,
    stops, delays, warps) as the engine constructors expect them."""
    parsed = parse_sm(_SM)
    chart = parsed['charts'][0]
    return {
        'sm_offset': parsed['offset'],
        'bpms': parsed['bpms'],
        'scrolls': chart.get('scrolls') or [],
        'speeds': chart.get('speeds') or [],
        'stops': chart.get('stops') or [],
        'delays': chart.get('delays') or [],
        'warps': chart.get('warps') or [],
    }


# ---------------------------------------------------------------------------
# Reference vs measure parity on the real chart
# ---------------------------------------------------------------------------


def test_real_chart_cumulative_parity(chart_data):
    """The new measure-based engine must agree with the reference engine
    across the entire chart's runtime, including all gimmick regions."""
    from analysis.player.sv.engine import BeatSpaceSVEngine
    from analysis.player.sv.measure_engine import beat_space_engine

    ref = BeatSpaceSVEngine(**chart_data)
    new = beat_space_engine(**chart_data)

    # 2000 samples covers every gimmick zone in the chart's runtime.
    samples = np.linspace(0.0, 600.0, 2000)
    ref_cum = ref.project_times(samples)
    new_cum = new.project_times(samples)

    # 1e-6 absolute is fine: the chart has BPM=500000 segments where a
    # tiny relative error blows up to bigger absolute differences.
    np.testing.assert_allclose(new_cum, ref_cum, atol=1e-6, rtol=1e-9,
                                err_msg='cumulative diverges on real chart')


# ---------------------------------------------------------------------------
# Visual continuity invariant
# ---------------------------------------------------------------------------


def test_cross_engine_continuous_at_negative_time():
    """When swapping from beat to time engine in the lead-in (t < 0),
    cumulative_at must agree at the playhead. Without the as_sections
    anchor, charts with non-1 first SCROLLS produce a cumulative sign
    flip across t=0 that visually shifts the playhead on swap."""
    from analysis.player.sv.engine import BeatSpaceSVEngine, TimeSpaceSVEngine

    # Synthetic chart with negative SCROLLS at start.
    beat = BeatSpaceSVEngine(
        scrolls=[(0.0, -1.0), (4.0, 1.0)],
        speeds=[],
        bpms=[(0.0, 120.0)],
        sm_offset=0.0,
    )
    sections = list(beat.as_sections())
    # Apply the same anchor render.py uses for the cross-engine slot.
    if sections and (sections[0][0] > 0.0 or abs(sections[0][1] - 1.0) > 1e-12):
        sections = [(0.0, 1.0)] + sections
    time = TimeSpaceSVEngine(sections)

    # Across the lead-in / boundary, cumulative must agree.
    for t in [-1.0, -0.5, -0.1, -0.001, 0.0, 0.001, 0.5, 1.0, 2.0]:
        assert beat.cumulative_at(t) == pytest.approx(
            time.cumulative_at(t), abs=1e-9), \
            f'cross-engine continuity broke at t={t}'


def test_beat_engine_handles_real_chart_warps_and_stops(chart_data):
    """The chart has 19 warps and 28 stops with no SCROLLS/SPEEDS. The
    cumulative function must be monotone non-decreasing across the entire
    runtime (warps add jumps, stops add plateaus, BPMs scale -- but
    nothing should make C decrease)."""
    from analysis.player.sv.measure_engine import beat_space_engine

    eng = beat_space_engine(**chart_data)
    samples = np.linspace(0.0, 600.0, 5000)
    cum = eng.project_times(samples)
    diffs = np.diff(cum)
    # Allow tiny negative numerical noise but no real regression.
    assert (diffs >= -1e-9).all(), \
        f'cumulative is non-monotone: min diff = {diffs.min()}'


def test_swap_engine_rebuilds_all_dependent_caches(chart_data):
    """If a chart has miss-holds, mines, lifts, or fakes, those have
    pre-cached SV positions on the Player. swap_engine must rebuild ALL
    of them; rebuilding only `_note_sv_cum` leaves the chart-stream
    sprites offset.
    """
    # Build a minimal player surface that mirrors what render.py touches.
    from analysis.player.sv.measure_engine import beat_space_engine
    from analysis.player.sv.render import SvRenderController

    class _Notes:
        ghost_times = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ghost_sv_times = None
        miss_hold_press = np.array([5.0, 6.0], dtype=np.float64)
        miss_hold_release = np.array([5.5, 6.8], dtype=np.float64)
        miss_hold_press_sv = None
        miss_hold_release_sv = None
        miss_hold_max_sv_dur = None
        mine_times = np.array([7.0], dtype=np.float64)
        mine_rows = np.array([336], dtype=np.int64)
        lift_times = np.array([], dtype=np.float64)
        lift_rows = np.array([], dtype=np.int64)
        fake_times = np.array([], dtype=np.float64)
        fake_rows = np.array([], dtype=np.int64)
        mine_sv = None
        lift_sv = None
        fake_sv = None

    class _FakePlayer:
        class _Clock:
            def set_sv_engine(self, e): pass
        class _Hud:
            open_flyout = None
        def __init__(self):
            self.times = np.linspace(0.0, 600.0, 1000)
            self._note_sv_cum = None
            self._render_timeline = None
            self._mode_state = {'linear_ms': {'value': 600.0, 'options': {}}}
            self.scroll_mode = 'linear_ms'
            self.SCROLL_MODE_MS = 'linear_ms'
            self.scroll_speed = 1.0
            self.H = 800
            self.hit_line_y_frac = 0.85
            self.game = 'etterna'
            self.hud = self._Hud()
            self._state_dict = {}
            self._clock = self._Clock()
            self.notes = _Notes()
            self.sv_enabled = False
            self._sv_engine = None
        def _state(self):
            return self._state_dict

    p = _FakePlayer()
    ctrl = SvRenderController(p)
    p.scroll_mode = 'linear_ms'

    replay = {
        '_etterna_scrolls': chart_data['scrolls'],
        '_etterna_speeds': chart_data['speeds'],
        '_etterna_bpms': chart_data['bpms'],
        '_etterna_stops': chart_data['stops'],
        '_etterna_delays': chart_data['delays'],
        '_etterna_warps': chart_data['warps'],
        '_etterna_offset': chart_data['sm_offset'],
    }
    ctrl.init(sv_sections=None, replay=replay)
    p.sv_render = ctrl

    # Build all caches under the native engine.
    ctrl.build_cumulative_sv()
    ctrl.build_ghost_sv_caches()

    note_cum_before = p._note_sv_cum.copy()
    ghost_before = p.notes.ghost_sv_times.copy()
    miss_press_before = p.notes.miss_hold_press_sv.copy()
    mine_before = p.notes.mine_sv.copy()

    # Swap engine.
    from analysis.player.sv.registry import KEY_OSU_TIME
    assert KEY_OSU_TIME in ctrl.available_engine_keys()
    ctrl.swap_engine(KEY_OSU_TIME)

    # After swap, _note_sv_cum is rebuilt. Verify it changed.
    assert not np.array_equal(p._note_sv_cum, note_cum_before), \
        'note cumulative did not rebuild after swap'

    # Now check ghost caches. THESE ARE THE BUG: if swap_engine doesn't
    # rebuild them, they're stale -- still in the old engine's space.
    # Test: they must agree with the new engine's projection of the
    # underlying chart-time arrays.
    new_engine = p._sv_engine

    expected_ghost = new_engine.project_times(p.notes.ghost_times)
    expected_miss_press = new_engine.project_times(p.notes.miss_hold_press)

    assert np.allclose(p.notes.ghost_sv_times, expected_ghost,
                       atol=1e-9), \
        ('ghost_sv_times stale after swap_engine: '
         'still in the old engine cumulative space')
    assert np.allclose(p.notes.miss_hold_press_sv, expected_miss_press,
                       atol=1e-9), \
        'miss_hold_press_sv stale after swap_engine'

    # Mines use project_beats on beat-space engines, project_times on
    # time-space. Either way the result must be in the active engine's
    # space.
    if hasattr(new_engine, 'project_beats'):
        expected_mine = new_engine.project_beats(
            p.notes.mine_rows.astype(np.float64) / 48.0)
    else:
        expected_mine = new_engine.project_times(p.notes.mine_times)
    assert np.allclose(p.notes.mine_sv, expected_mine, atol=1e-9), \
        'mine_sv stale after swap_engine'
