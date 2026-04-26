"""Tests for the SV engine registry and hot-swap mechanism.

Focus: the registry round-trips engine selection without losing chart
state, and swap_engine invalidates downstream caches correctly.
"""
import numpy as np
import pytest

from analysis.player.sv.registry import (KEY_ETTERNA_BEAT, KEY_IDENTITY,
                                          KEY_OSU_TIME, SVEngineRegistry)


# ---------------------------------------------------------------------------
# Registry unit tests
# ---------------------------------------------------------------------------


def test_registry_eager_native_is_active_and_built():
    built = []

    def factory():
        built.append('native')
        return 'native_engine'

    reg = SVEngineRegistry()
    reg.register('native', 'Native', factory, native=True, eager=True)

    assert reg.native_key() == 'native'
    assert reg.active_key() == 'native'
    assert reg.active() == 'native_engine'
    assert built == ['native']  # eager: built once at registration


def test_registry_lazy_factory_runs_on_first_access():
    built = []
    reg = SVEngineRegistry()
    reg.register('a', 'A', lambda: built.append('a') or 'engine_a',
                 native=True, eager=True)
    reg.register('b', 'B', lambda: built.append('b') or 'engine_b')

    assert built == ['a']      # b not built yet
    reg.set_active('b')
    assert built == ['a', 'b']
    # Second activation reuses cached instance.
    reg.set_active('a')
    reg.set_active('b')
    assert built == ['a', 'b']


def test_registry_next_key_wraps_around():
    reg = SVEngineRegistry()
    reg.register('x', 'X', lambda: 'X', native=True, eager=True)
    reg.register('y', 'Y', lambda: 'Y')
    reg.register('z', 'Z', lambda: 'Z')

    assert reg.next_key() == 'y'
    reg.set_active('y')
    assert reg.next_key() == 'z'
    reg.set_active('z')
    assert reg.next_key() == 'x'


def test_registry_unknown_key_raises():
    reg = SVEngineRegistry()
    reg.register('x', 'X', lambda: 'X', native=True, eager=True)
    with pytest.raises(KeyError):
        reg.set_active('no_such_key')


# ---------------------------------------------------------------------------
# Integration: SvRenderController.swap_engine
# ---------------------------------------------------------------------------


class _FakePlayer:
    """Minimal Player stand-in for the controller's surface."""

    class _Clock:
        def __init__(self):
            self.engine = None
        def set_sv_engine(self, e):
            self.engine = e

    class _Hud:
        open_flyout = None

    def __init__(self):
        self._sv_engine = None
        self.sv_enabled = False
        self.times = np.linspace(0.0, 10.0, 50)
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

    def _state(self):
        return self._state_dict


def _make_controller_with_etterna_chart():
    """Build an SvRenderController on a fake player loaded with Etterna
    timing data that the registry's native_key path recognizes."""
    from analysis.player.sv.render import SvRenderController
    from analysis.player.sv.replay_doc import SvReplayDoc, KIND_BEAT_SPACE

    p = _FakePlayer()
    ctrl = SvRenderController(p)

    # Chart with a BPM change inside a SCROLLS=1.0 region: time-space and
    # beat-space disagree on cumulative because beat-space's BPM-derived
    # weight `B'(tau) = bpm/60` changes mid-region while time-space treats
    # the SCROLLS multiplier as constant on chart-time.
    replay = {
        'sv': SvReplayDoc(
            engine_kind=KIND_BEAT_SPACE,
            engine_key='etterna_beat',
            scrolls=[(0.0, 1.0), (8.0, 2.0)],
            bpms=[(0.0, 120.0), (4.0, 240.0)],
        ),
    }

    # Skip mode_desc.on_enter by stubbing the lookup result. init() expects
    # scroll_registry.get(p.scroll_mode); use a benign mode that exists.
    p.scroll_mode = 'linear_ms'
    ctrl.init(replay=replay)
    return ctrl, p


def _make_controller_with_osu_chart():
    """Build an SvRenderController on an osu replay (sv_sections + the
    BPM map exposed for the cross-engine beat-space slot)."""
    from analysis.player.sv.render import SvRenderController
    from analysis.player.sv.replay_doc import SvReplayDoc, KIND_TIME_SPACE

    p = _FakePlayer()
    p.game = 'osu'
    ctrl = SvRenderController(p)
    p.scroll_mode = 'linear_ms'
    ctrl.init(replay={
        'sv': SvReplayDoc(
            engine_kind=KIND_TIME_SPACE,
            engine_key='osu_time',
            sections=[(0.0, 1.0), (4.0, 2.0)],
            bpms=[(0.0, 120.0), (16.0, 240.0)],
        ),
    })
    return ctrl, p


def test_osu_chart_offers_cross_engine_beat_space():
    ctrl, p = _make_controller_with_osu_chart()
    keys = ctrl.available_engine_keys()
    assert KEY_OSU_TIME in keys
    assert KEY_ETTERNA_BEAT in keys, \
        'osu charts with a BPM map should offer beat-space as a cross-engine'
    assert KEY_IDENTITY in keys
    assert ctrl.active_engine_key() == KEY_OSU_TIME


def test_osu_chart_swap_to_beat_changes_cumulative():
    ctrl, p = _make_controller_with_osu_chart()
    ctrl.build_cumulative_sv()
    cum_native = p._note_sv_cum.copy()

    ctrl.swap_engine(KEY_ETTERNA_BEAT)
    cum_swapped = p._note_sv_cum

    # Native osu time-space ignores BPM; beat-space integrates BPM into
    # the cumulative. With a BPM change in the chart, the two arrays
    # disagree.
    assert not np.array_equal(cum_swapped, cum_native), \
        'osu time-space and beat-space should produce different cumulative'


def test_controller_init_builds_registry_with_native_etterna():
    ctrl, p = _make_controller_with_etterna_chart()
    reg = ctrl.registry

    assert reg is not None
    assert reg.native_key() == KEY_ETTERNA_BEAT
    assert reg.active_key() == KEY_ETTERNA_BEAT
    # Cross-game engines should be available too.
    keys = reg.keys()
    assert KEY_ETTERNA_BEAT in keys
    assert KEY_OSU_TIME in keys
    assert KEY_IDENTITY in keys


def test_swap_engine_invalidates_cumulative_cache():
    ctrl, p = _make_controller_with_etterna_chart()

    ctrl.build_cumulative_sv()
    cum_native = p._note_sv_cum.copy()

    ctrl.swap_engine(KEY_OSU_TIME)
    cum_swapped = p._note_sv_cum

    # The cumulative array is rebuilt; identity comparison fails (new
    # numpy array) and values differ because the engines disagree on
    # how to integrate the chart.
    assert cum_swapped is not cum_native
    assert not np.array_equal(cum_swapped, cum_native), \
        "engines should produce different cumulative values"


def test_swap_engine_updates_clock_and_timeline():
    ctrl, p = _make_controller_with_etterna_chart()
    initial_engine = p._sv_engine
    initial_clock_engine = p._clock.engine

    assert initial_clock_engine is initial_engine

    ctrl.swap_engine(KEY_OSU_TIME)
    assert p._sv_engine is not initial_engine
    assert p._clock.engine is p._sv_engine
    # Render timeline is freshly constructed for the new engine.
    assert p._render_timeline is not None


def test_swap_engine_idempotent_on_same_key():
    ctrl, p = _make_controller_with_etterna_chart()
    engine_before = p._sv_engine
    ok = ctrl.swap_engine(ctrl.active_engine_key())
    assert ok is True
    assert p._sv_engine is engine_before    # no rebuild on no-op


def test_swap_engine_unknown_key_returns_false():
    ctrl, p = _make_controller_with_etterna_chart()
    assert ctrl.swap_engine('not_a_real_engine') is False


def test_cycle_engine_advances_active_key():
    ctrl, p = _make_controller_with_etterna_chart()
    keys = ctrl.available_engine_keys()
    started = ctrl.active_engine_key()

    seen = [started]
    for _ in range(len(keys)):
        ctrl.cycle_engine()
        seen.append(ctrl.active_engine_key())

    # After len(keys) cycles we're back where we started.
    assert seen[-1] == started
    # All keys appeared exactly once in the cycle (modulo the wrap).
    assert sorted(set(seen[:-1])) == sorted(set(keys))


def test_swap_engine_refreshes_sv_enabled():
    ctrl, p = _make_controller_with_etterna_chart()
    # Native Etterna engine is enabled (chart has SV).
    assert p.sv_enabled is True
    ctrl.swap_engine(KEY_IDENTITY)
    # Identity engine is NOT enabled.
    assert p.sv_enabled is False


def test_swap_engine_under_cmod_updates_saved_state():
    """When CMOD has stashed sv_enabled in p._state()['sv_enabled_saved'],
    swapping the engine should refresh the saved value (so on_exit restores
    the new engine's enabled state, not the stale one)."""
    ctrl, p = _make_controller_with_etterna_chart()
    p._state_dict['sv_enabled_saved'] = True   # simulate CMOD active

    ctrl.swap_engine(KEY_IDENTITY)

    # CMOD-suspended path: visible sv_enabled is left alone (CMOD owns it),
    # but the saved state now reflects the new engine.
    assert p._state_dict['sv_enabled_saved'] is False
