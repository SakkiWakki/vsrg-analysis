"""NotITG live-host guard `Surface`: operand resolution over a Lua env.

The surface backs the guard evaluator/compiler against a running host - clock
symbols read the live beat/time, other names read the shared Lua env, a nil
global is UNRESOLVED (not a fault), `perframe` is live range membership, and
`clock_reader('beat')` binds the compile path's seconds -> beat reader.
"""
import pytest

pytest.importorskip('lupa')

from analysis.games.notitg.guard_surface import NotitgGuardSurface
from analysis.player.render.expr.surface import UNRESOLVED, Surface
from analysis.player.render.lua.host import LuaHost


class _FakeEnv:
    """Minimal engine-host stand-in: the two clock accessors the surface
    reads plus a real Lua host, so lupa-table globals resolve for real."""

    def __init__(self, beat: float, seconds: float):
        self._clock_beat = beat
        self._seconds = seconds
        self._host = LuaHost(dialect='luajit21')

    def _song_time(self) -> float:
        return self._seconds

    def set_global(self, name: str, value) -> None:
        self._host.env[name] = self._host.to_lua(value)


def _surface(beat=100.0, seconds=30.0, to_beat=None):
    return NotitgGuardSurface(_FakeEnv(beat, seconds), to_beat=to_beat)


def test_conforms_to_surface_protocol():
    assert isinstance(_surface(), Surface)


def test_symbol_beat_reads_live_clock():
    assert _surface(beat=137.5).symbol('beat') == 137.5


@pytest.mark.parametrize('name', ['mod_time', 'time', 'curtime'])
def test_symbol_seconds_symbols_read_song_time(name):
    assert _surface(seconds=42.0).symbol(name) == 42.0


def test_symbol_measure_falls_back_to_beat_over_four():
    assert _surface(beat=16.0).symbol('measure') == 4.0


def test_symbol_nil_global_is_unresolved_not_fault():
    assert _surface().symbol('never_set') is UNRESOLVED


def test_symbol_numeric_and_bool_globals_resolve():
    surface = _surface()
    surface._env.set_global('fgcurcommand', 2)
    surface._env.set_global('flag', True)
    assert surface.symbol('fgcurcommand') == 2
    assert surface.symbol('flag') is True


def test_index_over_resolved_lua_table_is_one_indexed():
    surface = _surface()
    surface._env.set_global('e', [10, 40])
    table = surface.symbol('e')
    assert table is not UNRESOLVED
    assert surface.index(table, 1) == 10
    assert surface.index(table, 2) == 40
    assert surface.index(table, 3) is UNRESOLVED       # past the end


def test_index_propagates_unresolved_operands():
    surface = _surface()
    assert surface.index(UNRESOLVED, 1) is UNRESOLVED
    surface._env.set_global('e', [10])
    assert surface.index(surface.symbol('e'), UNRESOLVED) is UNRESOLVED


def test_perframe_two_arg_range_membership():
    surface = _surface(beat=150.0)
    assert surface.call('perframe', [128, 252]) is True
    assert surface.call('perframe', [10, 20]) is False


def test_perframe_one_arg_is_unit_window():
    inside = _surface(beat=50.4)
    outside = _surface(beat=51.0)                       # end is exclusive
    assert inside.call('perframe', [50]) is True
    assert outside.call('perframe', [50]) is False


def test_perframe_with_unresolved_arg_is_unresolved():
    assert _surface().call('perframe', [UNRESOLVED, 20]) is UNRESOLVED


def test_non_perframe_call_is_unresolved_never_executed():
    surface = _surface()
    surface._env.set_global('danger', 1)
    assert surface.call('danger', [1]) is UNRESOLVED
    assert surface.call('SetShaderFlag', ['x']) is UNRESOLVED


def test_clock_reader_beat_returns_working_seconds_to_beat():
    reader = _surface(to_beat=lambda s: s * 2.0).clock_reader('beat')
    assert reader is not None
    assert reader(15.0) == 30.0


@pytest.mark.parametrize('name', ['mod_time', 'time', 'curtime'])
def test_clock_reader_seconds_symbols_are_identity(name):
    reader = _surface().clock_reader(name)
    assert reader is not None
    assert reader(12.5) == 12.5


def test_clock_reader_non_driver_is_none():
    surface = _surface(to_beat=lambda s: s)
    assert surface.clock_reader('measure') is None
    assert surface.clock_reader('fgcurcommand') is None
