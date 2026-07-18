"""NotITG live-host `Surface`: resolve guard operands against the engine host.

The guard evaluator/compiler (`analysis/player/render/expr/`) reads names,
table elements, and calls off a `Surface`. This is NotITG's implementation,
backed by the running engine-loop host (`sim.env.SimEnvironment`): the SAME
live host whose beat/time clocks and Lua globals the chart's Update body
mutates, so a guard reads one source of truth with the sim.

Clock symbols (`beat`, `mod_time`, ...) resolve to the host's live clock
values. Every other name is read from the shared Lua env; a nil global is
UNRESOLVED (an absent operand, never a fault). `perframe(a, b)` resolves to
live range membership. `clock_reader` binds a driver's `seconds -> value`
reader for the compile path - `beat` needs a real inversion, supplied to the
constructor as `to_beat` (built by the integrator, not reconstructed here).
"""
from __future__ import annotations

from typing import Callable

from analysis.player.render.expr.surface import UNRESOLVED, Resolution


# Clock symbols that resolve to song-seconds (identity clock readers): these
# ARE the seconds axis, so a `seconds -> value` reader is the identity.
_SECONDS_SYMBOLS = ('mod_time', 'time', 'curtime')

_BEATS_PER_MEASURE = 4.0


def _is_lua_table(value) -> bool:
    """Duck-typed lupa-table check: a Lua table supports integer indexing
    but is not a Python string/bytes."""
    return hasattr(value, '__getitem__') and not isinstance(
        value, (str, bytes))


class _LuaIndexable:
    """A 1-indexed read view over a lupa table, so `index` can resolve
    `t[k]` lazily without eagerly converting the whole table. Wraps only the
    Lua table itself; element reads return the raw Lua value (a nested table
    becomes another `_LuaIndexable` when re-indexed)."""

    def __init__(self, table):
        self._table = table

    def at(self, key: int) -> Resolution:
        value = self._table[key]
        if value is None:
            return UNRESOLVED
        if _is_lua_table(value):
            return _LuaIndexable(value)
        return value


class NotitgGuardSurface:
    """`Surface` over a live engine host. Clock symbols read the host's
    current beat/time; other names read the shared Lua env; `perframe`
    resolves live range membership; `to_beat` (seconds -> beat) is supplied
    for the compile path's `beat` reader."""

    def __init__(self, env, to_beat: Callable[[float], float] | None = None):
        self._env = env
        self._to_beat = to_beat

    def _beat(self) -> float:
        return float(self._env._clock_beat)

    def _seconds(self) -> float:
        return float(self._env._song_time())

    def symbol(self, name: str) -> Resolution:
        match name:
            case 'beat':
                return self._beat()
            case 'measure':
                measure = self._read_global('measure')
                if measure is not UNRESOLVED:
                    return measure
                return self._beat() / _BEATS_PER_MEASURE
            case name if name in _SECONDS_SYMBOLS:
                return self._seconds()
            case _:
                return self._read_global(name)

    def _read_global(self, name: str) -> Resolution:
        value = self._env._host.env[name]
        if value is None:
            return UNRESOLVED
        if isinstance(value, bool) or isinstance(value, (int, float)):
            return value
        if _is_lua_table(value):
            return _LuaIndexable(value)
        return UNRESOLVED

    def index(self, base: Resolution, key: Resolution) -> Resolution:
        if base is UNRESOLVED or key is UNRESOLVED:
            return UNRESOLVED
        try:
            if isinstance(base, _LuaIndexable):
                return base.at(int(key))
            if isinstance(base, (list, tuple)):
                return base[int(key) - 1]      # Lua tables are 1-indexed
            if isinstance(base, dict):
                return base.get(key, UNRESOLVED)
        except (IndexError, ValueError, TypeError):
            return UNRESOLVED
        return UNRESOLVED

    def call(self, name: str, args: list) -> Resolution:
        if name != 'perframe' or not args or any(a is UNRESOLVED for a in args):
            return UNRESOLVED
        start = args[0]
        end = args[1] if len(args) > 1 else start + 1.0
        try:
            return start <= self._beat() < end
        except TypeError:
            return UNRESOLVED

    def clock_reader(self, name: str) -> Callable[[float], float] | None:
        match name:
            case 'beat':
                return self._to_beat
            case name if name in _SECONDS_SYMBOLS:
                return lambda seconds: seconds
            case _:
                return None
