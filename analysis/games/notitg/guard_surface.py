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

from analysis.games.notitg.lua_api import COMMAND_NAMES
from analysis.player.render.expr.surface import UNRESOLVED, Resolution


# Clock symbols that resolve to song-seconds (identity clock readers): these
# ARE the seconds axis, so a `seconds -> value` reader is the identity.
_SECONDS_SYMBOLS = ('mod_time', 'time', 'curtime')

_BEATS_PER_MEASURE = 4.0

# The method verbs that SCHEDULE rather than poke (queuecommand/playcommand),
# from the one authority; a `poke` effect routes these to _actor_command.
_COMMAND_VERBS = frozenset(COMMAND_NAMES)


def _is_lua_table(value) -> bool:
    """Duck-typed lupa-table check: a Lua table supports integer indexing
    but is not a Python string/bytes."""
    return hasattr(value, '__getitem__') and not isinstance(
        value, (str, bytes))


def _lua_index(table, key) -> Resolution:
    """Read `table[key]` from a raw lupa table, returning the RAW Lua value
    (a nested table stays a raw table, a function stays callable) so it round-
    trips back into Lua calls without becoming `userdata`. nil/missing ->
    UNRESOLVED."""
    try:
        value = table[key]
    except (KeyError, TypeError):
        return UNRESOLVED
    return UNRESOLVED if value is None else value


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
            # Return the RAW lupa table, not a wrapper: `index` handles a raw
            # table, and - crucially - a raw table round-trips back into a Lua
            # call (`table.insert(t, x)`) as a real table, where a Python
            # wrapper would arrive as `userdata` and fault the builtin.
            return value
        return UNRESOLVED

    def index(self, base: Resolution, key: Resolution) -> Resolution:
        if base is UNRESOLVED or key is UNRESOLVED:
            return UNRESOLVED
        try:
            if _is_lua_table(base):
                # A raw lupa table: string key = field access (`math.sin`),
                # numeric = element. Lua indexes both the same; lupa keeps its
                # own 1-based array keys, so pass the key through unchanged.
                return _lua_index(base, key if isinstance(key, str)
                                  else int(key))
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

    def method(self, recv: Resolution, name: str, args: list) -> Resolution:
        """`recv:name(args)` in VALUE position - a getter read against the live
        actor (`self:GetX()`, `SCREENMAN:GetTopScreen()`... - though singleton
        methods route through `_read_global`, so `recv` here is an actor
        recorder). Routes to the SAME executor entry the Lua bridge uses, so a
        getter read by the interpreter sees exactly what the Lua path would.
        A non-actor recv, or a getter that yields nil, is UNRESOLVED."""
        rec_id = self._rec_id(recv)
        if rec_id is None:
            # Not an actor: an engine SINGLETON method (GAMESTATE:GetSongBeat,
            # SCREENMAN:GetTopScreen) whose Lua table lives in the host env.
            # Route to that table's method - the transition bridge while the
            # load pass still populates singletons as Lua. A returned actor
            # recorder table flows on as an actor recv.
            return self._lua_method(recv, name, args)
        if name == 'GetChild':
            # The TOP SCREEN's GetChild seeds players at their engine start
            # position and registers the screen-child stream (the Lua path's
            # screen-recorder metatable routes here); a plain actor's GetChild
            # resolves an XML/synthetic child. The screen recorder is a real
            # actor (rec_id set), so dispatch on identity, not on rec_id being
            # None.
            arg = args[0] if args else None
            child = (self._env._screen_get_child(arg)
                     if rec_id == self._env._screen_id
                     else self._env._actor_get_child(rec_id, arg))
            return child if child is not None else UNRESOLVED
        if name == 'GetShader':
            # `GetShader()` chains the actor's own recorder back (the Lua
            # metatable returns `self` for this unmodeled verb), so a following
            # `:uniform1f(name, v)` pokes the frag-owning actor's uniform
            # channel. Return the recv unchanged to continue the chain.
            return recv
        value = self._env._actor_get(rec_id, name)
        return UNRESOLVED if value is None else value

    def poke(self, recv: Resolution, name: str, args: list) -> None:
        """`recv:name(args)` in EFFECT position - apply the setter/command to
        the live actor through the executor. `queuecommand`/`playcommand`
        schedule; every other verb is an actor poke (position, rotation,
        ApplyModifiers pass-throughs land on the same sinks the Lua path
        uses). A singleton effect (`GAMESTATE:ApplyGameCommand`) routes to the
        singleton's Lua method. An UNRESOLVED/nil recv is dropped (the engine
        no-ops a poke on a nil actor too)."""
        clean = [a for a in args if a is not UNRESOLVED]
        rec_id = self._rec_id(recv)
        if rec_id is None:
            self._lua_method(recv, name, clean)     # singleton effect, if any
            return
        if name in _COMMAND_VERBS:
            self._env._actor_command(rec_id, name,
                                     clean[0] if clean else None)
        else:
            self._env._actor_poke(rec_id, name, *clean)

    def _lua_method(self, recv, name, args) -> Resolution:
        """Invoke `recv:name(args)` when `recv` is a live Lua TABLE that is not
        one of our actor recorders (an engine singleton). Colon-call
        convention: the receiver is passed as the implicit first arg. A missing
        method, a nil result, or any fault is UNRESOLVED (never a hard fault -
        the interpreter degrades per-call, like the Lua path swallows)."""
        if not _is_lua_table(recv):
            return UNRESOLVED
        try:
            fn = recv[name]
        except (KeyError, TypeError):
            return UNRESOLVED
        if not callable(fn):
            return UNRESOLVED
        try:
            result = fn(recv, *args)
        except Exception:
            return UNRESOLVED
        return UNRESOLVED if result is None else result

    def _rec_id(self, recv: Resolution):
        """The recorder id behind an actor recv (a Lua recorder table), or
        None when `recv` is not a live actor (a singleton, a nil, a number)."""
        if recv is UNRESOLVED or recv is None:
            return None
        return self._env._table_rec_id(recv)

    def clock_reader(self, name: str) -> Callable[[float], float] | None:
        match name:
            case 'beat':
                return self._to_beat
            case name if name in _SECONDS_SYMBOLS:
                return lambda seconds: seconds
            case _:
                return None
