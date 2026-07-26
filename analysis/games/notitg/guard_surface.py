"""NotITG live-host `Surface`: resolve guard operands against the engine host.

The guard evaluator/compiler (`analysis/player/render/expr/`) reads names,
table elements, and calls off a `Surface`. This is NotITG's implementation,
backed by the running engine-loop host (`sim.env.SimEnvironment`): the SAME
live host whose beat/time clocks and Lua globals the chart's Update body
mutates, so a guard reads one source of truth with the sim.

The game-neutral lupa-table machinery (raw-table index/set/iterate/classify,
colon-method dispatch, nil<->UNRESOLVED marshalling) lives in the shared
`LuaHostSurface` base; this class overrides only NotITG vocabulary. Clock
symbols (`beat`, `mod_time`, ...) resolve to the host's live clock values.
Every other name is read from the shared Lua env; a nil global is UNRESOLVED
(an absent operand, never a fault). `perframe(a, b)` resolves to live range
membership. `clock_reader` binds a driver's `seconds -> value` reader for the
compile path - `beat` needs a real inversion, supplied to the constructor as
`to_beat` (built by the integrator, not reconstructed here).
"""
from __future__ import annotations

from typing import Callable

from analysis.games.notitg.lua_api import COMMAND_NAMES, _SCALAR_GETTERS
from analysis.games.notitg.sim import verb_surface
from analysis.player.render.expr.host_surface import LuaHostSurface
from analysis.player.render.expr.surface import UNRESOLVED, Resolution


# Clock symbols that resolve to song-seconds (identity clock readers): these
# ARE the seconds axis, so a `seconds -> value` reader is the identity.
_SECONDS_SYMBOLS = ('mod_time', 'time', 'curtime')

_BEATS_PER_MEASURE = 4.0

# The method verbs that SCHEDULE rather than poke (queuecommand/playcommand),
# from the one authority; a `poke` effect routes these to _actor_command.
_COMMAND_VERBS = frozenset(COMMAND_NAMES)

# Getter verbs the op-stream compiler can resolve to an integer property id and
# stop sending a name for.
#
# Derived from `_SCALAR_GETTERS` - the table `SimActor.read` itself consults -
# and NOT from `verb_surface.GETTERS`, so `actor_prop` answers each verb with
# exactly the property `read` would. Two traps that costs:
#   - `verb_surface.GETTERS` carries a READ_CURRENT / READ_DEST mode, but `read`
#     ignores it and routes every scalar getter through `get()`, the CURRENT
#     value. Filtering on that mode would have dropped GetZoom/GetRotation* for
#     the wrong reason.
#   - `GETTERS` also lists verbs `read` does NOT handle (GetCurrentRotationX,
#     getaux, ...), which `_actor_get` answers with a permissive table. Lowering
#     those would have made GET_PROP invent values the generic path never
#     returns.
# Sorted so the id space is stable across processes.
PROP_GETS: dict = {
    verb: index for index, verb in enumerate(sorted(_SCALAR_GETTERS))}
PROP_GET_TARGETS: tuple = tuple(
    _SCALAR_GETTERS[verb] for verb in sorted(_SCALAR_GETTERS))

# Property NAME -> GET_PROP id, for the settled mirror. Several verbs read the
# same property (GetZoom and GetZoomX are both scale_x), so a property maps to
# every id that reads it: the executor indexes by id, not by property.
# Setter verbs the compiler can resolve to an integer id: the SINGLE-property
# scalar setters and the add-setters. `zoom`/`basezoom`/`align` write two
# properties from one verb and keep crossing - the op carries one id.
# Each entry is (property, is_add); ids are sorted for stability.
_SETTER_VERBS: dict = {}
for _v, _p in sorted(verb_surface.SCALAR_SETTERS.items()):
    if isinstance(_p, str):
        _SETTER_VERBS[_v] = (_p, False)
for _v, _p in sorted(verb_surface.ADD_SETTERS.items()):
    _SETTER_VERBS[_v] = (_p, True)
PROP_SETS: dict = {v: i for i, v in enumerate(sorted(_SETTER_VERBS))}
PROP_SET_TARGETS: tuple = tuple(
    _SETTER_VERBS[v] for v in sorted(_SETTER_VERBS))

PROP_SLOTS: dict = {}
for _verb, _id in PROP_GETS.items():
    PROP_SLOTS.setdefault(PROP_GET_TARGETS[_id], []).append(_id)


class NotitgGuardSurface(LuaHostSurface):
    """`Surface` over a live engine host. Clock symbols read the host's
    current beat/time; other names read the shared Lua env; `perframe`
    resolves live range membership; `to_beat` (seconds -> beat) is supplied
    for the compile path's `beat` reader. Raw-table ops come from the
    `LuaHostSurface` base."""

    def __init__(self, env, to_beat: Callable[[float], float] | None = None):
        self._env = env
        self._to_beat = to_beat
        # Actor identity for callers that can carry an id instead of the
        # recorder table (the op-stream frontier). Bound as instance attributes
        # straight onto the lupa fn and the dict method - no wrapper frame on
        # either, both are per-crossing hot. Advertised ONLY when the env really
        # models actors; over a stub env the base class's None-returning
        # versions stand, which correctly reports "no actors here" rather than
        # half-wiring a fast path.
        host = getattr(env, '_host', None)
        tables = getattr(env, '_tables', None)
        rec_id_fn = host.env['__rec_id'] if host is not None else None
        if rec_id_fn is not None and tables is not None:
            self.actor_id = rec_id_fn
            self.actor_value = tables.get
        # `_G[name]` is a global read wearing a table index. Charts reach for
        # it constantly to build a name at runtime (`_G['P'..pn]`), and every
        # one was a metamethod call into Lua; the host's mirror answers it as a
        # dict lookup. Identity-compared against the sandbox table, so a
        # genuine table that merely holds globals is unaffected.
        self._globals_table = host.env if host is not None else None
        self._globals_read = host.global_value if host is not None else None

    def _beat(self) -> float:
        return float(self._env._clock_beat)

    def _seconds(self) -> float:
        return float(self._env._song_time())

    def _global(self, name: str):
        # Through the host's mirror, not the Lua table: with global writes
        # observed this is a dict lookup rather than a metamethod call, and it
        # is the single hottest read on the frontier.
        return self._env._host.global_value(name)

    def index(self, base: Resolution, key: Resolution) -> Resolution:
        if base is self._globals_table and self._globals_read is not None \
                and isinstance(key, str):
            value = self._globals_read(key)
            return UNRESOLVED if value is None else value
        return super().index(base, key)

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
        return self.actor_method(rec_id, name, args, recv=recv)

    def actor_method(self, actor_id: int, name: str, args: list,
                     recv: Resolution = None) -> Resolution:
        """`method` for a recv ALREADY resolved to an actor id - the entry a
        caller uses when it carries the id rather than the recorder table. The
        singleton branch is the caller's to skip; everything after it lives
        here so the two entries cannot drift.

        `recv` is the recorder table when the caller has it. `GetShader` chains
        the receiver back, so a caller that has only an id must supply one - or
        pass None and get the canonical table."""
        rec_id = actor_id
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
            return recv if recv is not None else self._env._tables.get(rec_id)
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
        rec_id = self._rec_id(recv)
        if rec_id is None:
            # singleton effect, if any
            self._lua_method(recv, name,
                             [a for a in args if a is not UNRESOLVED])
            return
        self.actor_poke(rec_id, name, args)

    def install_prop_mirror(self, on_prop, on_tween) -> None:
        """See `SimEnvironment.install_prop_mirror`."""
        self._env.install_prop_mirror(on_prop, on_tween)

    def mirror_actor(self, actor_id: int) -> None:
        """See `SimEnvironment.mirror_actor`."""
        self._env.mirror_actor(actor_id)

    def observe_global_writes(self, callback) -> None:
        """Register for per-name global-write notifications (see
        `SimEnvironment.observe_global_write`). A caller that caches a symbol
        across ticks uses this instead of guessing when it went stale."""
        self._env.observe_global_write(callback)

    def global_generation(self) -> int:
        """Counter the host bumps whenever code it cannot inspect may have
        rebound a global (a command body, a classic body, a mod-action
        payload). A caller caching a symbol across ticks drops everything when
        this moves - the write itself is invisible from Python, so the only
        sound response to "something ran" is to assume it wrote."""
        return self._env._global_gen

    def actor_prop(self, actor_id: int, prop_id: int):
        """A live actor property read by ID - the entry a caller uses when the
        verb was resolved to a property at COMPILE time. `PROP_GETS` publishes
        the id space; this is its only consumer, so the two cannot drift.

        Answers exactly what `actor_method` would for the source verb, minus the
        name decode and the verb routing. Returns None for an unknown actor, so
        the caller yields UNRESOLVED as the generic path does."""
        actor = self._env._actors.get(actor_id)
        if actor is None:
            return None
        return actor.get(PROP_GET_TARGETS[prop_id])

    def actor_set_prop(self, actor_id: int, set_id: int, value) -> None:
        """A setter whose verb the compiler resolved to a property id.

        Routes to the SAME `_set_scalar` / `_add_dest` the verb dispatch would
        reach, so the recording and tween-tail behaviour are unchanged - only
        the routing is skipped. A `None` value drops, exactly as `_arg_float`
        yielding None does on the verb path."""
        actor = self._env._actors.get(actor_id)
        if actor is None:
            return
        prop, is_add = PROP_SET_TARGETS[set_id]
        self._env._sync(actor_id)
        if is_add:
            actor._add_dest(prop, value)
        else:
            actor._set_scalar(prop, value)

    def actor_poke(self, actor_id: int, name: str, args: list) -> None:
        """`poke` for a recv ALREADY resolved to an actor id. UNRESOLVED args
        are REMOVED rather than nil-substituted, so a hole shifts every later
        positional arg left - engine-visible, and why this is one shared
        implementation rather than two."""
        clean = [a for a in args if a is not UNRESOLVED]
        if name in _COMMAND_VERBS:
            self._env._actor_command(actor_id, name,
                                     clean[0] if clean else None)
        else:
            self._env._actor_poke(actor_id, name, *clean)

    def _rec_id(self, recv: Resolution):
        """The recorder id behind an actor recv (a Lua recorder table), or
        None when `recv` is not a live actor (a singleton, a nil, a number)."""
        if recv is UNRESOLVED or recv is None:
            return None
        return self._env._table_rec_id(recv)

    def iter_table(self, table: Resolution) -> list | None:
        """`(key, value)` pairs for a lupa table a load pass created (`local
        prefix_plr = {}` then `table.insert` in the Update body), so a
        generic-for iterates it. An actor recorder is a Lua table too but is
        NOT a data container to iterate - exclude it (rec_id set). None for a
        non-lupa value (the interpreter's own LuaTable iterates itself)."""
        if self._rec_id(table) is not None:
            return None
        return super().iter_table(table)

    def clock_reader(self, name: str) -> Callable[[float], float] | None:
        match name:
            case 'beat':
                return self._to_beat
            case name if name in _SECONDS_SYMBOLS:
                return lambda seconds: seconds
            case _:
                return None
