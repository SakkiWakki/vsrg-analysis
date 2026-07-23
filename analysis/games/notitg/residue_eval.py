"""Pure lane-backed residue evaluation: the Update body at compile time.

The runtime sweep exists to run what the lift cannot close-form - but on
real charts that residue is a pure function of things the compiler
already holds: collection MEMBERSHIP (registration handlers lowered to
intervals), data-holder trajectories (action-preview lanes), and the
clock. So instead of simulating those windows in a live environment,
this module drives the EXISTING frame_eval interpreter over the body,
tick-by-tick within the residue windows only, against a `LaneSurface`
that resolves every read purely - no lupa, no SimEnvironment, no
recording machinery - and collects pokes as per-channel emissions.

Conservatism is per CHANNEL, not per window: any channel that ever
receives an UNRESOLVED-tainted or unclassifiable write is discarded
(the runtime sweep stays its source of truth); everything else becomes
whole-chart preview lanes at open. `ApplyModifiers` calls with resolved
modstrings become applied-mod rows - the per-frame painter curves.

`perframe(a, b)` guards rewrite to beat-range comparisons before
interpretation (the same idiom recognition the guard-window extraction
relies on); other chart-helper calls resolve UNRESOLVED and taint only
what they touch.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

from analysis.games.notitg.lua_api import _SCALAR_GETTERS
from analysis.games.notitg.schedule_lower import _setter_targets
from analysis.games.notitg.sim.actor import _rest
from analysis.player.render.expr import ast
from analysis.player.render.expr.frame_eval import (
    GlobalStore, Interpreter, UNRESOLVED)
from analysis.player.render.expr.parser import parse_body
from analysis.player.render.schedule import Add

_MOD_VERBS = frozenset({'ApplyModifiers', 'mod'})


@dataclass(slots=True)
class EvaluatedResidue:
    """emissions: (rec_id, prop) -> [(t, value)] in time order; applied:
    (t, beat, modstring, player) painter rows; tainted: channels the
    evaluation could not answer (left to the sweep); ticks run."""
    emissions: dict = field(default_factory=dict)
    applied: list = field(default_factory=list)
    tainted: set = field(default_factory=set)
    ticks: int = 0


class _GameState:
    __slots__ = ()


_GAMESTATE = _GameState()


class _GlobalsView:
    """`_G[name]` dynamic-global indexing (the walking rig's
    `_G['gat_pos'..i]` idiom): indexes resolve exactly like bare
    symbols."""
    __slots__ = ()


_G_VIEW = _GlobalsView()


class _ActorRef:
    __slots__ = ('rec_id',)

    def __init__(self, rec_id: int):
        self.rec_id = rec_id


class _MemberTable:
    """A registration collection: iteration yields the members whose
    registration fired at or before the world's current time."""

    __slots__ = ('rows',)

    def __init__(self, rows):
        self.rows = rows          # [(fire_s, _ActorRef)] time-sorted

    def members_at(self, t: float):
        return [ref for fire_s, ref in self.rows if fire_s <= t]


class _DictStore(GlobalStore):
    """The body's persistent globals plus the sim-owned tick drivers
    (beat / mod_time / curtime), all in one pure dict."""

    __slots__ = ('values',)

    def __init__(self):
        self.values = {}

    def has(self, name: str) -> bool:
        return name in self.values

    def get(self, name: str):
        return self.values.get(name, UNRESOLVED)

    def set(self, name: str, value) -> None:
        self.values[name] = value


class LaneSurface:
    """The pure Surface: actor globals resolve to lane-backed refs,
    registration tables to membership views, getters to value-at-now,
    pokes to emissions. Anything else is UNRESOLVED - taint, never
    guesswork."""

    def __init__(self, names_to_rec, actors, action_lanes, tables, out,
                 host_env=None, global_sets=None):
        self._names = names_to_rec          # actor global -> rec_id
        self._actors = actors               # rec_id -> SimActor (load state)
        self._lanes = action_lanes          # rec_id -> {prop: [lane]}
        self._tables = tables               # table global -> _MemberTable
        self._out = out                     # EvaluatedResidue
        self._host_env = host_env           # post-load globals (scalars)
        # name -> ([t...], [value...]) step timelines from handler
        # global writes; consulted before the post-load fallback.
        self._global_steps = {
            name: ([t for t, _v in sorted(rows)],
                   [v for _t, v in sorted(rows)])
            for name, rows in (global_sets or {}).items()}
        self.now = 0.0
        self.beat = 0.0
        self.player = 1
        self._state: dict = {}              # (rec_id, prop) -> last write

    # -- reads -----------------------------------------------------------

    def symbol(self, name: str):
        if name == 'GAMESTATE':
            return _GAMESTATE
        if name == '_G':
            return _G_VIEW
        table = self._tables.get(name)
        if table is not None:
            return table
        rec_id = self._names.get(name)
        if rec_id is not None:
            return _ActorRef(rec_id)
        return self._host_scalar(name)

    def _host_scalar(self, name: str):
        """A post-load global scalar (gat_splitm and friends): the same
        post-load-values approximation the action lowering documents. A
        global the body itself writes overlays through the store first,
        so this only serves load-time constants."""
        steps = self._global_steps.get(name)
        if steps is not None:
            i = bisect_right(steps[0], self.now) - 1
            if i >= 0:
                return steps[1][i]
        if self._host_env is None:
            return UNRESOLVED
        try:
            value = self._host_env[name] if name in self._host_env else None
        except Exception:
            return UNRESOLVED
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return value
        return UNRESOLVED

    def method(self, recv, verb: str, args):
        if recv is _GAMESTATE:
            match verb:
                case 'GetSongBeat':
                    return self.beat
                case 'GetSongTime':
                    return self.now
                case _:
                    return UNRESOLVED
        if not isinstance(recv, _ActorRef):
            return UNRESOLVED
        prop = _SCALAR_GETTERS.get(verb)
        if prop is None:
            return UNRESOLVED
        return self._value(recv.rec_id, prop)

    def _value(self, rec_id: int, prop: str):
        written = self._state.get((rec_id, prop))
        if written is not None:
            return written
        lanes = self._lanes.get(rec_id, {}).get(prop)
        if lanes and len(lanes) == 1:
            return lanes[0].sample(self.now)
        actor = self._actors.get(rec_id)
        current = None if actor is None else actor._current.get(prop)
        if isinstance(current, (int, float)) \
                and not isinstance(current, bool):
            return float(current)
        rest = _rest(prop)
        return float(rest) if isinstance(rest, (int, float)) else UNRESOLVED

    def index(self, base, key):
        if base is _G_VIEW and isinstance(key, str):
            return self.symbol(key)
        if isinstance(base, _MemberTable) and isinstance(key, (int, float)):
            members = base.members_at(self.now)
            i = int(key)
            if 1 <= i <= len(members):
                return members[i - 1]
        return UNRESOLVED

    def iter_table(self, table):
        if isinstance(table, _MemberTable):
            return ((float(i), ref) for i, ref in
                    enumerate(table.members_at(self.now), start=1))
        return iter(())

    def is_host_table(self, value) -> bool:
        return isinstance(value, _MemberTable)

    def call(self, name: str, args):
        if name == '__lane_len__' and args \
                and isinstance(args[0], _MemberTable):
            return float(len(args[0].members_at(self.now)))
        return UNRESOLVED

    def clock_reader(self, name: str):
        return None

    # -- writes ----------------------------------------------------------

    def poke(self, recv, verb: str, args) -> None:
        if not isinstance(recv, _ActorRef):
            return
        if verb in _MOD_VERBS:
            mods = args[0] if args else None
            if isinstance(mods, str):
                self._out.applied.append(
                    (self.now, self.beat, mods, self.player))
            return

        targets = _setter_targets(verb, [self._poke_value(a) for a in args])
        if targets is None:
            self._taint_unknown(recv.rec_id, verb)
            return
        for prop, value in targets.items():
            channel = (recv.rec_id, prop)
            if isinstance(value, Add):
                base = self._value(recv.rec_id, prop)
                if not isinstance(base, float):
                    self._out.tainted.add(channel)
                    continue
                value = base + value.delta
            if not isinstance(value, float):
                self._out.tainted.add(channel)
                continue
            self._state[channel] = value
            self._out.emissions.setdefault(channel, []).append(
                (self.now, value))

    def _poke_value(self, arg):
        if isinstance(arg, bool) or not isinstance(arg, (int, float)):
            return None if arg is UNRESOLVED else arg
        return float(arg)

    def _taint_unknown(self, rec_id: int, verb: str) -> None:
        # An unclassifiable verb may move any channel of this actor;
        # taint the actor's evaluated channels rather than guess.
        for channel in list(self._out.emissions):
            if channel[0] == rec_id:
                self._out.tainted.add(channel)
        self._out.tainted.add((rec_id, f'verb:{verb}'))

    def set_index(self, base, key, value) -> bool:
        return False


def _rewrite_perframe(node):
    """`perframe(a, b)` -> `beat >= a and beat < b` (the classic
    template's window helper, the idiom the guard extraction already
    keys on). Rewrites recursively through the statement tree."""
    match node:
        case ast.Call(fn=ast.Field(base=ast.Sym(name='table'),
                                   name='getn'), args=(arg,)):
            return ast.Call(span=node.span,
                            fn=ast.Sym(span=node.span, name='__lane_len__'),
                            args=(_rewrite_perframe(arg),))
        case ast.Call(fn=ast.Sym(name='perframe'), args=(lo, hi)):
            beat = ast.Sym(span=node.span, name='beat')
            return ast.Binary(
                span=node.span, op='and',
                left=ast.Binary(span=node.span, op='>=', left=beat,
                                right=_rewrite_perframe(lo)),
                right=ast.Binary(span=node.span, op='<', left=beat,
                                 right=_rewrite_perframe(hi)))
        case ast.Node():
            fields = {}
            changed = False
            for name in node.__dataclass_fields__:
                value = getattr(node, name)
                new = _rewrite_field(value)
                fields[name] = new
                changed = changed or new is not value
            return type(node)(**fields) if changed else node
        case _:
            return node


def _rewrite_field(value):
    match value:
        case ast.Node():
            return _rewrite_perframe(value)
        case tuple():
            return tuple(_rewrite_field(v) for v in value)
        case list():
            return [_rewrite_field(v) for v in value]
        case _:
            return value


def evaluate_residue(live, registrations, player: int = 1,
                     global_sets=None) -> EvaluatedResidue | None:
    """Evaluate the Update body over its residue windows against the
    lane world. `registrations` is schedule_lower's {table: [(t,
    rec_id)]}. Returns None when there is no body or no windows."""
    body = live._body
    windows = getattr(live, 'residue_windows', None)
    if not body or not windows:
        return None
    try:
        stmts, _diags = parse_body(body)
    except Exception:
        return None
    stmts = tuple(_rewrite_perframe(s) for s in stmts)

    env = live.env
    names_to_rec = {name: rec_id
                    for rec_id, name in env.named_actor_ids().items()}
    action_lanes = {rec_id: actor._seg_preview
                    for rec_id, actor in env._actors.items()
                    if actor._seg_preview}
    tables = {name: _MemberTable(sorted(rows))
              for name, rows in registrations.items()}

    out = EvaluatedResidue()
    surface = LaneSurface(names_to_rec, env._actors, action_lanes,
                          tables, out, host_env=env._host.env,
                          global_sets=global_sets)
    surface.player = player
    store = _DictStore()
    interp = Interpreter(surface, store=store)
    # Compile once to nested closures (the CompiledBody pattern): the
    # per-tick cost is a call chain, not an AST re-walk.
    from analysis.player.render.expr.frame_compile_exec import compile_body
    run_compiled = compile_body(stmts, interp)

    to_seconds = live._to_seconds
    step = live._body_step
    load_s = live._load_s
    for lo_beat, hi_beat in windows:
        lo = to_seconds(lo_beat)
        hi = min(to_seconds(hi_beat), live._end_seconds)
        k = max(0, int((lo - load_s) / step))
        t = load_s + k * step
        while t <= hi:
            if t >= lo:
                surface.now = t
                surface.beat = live._to_beats(t)
                store.values['beat'] = surface.beat
                store.values['mod_time'] = t
                try:
                    run_compiled(interp.root)
                except Exception:
                    return None
                out.ticks += 1
            t += step
    return out
