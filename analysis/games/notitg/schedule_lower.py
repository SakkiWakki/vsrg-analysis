"""Phase 2 of the Schedule IR: mod_actions -> preview lanes at compile.

The classic template's string-payload actions are MESSAGE broadcasts:
each names a `<name>MessageCommand` whose body lives in the parsed
document, in one of the engine's two command syntaxes - classic
`verb,arg;verb,arg` chains (args are Lua expressions evaluated at fire
time) or `%`-prefixed Lua. Both are overwhelmingly constant tween
chains, so instead of waiting for the sweep to reach and execute each
row, this pass lowers the liftable handlers through the Schedule fold
at their exact fire times and evaluates the result into PREVIEW lanes:
whole-chart value timelines available the moment the chart opens.

Handlers fold in FIRE ORDER via a task heap, so per-actor state and
queue-end carry correctly across interleaved actions - including
`queuemessage`, whose broadcast lands as a zero-entry on the sender's
queue and re-enters the heap at its exact resolved time for every
registered receiver.

The preview is the beyond-the-frontier read layer only: behind the
frontier the swept recording stays authoritative, so lowering is free
to be conservative - any verb or argument it cannot resolve makes the
WHOLE handler residue (executed by the sweep exactly as before, just
not previewable). Closure payloads (no recoverable source) are residue
by construction in this phase.

Deliberate approximations, preview-only by design: a mid-handler
`stoptweening` is rejected rather than modeled (the leading-position
idiom is lowered with a queue reset); globals in arguments resolve to
their POST-LOAD values (a global a driver mutates mid-chart makes the
argument wrong in preview, exact after the sweep).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from analysis.games.notitg.lua_api import _ADD_SETTERS, _SCALAR_SETTERS
from analysis.games.notitg.sim.actor import _SIM_TWEEN_EASING, _rest
from analysis.games.notitg.xml_actors import (
    _strip_lua_wrapper, parse_command_string)
from analysis.player.render.expr import ast
from analysis.player.render.expr.parser import parse_body
from analysis.player.render.schedule import Add, Seg, Seq, lower
from analysis.player.render.segment_timeline import SegmentTimeline

_STOP_VERBS = frozenset({'stoptweening', 'finishtweening'})
_MOD_VERBS = frozenset({'ApplyModifiers', 'mod'})
_EXTRA_SETTERS = {'diffusealpha': 'alpha', 'hidden': 'hidden'}
_COLOR_LANES = ('color:0', 'color:1', 'color:2')
_MAX_COMMAND_DEPTH = 8


@dataclass(slots=True)
class PreviewCompile:
    """lanes: rec_id -> {prop: [SegmentTimeline per component]};
    applied: (t, beat, modstring, player) rows in fire order; handler
    and action counts for the compile report."""
    lanes: dict = field(default_factory=dict)
    applied: list = field(default_factory=list)
    registrations: dict = field(default_factory=dict)
    global_sets: dict = field(default_factory=dict)
    seed_pokes: dict = field(default_factory=dict)
    emissions: dict = field(default_factory=dict)
    lifted_handlers: int = 0
    residue_handlers: int = 0
    residue_actions: int = 0


@dataclass(slots=True)
class _Broadcast:
    """A deferred message broadcast payload - distinct from a raw
    handler-body string so a queued named command's body is never
    mistaken for a message name."""
    message: str


class _Scope(dict):
    """The walk's name-resolution scope: actor-global bindings (the
    dict payload, so every `names.get` site reads it directly) plus
    the fold's live registration rows for collection-member indexing
    (`afts[3]` needs membership AT FIRE TIME - collections that are
    empty post-load fill in as registration handlers fold)."""

    __slots__ = ('registrations',)

    def __init__(self, actors, registrations):
        super().__init__(actors)
        self.registrations = registrations


@dataclass(slots=True)
class _DeferredBody:
    """A mod_message closure riding the task heap: its parsed statements
    plus the const upvalues it captured from the enclosing body (`local
    m_bl = 60/150` then `linear(m_bl*32)` inside the closure)."""
    stmts: tuple
    frame: dict


@dataclass(slots=True)
class _Handler:
    by_target: dict = field(default_factory=dict)
    resets_queue: bool = False
    applied: list = field(default_factory=list)
    registrations: list = field(default_factory=list)
    global_sets: list = field(default_factory=list)
    deferred: list = field(default_factory=list)

    def segs(self, target):
        return self.by_target.setdefault(target, [])


def lower_actions(env, player: int = 1, to_beats=None,
                  to_seconds=None) -> PreviewCompile:
    """Lower every liftable staged action into per-actor preview lanes.
    Call after `load_actors` + `prepare_mod_actions`; reads only, the
    env is never advanced or mutated. `to_beats` (seconds -> beat) lets
    queuemessage rebroadcasts resolve their clock; `to_seconds` (beat ->
    seconds) lets `mod_message` deferrals schedule; without them those
    paths are residue."""
    out = PreviewCompile()
    emissions: dict = {}
    state: dict = {}
    queue_end: dict = {}
    names = _Scope({name: rec_id
                    for rec_id, name in env.named_actor_ids().items()},
                   out.registrations)

    tasks: list = []
    order = 0
    for fire_s, beat, payload in env._staged_actions:
        if not isinstance(payload, str):
            out.residue_actions += 1
            continue
        for rec_id, body in env._message_commands.get(payload, ()):
            heapq.heappush(tasks, (fire_s, order, beat, rec_id, body, 0))
            order += 1

    # Load-run bodies (Init/On) already executed into post-load state,
    # so their setters and globals must not re-lower - but the FUTURE
    # they scheduled is invisible to that state: mod_message deferrals
    # and collection registrations are schedule facts only these bodies
    # hold.
    load_s = getattr(env, '_load_seconds', 0.0)
    for rec_id, body in getattr(env, '_load_bodies', ()):
        deferred = _load_schedule_facts(env, names, rec_id, body,
                                        load_s, out)
        order = _push_deferred(env, deferred, tasks, order, 0,
                               to_seconds, to_beats, out)

    while tasks:
        fire_s, _n, beat, rec_id, body, depth = heapq.heappop(tasks)
        handler = _lower_handler(env, names, rec_id, body, beat, fire_s,
                                 depth)
        if handler is None:
            out.residue_handlers += 1
            # Registrations, global writes, deferrals and instant
            # constant setters are facts about WHAT FIRES, not partial
            # pokes - harvest them even from handlers whose other verbs
            # cannot lower, so the residue evaluation knows membership,
            # section flags and data-holder placement regardless.
            deferred = _harvest_side_facts(env, names, rec_id, body,
                                           beat, fire_s, out)
            order = _push_deferred(env, deferred, tasks, order, depth,
                                   to_seconds, to_beats, out)
            continue
        out.lifted_handlers += 1
        fires = _fold_handler(rec_id, handler, fire_s, beat, player,
                              emissions, state, queue_end, out, env)
        order = _queue_broadcasts(env, fires, tasks, order, depth,
                                  to_beats, beat, out)
        order = _push_deferred(env, handler.deferred, tasks, order,
                               depth, to_seconds, to_beats, out)

    out.emissions = emissions
    for rec_id, per_prop in emissions.items():
        out.lanes[rec_id] = _finish_lanes(env, rec_id, per_prop)
    return out


def _queue_broadcasts(env, fires, tasks, order, depth, to_beats,
                      fallback_beat, out) -> int:
    for fire in fires:
        name = fire.effect
        handlers = env._message_commands.get(name, ())
        if not handlers:
            continue
        if to_beats is None or depth >= _MAX_COMMAND_DEPTH:
            out.residue_handlers += len(handlers)
            continue
        beat = to_beats(fire.t)
        for rec_id, body in handlers:
            heapq.heappush(tasks,
                           (fire.t, order, beat, rec_id, body, depth + 1))
            order += 1
    return order


def _push_deferred(env, rows, tasks, order, depth, to_seconds, to_beats,
                   out) -> int:
    """Deferral rows (beat, rec_id, payload, fire_s) onto the task
    heap. mod_message rows carry a fire BEAT (fire_s None, resolved via
    to_seconds); load-queue rows carry exact SECONDS (beat None). A
    string payload broadcasts that message's handlers; a _DeferredBody
    or named-command body string becomes the handler itself."""
    for b, rec_id, payload, fire_s in rows:
        if depth >= _MAX_COMMAND_DEPTH:
            out.residue_handlers += 1
            continue
        if fire_s is None:
            if to_seconds is None:
                out.residue_handlers += 1
                continue
            fire_s = to_seconds(b)
        if b is None:
            b = to_beats(fire_s) if to_beats is not None else 0.0
        if isinstance(payload, _Broadcast):
            handlers = env._message_commands.get(payload.message, ())
            for handler_rec, body in handlers:
                heapq.heappush(tasks, (fire_s, order, b, handler_rec,
                                       body, depth + 1))
                order += 1
        else:
            heapq.heappush(tasks, (fire_s, order, b, rec_id, payload,
                                   depth + 1))
            order += 1
    return order


def _fold_handler(rec_id, handler, fire_s, beat, player,
                  emissions, state, queue_end, out, env) -> list:
    for modstring in handler.applied:
        out.applied.append((fire_s, beat, modstring, player))
    for table_name, member in handler.registrations:
        out.registrations.setdefault(table_name, []).append(
            (fire_s, member))
    for global_name, value in handler.global_sets:
        out.global_sets.setdefault(global_name, []).append(
            (fire_s, value))

    fires: list = []
    for target, segs in handler.by_target.items():
        if not segs:
            continue
        # Engine tail-append: a chain fired while the previous one is
        # still running queues behind it; stoptweening (self-only)
        # clears the queue instead.
        resets = handler.resets_queue and target == rec_id
        start = fire_s if resets \
            else max(fire_s, queue_end.get(target, fire_s))
        seed = state.get(target)
        if seed is None:
            seed = _load_state(env, target)
        lowered = lower(Seq(*segs), t0=start, state=seed)
        state[target] = lowered.end_state
        queue_end[target] = lowered.end_t
        emissions.setdefault(target, []).extend(lowered.emissions)
        fires.extend(lowered.fires)
    return fires


def _load_state(env, rec_id) -> dict:
    actor = env._actors.get(rec_id)
    if actor is None:
        return {}
    seed = {}
    for prop, value in actor._current.items():
        match value:
            case bool():
                seed[prop] = float(value)
            case int() | float():
                seed[prop] = float(value)
            case (r, g, b, *_rest_c) if prop == 'color':
                seed['color:0'], seed['color:1'], seed['color:2'] = \
                    float(r), float(g), float(b)
    return seed


def _finish_lanes(env, rec_id, per_prop_emissions) -> dict:
    actor = env._actors.get(rec_id)
    lanes: dict = {}

    def lane_for(key):
        lane = lanes.get(key)
        if lane is None:
            lane = lanes[key] = SegmentTimeline(rest=_lane_rest(actor, key))
        return lane

    for e in sorted(per_prop_emissions, key=_emission_time):
        match e:
            case (t, key, value):
                lane_for(key).poke(t, value)
            case _ if hasattr(e, 't0'):
                lane_for(e.prop).add_ramp(e.t0, e.t1, e.v0, e.v1, e.ease)
            case _:
                lane_for(e.prop).add_hold(e.t, e.v)
    for lane in lanes.values():
        lane.finish()

    grouped: dict = {}
    for key in sorted(lanes):
        prop, _sep, _idx = key.partition(':')
        grouped.setdefault(prop, []).append(lanes[key])
    return grouped


def _lane_rest(actor, key) -> float:
    prop, _sep, idx = key.partition(':')
    current = None if actor is None else actor._current.get(prop)
    if idx and isinstance(current, tuple) and int(idx) < len(current):
        current = current[int(idx)]
    if isinstance(current, (int, float)):
        return float(current)
    rest = _rest(prop)
    return float(rest) if isinstance(rest, (int, float)) else 0.0


def _emission_time(e) -> float:
    match e:
        case (t, _key, _value):
            return t
        case _ if hasattr(e, 't0'):
            return e.t0
        case _:
            return e.t


def _harvest_side_facts(env, names, rec_id, body, beat, fire_s,
                        out) -> list:
    """Residue handlers still yield FACTS: registrations, global
    writes, mod_message deferrals and instantaneous constant setters
    all run unconditionally at fire time regardless of the verbs around
    them that cannot lower. The tolerant walk skips what it cannot read
    and never enters conditional bodies. Setter facts become seed pokes
    only for targets with no tween verb in the same handler (a tween
    chain reorders; the deferral path lowers those exactly). Returns
    the deferred rows for the task heap."""
    stmts = _handler_stmts(body)
    if stmts is None:
        return []
    frame = dict(body.frame) if isinstance(body, _DeferredBody) else {}
    steps: list = []
    _walk_steps(env, names, rec_id, stmts, beat, fire_s, steps,
                strict=False, frame=frame)

    deferred: list = []
    tweened = {target for target, verb, _v in steps
               if verb == 'sleep' or verb in _SIM_TWEEN_EASING}
    for target, verb, values in steps:
        match verb:
            case '__register__':
                out.registrations.setdefault(values[0], []).append(
                    (fire_s, target))
            case '__setglobal__':
                out.global_sets.setdefault(values[0], []).append(
                    (fire_s, values[1]))
            case '__defer__':
                deferred.append((values[0], target, values[1], None))
            case _ if target not in tweened:
                targets = _setter_targets(verb, values)
                for prop, value in (targets or {}).items():
                    if isinstance(value, float):
                        out.seed_pokes.setdefault(
                            (target, prop), []).append((fire_s, value))
    return deferred


def _load_schedule_facts(env, names, rec_id, body, load_s, out) -> list:
    """The schedule a load body leaves behind: registrations,
    mod_message deferrals, and QUEUED named commands (`sleep,0.02;
    queuecommand,SetMe` - the queue drains past load, so the named body
    replays at load + accumulated tween time). Setter and global facts
    are deliberately dropped - the post-load state is their exact
    result, while a harvested step could pin a value the rest of the
    load pass overwrote. Queue accounting per target goes conservative
    (None) on an unknowable duration or any skipped statement."""
    if isinstance(body, str) and not body.startswith('%'):
        steps = _classic_steps(env, rec_id, body, 0.0, load_s) or []
    else:
        stmts = _handler_stmts(body)
        if stmts is None:
            return []
        steps = []
        _walk_steps(env, names, rec_id, stmts, 0.0, load_s, steps,
                    strict=False, frame={})

    deferred: list = []
    delay: dict = {}
    skipped = False
    for target, verb, values in steps:
        match verb:
            case '__register__':
                out.registrations.setdefault(values[0], []).append(
                    (load_s, target))
            case '__defer__':
                deferred.append((values[0], target, values[1], None))
            case '__skipped__':
                skipped = True
            case _ if verb == 'sleep' or verb in _SIM_TWEEN_EASING:
                dur = values[0] if values else None
                waited = delay.get(target, 0.0)
                delay[target] = waited + dur \
                    if isinstance(dur, float) and waited is not None \
                    else None
            case 'queuecommand':
                name = values[0] if values else None
                queued = env._named_commands.get(target, {}).get(name) \
                    if isinstance(name, str) else None
                waited = delay.get(target, 0.0)
                if queued is not None and waited is not None \
                        and not skipped:
                    deferred.append((None, target, queued,
                                     load_s + waited))
    return deferred


def _handler_stmts(body):
    """A handler body as parsed statements: deferred closures ride the
    task heap pre-parsed, %-Lua strings parse here, classic strings have
    no Lua statement form."""
    if isinstance(body, _DeferredBody):
        return body.stmts
    if isinstance(body, tuple):
        return body
    if not isinstance(body, str) or not body.startswith('%'):
        return None
    try:
        stmts, _diags = parse_body(_strip_lua_wrapper(body))
    except Exception:
        return None
    return stmts


# -- one handler body -> Schedule pieces --------------------------------

def _lower_handler(env, names, rec_id, body, beat, fire_s,
                   depth: int = 0) -> _Handler | None:
    """One message/named-command body -> its per-target Schedules, or
    None when any verb or argument is outside the lowered subset (the
    whole handler is then residue: partial lowering would reorder)."""
    if depth > _MAX_COMMAND_DEPTH \
            or not isinstance(body, (str, tuple, _DeferredBody)):
        return None
    steps = _classic_steps(env, rec_id, body, beat, fire_s) \
        if isinstance(body, str) and not body.startswith('%') \
        else _lua_steps(env, names, rec_id, body, beat, fire_s)
    if steps is None:
        return None

    handler = _Handler()
    for index, (target, verb, values) in enumerate(steps):
        if not _apply_verb(env, names, rec_id, target, verb, values,
                           index, handler, beat, fire_s, depth):
            return None
    return handler


def _lua_steps(env, names, rec_id, body, beat, fire_s):
    """A %-Lua body (or a deferred closure's pre-parsed statements) as
    (target, verb, resolved-args) steps: a flat chain of method calls
    on `self` or on globals bound to actors, plus the statement forms
    that lower as schedule data."""
    stmts = _handler_stmts(body)
    if stmts is None:
        return None
    frame = dict(body.frame) if isinstance(body, _DeferredBody) else {}
    steps: list = []
    if _walk_steps(env, names, rec_id, stmts, beat, fire_s, steps,
                   strict=True, frame=frame):
        return steps
    return None


def _walk_steps(env, names, rec_id, stmts, beat, fire_s, steps,
                strict, frame) -> bool:
    """Statements -> steps. Strict mode fails the whole walk on the
    first statement outside the lowered subset (the lifting contract:
    partial lowering would reorder); tolerant mode skips it and keeps
    walking (the harvest contract: statements that provably execute at
    fire time, so a conditional body is entered only when its condition
    resolves constant). `frame` carries const local bindings forward
    (None entries poison rebound-non-const names)."""
    for stmt in stmts:
        if _step_from(env, names, rec_id, stmt, beat, fire_s, steps,
                      strict, frame):
            continue
        if strict:
            return False
        # A skipped statement could hide queue-time verbs; the marker
        # lets sequential consumers (load queue accounting) go
        # conservative from this point on.
        steps.append((rec_id, '__skipped__', []))
    return True


def _step_from(env, names, rec_id, stmt, beat, fire_s, steps,
               strict, frame) -> bool:
    match stmt:
        case ast.ExprStmt(expr=ast.Method(
                recv=ast.Sym(name='MESSAGEMAN'), name='Broadcast',
                args=(ast.Str(value=message), *_rest))):
            # An immediate broadcast: the message's handlers fire at
            # this handler's own clock (a zero-delay deferral).
            steps.append((rec_id, '__defer__',
                          [beat, _Broadcast(message)]))
            return True
        case ast.ExprStmt(expr=ast.Method(recv=recv, name=verb,
                                          args=args)):
            target = _recv_target(env, names, rec_id, recv, beat, fire_s,
                                  frame)
            if target is None:
                return False
            values = [_const_node(env, a, beat, fire_s, frame)
                      for a in args]
            steps.append((target, verb, values))
            return True
        case ast.ExprStmt(expr=ast.Call(
                fn=ast.Field(base=ast.Sym(name='table'), name='insert'),
                args=(ast.Sym(name=table_name), ast.Sym(name='self')))):
            # A registration: the actor enrolls itself in a driver
            # collection. Pure schedule data - membership becomes a
            # known interval for the residue evaluation.
            steps.append((rec_id, '__register__', [table_name]))
            return True
        case ast.ExprStmt(expr=ast.Call(fn=ast.Sym(name='mod_message'),
                                        args=(beat_node, payload,
                                              *_flags))):
            # The template's deferred scheduler: run a closure (or
            # broadcast a message) when the chart reaches a beat. Pure
            # schedule data - the payload re-enters the task heap at
            # its resolved fire time.
            fire_beat = _const_node(env, beat_node, beat, fire_s, frame)
            if not isinstance(fire_beat, float):
                return False
            match payload:
                case ast.Str(value=message):
                    steps.append((rec_id, '__defer__',
                                  [fire_beat, _Broadcast(message)]))
                case ast.FuncExpr(params=(), body=closure_body):
                    steps.append((rec_id, '__defer__',
                                  [fire_beat, _DeferredBody(
                                      closure_body, dict(frame))]))
                case _:
                    return False
            return True
        case ast.Assign(targets=(ast.Sym(),),
                        values=(ast.Sym(name='self'),)):
            # Load-time naming (`holder = self`): the env already owns
            # the binding, so this is a schedule no-op.
            return True
        case ast.Assign(targets=(ast.Sym(name=global_name),),
                        values=(value_node,)):
            # A handler global write (walk speeds, section flags):
            # becomes a step timeline the residue evaluation reads
            # at its exact fire time.
            value = _const_node(env, value_node, beat, fire_s, frame)
            if value is not None:
                steps.append((rec_id, '__setglobal__',
                              [global_name, value]))
                return True
            # The older template dispatches a literal action table
            # (`mod_actions = {{beat, closure}, ...}`) instead of
            # mod_message calls - the same schedule data. Tolerant-only:
            # a strict walk cannot represent the table binding itself,
            # so the handler goes residue and the harvest re-walk (this
            # branch) extracts the deferrals.
            rows = None if strict else _deferred_table_rows(
                env, value_node, beat, fire_s, frame)
            if rows:
                steps.extend((rec_id, '__defer__', row) for row in rows)
                return True
            return False
        case ast.Local(names=local_names, values=values):
            # Const locals join the frame (the `local m_bl = 60/150`
            # tween-length idiom); a non-const initializer poisons its
            # name so later references refuse instead of resolving to a
            # stale global of the same name.
            resolved = [_const_node(env, v, beat, fire_s, frame)
                        for v in values]
            resolved += [None] * (len(local_names) - len(resolved))
            for name, value in zip(local_names, resolved):
                frame[name] = value
            return strict is False or None not in resolved
        case ast.If(cond=cond, body=body, elifs=elifs, orelse=orelse):
            # A fire-time-const condition makes one branch the
            # unconditional path (the `if hw_flag then` hardware
            # fork) - the same post-load approximation arguments use.
            for branch_cond, branch_body in ((cond, body), *elifs):
                truth = _const_cond(env, branch_cond, beat, fire_s,
                                    frame)
                if truth is None:
                    return False
                if truth:
                    return _walk_steps(env, names, rec_id, branch_body,
                                       beat, fire_s, steps, strict,
                                       frame)
            return _walk_steps(env, names, rec_id, orelse, beat, fire_s,
                               steps, strict, frame)
        case ast.NumericFor(var=var, start=start, stop=stop,
                            step=step_node, body=loop_body):
            return _unroll_for(env, names, rec_id, var, start, stop,
                               step_node, loop_body, beat, fire_s,
                               steps, strict, frame)
        case _:
            return False


_UNROLL_CAP = 256


def _unroll_for(env, names, rec_id, var, start, stop, step_node,
                loop_body, beat, fire_s, steps, strict, frame) -> bool:
    """A literal-bounded numeric for unrolls to its iterations with the
    loop variable substituted (the `for i=1,4 do _G['holder'..i]...`
    placement idiom). A body that rebinds the variable refuses - the
    per-statement substitution cannot model the shadow."""
    lo = _const_node(env, start, beat, fire_s, frame)
    hi = _const_node(env, stop, beat, fire_s, frame)
    inc = 1.0 if step_node is None \
        else _const_node(env, step_node, beat, fire_s, frame)
    if not (isinstance(lo, float) and isinstance(hi, float)
            and isinstance(inc, float)) or inc == 0.0:
        return False
    if any(isinstance(s, (ast.Local, ast.FuncDef)) for s in loop_body):
        return False

    count = int((hi - lo) / inc) + 1 if (hi - lo) * inc >= 0 else 0
    if count > _UNROLL_CAP:
        return False
    ok = True
    for k in range(count):
        unrolled = tuple(_subst_var(s, var, lo + k * inc)
                         for s in loop_body)
        ok = _walk_steps(env, names, rec_id, unrolled, beat, fire_s,
                         steps, strict, frame) and ok
    return ok


def _recv_target(env, names, rec_id, recv, beat, fire_s, frame):
    match recv:
        case ast.Sym(name='self'):
            return rec_id
        case ast.Sym(name=name):
            return names.get(name)
        case ast.Index(base=ast.Sym(name='_G'), key=key_node):
            name = _const_node(env, key_node, beat, fire_s, frame)
            return names.get(name) if isinstance(name, str) else None
        case ast.Index(base=ast.Sym(name=table_name), key=key_node):
            # A collection member (`afts[i+1]:linear(...)` after loop
            # unroll): a const index resolves against the load-built
            # host table, or against fold-time registration rows for
            # collections that fill in mid-chart.
            key = _const_node(env, key_node, beat, fire_s, frame)
            if not isinstance(key, float):
                return None
            return _member_rec(env, names, table_name, int(key), fire_s)
        case _:
            return None


def _member_rec(env, names, table_name: str, index: int, fire_s: float):
    try:
        host = env._host.env
        table = host[table_name] if table_name in host else None
        member = None if table is None else table[index]
        rec = None if member is None else member['__recorder_id']
    except Exception:
        rec = None
    if isinstance(rec, (int, float)):
        return int(rec)

    rows = getattr(names, 'registrations', {}).get(table_name)
    if rows is None:
        return None
    members = [member for t, member in sorted(rows) if t <= fire_s]
    if 1 <= index <= len(members):
        return members[index - 1]
    return None


def _deferred_table_rows(env, node, beat, fire_s, frame):
    """`{{beat, payload [, persistent]}, ...}` action-table rows as
    deferral [fire_beat, payload] pairs - the mod_message signature
    stored as data (the older template's dispatch table). Payloads are
    message-name strings or zero-param closures; anything else refuses
    the WHOLE table (a partial read of a dispatch table would drop
    scheduled work silently), which keeps modstring tables ({beat, len,
    'mods...'} rows) unmatched."""
    match node:
        case ast.Table(array=array, fields=()) if array:
            rows = []
            for entry in array:
                match entry:
                    case ast.Table(array=(beat_node, payload,
                                          *_flags), fields=()):
                        fire_beat = _const_node(env, beat_node, beat,
                                                fire_s, frame)
                        if not isinstance(fire_beat, float):
                            return None
                        match payload:
                            case ast.Str(value=message):
                                rows.append([fire_beat,
                                             _Broadcast(message)])
                            case ast.FuncExpr(params=(), body=body):
                                rows.append([fire_beat, _DeferredBody(
                                    body, dict(frame))])
                            case _:
                                return None
                    case _:
                        return None
            return rows
        case _:
            return None


def _const_cond(env, node, beat, fire_s, frame):
    """A condition's fire-time truth value, or None when unknowable.
    Lua truth: nil and false are falsy, everything else (0 included)
    is truthy."""
    match node:
        case ast.Sym(name=name):
            if name in frame:
                return None if frame[name] is None else True
            try:
                host = env._host.env
                value = host[name] if name in host else None
            except Exception:
                return None
            if value is None or value is False:
                return False
            return True
        case ast.Unary(op='not', operand=inner):
            truth = _const_cond(env, inner, beat, fire_s, frame)
            return None if truth is None else not truth
        case _:
            value = _const_node(env, node, beat, fire_s, frame)
            return True if value is not None else None


def _subst_var(node, var: str, value: float):
    """The loop variable as a literal throughout a statement tree,
    stopping at scopes that rebind it (an inner `for` over the same
    name keeps its own body; its bounds still see the outer value)."""
    match node:
        case ast.Sym(name=name) if name == var:
            return ast.Num(span=node.span, value=float(value))
        case ast.NumericFor(var=inner) if inner == var:
            return ast.NumericFor(
                span=node.span, var=inner,
                start=_subst_var(node.start, var, value),
                stop=_subst_var(node.stop, var, value),
                step=None if node.step is None
                else _subst_var(node.step, var, value),
                body=node.body)
        case ast.FuncExpr(params=params) if var in params:
            return node
        case ast.Node():
            fields = {}
            changed = False
            for name in node.__dataclass_fields__:
                old = getattr(node, name)
                new = _subst_field(old, var, value)
                fields[name] = new
                changed = changed or new is not old
            return type(node)(**fields) if changed else node
        case _:
            return node


def _subst_field(field_value, var, value):
    match field_value:
        case ast.Node():
            return _subst_var(field_value, var, value)
        case tuple():
            return tuple(_subst_field(v, var, value) for v in field_value)
        case _:
            return field_value


def _classic_steps(env, rec_id, body, beat, fire_s):
    """A classic `verb,arg;verb,arg` body as self-targeted steps; args
    are Lua expressions evaluated against the post-load globals plus the
    fire-time clock (`_classic_arg` semantics)."""
    try:
        parsed = parse_command_string(body)
    except Exception:
        return None
    return [(rec_id, verb,
             [_const_arg(env, a, beat, fire_s) for a in args])
            for verb, args in parsed]


def _apply_verb(env, names, rec_id, target, verb, values, index,
                handler, beat, fire_s, depth) -> bool:
    if verb in _STOP_VERBS:
        # Only the self-leading idiom is modeled; a mid-chain stop
        # freezes in-flight values, which the preview cannot pin.
        if index != 0 or target != rec_id:
            return False
        handler.resets_queue = True
        return True
    if verb == 'sleep':
        return _open_tween(handler, target, values, ease_id=0)
    if verb in _SIM_TWEEN_EASING:
        return _open_tween(handler, target, values,
                           ease_id=_SIM_TWEEN_EASING[verb])
    if verb in _MOD_VERBS:
        mods = values[0] if values else None
        if not isinstance(mods, str):
            return False
        handler.applied.append(mods)
        return True
    if verb == 'queuemessage':
        name = values[0] if values else None
        if not isinstance(name, str):
            return False
        handler.segs(target).append(Seg(0.0, effect=name))
        return True
    if verb == '__register__':
        handler.registrations.append((values[0], target))
        return True
    if verb == '__setglobal__':
        handler.global_sets.append((values[0], values[1]))
        return True
    if verb == '__defer__':
        handler.deferred.append((values[0], rec_id, values[1], None))
        return True
    if verb in ('queuecommand', 'playcommand'):
        return _inline_named(env, names, rec_id, target, values, handler,
                             beat, fire_s, depth)
    return _apply_setter(verb, values, handler, target)


def _open_tween(handler, target, values, ease_id) -> bool:
    dur = values[0] if values else None
    if not isinstance(dur, float):
        return False
    handler.segs(target).append(Seg(dur=max(0.0, dur), ease=ease_id))
    return True


def _setter_targets(verb, values):
    if verb == 'diffuse':
        floats = [v for v in values if isinstance(v, float)]
        if len(floats) < 3 or len(floats) != len(values):
            return None
        targets = dict(zip(_COLOR_LANES, floats))
        if len(floats) > 3:
            targets['alpha'] = floats[3]
        return targets

    value = values[0] if values else None
    if not isinstance(value, float):
        return None
    adder = _ADD_SETTERS.get(verb)
    if adder is not None:
        return {adder: Add(value)}
    props = _EXTRA_SETTERS.get(verb) or _SCALAR_SETTERS.get(verb)
    if props is None:
        return None
    return {p: value for p in
            (props if isinstance(props, tuple) else (props,))}


def _apply_setter(verb, values, handler, target) -> bool:
    targets = _setter_targets(verb, values)
    if targets is None:
        return False

    segs = handler.segs(target)
    tail = segs[-1] if segs else None
    if tail is not None and tail.dur > 0.0 and tail.effect is None:
        segs[-1] = Seg(tail.dur, tail.ease, {**tail.targets, **targets})
    else:
        segs.append(Seg(0.0, targets=targets))
    return True


def _inline_named(env, names, rec_id, target, values, handler,
                  beat, fire_s, depth) -> bool:
    name = values[0] if values else None
    if not isinstance(name, str):
        return False
    body = env._named_commands.get(target, {}).get(name)
    if body is None:
        return False
    nested = _lower_handler(env, names, target, body, beat, fire_s,
                            depth + 1)
    if nested is None or nested.resets_queue:
        return False
    for sub_target, segs in nested.by_target.items():
        handler.segs(sub_target).extend(segs)
    handler.applied.extend(nested.applied)
    return True


# -- fire-time constant evaluation ---------------------------------------

def _const_arg(env, arg, beat, fire_s):
    """A classic-command argument resolved at fire time: numbers pass
    through, everything else parses as a Lua expression."""
    if isinstance(arg, (int, float)):
        return float(arg)
    if not isinstance(arg, str):
        return None
    try:
        stmts, _diags = parse_body(f'return ({arg})')
    except Exception:
        return arg
    match stmts:
        case (ast.Return(values=(expr,)),):
            resolved = _const_node(env, expr, beat, fire_s)
            return arg if resolved is None else resolved
        case _:
            return arg


def _const_node(env, node, beat, fire_s, frame=None):
    """The value a node provably evaluates to AT FIRE TIME: clock reads,
    const locals in `frame`, and post-load globals are constants here,
    which is what makes fire-time lowering so much more liftable than
    generic lifting."""
    match node:
        case ast.Num(value=v):
            return float(v)
        case ast.Str(value=s):
            return s
        case ast.Unary(op='-', operand=inner):
            v = _const_node(env, inner, beat, fire_s, frame)
            return -v if isinstance(v, float) else None
        case ast.Sym(name=name):
            if frame is not None and name in frame:
                return frame[name]
            return _global_number(env, name)
        case ast.Method(recv=ast.Sym(name='GAMESTATE'), name=getter):
            match getter:
                case 'GetSongBeat':
                    return float(beat)
                case 'GetSongTime':
                    return float(fire_s)
                case _:
                    return None
        case ast.Binary(op='..', left=left, right=right):
            a = _concat_part(env, left, beat, fire_s, frame)
            b = _concat_part(env, right, beat, fire_s, frame)
            return a + b if a is not None and b is not None else None
        case ast.Binary(op=op, left=left, right=right):
            a = _const_node(env, left, beat, fire_s, frame)
            b = _const_node(env, right, beat, fire_s, frame)
            if isinstance(a, float) and isinstance(b, float):
                return _arith(op, a, b)
            return None
        case _:
            return None


def _global_number(env, name):
    try:
        value = env._host.env[name] if name in env._host.env else None
    except Exception:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _concat_part(env, node, beat, fire_s, frame=None):
    value = _const_node(env, node, beat, fire_s, frame)
    match value:
        case str():
            return value
        case float():
            return str(int(value)) if value.is_integer() else repr(value)
        case _:
            return None


def _arith(op, a, b):
    match op:
        case '+':
            return a + b
        case '-':
            return a - b
        case '*':
            return a * b
        case '/':
            return a / b if b else None
        case '%':
            return a - (a // b) * b if b else None
        case _:
            return None


# -- Phase 3: the Update body's closed-form half as preview lanes --------

def lower_update_body(live) -> tuple[dict, str]:
    """Route the chart's Update body through the frame-IR closed-form
    split (`frame_compile.compile_update`) and bake the flattenable
    pokes into preview lanes: dense tick-grid samples enter the poke
    corridor, so smooth curves collapse to compact ramps. Returns
    ({rec_id: {prop: [lane]}}, report-note); the evaluated windows are
    left on `live.residue_windows` for the sweep."""
    body = live._body
    if not body:
        return {}, ''
    from analysis.games.notitg.frame_compile import compile_update
    from analysis.games.notitg.guard_surface import NotitgGuardSurface

    names_by_id = live.env.named_actor_ids()
    action_lanes = {
        name: live.env._actors[rec_id]._seg_preview
        for rec_id, name in names_by_id.items()
        if rec_id in live.env._actors
        and live.env._actors[rec_id]._seg_preview}
    try:
        plan = compile_update(body, NotitgGuardSurface(live.env),
                              live._to_seconds,
                              preview_lanes=action_lanes)
    except Exception as exc:
        return {}, f'; body preview failed: {exc}'
    live.residue_windows = list(plan.evaluated_windows)

    names = {name: rec_id for rec_id, name in names_by_id.items()}
    by_actor: dict = {}
    for (global_name, prop), kfs in plan.closed_keyframes.items():
        actor_id = names.get(global_name)
        actor = live.env._actors.get(actor_id)
        if actor is None:
            continue
        lane = SegmentTimeline(rest=_lane_rest(actor, prop))
        for kf in sorted(kfs, key=lambda k: k.t):
            lane.poke(kf.t, float(kf.values[0]))
        lane.finish()
        by_actor.setdefault(actor_id, {})[prop] = [lane]

    channels = sum(len(v) for v in by_actor.values())
    note = (f'; body preview: {plan.closed}/{plan.closed + plan.evaluated} '
            f'pokes closed-form -> {channels} channels, '
            f'{len(plan.evaluated_windows)} residue windows')
    return by_actor, note
