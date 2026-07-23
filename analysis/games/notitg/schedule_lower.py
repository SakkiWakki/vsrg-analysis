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
    emissions: dict = field(default_factory=dict)
    lifted_handlers: int = 0
    residue_handlers: int = 0
    residue_actions: int = 0


@dataclass(slots=True)
class _Handler:
    by_target: dict = field(default_factory=dict)
    resets_queue: bool = False
    applied: list = field(default_factory=list)
    registrations: list = field(default_factory=list)
    global_sets: list = field(default_factory=list)

    def segs(self, target):
        return self.by_target.setdefault(target, [])


def lower_actions(env, player: int = 1, to_beats=None) -> PreviewCompile:
    """Lower every liftable staged action into per-actor preview lanes.
    Call after `load_actors` + `prepare_mod_actions`; reads only, the
    env is never advanced or mutated. `to_beats` (seconds -> beat) lets
    queuemessage rebroadcasts resolve their clock; without it they are
    residue."""
    out = PreviewCompile()
    emissions: dict = {}
    state: dict = {}
    queue_end: dict = {}
    names = {name: rec_id
             for rec_id, name in env.named_actor_ids().items()}

    tasks: list = []
    order = 0
    for fire_s, beat, payload in env._staged_actions:
        if not isinstance(payload, str):
            out.residue_actions += 1
            continue
        for rec_id, body in env._message_commands.get(payload, ()):
            heapq.heappush(tasks, (fire_s, order, beat, rec_id, body, 0))
            order += 1

    while tasks:
        fire_s, _n, beat, rec_id, body, depth = heapq.heappop(tasks)
        handler = _lower_handler(env, names, rec_id, body, beat, fire_s,
                                 depth)
        if handler is None:
            out.residue_handlers += 1
            # Registrations and global writes are facts about WHAT FIRES,
            # not partial pokes - harvest them even from handlers whose
            # verbs cannot lower, so the residue evaluation knows
            # membership and section flags regardless.
            _harvest_side_facts(env, rec_id, body, beat, fire_s, out)
            continue
        out.lifted_handlers += 1
        fires = _fold_handler(rec_id, handler, fire_s, beat, player,
                              emissions, state, queue_end, out, env)
        order = _queue_broadcasts(env, fires, tasks, order, depth,
                                  to_beats, beat, out)

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


def _harvest_side_facts(env, rec_id, body, beat, fire_s, out) -> None:
    if not isinstance(body, str) or not body.startswith('%'):
        return
    try:
        stmts, _diags = parse_body(_strip_lua_wrapper(body))
    except Exception:
        return
    for stmt in stmts:
        match stmt:
            case ast.ExprStmt(expr=ast.Call(
                    fn=ast.Field(base=ast.Sym(name='table'), name='insert'),
                    args=(ast.Sym(name=table_name), ast.Sym(name='self')))):
                out.registrations.setdefault(table_name, []).append(
                    (fire_s, rec_id))
            case ast.Assign(targets=(ast.Sym(name=global_name),),
                            values=(value_node,)):
                value = _const_node(env, value_node, beat, fire_s)
                if value is not None:
                    out.global_sets.setdefault(global_name, []).append(
                        (fire_s, value))


# -- one handler body -> Schedule pieces --------------------------------

def _lower_handler(env, names, rec_id, body, beat, fire_s,
                   depth: int = 0) -> _Handler | None:
    """One message/named-command body -> its per-target Schedules, or
    None when any verb or argument is outside the lowered subset (the
    whole handler is then residue: partial lowering would reorder)."""
    if depth > _MAX_COMMAND_DEPTH or not isinstance(body, str):
        return None
    steps = _lua_steps(env, names, rec_id, body, beat, fire_s) \
        if body.startswith('%') \
        else _classic_steps(env, rec_id, body, beat, fire_s)
    if steps is None:
        return None

    handler = _Handler()
    for index, (target, verb, values) in enumerate(steps):
        if not _apply_verb(env, names, rec_id, target, verb, values,
                           index, handler, beat, fire_s, depth):
            return None
    return handler


def _lua_steps(env, names, rec_id, body, beat, fire_s):
    """A %-Lua body as (target, verb, resolved-args) steps: a flat chain
    of method calls on `self` or on globals bound to actors."""
    try:
        stmts, _diags = parse_body(_strip_lua_wrapper(body))
    except Exception:
        return None

    steps = []
    for stmt in stmts:
        match stmt:
            case ast.ExprStmt(expr=ast.Method(recv=ast.Sym(name=recv),
                                              name=verb, args=args)):
                target = rec_id if recv == 'self' else names.get(recv)
                if target is None:
                    return None
                values = [_const_node(env, a, beat, fire_s) for a in args]
                steps.append((target, verb, values))
            case ast.ExprStmt(expr=ast.Call(
                    fn=ast.Field(base=ast.Sym(name='table'), name='insert'),
                    args=(ast.Sym(name=table_name), ast.Sym(name='self')))):
                # A registration: the actor enrolls itself in a driver
                # collection. Pure schedule data - membership becomes a
                # known interval for the residue evaluation.
                steps.append((rec_id, '__register__', [table_name]))
            case ast.Assign(targets=(ast.Sym(name=global_name),),
                            values=(value_node,)):
                # A handler global write (walk speeds, section flags):
                # becomes a step timeline the residue evaluation reads
                # at its exact fire time.
                value = _const_node(env, value_node, beat, fire_s)
                if value is None:
                    return None
                steps.append((rec_id, '__setglobal__',
                              [global_name, value]))
            case _:
                return None
    return steps


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


def _const_node(env, node, beat, fire_s):
    """The value a node provably evaluates to AT FIRE TIME: clock reads
    and post-load globals are constants here, which is what makes
    fire-time lowering so much more liftable than generic lifting."""
    match node:
        case ast.Num(value=v):
            return float(v)
        case ast.Str(value=s):
            return s
        case ast.Unary(op='-', operand=inner):
            v = _const_node(env, inner, beat, fire_s)
            return -v if isinstance(v, float) else None
        case ast.Sym(name=name):
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
            a = _concat_part(env, left, beat, fire_s)
            b = _concat_part(env, right, beat, fire_s)
            return a + b if a is not None and b is not None else None
        case ast.Binary(op=op, left=left, right=right):
            a = _const_node(env, left, beat, fire_s)
            b = _const_node(env, right, beat, fire_s)
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


def _concat_part(env, node, beat, fire_s):
    value = _const_node(env, node, beat, fire_s)
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
