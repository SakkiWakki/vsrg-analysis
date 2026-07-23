"""The Schedule IR: the time algebra the SM actor framework implements.

The engine's tween queue is an exact composition of timed segments, not
a per-frame mechanism: `Actor::UpdateTweening` (openitg Actor.cpp:469)
drains entries with exact arithmetic (`min(timeLeft, dt)` plus a
remainder carried into the next entry), fires an entry's command when
it first becomes head (:484-495), and `Sleep`/`QueueCommand` are
themselves queue entries (:1068, :1074). Frame rate only chooses where
the composition is OBSERVED. `lower()` therefore evaluates a schedule
to its segments once, at compile - the same fold the engine runs per
frame, with time as a closed variable.

Nodes: `Seg` is one queue entry (duration, ease, absolute prop targets,
optional effect fired at entry START). `Seq` is the queue itself.
`Hibernate` is the Update-level prefix sleep (leftover-carry semantics,
Actor.cpp:545-554). `Loop` is the self-requeue fixpoint (the classic
`sleep; queuecommand` re-arm rig), unrolled to the evaluation horizon.
An effect may itself BE a schedule: the engine plays a command's verbs
when its entry begins, and fresh entries append to the queue tail, so a
nested schedule joins the fold at the tail (`Actor::BeginTweening`
appends; the depth-50 overflow guard at :617 bounds recursion).

Lowering emits the same shapes `SegmentTimeline` records: a ramp per
property whose target differs from its start (the emit-on-change rule),
a structural hold for zero-duration writes, and exact fire times for
opaque effects (the residue the sweep still owns).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

_QUEUE_DEPTH_BOUND = 50


@dataclass(frozen=True, slots=True)
class Add:
    """A relative target: resolves to start + delta at fold time (the
    engine's add-onto-dest verbs: addx, addrotationz, ...)."""
    delta: float


@dataclass(frozen=True, slots=True)
class Seg:
    dur: float
    ease: int = 0
    targets: dict = field(default_factory=dict)
    effect: object = None


@dataclass(frozen=True, slots=True)
class Seq:
    parts: tuple

    def __init__(self, *parts):
        object.__setattr__(self, 'parts', tuple(parts))


@dataclass(frozen=True, slots=True)
class Hibernate:
    dur: float


@dataclass(frozen=True, slots=True)
class Loop:
    period: float
    body: object


def sleep(dur: float) -> Seg:
    return Seg(dur=dur)


def command(effect) -> Seg:
    return Seg(dur=0.0, effect=effect)


@dataclass(slots=True)
class Ramp:
    prop: str
    t0: float
    t1: float
    v0: float
    v1: float
    ease: int


@dataclass(slots=True)
class Hold:
    prop: str
    t: float
    v: float


@dataclass(slots=True)
class Fire:
    t: float
    effect: object


@dataclass(slots=True)
class Lowered:
    """One chain's evaluation: emissions in time order, effect fire
    times, the clock and property state where the fold stopped."""
    emissions: list
    fires: list
    end_t: float
    end_state: dict


def lower(schedule, *, t0: float = 0.0, state: dict | None = None,
          until: float | None = None) -> Lowered:
    """Evaluate one actor chain to its emissions - the compile-time run
    of the engine's queue fold. `state` seeds property values (the
    ease-from side of the first segments); `until` bounds evaluation and
    is REQUIRED when the schedule contains a Loop."""
    state = dict(state or {})
    emissions: list = []
    fires: list = []

    t = float(t0)
    work: deque = deque()
    _push_tail(work, schedule, depth=0)
    while work:
        if until is not None and t >= until:
            break
        entry = work.popleft()
        match entry:
            case Hibernate(dur=dur):
                t += max(0.0, dur)
            case Loop(period=period, body=body):
                t = _unroll_loop(entry, t, until, work)
            case Seg():
                t = _run_seg(entry, t, state, emissions, fires, work)

    return Lowered(emissions, fires, t, state)


def to_timelines(lowered: Lowered, rests: dict | None = None) -> dict:
    """Emissions -> one finished SegmentTimeline per property."""
    from analysis.player.render.segment_timeline import SegmentTimeline

    rests = rests or {}
    lanes: dict = {}

    def lane(prop):
        if prop not in lanes:
            lanes[prop] = SegmentTimeline(rest=float(rests.get(prop, 0.0)))
        return lanes[prop]

    for e in lowered.emissions:
        match e:
            case Ramp():
                lane(e.prop).add_ramp(e.t0, e.t1, e.v0, e.v1, e.ease)
            case Hold():
                lane(e.prop).add_hold(e.t, e.v)
    for tl in lanes.values():
        tl.finish()
    return lanes


def _push_tail(work: deque, node, depth: int) -> None:
    if depth > _QUEUE_DEPTH_BOUND:
        raise RecursionError('schedule depth exceeds the engine tween '
                             f'bound ({_QUEUE_DEPTH_BOUND}); infinitely '
                             'recursing command?')
    match node:
        case Seq(parts=parts):
            for part in parts:
                _push_tail(work, part, depth + 1)
        case Seg() | Hibernate() | Loop():
            work.append(node)
        case _:
            raise TypeError(f'not a schedule node: {node!r}')


def _run_seg(seg: Seg, t: float, state: dict,
             emissions: list, fires: list, work: deque) -> float:
    # Entry START: the command fires first (Actor.cpp:484-495), and any
    # entries it queues join at the TAIL, after everything pending.
    if seg.effect is not None:
        if isinstance(seg.effect, (Seg, Seq, Hibernate, Loop)):
            _push_tail(work, seg.effect, depth=1)
        else:
            fires.append(Fire(t, seg.effect))

    end = t + max(0.0, seg.dur)
    for prop, dest in seg.targets.items():
        v0 = state.get(prop)
        if isinstance(dest, Add):
            dest = (0.0 if v0 is None else v0) + dest.delta
        changed = v0 is None or dest != v0
        if changed and seg.dur > 0.0:
            emissions.append(Ramp(prop, t, end, 0.0 if v0 is None else v0,
                                  float(dest), seg.ease))
        elif changed:
            emissions.append(Hold(prop, t, float(dest)))
        state[prop] = float(dest)
    return end


def _unroll_loop(loop: Loop, t: float, until: float | None,
                 work: deque) -> float:
    """Expand one iteration and re-queue the loop behind it: the
    self-requeue rig re-arms itself each pass, so expansion is lazy and
    naturally bounded by `until`."""
    if until is None:
        raise ValueError('lowering a Loop requires an `until` horizon')
    if loop.period <= 0.0:
        raise ValueError('Loop period must be positive')
    if t < until:
        head: deque = deque()
        _push_tail(head, loop.body, depth=1)
        head.append(Hibernate(max(0.0, loop.period - _body_duration(loop.body))))
        head.append(loop)
        work.extendleft(reversed(head))
    return t


def _body_duration(node) -> float:
    match node:
        case Seg(dur=dur):
            return max(0.0, dur)
        case Hibernate(dur=dur):
            return max(0.0, dur)
        case Seq(parts=parts):
            return sum(_body_duration(p) for p in parts)
        case Loop():
            return 0.0
        case _:
            return 0.0
