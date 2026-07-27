"""Interpolated event timeline shared by transform effects.

Every fluXis `.ffx` transform stream (playfieldmove/scale/rotate,
camera, ...) is the same shape: timestamped keyframes each easing a
value over `[time, time + duration]`, holding the last value after.
`EventTimeline` samples one such stream; concrete effects map the
sampled value(s) to a `QTransform`.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

from analysis.player.render.effects.easing import ease


def bloom(elapsed, total, in_frac, easing, rest, peak) -> float:
    """One grow-then-shrink cycle: eases `rest` -> `peak` over the first
    `in_frac` of `total`, then `peak` -> `rest` over the rest, holding
    `rest` once `elapsed` reaches `total`. Shared by the pulse/beatpulse
    effects (osu.Framework Then()-chained absolute transforms)."""
    if elapsed < 0.0 or elapsed >= total:
        return rest
    in_dur = total * in_frac
    if elapsed < in_dur:
        u = elapsed / in_dur if in_dur > 0.0 else 1.0
        return rest + (peak - rest) * ease(easing, u)
    out_dur = total - in_dur
    u = (elapsed - in_dur) / out_dur if out_dur > 0.0 else 1.0
    return peak + (rest - peak) * ease(easing, u)


@dataclass(frozen=True, slots=True)
class Keyframe:
    # slots: ~775K are constructed per heavy-chart bake; dropping the per-instance
    # __dict__ speeds allocation + attribute access in simplify/sample. Nothing
    # sets ad-hoc attributes on a Keyframe.
    t: float          # seconds
    values: tuple     # target value(s) at t + duration
    duration: float   # seconds
    easing: int
    start: tuple | None = None   # ease-from override (fluXis use-start)


# Collapsed run points must be reproduced by the EMITTED keyframes to
# within this under EventTimeline's own playback (step-hold for
# instants, eased ramp for tweens). In design pixels / degrees this is
# sub-visible.
SIMPLIFY_EPS = 1e-3

_EASE_LINEAR = 0

# `breakpoints` emission bounds. `_REST_EPS` is how close two consecutive
# holds must be before the second is dropped as redundant; `_TRACE_DT` is
# how finely a CURVED ease is walked, since the consumer's ramp is linear
# and cannot carry the curve itself.
_REST_EPS = 1e-4
_TRACE_DT = 1.0 / 30.0


def simplify_instants(frames):
    """Collapse runs of instant (duration 0) keyframes into the exact
    shape EventTimeline plays back.

    EventTimeline holds an instant's value until the next keyframe - it
    never interpolates between instants - so a dropped point must be
    reproduced by what remains, not by interpolation the sampler does
    not do. A run of collinear per-frame-driver instants therefore
    becomes ONE linear tween easing from the run head's value to the
    last value over the run's span, and a constant run becomes its head
    alone: the transition stays at the run START (a `hidden` flip
    recorded at its true time never migrates to the run's end).

    Only single-value scalar numeric instants join runs; any keyframe
    carrying a tween, a multi-component value, or an ease-from `start`
    override is structural, kept verbatim, and breaks the run. So does
    a same-time pair (a zero-tween chain step)."""
    if len(frames) < 3:
        return frames
    out = []
    i, n = 0, len(frames)
    while i < n:
        head = frames[i]
        if not _plain_instant(head):
            out.append(head)
            i += 1
            continue
        # Grow the run while the chord from `head` to each new point
        # reproduces every interior point to SIMPLIFY_EPS: each accepted
        # point narrows the feasible slope corridor, and a point is
        # accepted only when its own chord slope lies inside it. A point
        # not strictly after its predecessor is a zero-tween chain step
        # (structural), so it breaks the run wherever it appears - not
        # only at the head, or a duplicate write at the run tail would
        # count as a third sample and upgrade a two-point step pair into
        # a ramp bridging the whole gap.
        j = i + 1
        lo, hi = float('-inf'), float('inf')
        while j < n and _plain_instant(frames[j]):
            if frames[j].t <= frames[j - 1].t:
                break
            dt = frames[j].t - head.t
            slope = (frames[j].values[0] - head.values[0]) / dt
            if not lo <= slope <= hi:
                break
            tol = SIMPLIFY_EPS / dt
            lo, hi = max(lo, slope - tol), min(hi, slope + tol)
            j += 1
        out.extend(_collapse_run(frames[i:j]))
        i = j
    return out


def _plain_instant(kf) -> bool:
    return (kf.duration <= 0.0 and kf.start is None and len(kf.values) == 1
            and isinstance(kf.values[0], (int, float)))


def _collapse_run(run):
    if len(run) < 3:
        return run
    head, last = run[0], run[-1]
    if abs(last.values[0] - head.values[0]) <= SIMPLIFY_EPS:
        return [head]
    return [Keyframe(head.t, last.values, last.t - head.t, _EASE_LINEAR,
                     start=head.values)]


def keyframes_from_events(events, value_keys, rest) -> list:
    """Keyframes from `.ffx`-shaped event dicts: each event carries
    ms-keyed `time`/`duration`, an `ease` id, and one value per key in
    `value_keys` (missing values fall back to `rest`)."""
    out = []
    for e in events or []:
        if not isinstance(e, dict):
            continue
        values = tuple(float(e.get(k, r))
                       for k, r in zip(value_keys, rest))
        out.append(Keyframe(
            t=float(e.get('time', 0.0)) / 1000.0,
            values=values,
            duration=max(0.0, float(e.get('duration', 0.0))) / 1000.0,
            easing=int(e.get('ease', 0)),
        ))
    return out


class EventTimeline:
    """Piecewise-eased sampler. Before the first keyframe returns
    `rest`; between keyframes eases from the previous target (or the
    keyframe's own `start` override) toward the current one over the
    current keyframe's duration, then holds."""

    def __init__(self, keyframes, rest):
        self._kf = sorted(keyframes, key=lambda k: k.t)
        self._starts = [k.t for k in self._kf]
        self._rest = tuple(rest)

    def __bool__(self):
        return bool(self._kf)

    def is_static(self) -> bool:
        """True when `sample` provably returns `rest` at every t, so an
        exporter can skip discovering the shape (mirrors
        `seg_read.SegCurve.is_static`).

        Almost always this is a stream with no keyframes at all - a property
        the chart never wrote. Without the probe, a composite over two of
        them (a fit rect's edges, a fill's absolute size versus its natural
        w/h) has to assume it moves and dense-samples the whole chart to
        rediscover one constant: 806 of gat 2's 845 dense exports were that.

        A keyframe that only restates the rest still holds it, ease and all -
        but only if it does not ease FROM somewhere else, which a `start`
        override can do."""
        return all(kf.values == self._rest
                   and (kf.start is None or kf.start == self._rest)
                   for kf in self._kf)

    def sample(self, t_now: float) -> tuple:
        idx = bisect_right(self._starts, float(t_now)) - 1
        if idx < 0:
            return self._rest
        kf = self._kf[idx]
        if kf.duration <= 0 or t_now >= kf.t + kf.duration:
            return kf.values
        if kf.start is not None:
            prev = kf.start
        else:
            prev = self._kf[idx - 1].values if idx > 0 else self._rest
        f = ease(kf.easing, (t_now - kf.t) / kf.duration)
        return tuple(a + (b - a) * f for a, b in zip(prev, kf.values))

    def breakpoints(self, t0: float, t1: float, index: int = 0):
        """`(ts, vals, durs, eases)` reproducing `sample(t)[index]` for a
        piecewise linear-ramp consumer: one that serves this timeline's rest
        before `ts[0]`, then at breakpoint `i` holds `vals[i]` when
        `durs[i] <= 0` and otherwise ramps `vals[i] -> vals[i+1]` over
        `durs[i]`.

        EXACT, and independent of `[t0, t1]` - the keyframes ARE the shape,
        so the whole stream is translated whatever window is asked for. An
        instant becomes one hold; a linearly-eased ramp becomes its two
        endpoints; a CURVED ease is traced at `_TRACE_DT`, because one linear
        ramp cannot carry a curve. Rest rides the consumer's channel, so no
        pre-roll breakpoint is emitted.

        Examples: the storyboard doc exporters call this through
        `export_channel`, and `field_compose._SumTimeline` calls it to learn
        where a link changes before re-reading the sum there.
        """
        ts: list[float] = []
        vals: list[float] = []
        durs: list[float] = []

        def emit(bt: float, value: float, dur: float) -> None:
            # Collapse a redundant hold onto the previous breakpoint of equal
            # value (keeps the channel minimal; the sampler is unaffected).
            if ts and durs[-1] <= 0.0 and abs(vals[-1] - value) <= _REST_EPS \
                    and dur <= 0.0:
                return
            ts.append(bt)
            vals.append(value)
            durs.append(dur)

        prev = float(self._rest[index])
        for kf in self._kf:
            target = float(kf.values[index])
            if kf.duration <= 0.0:
                emit(kf.t, target, 0.0)
            elif kf.easing == _EASE_LINEAR:
                start = float(kf.start[index]) if kf.start is not None else prev
                emit(kf.t, start, kf.duration)
                emit(kf.t + kf.duration, target, 0.0)
            else:
                self._trace(index, kf.t, kf.t + kf.duration, ts, vals, durs)
                emit(kf.t + kf.duration, target, 0.0)
            prev = target
        return ts, vals, durs, [_EASE_LINEAR] * len(ts)

    def _trace(self, index, a: float, b: float, ts, vals, durs) -> None:
        """Append linear-ramp breakpoints following this timeline across
        `[a, b)` at `_TRACE_DT`, so the reconstruction follows the curve."""
        n = max(1, int(math.ceil((b - a) / _TRACE_DT)))
        step = (b - a) / n
        for k in range(n):
            bt = a + k * step
            ts.append(bt)
            vals.append(float(self.sample(bt)[index]))
            durs.append(step)
