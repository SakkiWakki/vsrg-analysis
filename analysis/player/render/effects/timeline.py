"""Interpolated event timeline shared by transform effects.

Every fluXis `.ffx` transform stream (playfieldmove/scale/rotate,
camera, ...) is the same shape: timestamped keyframes each easing a
value over `[time, time + duration]`, holding the last value after.
`EventTimeline` samples one such stream; concrete effects map the
sampled value(s) to a `QTransform`.
"""
from __future__ import annotations

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


@dataclass(frozen=True)
class Keyframe:
    t: float          # seconds
    values: tuple     # target value(s) at t + duration
    duration: float   # seconds
    easing: int
    start: tuple | None = None   # ease-from override (fluXis use-start)


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
