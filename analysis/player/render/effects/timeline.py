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

from analysis.player.render.easing import ease


@dataclass(frozen=True)
class Keyframe:
    t: float          # seconds
    values: tuple     # target value(s) at t + duration
    duration: float   # seconds
    easing: int


class EventTimeline:
    """Piecewise-eased sampler. Before the first keyframe returns
    `rest`; between keyframes eases from the previous target toward the
    current one over the current keyframe's duration, then holds."""

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
        prev = self._kf[idx - 1].values if idx > 0 else self._rest
        if kf.duration <= 0 or t_now >= kf.t + kf.duration:
            return kf.values
        f = ease(kf.easing, (t_now - kf.t) / kf.duration)
        return tuple(a + (b - a) * f for a, b in zip(prev, kf.values))
