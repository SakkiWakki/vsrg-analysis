"""Playfield shake from `.ffx`-style shake events.

Ports fluXis's `Drawable.Shake` extension: four bounce positions
drawn uniformly from [-magnitude, magnitude]^2, visited as five
equal-length eased segments (OutSine into the first bounce, InOutSine
between bounces, InSine back to rest). fluXis rolls the positions
with a live RNG; here each event seeds its own generator so playback
and scrubbing are deterministic.

Magnitude is in fluXis's 1366x768 reference draw space, scaled to the
chart region like the playfield-move effect. fluXis shakes the whole
gameplay screen; this shakes the chart layers and leaves the map
background still, consistent with the other playfield transforms.
"""
from __future__ import annotations

import random
from bisect import bisect_right

from PySide6.QtGui import QTransform

from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.easing import ease

_REF_W = 1366.0
_REF_H = 768.0
_BOUNCES = 4

_EASE_OUT_SINE = 16
_EASE_IN_SINE = 15
_EASE_IN_OUT_SINE = 17


def _waypoints(time_ms, magnitude) -> tuple:
    rng = random.Random(f'{time_ms}:{magnitude}')
    bounces = tuple((rng.uniform(-magnitude, magnitude),
                     rng.uniform(-magnitude, magnitude))
                    for _ in range(_BOUNCES))
    return ((0.0, 0.0),) + bounces + ((0.0, 0.0),)


class ShakeEffect:
    def __init__(self, events):
        shakes = []
        for event in events or []:
            if not isinstance(event, dict):
                continue
            time_ms = float(event.get('time', 0.0))
            duration = max(0.0, float(event.get('duration', 0.0) or 0.0))
            magnitude = float(event.get('magnitude', 10.0) or 0.0)
            if duration <= 0.0 or magnitude == 0.0:
                continue
            shakes.append((time_ms / 1000.0, duration / 1000.0,
                           _waypoints(time_ms, magnitude)))
        self._shakes = sorted(shakes)
        self._starts = [s[0] for s in self._shakes]

    def __bool__(self):
        return bool(self._shakes)

    def _offset(self, t_now):
        idx = bisect_right(self._starts, float(t_now)) - 1
        if idx < 0:
            return None
        start, duration, waypoints = self._shakes[idx]
        progress = (t_now - start) / duration
        if progress >= 1.0:
            return None

        segments = len(waypoints) - 1
        seg = min(segments - 1, int(progress * segments))
        local = progress * segments - seg
        if seg == 0:
            easing = _EASE_OUT_SINE
        elif seg == segments - 1:
            easing = _EASE_IN_SINE
        else:
            easing = _EASE_IN_OUT_SINE
        f = ease(easing, local)

        (x0, y0), (x1, y1) = waypoints[seg], waypoints[seg + 1]
        return (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)

    def at(self, ctx) -> EffectFrame | None:
        offset = self._offset(ctx.t_now)
        if offset is None:
            return None
        _rx, _ry, w, h = ctx.chart_rect
        dx = offset[0] * w / _REF_W
        dy = offset[1] * h / _REF_H
        if dx == 0.0 and dy == 0.0:
            return None
        return EffectFrame(transform=QTransform().translate(dx, dy))
