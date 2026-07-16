"""Rhythmic playfield zoom from `.ffx` beatpulse events.

Ports fluXis's BeatPulseManager: from each event's time until the next
event (or the chart's end), it emits one zoom pulse per beat. A beat's
length is `timing_point.MsPerBeat * clamp(interval, 0.01, 4)`; within it
the playfield scales 1 -> `strength` over `zoom` of the beat (OutQuint),
then `strength` -> 1 over the rest (OutQuint). Events with strength ~= 1
or interval < 0.01 emit nothing, matching the manager's skip.

fluXis scales `keybindContainer` -- the whole playfield -- about its
center. We zoom about the chart-region center, the convention the other
playfield transforms use, so the pulse composes cleanly with them.
"""
from __future__ import annotations

from bisect import bisect_right

from PySide6.QtGui import QTransform

from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.timeline import bloom

_EASE_OUT_QUINT = 13
_INTERVAL_MIN = 0.01
_INTERVAL_MAX = 4.0
_STRENGTH_EPS = 0.0001


def _ms_per_beat(timing_points, t_ms) -> float:
    """MapInfo.GetTimingPoint(t).MsPerBeat: last point at or before `t`,
    falling back to the first point (60 BPM when none exist)."""
    if not timing_points:
        return 60000.0 / 60.0
    bpm = timing_points[0][1]
    for point_ms, point_bpm in timing_points:
        if point_ms > t_ms:
            break
        bpm = point_bpm
    return 60000.0 / bpm if bpm else 60000.0 / 120.0


def _beats(event, end_ms, timing_points):
    strength = float(event.get('strength', 1.05))
    interval = float(event.get('interval', 1.0))
    zoom = float(event.get('zoom', 0.25))
    if abs(strength - 1.0) < _STRENGTH_EPS or interval < _INTERVAL_MIN:
        return
    clamped = max(_INTERVAL_MIN, min(_INTERVAL_MAX, interval))
    t = float(event.get('time', 0.0))
    while t < end_ms:
        ms = _ms_per_beat(timing_points, t) * clamped
        yield (t / 1000.0, ms / 1000.0, strength, zoom)
        t += ms


def _all_beats(events, timing_points, end_ms):
    """Every event's per-beat pulses; each event runs until the next
    event's time, or `end_ms` for the last one (BeatPulseManager spans)."""
    for i, event in enumerate(events):
        span_end = (float(events[i + 1].get('time', 0.0))
                    if i + 1 < len(events) else float(end_ms))
        yield from _beats(event, span_end, timing_points)


class BeatPulseEffect:
    def __init__(self, events, timing_points, end_ms):
        events = sorted((e for e in events or [] if isinstance(e, dict)),
                        key=lambda e: float(e.get('time', 0.0)))
        timing_points = sorted(timing_points or [], key=lambda p: p[0])
        self._beats = sorted(_all_beats(events, timing_points, end_ms))
        self._starts = [b[0] for b in self._beats]

    def __bool__(self):
        return bool(self._beats)

    def _scale(self, t_now) -> float:
        idx = bisect_right(self._starts, float(t_now)) - 1
        if idx < 0:
            return 1.0
        start, beat, strength, zoom = self._beats[idx]
        return bloom(t_now - start, beat, zoom, _EASE_OUT_QUINT,
                     rest=1.0, peak=strength)

    def at(self, ctx) -> EffectFrame | None:
        scale = self._scale(ctx.t_now)
        if scale == 1.0:
            return None
        rx, ry, w, h = ctx.chart_rect
        cx, cy = rx + w / 2.0, ry + h / 2.0
        transform = QTransform()
        transform.translate(cx, cy)
        transform.scale(scale, scale)
        transform.translate(-cx, -cy)
        return EffectFrame(transform=transform)
