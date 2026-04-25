"""Etterna `#SPEEDS` evaluator.

Per DESIGN.tex, SPEEDS is the position-dependent zoom z(t) -- evaluated at
the current playhead, applied as a uniform multiplier on the cumulative
delta. It is NOT integrated into the cumulative cache, because doing so
would break the invariant that `cumulative_at(b) - cumulative_at(a) ==
distance(a, b)` outside transition windows.

The evaluator matches the reference engine's `_speed_percent_at` exactly;
both the reference `BeatSpaceSVEngine` and the new measure engine consume
this class.
"""
from __future__ import annotations

import bisect

from analysis.player.sv.timing import TimingMap


def normalize_speed_segment(seg) -> tuple[float, float, float, int]:
    """Coerce a #SPEEDS row to the canonical 4-tuple
    (beat, ratio, delay, unit) where unit is 0 (beats) or 1 (seconds)."""
    if len(seg) >= 4:
        beat, ratio, delay, unit = seg[:4]
        return (float(beat), float(ratio), float(delay),
                0 if int(unit) == 0 else 1)
    if len(seg) == 3:
        beat, ratio, delay = seg
        return float(beat), float(ratio), float(delay), 0
    beat, ratio = seg[:2]
    return float(beat), float(ratio), 0.0, 0


class SpeedsEvaluator:
    """Position-dependent zoom z(t) from Etterna `#SPEEDS`.

    Each segment is `(beat, ratio, delay, unit)`. `delay` is the lerp
    duration from the prior ratio; `unit` is 0 (delay measured in beats)
    or 1 (delay measured in seconds). At a given (beat, music_seconds):

    - Before the first segment: ratio = 1.
    - During a segment's delay window: lerp from prior_ratio to seg_ratio.
    - After the delay: hold at seg_ratio until the next segment.

    Delays from `#DELAYS` events at the segment's beat shift the start
    time backward (matching Etterna's GetDisplayedSpeedPercent behavior on
    rows with both DELAY and SPEEDS).
    """

    def __init__(self, speeds, timing: TimingMap, delay_at_beats=None):
        self._speeds = [normalize_speed_segment(s) for s in (speeds or [])]
        self._beats = [s[0] for s in self._speeds]
        self._timing = timing
        self._delay_at_beats = dict(delay_at_beats or {})

    def __bool__(self) -> bool:
        return bool(self._speeds)

    def _delay_at_beat(self, beat: float) -> float:
        return self._delay_at_beats.get(float(beat), 0.0)

    def percent_at(self, beat: float, music_seconds: float) -> float:
        """Evaluate Etterna's GetDisplayedSpeedPercent(songBeat, songSec)."""
        if not self._speeds:
            return 1.0
        idx = bisect.bisect_right(self._beats, beat) - 1
        if idx < 0:
            return 1.0
        seg_beat, seg_ratio, seg_delay, seg_unit = self._speeds[idx]
        start_time = self._timing.beat_to_time(seg_beat) - self._delay_at_beat(seg_beat)
        if seg_unit == 1:
            end_time = start_time + seg_delay
        else:
            end_beat = seg_beat + seg_delay
            end_time = self._timing.beat_to_time(end_beat) - self._delay_at_beat(end_beat)

        first = self._speeds[0]
        if idx == 0 and first[2] > 0.0 and music_seconds < start_time:
            return 1.0
        if end_time >= music_seconds and (idx > 0 or first[2] > 0.0):
            prior_ratio = 1.0 if idx == 0 else self._speeds[idx - 1][1]
            duration = end_time - start_time
            ratio_used = 1.0 if duration == 0.0 else (music_seconds - start_time) / duration
            return prior_ratio + (seg_ratio - prior_ratio) * ratio_used
        return seg_ratio
