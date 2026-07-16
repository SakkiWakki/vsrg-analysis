"""Animated lane geometry for charts with lane switches.

Ports fluXis's `LaneSwitchManager` presentation: each lane is a
width-animated slot -- active lanes at full width, inactive lanes at
zero -- resized over the switch event's duration with its easing, and
the packed row re-centers on the playfield. (fluXis durations are plain
milliseconds from the event's `speed` field; there is no BPM coupling
in their implementation.)

The timeline consumed here is the adapter `lane_mask()` output:
`[(t_start_s, mask, duration_s, easing_id)]`, time-sorted.
"""
from __future__ import annotations

from bisect import bisect_right

from analysis.player.render.easing import ease


def column_layout(timeline, keycount, t_now, x0, lane_w):
    """`(xs, widths)` tuples (len `keycount`, px) for the current
    instant, or None when every lane is at full width (static layout
    fast path -- callers keep the uniform-geometry drawing)."""
    factors = _width_factors(timeline, keycount, t_now)
    if factors is None:
        return None

    total = sum(factors) * lane_w
    center = x0 + keycount * lane_w / 2.0
    x = center - total / 2.0
    xs, widths = [], []
    for f in factors:
        w = f * lane_w
        xs.append(x)
        widths.append(w)
        x += w
    return tuple(xs), tuple(widths)


def _width_factors(timeline, keycount, t_now):
    idx = bisect_right(timeline, float(t_now), key=lambda e: e[0]) - 1
    if idx < 0:
        return None   # before the first switch: full layout

    t0, mask, duration, easing = timeline[idx]
    prev_mask = timeline[idx - 1][1] if idx > 0 else (1,) * keycount

    if duration <= 0 or t_now >= t0 + duration:
        factors = [float(m) for m in mask]
    else:
        f = ease(easing, (t_now - t0) / duration)
        factors = [a + (b - a) * f for a, b in zip(prev_mask, mask)]

    if all(x >= 1.0 for x in factors):
        return None
    return factors
