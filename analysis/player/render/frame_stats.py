"""Render-frame cadence tracker.

Owned by the renderer, which calls `tick()` once per rendered frame;
consumers (the frame-analyzer sidebar component) read `summary()`.
Split from the component because sidebar draws are throttled to the
HUD cache cadence and no longer run once per rendered frame.
"""
from __future__ import annotations

import time
from collections import deque
from math import sqrt


class FrameStats:
    def __init__(self, max_samples: int = 300) -> None:
        self._last_wall: float | None = None
        self._dts = deque(maxlen=max_samples)

    def tick(self) -> float | None:
        now = time.monotonic()
        last = self._last_wall
        self._last_wall = now
        if last is None:
            return None
        dt = now - last
        # Ignore obviously bogus intervals (system suspend, clock jumps,
        # accidental long stalls) so one bad sample does not ruin the panel.
        if 0.0 < dt < 1.0:
            self._dts.append(dt)
            return dt
        return None

    def summary(self) -> dict | None:
        if not self._dts:
            return None
        arr = list(self._dts)
        n = len(arr)
        avg = sum(arr) / n

        var = 0.0
        if n > 1:
            var = sum((v - avg) ** 2 for v in arr) / n

        sorted_arr = sorted(arr)
        p95 = sorted_arr[max(0, min(n - 1, int(0.95 * (n - 1))))]
        p99 = sorted_arr[max(0, min(n - 1, int(0.99 * (n - 1))))]

        # Count rough hitches in-window: frames slower than 2x the average.
        hitch_threshold = avg * 2.0
        hitches = sum(1 for v in arr if v >= hitch_threshold)

        return {
            'n': n,
            'inst_dt': arr[-1],
            'avg_dt': avg,
            'min_dt': min(arr),
            'max_dt': max(arr),
            'std_dt': sqrt(var),
            'p95_dt': p95,
            'p99_dt': p99,
            'hitches': hitches,
        }
