"""Built-in sidebar section: render-frame timing analyzer.

Tracks wall-clock deltas between component draw calls to estimate GUI render
cadence and jitter in real time.
"""
from __future__ import annotations

from collections import deque
from math import sqrt

from analysis.components import Manifest, SURFACE_GUI
from analysis.plugins.host_api import monotonic_seconds
from plugins.builtin.sidepanel import SidebarFields


MANIFEST = Manifest(
    key='builtin:frame_analyzer',
    name='Frame Analyzer',
    supported_surfaces={SURFACE_GUI},
    requires_data={'t_now', 'paused'},
    plugin_fields={
        'sidebar': SidebarFields(
            priority=110,
            draggable=True,
            default_free_xy=(0.02, 0.44),
            default_size=(240, 170),
        ),
    },
)


class _FrameStats:
    def __init__(self, max_samples: int = 300) -> None:
        self._last_wall: float | None = None
        self._dts = deque(maxlen=max_samples)

    def tick(self) -> float | None:
        now = monotonic_seconds()
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
        inst = arr[-1]
        mn = min(arr)
        mx = max(arr)

        var = 0.0
        if n > 1:
            var = sum((v - avg) ** 2 for v in arr) / n
        std = sqrt(var)

        sorted_arr = sorted(arr)
        p95_idx = max(0, min(n - 1, int(0.95 * (n - 1))))
        p99_idx = max(0, min(n - 1, int(0.99 * (n - 1))))
        p95 = sorted_arr[p95_idx]
        p99 = sorted_arr[p99_idx]

        # Count rough hitches in-window: frames slower than 2x the average.
        hitch_threshold = avg * 2.0
        hitches = sum(1 for v in arr if v >= hitch_threshold)

        return {
            'n': n,
            'inst_dt': inst,
            'avg_dt': avg,
            'min_dt': mn,
            'max_dt': mx,
            'std_dt': std,
            'p95_dt': p95,
            'p99_dt': p99,
            'hitches': hitches,
        }


_STATS = _FrameStats()


def _ms(v: float) -> str:
    return f'{v * 1000.0:6.2f}ms'


def _fps(v: float) -> str:
    if v <= 1e-9:
        return '   inf'
    return f'{1.0 / v:6.1f}'


def _draw(ctx):
    if getattr(ctx, 'measure_only', False):
        return

    _ = ctx.data.t_now()
    _STATS.tick()
    s = _STATS.summary()

    state = 'PAUSED' if ctx.data.paused() else 'PLAYING'
    if s is None:
        ctx.draw_text('Frame Analyzer')
        ctx.draw_text('collecting samples...')
        ctx.draw_text(f'state: {state}')
        return

    lines = (
        'Frame Analyzer',
        f'state: {state}',
        f'fps  now/avg = {_fps(s["inst_dt"])} / {_fps(s["avg_dt"])}',
        f'dt   now/avg = {_ms(s["inst_dt"])} / {_ms(s["avg_dt"])}',
        f'dt   p95/p99 = {_ms(s["p95_dt"])} / {_ms(s["p99_dt"])}',
        f'dt   min/max = {_ms(s["min_dt"])} / {_ms(s["max_dt"])}',
        f'jitter (std) = {_ms(s["std_dt"])}',
        f'hitches (2x) = {s["hitches"]} / {s["n"]}',
    )
    for line in lines:
        ctx.draw_text(line)


def register_components(add):
    add(MANIFEST, _draw)
