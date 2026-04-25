"""Optional cProfile + per-frame metric harness for the player canvas.

Enable with ``VSRG_PROFILE_PAINT=N`` where N is the number of paint
frames to capture before dumping. The profiler wraps each
``paintEvent`` call inside a single shared cProfile.Profile so the
collected data covers the actual hot-path work the user sees as
hitches. After N frames the cumulative stats are written to
``$VSRG_PROFILE_OUT`` (or ``/tmp/vsrg_paint.prof`` if unset) in
pstats format and a short text summary is printed to stderr.

The point is to capture *real* paints under real load -- a synthetic
harness can't reproduce Qt's painter state, the GPU compositor, or
the actual sprite-cache hit pattern at the chart's start. So we sample
inline and dump.

Use:
    VSRG_PROFILE_PAINT=300 python -m analysis ...
    # play through the dense start of a chart
    # after 300 paints (~2.5 s at 120 Hz) /tmp/vsrg_paint.prof is ready
    python -m pstats /tmp/vsrg_paint.prof
"""
from __future__ import annotations

import cProfile
import os
import pstats
import sys
import threading
import time
from io import StringIO


class PaintProfiler:
    """Lazy-init cProfile collector keyed on env vars.

    Returns a no-op handle when the profiler is disabled so the paint
    site stays a single ``with`` statement either way.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._profile: cProfile.Profile | None = None
        self._frames_remaining = 0
        self._dumped = False
        self._out_path = '/tmp/vsrg_paint.prof'
        # Per-frame metrics so we can correlate frame-time with visible
        # work. List of dicts; appended by `record_frame`. Dumped to a
        # CSV next to the prof file so the user can pivot in their
        # tool of choice.
        self._frame_metrics: list[dict] = []
        self._frame_start_wall: float = 0.0
        self._enabled = self._init_from_env()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _init_from_env(self) -> bool:
        raw = os.environ.get('VSRG_PROFILE_PAINT')
        if not raw:
            return False
        try:
            n = int(raw)
        except ValueError:
            return False
        if n <= 0:
            return False
        self._frames_remaining = n
        self._out_path = os.environ.get('VSRG_PROFILE_OUT',
                                         '/tmp/vsrg_paint.prof')
        self._profile = cProfile.Profile()
        print(f'[paint_profiler] capturing {n} paint frames -> '
              f'{self._out_path}', file=sys.stderr)
        return True

    def begin_frame(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            if self._profile is None or self._frames_remaining <= 0:
                return
            self._frame_start_wall = time.perf_counter()
            self._profile.enable()

    def end_frame(self) -> None:
        if not self._enabled:
            return
        dump = False
        with self._lock:
            if self._profile is None:
                return
            self._profile.disable()
            self._frames_remaining -= 1
            if self._frames_remaining <= 0 and not self._dumped:
                self._dumped = True
                dump = True
        if dump:
            self._dump()

    def record_frame(self, ctx) -> None:
        """Snapshot scalar metrics for the just-finished frame. Called
        from the renderer right before paintEvent's painter.end()."""
        if not self._enabled:
            return
        with self._lock:
            if self._profile is None:
                return
            try:
                wall_ms = (time.perf_counter() - self._frame_start_wall) * 1000
                p = ctx.player
                self._frame_metrics.append({
                    'frame': len(self._frame_metrics),
                    'wall_ms': float(wall_ms),
                    't_now': float(getattr(ctx, 't_now', 0.0)),
                    'candidates': int(len(getattr(ctx, 'candidates', []))),
                    'visible_miss_holds': int(len(
                        getattr(ctx, 'visible_miss_holds', []))),
                    'visible_ghost_taps': int(len(
                        getattr(ctx, 'visible_ghost_taps', []))),
                    'screen_h': int(getattr(p, 'H', 0)),
                    'use_sv': bool(getattr(ctx, 'use_sv_space', False)),
                    'scroll_speed': float(getattr(p, 'scroll_speed', 0.0)),
                    'target_lo': float(getattr(ctx, 'target_lo', 0.0)),
                    'target_hi': float(getattr(ctx, 'target_hi', 0.0)),
                })
            except Exception:
                pass

    def _dump(self) -> None:
        prof = self._profile
        if prof is None:
            return
        try:
            prof.dump_stats(self._out_path)
        except Exception as e:
            print(f'[paint_profiler] dump_stats failed: {e}', file=sys.stderr)
            return
        # Also print a short summary so the user sees the top functions
        # without having to open pstats interactively.
        buf = StringIO()
        try:
            stats = pstats.Stats(prof, stream=buf)
            stats.strip_dirs().sort_stats('cumulative').print_stats(25)
        except Exception:
            pass
        print('[paint_profiler] top by cumulative time:', file=sys.stderr)
        print(buf.getvalue(), file=sys.stderr)
        print(f'[paint_profiler] full stats at {self._out_path}',
              file=sys.stderr)

        self._dump_frame_metrics()

    def _dump_frame_metrics(self) -> None:
        if not self._frame_metrics:
            return
        csv_path = self._out_path.rsplit('.', 1)[0] + '_frames.csv'
        try:
            with open(csv_path, 'w') as f:
                keys = list(self._frame_metrics[0].keys())
                f.write(','.join(keys) + '\n')
                for row in self._frame_metrics:
                    f.write(','.join(str(row.get(k, '')) for k in keys) + '\n')
        except Exception as e:
            print(f'[paint_profiler] csv dump failed: {e}', file=sys.stderr)
            return

        # Print a small summary so the user sees what's hot at a glance.
        wall_ms = [r['wall_ms'] for r in self._frame_metrics]
        cands = [r['candidates'] for r in self._frame_metrics]
        wall_ms_sorted = sorted(wall_ms)
        n = len(wall_ms_sorted)
        def pct(p):
            return wall_ms_sorted[min(n - 1, int(p * n / 100))]

        print('[paint_profiler] frame metrics:', file=sys.stderr)
        print(f'  frames        = {n}', file=sys.stderr)
        print(f'  wall_ms       p50={pct(50):.2f}  p90={pct(90):.2f}  '
              f'p99={pct(99):.2f}  max={max(wall_ms):.2f}', file=sys.stderr)
        print(f'  candidates    p50={sorted(cands)[n//2]}  '
              f'p99={sorted(cands)[min(n-1, n*99//100)]}  '
              f'max={max(cands)}', file=sys.stderr)
        # Correlation: are slow frames the high-candidate-count frames?
        # Show the 10 slowest frames with their candidate counts so the
        # user can eyeball the relationship.
        slowest = sorted(self._frame_metrics, key=lambda r: -r['wall_ms'])[:10]
        print('[paint_profiler] 10 slowest frames:', file=sys.stderr)
        print('  frame  wall_ms  cands  miss_holds  ghosts  t_now', file=sys.stderr)
        for r in slowest:
            print(f"  {r['frame']:5d}  {r['wall_ms']:7.2f}  "
                  f"{r['candidates']:5d}  {r['visible_miss_holds']:10d}  "
                  f"{r['visible_ghost_taps']:6d}  {r['t_now']:6.3f}",
                  file=sys.stderr)
        print(f'[paint_profiler] per-frame CSV at {csv_path}', file=sys.stderr)


_PROFILER = PaintProfiler()


def begin_frame() -> None:
    _PROFILER.begin_frame()


def end_frame() -> None:
    _PROFILER.end_frame()


def record_frame(ctx) -> None:
    _PROFILER.record_frame(ctx)


def enabled() -> bool:
    return _PROFILER.enabled
