"""Render app-stack frames of any chart/replay at given timestamps.

The fast way to SEE what the player renders at time T without launching
the GUI: builds the real Player + QtPlayerRenderer offscreen and paints
one PNG per requested time.

    python tools/render_frames.py <path> [options] t1 [t2 ...]

    <path>   a replay file, or a chart ref the game adapter understands
             (NotITG/Etterna: "song.sm" or "song.sm::<chart index>")

    --game NAME    force the adapter (default: first that claims the
                   path; NotITG .sm files need --game notitg or the
                   etterna adapter claims them)
    --out DIR      output directory (default: alongside this repo in
                   ./frames)
    --size WxH     window size (default 1280x900)
    --wait-sweep   NotITG lazy compile only: wait for the FULL
                   background sweep before rendering. Without it the
                   tool advances the live sim just past the last
                   requested time - much faster, and actor/element
                   state is exact; only the driver-APPLIED mod channels
                   (ApplyModifiers streams) and swept-only surfaces
                   (complete proxy topology) may still be partial,
                   since those hot-swap in at sweep end.

Frames paint through the raster backend (no GL context), so GL-only
tiers degrade exactly as the app's raster fallback does: fullscreen/
per-actor shaders skip, Polygon meshes draw as their flat quad.
Geometry, storyboard, mods, and field-instance compositing are the real
app path.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from analysis.core import game as game_mod  # noqa: E402
from analysis.player.player_api.base import Player  # noqa: E402
from analysis.player.render.qt_renderer import QtPlayerRenderer  # noqa: E402


class _NoopSidebar:
    def top_sections(self):
        return []

    def bottom_sections(self):
        return []

    def free_sections(self):
        return []

    def flyout_section(self, _key):
        return None


class _NoopPlugins:
    layers = None
    sidebar = _NoopSidebar()

    def draw(self, _stage, _ctx):
        return None


def _resolve(path: str, forced_game: str | None):
    """(game, replay, bpms, offset) via the forced adapter's
    parse_replay/resolve_all, else the standalone resolver sweep."""
    if forced_game:
        adapter = game_mod.get(forced_game)
        replay = adapter.parse_replay(path)
        bpms, offset, _audio = adapter.resolve_all(replay)
        return forced_game, replay, bpms, offset
    name, replay, bpms, offset, _audio, _extra = (
        game_mod.resolve_standalone_replay(path, args=[]))
    return name, replay, bpms, offset


def _advance_live_sim(replay, target: float, wait_sweep: bool) -> None:
    """NotITG lazy compile: bring the shared live sim's frontier past
    `target` (or to the chart end with --wait-sweep) so the frames read
    resolved state instead of pre-load rests. No-op for eager compiles
    and other games."""
    compiled = replay.get('_notitg_modfile') or {}
    live = compiled.get('_live_sim')
    if live is None:
        return
    end = getattr(live, '_end_s', None)
    goal = end if wait_sweep and end is not None else min(
        target + 1.0, end if end is not None else target + 1.0)
    print(f'advancing live sim to {goal:.1f}s ...', flush=True)
    lock = getattr(live, 'sweep_lock', None)
    started = time.monotonic()
    while getattr(live, 'now', 0.0) < goal:
        # Target computed INSIDE the lock: a stale target the daemon
        # sweep already passed reads as a backward seek and resets the
        # whole sim (see producers._spawn_background_upgrade).
        if lock is not None:
            with lock:
                live.advance_frontier(min(live.now + 2.0, goal))
        else:
            live.advance_frontier(min(live.now + 2.0, goal))
        if time.monotonic() - started > 600:
            print('live-sim advance timed out; rendering partial state',
                  file=sys.stderr)
            break
    if wait_sweep:
        # The sweep worker performs the end-of-sweep hot-swaps (driver
        # mod channels, complete topology); give it a beat to land them
        # once the frontier is at the end.
        time.sleep(3.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path')
    parser.add_argument('times', nargs='+', type=float)
    parser.add_argument('--game', default=None)
    parser.add_argument('--out', default='frames')
    parser.add_argument('--size', default='1280x900')
    parser.add_argument('--wait-sweep', action='store_true')
    args = parser.parse_args()

    width, height = (int(v) for v in args.size.lower().split('x'))
    _app = QApplication.instance() or QApplication([sys.argv[0]])

    game, replay, bpms, offset = _resolve(args.path, args.game)
    adapter = game_mod.get(game)
    player = Player(replay, game=game, bpms=bpms, sm_offset=offset,
                    audio_path=None, window_w=width, window_h=height,
                    **adapter.player_kwargs(replay))
    renderer = QtPlayerRenderer(_NoopPlugins())
    print(f'game={game} notes={len(replay.get("noterows", ()))} '
          f'window={width}x{height}')

    _advance_live_sim(replay, max(args.times), args.wait_sweep)

    os.makedirs(args.out, exist_ok=True)
    for t in args.times:
        print(f'rendering t={t:.3f} ...', flush=True)
        image = QImage(width, height, QImage.Format.Format_RGB32)
        image.fill(0xFF000000)
        painter = QPainter(image)
        try:
            renderer.draw(player, painter, float(t))
        finally:
            painter.end()
        out_path = os.path.join(args.out, f'frame_{t:.3f}.png')
        image.save(out_path)
        print('wrote', out_path, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
