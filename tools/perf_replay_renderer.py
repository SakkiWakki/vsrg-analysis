from __future__ import annotations

import argparse
import bisect
import cProfile
import io
import pstats
import statistics
import time
from dataclasses import dataclass, field

from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from analysis.core import game as game_mod
from analysis.core.game import resolve_standalone_replay
from analysis.player.player_api.base import Player
from analysis.player.render import culling
from analysis.player.render.render_context import RenderContext
from analysis.player.render.qt_renderer import QtPlayerRenderer


WINDOW_W = 1280
WINDOW_H = 900


@dataclass
class _FrameSnapshot:
    t_now: float
    note_views: list = field(default_factory=list)
    chart_extras: list[tuple[str, int, float]] = field(default_factory=list)
    miss_holds: list[tuple[int, float, float, float, float]] = field(
        default_factory=list)
    ghost_taps: list[tuple[int, float]] = field(default_factory=list)


class _NoopSidebar:
    def top_sections(self):
        return []

    def bottom_sections(self):
        return []

    def flyout_section(self, _key):
        return None


class _CorePlugins:
    layers = None
    sidebar = _NoopSidebar()

    def draw(self, _stage, _ctx):
        return None


class _FullPlugins:
    def __init__(self, real):
        self.layers = getattr(real, 'layers', None)
        self.sidebar = getattr(real, 'sidebar', _NoopSidebar())
        self._real = real

    def draw(self, stage, ctx):
        return self._real.draw(stage, ctx)


def _build_player(replay_path: str) -> Player:
    game, replay, bpms, sm_offset, _audio, _extra = (
        resolve_standalone_replay(replay_path, args=[]))
    adapter = game_mod.get(game)
    player_kwargs = adapter.player_kwargs(replay)
    return Player(
        replay,
        game=game,
        bpms=bpms,
        sm_offset=sm_offset,
        audio_path=None,
        window_w=WINDOW_W,
        window_h=WINDOW_H,
        **player_kwargs,
    )


def _candidate_count(player: Player, renderer: QtPlayerRenderer,
                     image: QImage, t_now: float) -> int:
    painter = QPainter(image)
    try:
        ctx = renderer.build_context(player, painter, t_now)
        return len(ctx.candidates)
    finally:
        painter.end()


def _peak_window(player: Player, image: QImage, sample_points: int):
    renderer = QtPlayerRenderer(_CorePlugins())
    renderer._draw_hud = lambda ctx, painter: None
    renderer._draw_free_sections = lambda ctx, painter: None

    sample_ts = [
        player.t_min + (player.t_max - player.t_min) * i
        / max(1, sample_points - 1)
        for i in range(sample_points)
    ]
    counts = [
        (t_now, _candidate_count(player, renderer, image, t_now))
        for t_now in sample_ts
    ]
    peak_t, peak_count = max(counts, key=lambda item: item[1])
    count_values = [count for _t, count in counts]
    return peak_t, peak_count, count_values


def _build_frame_context(renderer: QtPlayerRenderer, player: Player,
                         t_now: float, *, painter=None,
                         with_culling: bool) -> RenderContext:
    x0, lane_w = player._lane_geom()
    ctx = RenderContext(
        player=player,
        screen=None,
        pygame=renderer.compat,
        colors=player.judge_colors,
        t_now=float(t_now),
        x0=x0,
        lane_w=lane_w,
        judge_y=int(player.H * player.hit_line_y_frac),
        painter=painter,
        _scroll_speed=float(player.scroll_speed),
    )
    ctx.drawers = renderer._resolve_drawers(player)
    if with_culling:
        culling.prepare_time_window(ctx)
        ctx.candidates = culling.select_note_candidates(ctx)
    return ctx


def _build_chart_extra_snapshot(ctx: RenderContext, kind: str,
                                times, cols) -> list[tuple[str, int, float]]:
    if not times.size:
        return []
    lo = bisect.bisect_left(times, ctx.target_lo)
    hi = bisect.bisect_right(times, ctx.target_hi)
    out = []
    for k in range(lo, hi):
        col = int(cols[k])
        if col >= ctx.player.keycount:
            continue
        out.append((kind, col, ctx.time_to_y(float(times[k]))))
    return out


def _build_snapshot(renderer: QtPlayerRenderer, player: Player,
                    t_now: float) -> _FrameSnapshot:
    ctx = _build_frame_context(
        renderer,
        player,
        t_now,
        with_culling=True,
    )
    note_views = []
    for i in ctx.candidates:
        note = renderer._build_note_view(ctx, i)
        if note is not None:
            note_views.append(note)

    chart_extras = []
    chart_extras.extend(_build_chart_extra_snapshot(
        ctx, 'mine', player.notes.mine_times, player.notes.mine_cols))
    chart_extras.extend(_build_chart_extra_snapshot(
        ctx, 'lift', player.notes.lift_times, player.notes.lift_cols))
    chart_extras.extend(_build_chart_extra_snapshot(
        ctx, 'fake', player.notes.fake_times, player.notes.fake_cols))

    miss_holds = []
    for k in renderer._visible_miss_hold_indices(ctx):
        col = int(player.notes.miss_hold_cols[k])
        if col >= player.keycount:
            continue
        y_press = ctx.time_to_y(float(player.notes.miss_hold_press[k]))
        y_release = ctx.time_to_y(float(player.notes.miss_hold_release[k]))
        clipped = renderer._clip_to_screen(y_press, y_release, player.H)
        if clipped is None:
            continue
        top, bot = clipped
        miss_holds.append((col, top, bot, y_press, y_release))

    ghost_taps = []
    if player.notes.ghost_times.size:
        ghost_key = (player.notes.ghost_sv_times if ctx.use_sv_space
                     else player.notes.ghost_times)
        g_lo = bisect.bisect_left(ghost_key, ctx.target_lo)
        g_hi = bisect.bisect_right(ghost_key, ctx.target_hi)
        for k in range(g_lo, g_hi):
            col = int(player.notes.ghost_cols[k])
            if col >= player.keycount:
                continue
            ghost_taps.append((col, ctx.time_to_y(float(player.notes.ghost_times[k]))))

    return _FrameSnapshot(
        t_now=float(t_now),
        note_views=note_views,
        chart_extras=chart_extras,
        miss_holds=miss_holds,
        ghost_taps=ghost_taps,
    )


def _draw_snapshot(renderer: QtPlayerRenderer, player: Player,
                   painter: QPainter, snapshot: _FrameSnapshot) -> None:
    ctx = _build_frame_context(
        renderer,
        player,
        snapshot.t_now,
        painter=painter,
        with_culling=False,
    )
    renderer._draw_background(ctx, painter)
    renderer._draw_lanes(ctx, painter)
    renderer._draw_judgment(ctx, painter)

    for note in snapshot.note_views:
        if note.is_ln:
            renderer._draw_ln_parts(ctx, painter, note)
        head_visible = renderer._draw_note_head_if_visible(ctx, painter, note)
        if head_visible:
            renderer._draw_press_mark(ctx, painter, note)
            if note.miss:
                renderer._draw_miss_x(ctx, painter, note)

    for kind, col, y in snapshot.chart_extras:
        lane_x = ctx.lane_x(col)
        match kind:
            case 'mine':
                ctx.drawers['mine'](painter, lane_x, y, ctx.lane_w)
            case 'lift':
                ctx.drawers['lift'](
                    painter, player.skin, lane_x, y,
                    ctx.lane_w, ctx.note_h, player.palette[col])
            case 'fake':
                ctx.drawers['fake'](
                    painter, player.skin, lane_x, y,
                    ctx.lane_w, ctx.note_h, player.palette[col])

    miss_hold_draw = ctx.drawers['miss_hold_stroke']
    miss_color = player.judge_colors['miss']
    for col, top, bot, y_press, y_release in snapshot.miss_holds:
        miss_hold_draw(
            painter,
            int(ctx.lane_x(col)),
            ctx.lane_w,
            top,
            bot,
            y_press,
            y_release,
            miss_color,
        )

    ghost_draw = ctx.drawers['ghost_tap']
    for col, y in snapshot.ghost_taps:
        ghost_draw(painter, ctx.lane_x(col), y, ctx.lane_w, ctx.note_h)


def _time_snapshot_build(renderer: QtPlayerRenderer, player: Player,
                         frame_ts: list[float]) -> tuple[float, list[_FrameSnapshot]]:
    wall_start = time.perf_counter()
    snapshots = [_build_snapshot(renderer, player, t_now) for t_now in frame_ts]
    return time.perf_counter() - wall_start, snapshots


def _time_snapshot_paint(renderer: QtPlayerRenderer, player: Player,
                         image: QImage,
                         snapshots: list[_FrameSnapshot]) -> float:
    wall_start = time.perf_counter()
    for snapshot in snapshots:
        image.fill(0)
        painter = QPainter(image)
        try:
            _draw_snapshot(renderer, player, painter, snapshot)
        finally:
            painter.end()
    return time.perf_counter() - wall_start


def _profile(player: Player, renderer: QtPlayerRenderer,
             image: QImage, frame_ts: list[float], stat_filter: str) -> tuple[float, str]:
    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()
    for t_now in frame_ts:
        image.fill(0)
        painter = QPainter(image)
        try:
            renderer.draw(player, painter, t_now)
        finally:
            painter.end()
    profiler.disable()
    wall_elapsed = time.perf_counter() - wall_start

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative').print_stats(stat_filter, 25)
    return wall_elapsed, stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_path')
    parser.add_argument('--sample-points', type=int, default=180)
    parser.add_argument('--frames', type=int, default=240)
    parser.add_argument('--fps', type=float, default=120.0)
    parser.add_argument(
        '--stat-filter',
        default='analysis/player|analysis/components|plugins/builtin/sidepanel',
    )
    args = parser.parse_args()

    app = QApplication.instance() or QApplication([])
    _ = app

    player = _build_player(args.replay_path)
    player.paused = False
    image = QImage(WINDOW_W, WINDOW_H, QImage.Format_ARGB32_Premultiplied)

    peak_t, peak_count, count_values = _peak_window(
        player,
        image,
        args.sample_points,
    )
    frame_dt = 1.0 / max(1e-6, args.fps)
    start_t = max(
        player.t_min,
        peak_t - (args.frames * frame_dt) / 2.0,
    )
    frame_ts = [start_t + i * frame_dt for i in range(args.frames)]

    core_renderer = QtPlayerRenderer(_CorePlugins())
    core_renderer._draw_hud = lambda ctx, painter: None
    core_renderer._draw_free_sections = lambda ctx, painter: None

    full_renderer = QtPlayerRenderer(_FullPlugins(player.plugins))

    core_wall, core_stats = _profile(
        player,
        core_renderer,
        image,
        frame_ts,
        args.stat_filter,
    )
    full_wall, full_stats = _profile(
        player,
        full_renderer,
        image,
        frame_ts,
        args.stat_filter,
    )
    snapshot_build_wall, snapshots = _time_snapshot_build(
        core_renderer,
        player,
        frame_ts,
    )
    snapshot_paint_wall = _time_snapshot_paint(
        core_renderer,
        player,
        image,
        snapshots,
    )
    swap_estimate_wall = max(snapshot_build_wall, snapshot_paint_wall)

    print('Replay:', args.replay_path)
    print('Game:', player.game)
    print('Keycount:', player.keycount)
    print('Replay length:', round(player.t_max, 3))
    print(
        'Sample candidate counts: min=%d median=%d max=%d mean=%.1f'
        % (
            min(count_values),
            int(statistics.median(count_values)),
            max(count_values),
            statistics.mean(count_values),
        )
    )
    print('Peak dense sample: t=%.3f count=%d' % (peak_t, peak_count))
    print(
        'Profile window: start=%.3f frames=%d dt=%.5f'
        % (start_t, args.frames, frame_dt)
    )
    print(
        'Core renderer: total %.3fs, %.3f ms/frame'
        % (core_wall, core_wall * 1000.0 / args.frames)
    )
    print(core_stats)
    print(
        'Full renderer: total %.3fs, %.3f ms/frame'
        % (full_wall, full_wall * 1000.0 / args.frames)
    )
    print(full_stats)
    print(
        'Swap-buffer experiment: build %.3fs, %.3f ms/frame'
        % (
            snapshot_build_wall,
            snapshot_build_wall * 1000.0 / args.frames,
        )
    )
    print(
        'Swap-buffer experiment: paint %.3fs, %.3f ms/frame'
        % (
            snapshot_paint_wall,
            snapshot_paint_wall * 1000.0 / args.frames,
        )
    )
    print(
        'Swap-buffer experiment: ideal throughput %.3fs, %.3f ms/frame'
        % (
            swap_estimate_wall,
            swap_estimate_wall * 1000.0 / args.frames,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
