"""Unified curve-native LN renderer (analysis/player/render/layers/notes.py).

Every hold body is a path `(xs, ys)`; the renderer strokes a constant-width
ribbon along it and seats the head/tail caps on its endpoints by tangent.
These tests pin the three producers (straight fast-path, SV fold, mod bend)
into the one consumer, the cap-seating tangent, and the fold-aware clip.

Draw primitives are recorded via fake painters (no real QPainter needed);
`_build` / `_sv_fold_path` run against SimpleNamespace fakes shaped like the
real player + render context.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.player.render.layers import notes as nl
from analysis.player.render.qt_renderer import _NoteView


# --- fakes ----------------------------------------------------------

def _ln_view(**kw):
    """`_NoteView` for a held LN with the fields the LN drawers touch."""
    base = dict(
        i=0, col=0, y=100.0, y_end=300.0, press_y=100.0, lx=40, off=0.0,
        press_t=0.0, release_t=None, rel_off=None, end_t=2.0,
        is_ln=True, is_roll=False, miss=False, state='held',
        note_color=(255, 255, 255), jcolor=(255, 0, 0),
    )
    base.update(kw)
    return _NoteView(**base)


class _RecordPainter:
    """Records the draw calls the LN body/tail path makes so tests can
    assert rect vs stroke and the tail cap's rotation."""
    def __init__(self):
        self.tiled = []       # QRectF body-rect blits (fast-path)
        self.paths = []       # stroked ribbon paths (curve-native body)
        self.pixmaps = []     # tail/head cap blits (point form)
        self._rotate = 0.0
        self.rotations = []   # rotation-deg at each drawPixmap

    def drawTiledPixmap(self, rect, pm):
        self.tiled.append(rect)

    def drawPath(self, path):
        self.paths.append(path)

    def drawPixmap(self, point, pm):
        self.pixmaps.append((float(point.x()), float(point.y())))
        self.rotations.append(self._rotate)

    def save(self):
        pass

    def restore(self):
        self._rotate = 0.0

    def setPen(self, *a):
        pass

    def setBrush(self, *a):
        pass

    def translate(self, *a):
        pass

    def rotate(self, deg):
        self._rotate += deg

    def scale(self, *a):
        pass


def _ctx(judge_y=500, lane_w=80, H=600, press_hide=False, sprite=None):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    pm = sprite
    if pm is None:
        pm = QPixmap(lane_w, 20)
        pm.fill(Qt.transparent)
    sprite_cache = SimpleNamespace(get=lambda *a, **k: pm)
    player = SimpleNamespace(press_hide=press_hide, H=H)
    return SimpleNamespace(
        player=player, judge_y=judge_y, lane_w=lane_w, screen_margin=100,
        sprite_cache=sprite_cache,
        lane_x=lambda col: 40.0 + col * lane_w,
        lane_width=lambda col: lane_w,
    )


# --- (a) straight body => rect fast-path ----------------------------

def test_straight_body_uses_rect_fast_path():
    # Downscroll convention: head is deeper (larger y=300), tail higher
    # up the lane (y_end=100). The straight body spans [tail, head].
    ctx = _ctx()
    painter = _RecordPainter()
    n = _ln_view(body_path=None, y=300.0, y_end=100.0, state='held')
    nl._draw_ln(ctx, painter, n)
    # No path stroked; exactly one tiled rect spanning the head->tail body.
    assert painter.paths == []
    assert len(painter.tiled) == 1
    rect = painter.tiled[0]
    assert rect.top() == pytest.approx(100.0)      # top = tail (higher up)
    assert rect.height() == pytest.approx(200.0)


def test_bent_body_strokes_ribbon_not_rect():
    ctx = _ctx()
    painter = _RecordPainter()
    xs = np.array([40.0, 60.0, 40.0])       # bows right then back
    ys = np.array([100.0, 200.0, 300.0])
    n = _ln_view(body_path=(xs, ys), state='held')
    nl._draw_ln(ctx, painter, n)
    # Curve-native body: a stroked ribbon path, no rect fast-path.
    assert painter.tiled == []
    assert len(painter.paths) == 1


# --- (b) SV fold path from a synthetic negative-SV hold -------------

def _fold_player(change_t, head_y, mid_y, tail_y):
    """Player whose `batch_time_to_y` maps the three sample times
    (head, change, end) to `head_y`, `mid_y`, `tail_y` -- a hold that
    scrolls out to `mid_y` then FOLDS back to `tail_y`."""
    times = np.array([0.0])
    ln_tail_times = np.array([2.0])
    change_times = [np.array([change_t])]
    y_by_t = {0.0: head_y, change_t: mid_y, 2.0: tail_y}

    def batch_time_to_y(sample_t, frame, groups=None):
        return np.array([y_by_t[float(t)] for t in np.asarray(sample_t)])

    notes = SimpleNamespace(ln_tail_times=ln_tail_times,
                            columns_list=[0])
    return SimpleNamespace(
        times=times, notes=notes, _ln_change_times=change_times,
        batch_time_to_y=batch_time_to_y,
    )


def test_sv_fold_path_traces_projected_samples():
    # head y=100, out to y=40 at the SV sign-change, then FOLDS back to
    # tail y=260. The path visits all three, in trace order.
    p = _fold_player(change_t=1.0, head_y=100.0, mid_y=40.0, tail_y=260.0)
    ctx = _ctx()
    ctx.frame = SimpleNamespace(use_sv=True)
    ctx.t_now = -1.0        # upcoming: head sample pinned to head_y
    path = nl._sv_fold_path(ctx, i=0, pos=0, p=p, head_y=100.0, tail_y=260.0)
    assert path is not None
    xs, ys = path
    assert list(ys) == pytest.approx([100.0, 40.0, 260.0])
    # x is the lane's left edge at every sample (SV never shifts x).
    assert list(xs) == pytest.approx([40.0, 40.0, 40.0])


def test_sv_monotone_hold_has_no_fold_path():
    # No sign-change waypoints => monotone body => None (rect fast-path).
    p = _fold_player(change_t=1.0, head_y=100.0, mid_y=40.0, tail_y=260.0)
    p._ln_change_times = [np.zeros(0)]
    ctx = _ctx()
    ctx.frame = SimpleNamespace(use_sv=True)
    ctx.t_now = -1.0
    assert nl._sv_fold_path(ctx, 0, 0, p, 100.0, 260.0) is None


def test_sv_fold_path_none_without_sv():
    p = _fold_player(change_t=1.0, head_y=100.0, mid_y=40.0, tail_y=260.0)
    ctx = _ctx()
    ctx.frame = SimpleNamespace(use_sv=False)
    ctx.t_now = -1.0
    assert nl._sv_fold_path(ctx, 0, 0, p, 100.0, 260.0) is None


# --- (c) cap seating + tangent at the path endpoints ----------------

def test_straight_tail_blits_unrotated_at_tail_y():
    ctx = _ctx()
    painter = _RecordPainter()
    n = _ln_view(body_path=None, y_end=300.0, state='held')
    nl._draw_ln_tail_sprite(ctx, painter, n)
    assert len(painter.pixmaps) == 1
    assert painter.rotations[0] == pytest.approx(0.0)


def test_fold_tail_tangent_points_back_up_the_lane():
    # Final segment runs UPWARD (ys decreasing): the fold's end tangent
    # is (0, -1), so the cap rotates ~180deg vs the straight-down cap --
    # this is how the old flip_tail mirror emerges from the path.
    ctx = _ctx()
    painter = _RecordPainter()
    xs = np.array([40.0, 40.0, 40.0])
    ys = np.array([100.0, 300.0, 200.0])     # out to 300, folds up to 200
    n = _ln_view(body_path=(xs, ys), y_end=200.0)
    ok = nl._draw_tail_on_curve(ctx, painter, n, ctx.sprite_cache.get())
    assert ok
    # dy = 200-300 = -100 (dx=0) => atan2(-100,0) = -90deg; -90 - 90 = -180.
    assert painter.rotations[0] == pytest.approx(-180.0)


def test_bent_tail_tangent_follows_last_segment():
    ctx = _ctx()
    painter = _RecordPainter()
    # last segment goes down-and-right at 45deg: dx=dy=+100.
    xs = np.array([40.0, 40.0, 140.0])
    ys = np.array([100.0, 200.0, 300.0])
    n = _ln_view(body_path=(xs, ys), y_end=300.0)
    ok = nl._draw_tail_on_curve(ctx, painter, n, ctx.sprite_cache.get())
    assert ok
    # atan2(100,100)=45deg; 45 - 90 = -45.
    assert painter.rotations[0] == pytest.approx(-45.0)


def test_degenerate_path_falls_back_to_straight_blit():
    ctx = _ctx()
    painter = _RecordPainter()
    # duplicate last point => zero tangent => curve seat refuses.
    xs = np.array([40.0, 40.0])
    ys = np.array([300.0, 300.0])
    n = _ln_view(body_path=(xs, ys), y_end=300.0)
    assert not nl._draw_tail_on_curve(ctx, painter, n, ctx.sprite_cache.get())


# --- (d) clip correctness on a folded path --------------------------

def test_clip_preserves_fold_order_and_cuts_at_edges():
    # A fold: y runs 100 -> 400 -> 100. Clip to [150, 350]: each arm is
    # cut at both edges, and PATH ORDER survives (down-arm then up-arm),
    # unlike a naive y-sort which would interleave them.
    xs = np.array([40.0, 40.0, 40.0])
    ys = np.array([100.0, 400.0, 100.0])
    out = nl._clip_body_samples(xs, ys, top=150.0, bot=350.0)
    assert out is not None
    _cx, cy = out
    # Down-arm enters at 150, exits at 350; up-arm enters at 350, exits 150.
    assert list(cy) == pytest.approx([150.0, 350.0, 350.0, 150.0])


def test_clip_returns_none_when_fully_outside():
    xs = np.array([40.0, 40.0])
    ys = np.array([500.0, 600.0])
    assert nl._clip_body_samples(xs, ys, top=100.0, bot=200.0) is None


def test_clip_straight_body_unchanged_inside_window():
    xs = np.array([40.0, 40.0, 40.0])
    ys = np.array([100.0, 200.0, 300.0])
    out = nl._clip_body_samples(xs, ys, top=50.0, bot=350.0)
    assert out is not None
    _cx, cy = out
    assert list(cy) == pytest.approx([100.0, 200.0, 300.0])


# --- fold-aware body span clip window -------------------------------

def test_body_span_window_is_path_extent_for_fold():
    ctx = _ctx()
    xs = np.array([40.0, 40.0, 40.0])
    ys = np.array([100.0, 420.0, 260.0])     # deepest point is the fold apex
    n = _ln_view(body_path=(xs, ys), state='held')
    span = nl._ln_body_span(ctx, n, hide=False)
    assert span is not None
    top, bot, state = span
    assert top == pytest.approx(100.0)
    assert bot == pytest.approx(420.0)       # includes the apex past the tail
    assert state == 'normal'


# --- straight-span orientation + consumed holds ---------------------

def test_upscroll_straight_body_spans_head_to_tail():
    # NotITG's mirrored field scrolls up: head near the receptor at the
    # TOP (small y), tail deeper below. The window is ordered min/max, so
    # the body draws in both scroll orientations.
    ctx = _ctx()
    n = _ln_view(state='upcoming', y=115.0, y_end=300.0)
    assert nl._ln_body_span(ctx, n, hide=False) == (115.0, 300.0, 'normal')


def test_held_upscroll_body_runs_receptor_to_tail():
    # The consumer pins a held head at the receptor upstream; the body
    # window is then exactly receptor -> tail (nothing past the receptor).
    ctx = _ctx()
    n = _ln_view(state='held', y=115.0, y_end=300.0)
    assert nl._ln_body_span(ctx, n, hide=False) == (115.0, 300.0, 'normal')


def test_held_hide_clamps_at_judge_without_crossing_tail():
    ctx = _ctx(judge_y=500)
    n = _ln_view(state='held', y=560.0, y_end=300.0)
    assert nl._ln_body_span(ctx, n, hide=True) == (300.0, 500.0, 'normal')
    # Once the tail crosses the judge line the window degenerates: the
    # clamp never extends the body past its own tail.
    n = _ln_view(state='held', y=560.0, y_end=520.0)
    top, bot, _state = nl._ln_body_span(ctx, n, hide=True)
    assert top == bot


def test_display_judge_follows_receptor_dy():
    ctx = _ctx(judge_y=500)
    assert nl._display_judge_y(ctx, 0) == 500.0
    ctx.receptor_offsets = {'dy': np.array([-385.0, 0.0])}
    assert nl._display_judge_y(ctx, 0) == 115.0


def test_released_hold_without_release_data_draws_nothing():
    # Autoplay streams carry no release offsets: a hold is fully consumed
    # at its tail -- no body slab, no tail cap drifting past the receptor.
    ctx = _ctx()
    n = _ln_view(state='released', rel_off=None, y=115.0, y_end=90.0)
    assert nl._ln_body_span(ctx, n, hide=False) is None
    painter = _RecordPainter()
    nl._draw_ln(ctx, painter, n)
    assert not painter.tiled and not painter.paths and not painter.pixmaps


def test_released_hold_with_release_data_keeps_guide_window():
    ctx = _ctx(judge_y=500)
    n = _ln_view(state='released', rel_off=-0.05, y_end=420.0)
    assert nl._ln_body_span(ctx, n, hide=False) == (420.0, 500, 'released')


def test_ribbon_width_matches_body_sprite_strip():
    # The stroked ribbon and the rect tile are the SAME noodle: a body
    # flipping between producers (constant-dx frames skip the polyline)
    # must not change thickness. Vertical path => bounding width is the
    # stroke width, which is the sprite strip's, not the full lane's.
    from analysis.player.render.layers.note_sprites import ln_body_width
    ctx = _ctx(lane_w=80)
    painter = _RecordPainter()
    xs = np.full(3, 40.0)
    ys = np.array([100.0, 200.0, 300.0])
    n = _ln_view(body_path=(xs, ys), state='held')
    nl._draw_ln(ctx, painter, n)
    assert len(painter.paths) == 1
    assert painter.paths[0].boundingRect().width() == pytest.approx(
        ln_body_width('bar', 80))
