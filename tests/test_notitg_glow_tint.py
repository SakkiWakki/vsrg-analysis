"""Per-note glow COLOR (stealthglow rgb companions), end to end.

The engine hides a stealthglowed note's fill and re-renders it as glow
tinted by the stealthglowred/green/blue companion channels (GetRedDiff/
GetGreenDiff/GetBlueDiff, additive per-column + global). Covers the
consumer stash (candidate_glow_rgb -> view.glow_rgb), the draw bracket
(the additive glow pass tints its sprite; the fill pass never does),
the tinted-pixmap cache, and the rest identity (no rgb channels driven
-> the exact untinted pixmap objects of today).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap

from analysis.games.notitg.mod_channels import compile_mod_channels
from analysis.games.notitg.note_mods import NotitgNoteMods
from analysis.player.render.layers import notes as _notes
from analysis.player.render.layers.notes import _NoteView


class _RecordCache:
    def __init__(self, size=8):
        self.pm = QPixmap(size, size)
        self.pm.fill(Qt.white)

    def get(self, name, ctx, **kw):
        return self.pm


class _RecordPainter:
    """Records (pixmap, opacity, composition) per blit."""

    def __init__(self):
        self.blits = []
        self._opacity = 1.0
        self._composition = 'normal'
        self._stack = []

    def drawPixmap(self, point, pm):
        self.blits.append((pm, self._opacity, self._composition))

    def opacity(self):
        return self._opacity

    def setOpacity(self, value):
        self._opacity = float(value)

    def setCompositionMode(self, mode):
        self._composition = mode

    def save(self):
        self._stack.append((self._opacity, self._composition))

    def restore(self):
        self._opacity, self._composition = self._stack.pop()


def _tap_view(**overrides):
    kwargs = dict(
        i=0, col=0, y=50.0, y_end=0, press_y=50.0, lx=100,
        off=0.0, press_t=99.0, release_t=None, rel_off=None, end_t=None,
        is_ln=False, is_roll=False, miss=False, state='upcoming',
        note_color=(255, 255, 255), jcolor=(255, 0, 0),
        alpha=0.0, glow=0.5)
    kwargs.update(overrides)
    return _NoteView(**kwargs)


def _draw_ctx():
    player = SimpleNamespace(press_hide=False, H=400)
    return SimpleNamespace(player=player, t_now=0.0, screen_margin=80,
                           lane_w=64.0, lane_width=lambda col: 64.0,
                           sprite_cache=_RecordCache())


# -- consumer stash --------------------------------------------------

def _apply(modstring, ctx):
    channels = compile_mod_channels(
        [{'t_start': 0.0, 't_end': 100.0, 'modstring': modstring,
          'player': None}])
    NotitgNoteMods(channels, [(0.0, 120.0)]).apply(ctx)


def _tap_ctx():
    notes = SimpleNamespace(noterows_list=[144], columns_list=[0],
                            ln_tail_times=np.full(1, np.nan))
    player = SimpleNamespace(
        keycount=4, columns=np.array([0], dtype=np.int64),
        times=np.array([3.0]), offsets=np.array([0.0]),
        misses=np.array([False], dtype=bool), notes=notes,
        palette=[(255, 255, 255)] * 4,
        judge_colors={None: (0, 255, 0)}, note_judges=[None],
        hold_release_offsets={}, press_hide=False)
    return SimpleNamespace(
        player=player, t_now=0.0, lane_w=64.0, judge_y=100,
        chart_rect=(0.0, 0.0, 400.0, 400.0),
        candidates=[0],
        candidate_head_y=np.full(1, 40.0),
        candidate_tail_y=np.full(1, np.nan),
        candidate_press_y=np.full(1, 40.0),
        lane_x=lambda col: 100.0 + col * 64.0,
        lane_width=lambda col: 64.0)


def test_rgb_channels_reach_the_view():
    ctx = _tap_ctx()
    _apply('*-1 stealthglow|0|1|0', ctx)
    assert ctx.candidate_glow_rgb is not None

    _notes.prepare(ctx)
    (view,) = ctx.note_views
    assert view.glow > 0.0
    assert view.glow_rgb == pytest.approx((0.0, 1.0, 0.0))


def test_no_rgb_channels_leave_view_untinted():
    ctx = _tap_ctx()
    _apply('*-1 stealthglow', ctx)
    _notes.prepare(ctx)
    (view,) = ctx.note_views
    assert view.glow > 0.0
    assert view.glow_rgb is None


# -- draw bracket ----------------------------------------------------

def test_glow_pass_blits_a_tinted_sprite():
    ctx = _draw_ctx()
    painter = _RecordPainter()
    view = _tap_view(glow_rgb=(1.0, 0.2, 0.9))
    _notes._draw_view(ctx, painter, view, _notes._draw_replay_note)

    # Only the additive glow pass drew (fill alpha 0), with a TINTED
    # copy of the sprite, and the tint never leaks past the bracket.
    (pm, opacity, composition), = painter.blits
    assert composition == QPainter.CompositionMode_Plus
    assert opacity == pytest.approx(0.5)
    assert pm is not ctx.sprite_cache.pm
    img = pm.toImage()
    color = img.pixelColor(4, 4)
    assert color.red() == 255
    assert color.green() < 100
    assert 200 < color.blue()
    assert color.alpha() == 255
    assert getattr(ctx, 'glow_tint', None) is None


def test_untinted_glow_blits_the_original_sprite():
    ctx = _draw_ctx()
    painter = _RecordPainter()
    _notes._draw_view(ctx, painter, _tap_view(), _notes._draw_replay_note)
    (pm, _opacity, composition), = painter.blits
    assert composition == QPainter.CompositionMode_Plus
    assert pm is ctx.sprite_cache.pm


def test_fill_pass_stays_untinted_under_partial_stealth():
    # A half-faded stealthglow note draws fill + glow; only the glow
    # pass carries the tint.
    ctx = _draw_ctx()
    painter = _RecordPainter()
    view = _tap_view(alpha=0.5, glow_rgb=(0.0, 1.0, 0.0))
    _notes._draw_view(ctx, painter, view, _notes._draw_replay_note)
    (fill_pm, _o1, fill_comp), (glow_pm, _o2, glow_comp) = painter.blits
    assert fill_comp == 'normal'
    assert fill_pm is ctx.sprite_cache.pm
    assert glow_comp == QPainter.CompositionMode_Plus
    assert glow_pm is not ctx.sprite_cache.pm


# -- tinted-pixmap cache ---------------------------------------------

def test_tinted_pixmap_preserves_alpha_and_caches():
    pm = QPixmap(4, 4)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.fillRect(0, 0, 4, 2, Qt.white)
    painter.end()

    tinted = _notes._glow_tinted(pm, (1.0, 0.0, 0.0))
    img = tinted.toImage()
    assert img.pixelColor(1, 0).red() == 255
    assert img.pixelColor(1, 0).green() == 0
    assert img.pixelColor(1, 0).alpha() == 255
    assert img.pixelColor(1, 3).alpha() == 0

    # Same sprite + rgb within one quantization step -> the cached copy.
    again = _notes._glow_tinted(pm, (1.0, 0.001, 0.001))
    assert again is tinted
    other = _notes._glow_tinted(pm, (0.0, 1.0, 0.0))
    assert other is not tinted
