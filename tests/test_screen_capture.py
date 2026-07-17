"""Renderer 'screen'-scope field copies (previous-frame chart-area
composite, SM ActorFrameTexture feedback semantics - scoping items 64/74).

A 'screen' copy blits the PREVIOUS frame's whole chart-area composite.
The renderer composites the chart region offscreen when any 'screen' copy
is live, retains it as `_prev_screen` for the next frame, and invalidates
that retention on a seek. These tests drive the composite lifecycle
directly with real QPixmap/QPainter and a minimal ctx - no full player."""
from types import SimpleNamespace

import pytest

pytest.importorskip('PySide6')

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap

from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.qt_renderer import QtPlayerRenderer, _SEEK_GAP_S


def _renderer():
    return QtPlayerRenderer(plugin_manager=SimpleNamespace())


def _ctx(t):
    """Minimal ctx: a player with W/H (for capture-pixmap sizing) and a
    chart_rect (the clip/blit region)."""
    player = SimpleNamespace(W=200, H=150)
    return SimpleNamespace(t_now=float(t), player=player,
                           chart_rect=(0, 0, 160, 150))


def _host():
    """A real QImage-backed painter to stand in for the widget target."""
    image = QImage(200, 150, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.black)
    return image, QPainter(image)


def _fields(*scopes):
    return EffectFrame(fields=tuple((None, 1.0, s) for s in scopes))


# -- scope detection ------------------------------------------------------

def test_has_screen_copy_detects_only_screen_scope():
    assert QtPlayerRenderer._has_screen_copy(_fields('screen'))
    assert QtPlayerRenderer._has_screen_copy(_fields('field', 'screen'))
    assert not QtPlayerRenderer._has_screen_copy(_fields('field', 'full'))
    assert not QtPlayerRenderer._has_screen_copy(EffectFrame())


# -- composite lifecycle --------------------------------------------------

def test_screen_composite_retains_pixmap_for_next_frame():
    r = _renderer()
    ctx = _ctx(1.0)
    image, painter = _host()
    r._sync_prev_screen(ctx)
    target = r._begin_screen_composite(_fields('screen'), ctx, painter)
    assert target is not None and target is not painter
    # Paint something recognizable into the composite, then close it.
    target.fillRect(0, 0, 160, 150, QColor(10, 200, 30))
    r._end_screen_composite(painter, ctx)
    painter.end()
    # The composite is retained with its chart time.
    assert r._prev_screen is not None
    assert r._prev_screen_t == pytest.approx(1.0)
    # And it was blitted to the host target.
    assert QColor(image.pixel(80, 75)).green() > 150


def test_no_screen_copies_leaves_prev_screen_untouched():
    r = _renderer()
    ctx = _ctx(1.0)
    image, painter = _host()
    r._sync_prev_screen(ctx)
    target = r._begin_screen_composite(_fields('field', 'full'), ctx, painter)
    painter.end()
    assert target is None
    assert r._prev_screen is None
    assert r._prev_screen_t is None


# -- seek invalidation ----------------------------------------------------

def test_smooth_advance_keeps_prev_screen():
    r = _renderer()
    r._prev_screen = QPixmap(4, 4)
    r._prev_screen_t = 10.0
    # A small forward step (one frame) is smooth playback.
    r._sync_prev_screen(_ctx(10.0 + _SEEK_GAP_S / 2))
    assert r._prev_screen is not None


def test_forward_seek_invalidates_prev_screen():
    r = _renderer()
    r._prev_screen = QPixmap(4, 4)
    r._prev_screen_t = 10.0
    r._sync_prev_screen(_ctx(10.0 + _SEEK_GAP_S + 1.0))
    assert r._prev_screen is None
    assert r._prev_screen_t is None


def test_backward_seek_invalidates_prev_screen():
    r = _renderer()
    r._prev_screen = QPixmap(4, 4)
    r._prev_screen_t = 10.0
    r._sync_prev_screen(_ctx(9.5))
    assert r._prev_screen is None


# -- blit: first-frame skip + feedback blit -------------------------------

def test_screen_copy_skipped_on_first_frame_then_appears():
    """No retained composite (first frame / just after a seek): the screen
    copy is skipped for one frame, but this frame's composite is still
    built and retained, so the copy reappears next frame."""
    r = _renderer()
    ctx = _ctx(1.0)
    image, painter = _host()
    frame = _fields('screen')
    # Simulate the draw() sequence: sync, begin composite, (chart draws
    # happen here incl. the field blit), end composite.
    r._sync_prev_screen(ctx)
    target = r._begin_screen_composite(frame, ctx, painter)
    # The field blit runs into the composite target; _prev_screen is None,
    # so the screen copy is skipped this frame (no crash, nothing drawn).
    r._field_pixmap = None
    r._blit_field_instances(frame, ctx, target)
    target.fillRect(0, 0, 160, 150, QColor(0, 0, 120))
    r._end_screen_composite(painter, ctx)
    painter.end()
    prev = r._prev_screen
    assert prev is not None  # retained despite the skipped copy

    # Next frame: _prev_screen is present, so the screen copy blits it.
    ctx2 = _ctx(1.008)
    image2, painter2 = _host()
    r._sync_prev_screen(ctx2)
    target2 = r._begin_screen_composite(frame, ctx2, painter2)
    r._blit_field_instances(frame, ctx2, target2)
    r._end_screen_composite(painter2, ctx2)
    painter2.end()
    # The previous blue composite fed forward into this frame's composite.
    assert QColor(image2.pixel(80, 75)).blue() > 80
