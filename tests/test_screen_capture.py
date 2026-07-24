"""Renderer screen-scope field copies (SM ActorFrameTexture capture
semantics - scoping items 64/74).

The AFT node captures the chart area at its draw position each frame:
the renderer snapshots the in-progress offscreen composite during the
instance blits. A 'screen' copy (sampler drawn after the node) blits
THIS frame's snapshot; a 'screen_prev' copy (drawn before it) blits the
previous frame's retained one - the one-frame feedback leg. The final
composite is never retained: it includes the screen blits themselves,
and feeding it back makes an identity opaque sampler a fixed point that
freezes the chart area. These tests drive the composite lifecycle
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


def _pixel(pixmap, x, y):
    return QColor(pixmap.toImage().pixel(x, y))


# -- scope detection ------------------------------------------------------

def test_has_screen_copy_detects_both_screen_scopes():
    assert QtPlayerRenderer._has_screen_copy(_fields('screen'))
    assert QtPlayerRenderer._has_screen_copy(_fields('screen_prev'))
    assert QtPlayerRenderer._has_screen_copy(_fields('field', 'screen'))
    assert not QtPlayerRenderer._has_screen_copy(_fields('field', 'full'))
    assert not QtPlayerRenderer._has_screen_copy(EffectFrame())


# -- composite lifecycle --------------------------------------------------

def test_capture_taken_at_blit_and_retained():
    """The node capture snapshots the composite at the screen blit, so
    content painted AFTER the blit (post-node draws) never enters it."""
    r = _renderer()
    ctx = _ctx(1.0)
    image, painter = _host()
    frame = _fields('screen')
    r._sync_prev_screen(ctx)
    target = r._begin_screen_composite(frame, ctx, painter)
    assert target is not None and target is not painter
    target.fillRect(0, 0, 160, 150, QColor(10, 200, 30))
    r._field_pixmap = None
    r._blit_field_instances(frame, ctx, target)
    target.fillRect(0, 0, 160, 150, QColor(200, 10, 10))
    r._end_screen_composite(painter, ctx)
    painter.end()
    # Retained with its chart time; pre-blit green, not post-blit red.
    assert r._prev_screen is not None
    assert r._prev_screen_t == pytest.approx(1.0)
    assert _pixel(r._prev_screen, 80, 75).green() > 150
    # The host got the full composite (red painted last).
    assert QColor(image.pixel(80, 75)).red() > 150


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


# -- blit: same-frame 'screen', one-frame 'screen_prev' feedback ----------

def test_screen_copy_blits_this_frames_capture():
    """A 'screen' (post-node) sampler shows same-frame content: on the
    very first frame - no retention yet - it already blits what the
    composite holds at its draw position."""
    r = _renderer()
    ctx = _ctx(1.0)
    image, painter = _host()
    frame = _fields('screen')
    r._sync_prev_screen(ctx)
    target = r._begin_screen_composite(frame, ctx, painter)
    target.fillRect(0, 0, 160, 150, QColor(10, 200, 30))
    r._field_pixmap = None
    r._blit_field_instances(frame, ctx, target)
    r._end_screen_composite(painter, ctx)
    painter.end()
    assert QColor(image.pixel(80, 75)).green() > 150


def test_screen_prev_skipped_first_frame_then_feeds_back():
    """A 'screen_prev' (pre-node) sampler skips the first frame (nothing
    retained), but the node still captures, so next frame it blits the
    previous frame's content - the trail feedback leg."""
    r = _renderer()
    ctx = _ctx(1.0)
    image, painter = _host()
    frame = _fields('screen_prev')
    r._sync_prev_screen(ctx)
    target = r._begin_screen_composite(frame, ctx, painter)
    target.fillRect(0, 0, 160, 150, QColor(0, 0, 120))
    r._field_pixmap = None
    r._blit_field_instances(frame, ctx, target)
    r._end_screen_composite(painter, ctx)
    painter.end()
    assert r._prev_screen is not None  # retained despite the skipped copy

    # Next frame: nothing painted directly, the copy alone carries the
    # previous frame's blue into this frame's composite.
    ctx2 = _ctx(1.008)
    image2, painter2 = _host()
    r._sync_prev_screen(ctx2)
    target2 = r._begin_screen_composite(frame, ctx2, painter2)
    r._blit_field_instances(frame, ctx2, target2)
    r._end_screen_composite(painter2, ctx2)
    painter2.end()
    assert QColor(image2.pixel(80, 75)).blue() > 80


# -- crop on instance blits (AFT sampler croptop/... pokes) ---------------

def _crop_blit_image(crop):
    """One 'screen' copy shifted down 60px, with `crop` fractions as the
    entry's 5th element (None = today's uncropped entry). The source
    content is a green strip across the top 20 rows; the copy makes the
    strip land at y 60..80, cropped to the un-hidden columns."""
    from PySide6.QtGui import QTransform

    r = _renderer()
    ctx = _ctx(1.0)
    image, painter = _host()
    entry = (QTransform.fromTranslate(0, 60), 1.0, 'screen', None, crop)
    frame = EffectFrame(fields=(entry,))
    r._sync_prev_screen(ctx)
    target = r._begin_screen_composite(frame, ctx, painter)
    target.fillRect(0, 0, 160, 20, QColor(10, 200, 30))
    r._field_pixmap = None
    r._blit_field_instances(frame, ctx, target)
    r._end_screen_composite(painter, ctx)
    painter.end()
    return image


def test_cropleft_clips_the_copys_source_half():
    image = _crop_blit_image((0.5, 0.0, 0.0, 0.0))
    # Right half of the strip copied (source x >= 80), left half not.
    assert QColor(image.pixel(120, 70)).green() > 150
    assert QColor(image.pixel(40, 70)).green() < 50


def test_rest_crop_none_keeps_the_full_copy():
    image = _crop_blit_image(None)
    assert QColor(image.pixel(120, 70)).green() > 150
    assert QColor(image.pixel(40, 70)).green() > 150


def test_croptop_clips_in_source_space():
    # The whole strip is inside the hidden top 20% of the source, so a
    # croptop crop leaves the copy region untouched.
    image = _crop_blit_image((0.0, 0.2, 0.0, 0.0))
    assert QColor(image.pixel(120, 70)).green() < 50
    assert QColor(image.pixel(40, 70)).green() < 50


def _fill_image(crop):
    """One curtain-fill entry over the chart region, with `crop`
    fractions as the entry's 5th element."""
    r = _renderer()
    ctx = _ctx(1.0)
    image, painter = _host()
    entry = (None, 1.0, 'fill', (0.0, 1.0, 0.0), crop)
    r._blit_field_instances(EffectFrame(fields=(entry,)), ctx, painter)
    painter.end()
    return image


def test_fill_crop_insets_curtain_band():
    # chart_rect (0, 0, 160, 150): the crop leaves the band x 40..160,
    # y 0..75 filled; the hidden left/bottom fractions never draw.
    image = _fill_image((0.25, 0.0, 0.0, 0.5))
    assert QColor(image.pixel(100, 40)).green() > 150
    assert QColor(image.pixel(20, 40)).green() < 50
    assert QColor(image.pixel(100, 100)).green() < 50


def test_fill_full_crop_draws_nothing():
    image = _fill_image((0.6, 0.0, 0.6, 0.0))
    assert QColor(image.pixel(80, 75)).green() < 50


def test_fill_rest_crop_none_covers_region():
    image = _fill_image(None)
    for x, y in ((100, 40), (20, 40), (100, 100)):
        assert QColor(image.pixel(x, y)).green() > 150
