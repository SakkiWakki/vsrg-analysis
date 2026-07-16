"""Effects framework: compositor, timeline sampling, transform math."""
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QTransform

from analysis.player.render.effects import composite
from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.timeline import EventTimeline, Keyframe
from analysis.player.render.effects.playfield_transform import (
    PlayfieldTransformEffect)


class _Effect:
    def __init__(self, frame):
        self._frame = frame

    def at(self, ctx):
        return self._frame


def _ctx(t=0.0):
    player = SimpleNamespace(keycount=4)
    return SimpleNamespace(t_now=t, x0=100.0, lane_w=40.0, judge_y=600,
                           player=player)


def test_composite_identity_when_no_effects():
    assert composite([], _ctx()).is_identity


def test_composite_skips_inactive_effects():
    frame = composite([_Effect(None), _Effect(None)], _ctx())
    assert frame.is_identity


def test_begin_effect_transform_confined_to_chart_rect():
    # Opacity + transform apply only inside the effect bracket; the HUD
    # paints outside it, so an invisibility effect can never hide the
    # sidebar. Assert the bracket clips to chart_rect and is restorable.
    from analysis.player.render.qt_renderer import QtPlayerRenderer
    from analysis.player.render.effects.base import CompositeFrame

    class FakePainter:
        def __init__(self):
            self.saved = 0
            self.clip = None
            self.opacity = 1.0
        def save(self): self.saved += 1
        def restore(self): self.saved -= 1
        def setClipRect(self, r): self.clip = r
        def setOpacity(self, o): self.opacity = o
        def setTransform(self, t, combine): pass

    ctx = _ctx_win()
    p = FakePainter()
    frame = CompositeFrame(transform=None, opacity=0.0)
    wrapped = QtPlayerRenderer._begin_effect_transform(frame, p, ctx)
    assert wrapped and p.saved == 1
    assert p.opacity == 0.0
    assert (p.clip.width(), p.clip.height()) == pytest.approx(
        (ctx.chart_rect[2], ctx.chart_rect[3]))
    QtPlayerRenderer._end_effect_transform(p)
    assert p.saved == 0   # opacity/clip fully unwound before HUD paints


def test_composite_multiplies_opacity_and_composes_transforms():
    a = QTransform().translate(10, 0)
    b = QTransform().scale(2, 2)
    frame = composite([_Effect(EffectFrame(transform=a, opacity=0.5)),
                       _Effect(EffectFrame(transform=b, opacity=0.5))],
                      _ctx())
    assert frame.opacity == pytest.approx(0.25)
    assert frame.transform is not None
    assert not frame.is_identity


def test_composite_splits_draws_by_z():
    below = (-1, lambda c, p: None)
    above = (2, lambda c, p: None)
    mid = (0, lambda c, p: None)
    frame = composite([_Effect(EffectFrame(draws=(above, below, mid)))],
                      _ctx())
    assert [z for z, _ in frame.below] == [-1]
    assert [z for z, _ in frame.above] == [0, 2]


# ── timeline ----------------------------------------------------------

def test_timeline_rest_before_first_keyframe():
    tl = EventTimeline([Keyframe(5.0, (10.0,), 0.0, 0)], rest=(0.0,))
    assert tl.sample(1.0) == (0.0,)


def test_timeline_eases_then_holds():
    # linear (easing id 0 = None) ease from 0 -> 10 over 2s at t=5.
    tl = EventTimeline([Keyframe(5.0, (10.0,), 2.0, 0)], rest=(0.0,))
    assert tl.sample(6.0) == pytest.approx((5.0,))    # halfway
    assert tl.sample(8.0) == pytest.approx((10.0,))   # settled
    assert tl.sample(99.0) == pytest.approx((10.0,))  # holds


# ── playfield transform ----------------------------------------------

def _ctx_win(t=0.0, W=1366, H=768):
    player = SimpleNamespace(keycount=4, W=W, H=H)
    return SimpleNamespace(t_now=t, x0=100.0, lane_w=40.0, judge_y=600,
                           player=player, chart_rect=(0, 0, W, H))


def test_playfield_rotate_is_about_field_center():
    eff = PlayfieldTransformEffect(
        rotate=[{'time': 0, 'roll': 90.0, 'duration': 0, 'ease': 0}])
    ctx = _ctx_win(t=1.0)
    frame = eff.at(ctx)
    cx = ctx.x0 + ctx.player.keycount * ctx.lane_w / 2   # 180
    cy = ctx.judge_y                                     # 600
    mapped = frame.transform.map(cx, cy)
    assert mapped[0] == pytest.approx(cx)   # receptor is fixed under rotation
    assert mapped[1] == pytest.approx(cy)


def test_playfield_z_depth_zooms_about_screen_center():
    # z = +100 halves the perspective scale (100/(100+100)); the field
    # shrinks about screen center.
    eff = PlayfieldTransformEffect(
        move=[{'time': 0, 'x': 0, 'y': 0, 'z': 100.0,
               'duration': 0, 'ease': 0}])
    ctx = _ctx_win(t=1.0)
    frame = eff.at(ctx)
    scx, scy = ctx.player.W / 2, ctx.player.H / 2
    assert frame.transform.map(scx, scy) == pytest.approx((scx, scy))
    m11 = frame.transform.m11()
    assert m11 == pytest.approx(0.5)


def test_playfield_move_scales_to_window():
    # x = ref width -> full-window shift, clamped to _MAX_OFFSET_FRAC.
    eff = PlayfieldTransformEffect(
        move=[{'time': 0, 'x': 1366.0, 'y': 0, 'z': 0,
               'duration': 0, 'ease': 0}])
    ctx = _ctx_win(t=1.0, W=1366, H=768)
    frame = eff.at(ctx)
    # raw dx would be 1366px; clamped to 0.5 * W = 683.
    assert frame.transform.dx() == pytest.approx(683.0)


def test_playfield_inactive_returns_none():
    eff = PlayfieldTransformEffect(move=[], scale=[], rotate=[])
    assert eff.at(_ctx()) is None
    assert not eff


def test_playfield_move_scales_with_chart_rect_width():
    # The same authored move shifts proportionally more in a wider
    # viewport -- effects reference the playfield/chart region size.
    eff = PlayfieldTransformEffect(
        move=[{'time': 0, 'x': 100.0, 'y': 0, 'z': 0,
               'duration': 0, 'ease': 0}])
    narrow = _ctx_win(t=1.0, W=800, H=600)
    wide = _ctx_win(t=1.0, W=1600, H=600)
    assert (eff.at(wide).transform.dx()
            > eff.at(narrow).transform.dx())
