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

def test_playfield_rotate_is_about_field_center():
    eff = PlayfieldTransformEffect(
        rotate=[{'time': 0, 'roll': 90.0, 'duration': 0, 'ease': 0}])
    ctx = _ctx(t=1.0)
    frame = eff.at(ctx)
    cx = ctx.x0 + ctx.player.keycount * ctx.lane_w / 2   # 180
    cy = ctx.judge_y                                     # 600
    mapped = frame.transform.map(cx, cy)
    assert mapped[0] == pytest.approx(cx)   # center is fixed under rotation
    assert mapped[1] == pytest.approx(cy)


def test_playfield_inactive_returns_none():
    eff = PlayfieldTransformEffect(move=[], scale=[], rotate=[])
    assert eff.at(_ctx()) is None
    assert not eff
