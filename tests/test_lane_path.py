"""The lane curve everything in a lane is a sample of.

The gate is that the DEGENERATE case (a straight lane, which is most games)
and the displaced one differ only by the hook, and that the derivations the
note pipeline does by hand today - a body span whose ends stay attached, a
cap that follows the curve rather than the lane - fall out of the same
object.
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.player.render import lane_path


_LANE_W = 64.0
_JUDGE_Y = 400.0


def _lane_center(col):
    return col * _LANE_W + _LANE_W / 2.0


def _note_y(y_offset):
    return _JUDGE_Y - np.asarray(y_offset, dtype=np.float64)


def _straight():
    return lane_path.straight(_lane_center, _note_y)


def test_a_straight_lane_puts_the_receptor_on_the_hit_line():
    x, y, rot, zoom = _straight().at(2, 0.0)
    assert (x, y) == (_lane_center(2), _JUDGE_Y)
    assert (rot, zoom) == (0.0, 1.0)


def test_a_straight_lane_is_the_degenerate_displaced_one():
    """Most games' lanes are this: the same object with no hook."""
    zero = lane_path.LanePath(
        _lane_center, _note_y,
        displace=lambda cols, ys: (np.zeros(len(cols)), np.zeros(len(cols)),
                                   np.zeros(len(cols)), np.ones(len(cols))))
    cols, offsets = (0, 1, 3), (0.0, 120.0, 300.0)
    plain = _straight().sample(cols, offsets)
    hooked = zero.sample(cols, offsets)
    assert plain.x == pytest.approx(hooked.x)
    assert plain.y == pytest.approx(hooked.y)
    assert plain.zoom == pytest.approx(hooked.zoom)


def test_displacement_rides_on_top_of_the_lane():
    # dx bends the lane sideways by the offset; zoom MULTIPLIES rather than
    # replaces, so a game hook cannot silently discard the lane's own scale.
    def bend(cols, ys):
        return (ys * 0.5, np.zeros(len(cols)),
                np.full(len(cols), 30.0), np.full(len(cols), 2.0))

    path = lane_path.LanePath(_lane_center, _note_y, displace=bend)
    x, y, rot, zoom = path.at(0, 100.0)
    assert x == pytest.approx(_lane_center(0) + 50.0)
    assert y == pytest.approx(_JUDGE_Y - 100.0)
    assert (rot, zoom) == (30.0, 2.0)


def test_a_body_span_keeps_its_ends_attached():
    """A hold body is the span between the head and tail offsets, and its
    endpoints must land exactly where those were placed independently -
    otherwise the ribbon detaches from its own cap."""
    path = _straight()
    span = path.between(1, 40.0, 300.0, samples=9)
    assert len(span) == 9
    assert span.at(0) == pytest.approx(path.at(1, 40.0))
    assert span.at(-1) == pytest.approx(path.at(1, 300.0))


def test_a_span_of_one_still_has_two_ends():
    # A degenerate request must not produce a zero-length ribbon.
    assert len(_straight().between(0, 10.0, 20.0, samples=1)) == 2


def test_the_cap_follows_the_curve_not_the_lane():
    """A folded hold's end segment runs back UP the lane; the tangent is
    what flips the cap, with no special case for the fold."""
    def fold(cols, ys):
        # The lane already travels UP (`note_y` subtracts the offset), so a
        # fold has to OUTRUN that: past offset 200 the displacement adds y
        # faster than the lane removes it, and the curve doubles back.
        ys = np.asarray(ys, dtype=np.float64)
        dy = np.where(ys > 200.0, 3.0 * (ys - 200.0), 0.0)
        return (np.zeros(len(cols)), dy, np.zeros(len(cols)),
                np.ones(len(cols)))

    path = lane_path.LanePath(_lane_center, _note_y, displace=fold)
    rising = path.between(0, 0.0, 100.0, samples=3)
    folded = path.between(0, 250.0, 350.0, samples=3)
    # Screen y grows downward, so an arrow travelling UP the lane heads one
    # way and the folded tail the other.
    assert rising.tangent_deg(0, -1) * folded.tangent_deg(0, -1) < 0.0


def test_a_still_point_has_no_heading():
    path = _straight()
    span = path.between(0, 50.0, 50.0, samples=4)
    assert span.tangent_deg(0, -1) == 0.0
