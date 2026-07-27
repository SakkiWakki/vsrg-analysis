"""The lane curve everything in a lane is a sample of.

The gate is that the DEGENERATE case (a straight lane, which is most games)
and the displaced one differ only by the hook, and that the derivations the
note pipeline used to do by hand - a body span whose ends stay attached, a
cap that follows the curve rather than the lane, a coarse span that must
not invent shapes it cannot resolve - fall out of the same object.
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


def _bend(dx=None, dy=None, rot=None, zoom=None, z=None, alpha=None):
    """A displace hook from whichever terms the test cares about; the rest
    rest at their identity."""
    def displace(cols, offsets, note_beats, cell):
        n = len(cols)

        def term(fn, rest):
            return (np.full(n, rest, dtype=np.float64) if fn is None
                    else np.asarray(fn(cols, offsets, note_beats, cell),
                                    dtype=np.float64))

        return (term(dx, 0.0), term(dy, 0.0), term(rot, 0.0),
                term(zoom, 1.0), term(z, 0.0), term(alpha, 1.0))

    return lane_path.LanePath(_lane_center, lambda cols, offs: _note_y(offs),
                              displace=displace)


def test_a_straight_lane_puts_the_receptor_on_the_hit_line():
    sample = _straight().at(2, 0.0)
    assert (sample.x, sample.y) == (_lane_center(2), _JUDGE_Y)
    assert (sample.rotation_deg, sample.zoom, sample.z) == (0.0, 1.0, 0.0)
    assert sample.alpha == 1.0


def test_a_straight_lane_is_the_degenerate_displaced_one():
    """Most games' lanes are this: the same object with no hook."""
    cols, offsets = (0, 1, 3), (0.0, 120.0, 300.0)
    plain = _straight().sample(cols, offsets)
    hooked = _bend().sample(cols, offsets)
    assert plain.x == pytest.approx(hooked.x)
    assert plain.y == pytest.approx(hooked.y)
    assert plain.zoom == pytest.approx(hooked.zoom)


def test_a_straight_lane_does_not_claim_to_displace():
    assert not _straight().displaces
    assert _bend().displaces


def test_displacement_rides_on_top_of_the_lane():
    # dx bends the lane sideways by the offset; the turn terms REPLACE (the
    # lane itself has no turn of its own to compose with).
    path = _bend(dx=lambda cols, offs, beats, cell: offs * 0.5,
                 rot=lambda cols, offs, beats, cell: np.full(len(cols), 30.0),
                 zoom=lambda cols, offs, beats, cell: np.full(len(cols), 2.0))
    sample = path.at(0, 100.0)
    assert sample.x == pytest.approx(_lane_center(0) + 50.0)
    assert sample.y == pytest.approx(_JUDGE_Y - 100.0)
    assert (sample.rotation_deg, sample.zoom) == (30.0, 2.0)


def test_the_scroll_remap_reaches_both_hooks():
    """The axis reshape (NotITG's accel family) must not be something the
    game applies twice or in one place only: `note_y` and `displace` have
    to answer for the SAME offset or the bend detaches from the axis."""
    seen = []

    def displace(cols, offsets, note_beats, cell):
        seen.append(np.asarray(offsets, dtype=np.float64))
        zeros = np.zeros(len(cols))
        return (zeros, zeros, zeros, np.ones(len(cols)), zeros,
                np.ones(len(cols)))

    path = lane_path.LanePath(_lane_center, lambda cols, offs: _note_y(offs),
                              displace=displace,
                              scroll_remap=lambda offs: offs * 2.0)
    assert path.at(0, 50.0).y == pytest.approx(_JUDGE_Y - 100.0)
    assert seen[0] == pytest.approx([100.0])


def test_a_column_can_place_the_axis_for_itself():
    """NotITG's reverse family slides one column's receptor to the mirror
    line while its neighbour stays put, so the undisplaced axis is a
    function of the column too."""
    def note_y(cols, offsets):
        flip = np.where(np.asarray(cols) % 2 == 0, 1.0, -1.0)
        return _JUDGE_Y - flip * offsets

    path = lane_path.LanePath(_lane_center, note_y)
    samples = path.sample((0, 1), (100.0, 100.0))
    assert samples.y == pytest.approx([_JUDGE_Y - 100.0, _JUDGE_Y + 100.0])


def test_what_travels_reaches_the_hook():
    """A displacement may depend on the thing's own beat, not just where
    it is (dizzy spins a note by its beats-until-step)."""
    path = _bend(rot=lambda cols, offs, beats, cell:
                 np.zeros(len(cols)) if beats is None
                 else np.asarray(beats, dtype=np.float64) * 90.0)
    assert path.at(0, 0.0).rotation_deg == 0.0
    assert path.at(0, 0.0, note_beat=2.0).rotation_deg == pytest.approx(180.0)


def test_a_body_span_keeps_its_ends_attached():
    """A hold body is the span between the head and tail offsets, and its
    endpoints must land exactly where those were placed independently -
    otherwise the ribbon detaches from its own cap."""
    path = _straight()
    span = path.between(1, 40.0, 300.0, 9)
    assert len(span) == 9
    assert span.at(0) == path.at(1, 40.0)
    assert span.at(-1) == path.at(1, 300.0)


def test_a_span_of_one_still_has_two_ends():
    # A degenerate request must not produce a zero-length ribbon.
    assert len(_straight().between(0, 10.0, 20.0, 1)) == 2


def test_spans_are_one_displacement_call():
    """Evaluating each hold separately pays the vectorized pipeline's fixed
    cost per hold instead of per frame, which is the whole reason the batch
    shape is the primitive."""
    calls = []

    def displace(cols, offsets, note_beats, cell):
        calls.append(len(cols))
        zeros = np.zeros(len(cols))
        return (zeros, zeros, zeros, np.ones(len(cols)), zeros,
                np.ones(len(cols)))

    path = lane_path.LanePath(_lane_center, lambda cols, offs: _note_y(offs),
                              displace=displace)
    bodies = path.spans((0, 2, 3), (0.0, 10.0, 20.0), (100.0, 50.0, 80.0),
                        (4, 6, 8))
    assert [len(b) for b in bodies] == [4, 6, 8]
    assert calls == [18]
    assert bodies[1].x == pytest.approx(np.full(6, _lane_center(2)))


def test_a_coarse_span_averages_what_it_cannot_resolve():
    """A ripple finer than the sample spacing must not be point-sampled
    into invented lobes: the engine draws strips every few px and converges
    to the local mean band, and the box filter is that band."""
    # Wavelength 0.7 against a sample spacing of 2.0: far under Nyquist,
    # and deliberately not a divisor of the spacing, so point sampling
    # yields a slow ghost curve rather than a flat line.
    ripple = _bend(dx=lambda cols, offs, beats, cell:
                   np.sin(offs * 2.0 * np.pi / 0.7) * 20.0)
    span = ripple.between(0, 0.0, 8.0, 5, taps=16)
    assert len(span) == 5
    unfiltered = ripple.between(0, 0.0, 8.0, 5)
    assert np.ptp(span.x) < 0.2 * np.ptp(unfiltered.x)
    assert span.x == pytest.approx(np.full(5, _lane_center(0)), abs=4.0)


def test_the_sample_spacing_reaches_the_hook():
    """A game band-limits its own kernels, so it has to be told how far
    apart the samples it is being asked for are."""
    cells = []

    def displace(cols, offsets, note_beats, cell):
        cells.append(cell)
        zeros = np.zeros(len(cols))
        return (zeros, zeros, zeros, np.ones(len(cols)), zeros,
                np.ones(len(cols)))

    path = lane_path.LanePath(_lane_center, lambda cols, offs: _note_y(offs),
                              displace=displace)
    path.at(0, 0.0)
    path.between(0, 0.0, 100.0, 5, cell=25.0)
    assert cells == [0.0, 25.0]


def test_the_cap_follows_the_curve_not_the_lane():
    """A folded hold's end segment runs back UP the lane; the tangent is
    what flips the cap, with no special case for the fold."""
    def fold(cols, offs, beats, cell):
        # The lane already travels UP (`note_y` subtracts the offset), so a
        # fold has to OUTRUN that: past offset 200 the displacement adds y
        # faster than the lane removes it, and the curve doubles back.
        return np.where(offs > 200.0, 3.0 * (offs - 200.0), 0.0)

    path = _bend(dy=fold)
    rising = path.between(0, 0.0, 100.0, 3)
    folded = path.between(0, 250.0, 350.0, 3)
    # Screen y grows downward, so an arrow travelling UP the lane heads one
    # way and the folded tail the other.
    assert rising.tangent_deg(0, -1) * folded.tangent_deg(0, -1) < 0.0


def test_a_still_point_has_no_heading():
    span = _straight().between(0, 50.0, 50.0, 4)
    assert span.tangent_deg(0, -1) == 0.0
