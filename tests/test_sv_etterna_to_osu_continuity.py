"""Etterna -> osu_time projection continuity.

`measure_engine.project_beat_to_time(beat_engine)` must keep
`cumulative_at(0.0)` == 0 even when the chart starts with a non-1.0
SCROLLS rate ; the synthetic `(0, 1.0)` prepend makes the time-space
engine extrapolate identically to beat-space's `GetDisplayedBeat`
fallthrough for t<0. Without the prepend, switching engines at the
song start would visually snap the field.
"""
from __future__ import annotations

from analysis.player.sv.measure_engine import (beat_space_engine,
                                                project_beat_to_time)


def _build_beat(scrolls):
    return beat_space_engine(scrolls=scrolls, speeds=[],
                             bpms=[(0.0, 120.0)], sm_offset=0.0)


def test_lead_in_prepend_keeps_zero_cumulative_continuous():
    """Chart that opens with SCROLLS=2.0 ; without the prepend, the
    time-space engine's pre-first-segment extrapolation would bake in
    the 2.0 multiplier and disagree with beat-space at t=0."""
    beat = _build_beat([(0.0, 2.0)])
    osu_time = project_beat_to_time(beat)
    # Both engines must agree at t=0.
    assert abs(beat.cumulative_at(0.0) - osu_time.cumulative_at(0.0)) < 1e-9


def test_lead_in_prepend_skipped_when_first_section_is_unit():
    """No prepend needed when the chart's first SCROLLS rate is exactly
    1.0 starting at t=0 ; the projection should still produce a working
    engine that agrees with beat-space at the origin."""
    beat = _build_beat([(0.0, 1.0)])
    osu_time = project_beat_to_time(beat)
    assert abs(beat.cumulative_at(0.0) - osu_time.cumulative_at(0.0)) < 1e-9


def test_lead_in_prepend_when_first_section_starts_after_zero():
    """Beat-space charts whose first SCROLLS section starts at t>0 need
    the prepend so the time-space engine's t<first_t extrapolation
    matches beat-space's ratio=1.0 fallthrough."""
    # Single SCROLLS=3.0 segment starting at beat 4 ; that's some real
    # time > 0 under 120 BPM (~2 seconds in).
    beat = _build_beat([(4.0, 3.0)])
    osu_time = project_beat_to_time(beat)
    # Still continuous at t=0 ; the prepend keeps the integrator from
    # treating the projected sections' first multiplier as the t<0 value.
    assert abs(beat.cumulative_at(0.0) - osu_time.cumulative_at(0.0)) < 1e-9
