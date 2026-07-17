"""Render scheduler: events + live-evaluated curves over clocks."""
import math

import pytest

from analysis.player.render.effects.timeline import EventTimeline, Keyframe
from analysis.player.render.scheduler import (
    Channel, Event, EventSchedule, IntegralClock, LoopClock, SongTimeClock,
    timeline_channel)


def test_song_time_clock_is_identity():
    clock = SongTimeClock()
    assert clock.at(3.5) == 3.5


def test_integral_clock_applies_the_integral():
    # beat = 2 * seconds (120 bpm)
    clock = IntegralClock(lambda t: t * 2.0)
    assert clock.at(4.0) == 8.0


def test_loop_clock_wraps_at_period():
    clock = LoopClock(SongTimeClock(), period=1.0)
    assert clock.at(0.25) == pytest.approx(0.25)
    assert clock.at(1.25) == pytest.approx(0.25)
    assert clock.at(2.75) == pytest.approx(0.75)


def test_channel_evaluates_analytic_curve_live():
    # An oscillator: no stored samples, computed at t.
    osc = Channel(curve=lambda phase: math.sin(phase * 2 * math.pi),
                  clock=SongTimeClock())
    assert osc.at(0.0) == pytest.approx(0.0)
    assert osc.at(0.25) == pytest.approx(1.0)
    assert osc.at(0.5) == pytest.approx(0.0, abs=1e-9)


def test_channel_over_a_beat_clock():
    # sin over beats at 120bpm (beat = 2*sec): a quarter cycle per beat.
    beat = IntegralClock(lambda t: t * 2.0)
    osc = Channel(curve=lambda b: math.sin(b * math.pi / 2.0), clock=beat)
    # t=0.5s -> beat 1.0 -> sin(pi/2) = 1.
    assert osc.at(0.5) == pytest.approx(1.0)


def test_none_curve_returns_rest():
    assert Channel(curve=None, rest=7.0).at(99.0) == 7.0


def test_timeline_channel_bridges_keyframes():
    # A keyframe eases from the PREVIOUS target toward its own over its
    # duration: an instant point at 0, then a 1s ramp 0->100.
    tl = EventTimeline([Keyframe(0.0, (0.0,), 0.0, 0),
                        Keyframe(1.0, (100.0,), 1.0, 0)], rest=(0.0,))
    ch = timeline_channel(tl)
    assert ch.at(0.0) == pytest.approx(0.0)
    assert ch.at(1.5) == pytest.approx(50.0)  # halfway through the 1s ramp
    assert ch.at(2.0) == pytest.approx(100.0)


def test_empty_timeline_channel_rests():
    ch = timeline_channel(EventTimeline([], rest=(0.0,)))
    assert ch.at(5.0) == 0.0


def test_event_schedule_due_is_ordered_half_open():
    sched = EventSchedule([Event(1.0, 'a'), Event(3.0, 'c'), Event(2.0, 'b')])
    due = sched.due(1.0, 3.0)
    assert [e.payload for e in due] == ['a', 'b']  # [1,3) excludes c@3
    assert [e.payload for e in sched.due(3.0, 10.0)] == ['c']


def test_event_schedule_add_keeps_sorted():
    sched = EventSchedule([Event(1.0, 'a'), Event(5.0, 'c')])
    sched.add(Event(3.0, 'b'))
    assert [e.payload for e in sched.events] == ['a', 'b', 'c']


def test_event_active_spans_duration():
    sched = EventSchedule([Event(1.0, 'win', duration=2.0)])
    assert [e.payload for e in sched.active(0.5)] == []
    assert [e.payload for e in sched.active(1.5)] == ['win']
    assert [e.payload for e in sched.active(3.5)] == []  # [1,3) excludes 3.5


def test_event_instant_active_at_exact_time():
    sched = EventSchedule([Event(2.0, 'ping', duration=0.0)])
    assert [e.payload for e in sched.active(2.0)] == ['ping']
