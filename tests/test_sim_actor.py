"""SimActor engine-semantics tests.

Every behavior here is pinned to openitg source (refs/notitg/openitg-src
when present; citations inline), per DESIGN_engine_loop.md: the sim
matches Actor.cpp, and the recorded keyframes replay exactly what the
sim displayed.
"""
import math

import pytest

from analysis.games.notitg.sim import SimActor
from analysis.player.render.effects.easing import (
    EASE_SM_BOUNCE_END, EASE_SM_SPRING, ease)
from analysis.player.render.effects.timeline import EventTimeline


def _timeline(actor, prop, rest=0.0):
    return EventTimeline(actor.keyframes().get(prop, []), rest=(rest,))


# -- immediate writes --------------------------------------------------------

def test_immediate_set_with_empty_queue():
    a = SimActor()
    a.poke('x', [100])
    assert a.get('x') == 100
    assert _timeline(a, 'x').sample(0.0) == (100,)


def test_unpoked_property_rests():
    a = SimActor()
    assert a.get('x') == 0.0
    assert a.get('scale_x') == 1.0
    assert a.keyframes() == {}


# -- tween interpolation (Actor.h:107 - GetX returns m_current) --------------

def test_linear_tween_interpolates_current():
    a = SimActor()
    a.poke('linear', [1.0])
    a.poke('x', [100])
    a.update_to(0.5)
    assert a.get('x') == pytest.approx(50.0)
    a.update_to(1.0)
    assert a.get('x') == pytest.approx(100.0)


def test_accelerate_is_in_quad():
    # Actor.cpp:522 - fPercentAlongPath = u * u.
    a = SimActor()
    a.poke('accelerate', [1.0])
    a.poke('x', [100])
    a.update_to(0.5)
    assert a.get('x') == pytest.approx(25.0)


def test_decelerate_is_out_quad():
    # Actor.cpp:523 - 1 - (1-u)^2.
    a = SimActor()
    a.poke('decelerate', [1.0])
    a.poke('x', [100])
    a.update_to(0.5)
    assert a.get('x') == pytest.approx(75.0)


def test_sm_bounce_and_spring_formulas():
    # Actor.cpp:524-526, verbatim.
    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert ease(EASE_SM_BOUNCE_END, u) == pytest.approx(
            math.sin(1.1 + (1 - u) * (math.pi - 1.1)) / 0.89)
        assert ease(EASE_SM_SPRING, u) == pytest.approx(
            1 - math.cos(u * math.pi * 2.5) / (1 + u * 3))


def test_spring_verb_maps_to_sm_curve():
    # The old recorder silently dropped bouncebegin/bounceend/spring.
    a = SimActor()
    a.poke('spring', [1.0])
    a.poke('x', [100])
    a.update_to(0.5)
    expected = 100 * ease(EASE_SM_SPRING, 0.5)
    assert a.get('x') == pytest.approx(expected)


# -- destination reads (Actor.h:110/117) -------------------------------------

def test_addx_stacks_on_destination_mid_tween():
    # AddX(x) = SetX(GetDestX()+x) - the DESTINATION, not the in-flight
    # current (Actor.h:117).
    a = SimActor()
    a.poke('linear', [1.0])
    a.poke('x', [100])
    a.update_to(0.5)
    a.poke('addx', [10])
    assert a.get_dest('x') == pytest.approx(110.0)
    a.update_to(2.0)
    assert a.get('x') == pytest.approx(110.0)


def test_chained_tweens_accumulate():
    a = SimActor()
    a.poke('linear', [1.0])
    a.poke('x', [100])
    a.poke('linear', [2.0])
    a.poke('y', [50])
    a.update_to(1.0)
    assert a.get('x') == pytest.approx(100.0)
    assert a.get('y') == pytest.approx(0.0)
    a.update_to(2.0)
    assert a.get('y') == pytest.approx(25.0)
    a.update_to(3.0)
    assert a.get('y') == pytest.approx(50.0)
    kf_y = a.keyframes()['y'][0]
    assert kf_y.t == pytest.approx(1.0)
    assert kf_y.duration == pytest.approx(2.0)


def test_retarget_started_head_reroutes_current():
    a = SimActor()
    a.poke('linear', [2.0])
    a.poke('x', [100])
    a.update_to(1.0)
    a.poke('x', [200])
    assert a.get('x') == pytest.approx(100.0)  # lerp(0, 200, 0.5)
    a.update_to(2.0)
    assert a.get('x') == pytest.approx(200.0)
    assert _timeline(a, 'x').sample(1.0) == (pytest.approx(100.0),)


# -- queue-borne commands (Actor.cpp:1068-1086) ------------------------------

def test_queuecommand_fires_when_queue_drains():
    a = SimActor()
    fired = []
    a.poke('sleep', [1.0])
    a.queue_command('N')
    a.update_to(0.5, fired.append)
    assert fired == []
    a.update_to(1.5, fired.append)
    assert fired == ['N']


def test_queuecommand_fire_time_is_queue_position():
    # Two stacked tweens then the command: fires at their sum, and the
    # actor's clock reads that moment inside the callback.
    a = SimActor()
    seen = []
    a.poke('linear', [1.0])
    a.poke('x', [10])
    a.poke('linear', [0.5])
    a.poke('y', [5])
    a.queue_command('N')
    a.update_to(5.0, lambda name: seen.append((name, a.now)))
    assert seen == [('N', pytest.approx(1.5))]


def test_queuemessage_carries_broadcast_marker():
    # Actor.cpp:1080 - "!" marks a broadcast instead of a command.
    a = SimActor()
    fired = []
    a.queue_message('M')
    a.update_to(0.1, fired.append)
    assert fired == ['!M']


def test_zero_dt_update_fires_nothing():
    # UpdateTweening early-outs on fDeltaTime == 0 (Actor.cpp:476).
    a = SimActor()
    fired = []
    a.queue_command('N')
    a.update_to(0.0, fired.append)
    assert fired == []
    a.update_to(0.01, fired.append)
    assert fired == ['N']


def test_command_pokes_defer_to_next_update():
    # A fired command's pokes append to the live queue but run on the
    # NEXT update (one queue pass per frame, engine-true): the appended
    # tween starts where this update left off.
    a = SimActor()

    def body(name):
        a.poke('linear', [1.0])
        a.poke('x', [100])

    a.poke('sleep', [1.0])
    a.queue_command('N')
    a.update_to(1.5, body)
    assert a.get('x') == pytest.approx(0.0)
    a.update_to(2.0, body)
    assert a.get('x') == pytest.approx(50.0)
    a.update_to(2.5, body)
    assert a.get('x') == pytest.approx(100.0)


def test_command_pokes_expand_in_place_when_not_deferred():
    # defer_queued=False (the loop's final drain) expands the appended
    # tweens within the same update.
    a = SimActor()

    def body(name):
        a.poke('linear', [1.0])
        a.poke('x', [100])

    a.poke('sleep', [1.0])
    a.queue_command('N')
    a.update_to(1.5, body, defer_queued=False)
    assert a.get('x') == pytest.approx(50.0)


def test_sleep_defers_following_set():
    a = SimActor()
    a.poke('sleep', [1.0])
    a.poke('x', [5])
    a.update_to(0.5)
    assert a.get('x') == pytest.approx(0.0)
    # UpdateTweening early-outs once dt is fully consumed, BEFORE
    # beginning the next queued tween (Actor.cpp:476) - at exactly t=1.0
    # the zero-tween carrying x has not begun; the next update lands it.
    a.update_to(1.0)
    assert a.get('x') == pytest.approx(0.0)
    a.update_to(1.0 + 1 / 60)
    assert a.get('x') == pytest.approx(5.0)


# -- stop / finish (Actor.cpp:652/657) ---------------------------------------

def test_stoptweening_abandons_mid_flight():
    a = SimActor()
    a.poke('linear', [2.0])
    a.poke('x', [100])
    a.update_to(1.0)
    a.poke('stoptweening', [])
    assert a.get('x') == pytest.approx(50.0)
    a.update_to(3.0)
    assert a.get('x') == pytest.approx(50.0)
    # The recorded timeline replays the abandonment: the pin wins from
    # the stop instant on.
    assert _timeline(a, 'x').sample(2.5) == (pytest.approx(50.0),)


def test_finishtweening_jumps_to_final_state():
    a = SimActor()
    a.poke('linear', [2.0])
    a.poke('x', [100])
    a.poke('linear', [1.0])
    a.poke('y', [7])
    a.poke('finishtweening', [])
    assert a.get('x') == pytest.approx(100.0)
    assert a.get('y') == pytest.approx(7.0)


def test_finishtweening_drops_queued_commands():
    # FinishTweening assigns DestTweenState and clears; queued commands
    # never play (Actor.cpp:657-660).
    a = SimActor()
    fired = []
    a.queue_command('N')
    a.poke('finishtweening', [])
    a.update_to(1.0, fired.append)
    assert fired == []


# -- immediate engine bits ---------------------------------------------------

def test_hidden_is_immediate_while_alpha_tweens():
    # SetHidden writes m_bVisible directly (Actor.h:311); diffusealpha
    # rides the tween queue.
    a = SimActor()
    a.poke('linear', [1.0])
    a.poke('diffusealpha', [0.0])
    a.poke('hidden', [1])
    assert a.get('hidden') == 1.0
    a.update_to(0.5)
    assert a.get('alpha') == pytest.approx(0.5)
    a.update_to(1.0)
    assert a.get('hidden') == 1.0  # survives tween completion


def test_vanish_point_immediate():
    a = SimActor()
    a.poke('linear', [1.0])
    a.poke('SetVanishPoint', [100, 200])
    assert a.get('vanish_x') == 100
    assert a.get('vanish_y') == 200


def test_diffuse_color_tweens():
    a = SimActor()
    a.poke('linear', [1.0])
    a.poke('diffuse', [0.0, 0.0, 0.0, 1.0])
    a.update_to(0.5)
    assert a.get('color') == tuple(pytest.approx(0.5) for _ in range(3))


# -- recorded timeline replays the sim exactly -------------------------------

def test_round_trip_recorded_timeline_matches_live_sim():
    def drive(actor, run=None):
        actor.poke('x', [10])
        actor.poke('linear', [1.0])
        actor.poke('x', [100])
        actor.poke('decelerate', [2.0])
        actor.poke('x', [-40])
        actor.poke('sleep', [0.5])
        actor.poke('x', [7])

    live = SimActor()
    drive(live)

    recorded = SimActor()
    drive(recorded)
    recorded.update_to(10.0)
    timeline = _timeline(recorded, 'x')

    for t in (0.0, 0.3, 0.9, 1.0, 1.7, 2.4, 3.0, 3.4, 3.6, 9.0):
        fresh = SimActor()
        drive(fresh)
        fresh.update_to(t)
        assert timeline.sample(t)[0] == pytest.approx(fresh.get('x')), t


# -- overflow guard (Actor.cpp:616) ------------------------------------------

def test_tween_overflow_finishes_queue():
    a = SimActor()
    for i in range(60):
        a.poke('linear', [1.0])
        a.poke('x', [i])
    # The guard finished the early queue rather than growing unbounded;
    # the actor stays consistent and the final destination is the last.
    assert a.get_dest('x') == 59
    a.update_to(100.0)
    assert a.get('x') == 59


# -- oscillator spans + driven spans -----------------------------------------

def test_oscillator_span_records_and_reads():
    a = SimActor()
    a.poke('vibrate', [])
    a.poke('effectmagnitude', [4, 4, 0])
    a.poke('effectperiod', [0.25])
    a.update_to(2.0)
    a.poke('stopeffect', [])
    (span,) = a.oscillator_spans()
    assert span.kind == 'vibrate'
    assert span.period == pytest.approx(0.25)
    assert span.magnitude_at(1.0) == (4, 4, 0)
    assert span.end == pytest.approx(2.0)


def test_secs_into_effect_timer_clock_wraps_at_period():
    # CLOCK_TIMER accumulates and wraps at period + delay
    # (Actor.cpp:571-575); bob's setter defaults the period to 2.0.
    a = SimActor()
    a.poke('bob', [])
    a.poke('effectmagnitude', [0, 10, 0])
    a.update_to(1.5)
    assert a.read('GetSecsIntoEffect') == pytest.approx(1.5)
    a.update_to(2.5)
    assert a.read('GetSecsIntoEffect') == pytest.approx(0.5)


def test_secs_into_effect_music_clock_is_song_time():
    # CLOCK_BGM_TIME tracks the song clock outright (Actor.cpp:581-583);
    # charts set `effectclock,music` with NO effect running to use this
    # as their mod-clock rig.
    a = SimActor()
    a.poke('effectclock', ['music'])
    a.update_to(12.25)
    assert a.read('GetSecsIntoEffect') == pytest.approx(12.25)
    a.poke('settext', [a.read('GetSecsIntoEffect')])
    assert float(a.read('GetText')) == pytest.approx(12.25)


def test_driven_spans_merge_ticks_and_split_sections():
    a = SimActor()
    a.set_driven(True)
    for i in range(10):
        a.update_to(i / 60.0)
        a.poke('x', [float(i)])
    a.update_to(5.0)
    a.poke('x', [99.0])
    a.set_driven(False)
    spans = a.driven_spans()
    assert len(spans) == 2
    assert spans[0][0] == pytest.approx(0.0)
    assert spans[0][1] == pytest.approx(9 / 60.0)
    assert spans[1] == (pytest.approx(5.0), pytest.approx(5.0))
