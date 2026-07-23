"""verbs_tween axis: the Schedule fold vs the simulated tween queue.

The engine drains its queue with exact time arithmetic at any frame
rate (Actor.cpp:469), so lowering a chain at compile time must produce
the same value trajectory the sim records tick-by-tick - and the sim's
own recording must be tick-rate independent. Both are asserted here on
synthetic chains, per the feature-suite plan.
"""
import pytest

from analysis.games.notitg.lua_api import _TWEEN_EASING
from analysis.games.notitg.sim.actor import SimActor
from analysis.player.render.schedule import (
    Hibernate, Loop, Seg, Seq, command, lower, sleep, to_timelines)

LINEAR = _TWEEN_EASING['linear']
ACCELERATE = _TWEEN_EASING['accelerate']
DECELERATE = _TWEEN_EASING['decelerate']

RESTS = {'x': 0.0, 'y': 0.0, 'rotation': 0.0}


def _actor_lanes(chain_pokes, end_t, step):
    actor = SimActor(now=0.0)
    for verb, args in chain_pokes:
        actor.poke(verb, args)
    t = 0.0
    while t < end_t:
        t = min(t + step, end_t)
        actor.update_to(t)
    return {prop: lanes[0] for prop, lanes in actor._seg.items()}


CHAIN_POKES = [
    ('linear', [0.5]), ('x', [100.0]),
    ('sleep', [0.25]),
    ('decelerate', [1.0]), ('x', [-40.0]), ('y', [64.0]),
    ('accelerate', [0.4]), ('y', [0.0]),
]

CHAIN_SCHEDULE = Seq(
    Seg(0.5, LINEAR, {'x': 100.0}),
    sleep(0.25),
    Seg(1.0, DECELERATE, {'x': -40.0, 'y': 64.0}),
    Seg(0.4, ACCELERATE, {'y': 0.0}),
)

GRID = [i * 0.037 for i in range(75)]


def test_lowered_chain_matches_simulated_actor():
    sim_lanes = _actor_lanes(CHAIN_POKES, end_t=3.0, step=1.0 / 60.0)
    lowered = lower(CHAIN_SCHEDULE, state=dict(RESTS))
    lanes = to_timelines(lowered, rests=RESTS)

    assert set(lanes) == set(sim_lanes)
    for prop, lane in lanes.items():
        for t in GRID:
            assert lane.sample(t) == pytest.approx(
                sim_lanes[prop].sample(t), abs=1e-9), (prop, t)


@pytest.mark.parametrize('step', [1.0 / 60.0, 1.0 / 7.3, 3.0])
def test_sim_recording_is_tick_rate_independent(step):
    fine = _actor_lanes(CHAIN_POKES, end_t=3.0, step=1.0 / 60.0)
    coarse = _actor_lanes(CHAIN_POKES, end_t=3.0, step=step)
    for prop, lane in fine.items():
        for t in GRID:
            assert lane.sample(t) == pytest.approx(
                coarse[prop].sample(t), abs=1e-9), (prop, t, step)


def test_zero_duration_write_is_a_hold():
    lowered = lower(Seq(sleep(0.5), Seg(0.0, LINEAR, {'x': 7.0})),
                    state=dict(RESTS))
    lanes = to_timelines(lowered, rests=RESTS)
    assert lanes['x'].sample(0.49) == 0.0
    assert lanes['x'].sample(0.5) == 7.0


def test_unchanged_target_emits_nothing():
    lowered = lower(Seg(1.0, LINEAR, {'x': 0.0}), state=dict(RESTS))
    assert lowered.emissions == []


def test_command_fires_at_exact_offsets():
    lowered = lower(Seq(sleep(0.3), command('boom'),
                        sleep(0.2), command('bam')))
    assert [(f.t, f.effect) for f in lowered.fires] == [
        (pytest.approx(0.3), 'boom'), (pytest.approx(0.5), 'bam')]


def test_hibernate_is_a_prefix_sleep():
    lowered = lower(Seq(Hibernate(1.5), Seg(0.5, LINEAR, {'x': 1.0})),
                    state=dict(RESTS))
    ramp = lowered.emissions[0]
    assert (ramp.t0, ramp.t1) == (pytest.approx(1.5), pytest.approx(2.0))


def test_loop_unrolls_to_horizon():
    lowered = lower(Loop(0.5, command('tick')), t0=0.1, until=2.0)
    assert [pytest.approx(f.t) for f in lowered.fires] \
        == [0.1, 0.6, 1.1, 1.6]


def test_loop_with_timed_body_advances_by_body_time():
    lowered = lower(Loop(0.5, Seg(0.7, LINEAR, {'x': 1.0})), until=2.0)
    starts = [e.t0 for e in lowered.emissions]
    assert starts == [pytest.approx(0.0)]
    assert lowered.end_t >= 2.0


def test_nested_effect_schedule_joins_the_tail():
    nested = Seq(sleep(1.0), Seg(0.5, LINEAR, {'y': 5.0}))
    lowered = lower(Seq(Seg(0.0, effect=nested),
                        Seg(1.0, LINEAR, {'x': 10.0})),
                    state=dict(RESTS))
    lanes = to_timelines(lowered, rests=RESTS)
    assert lanes['x'].sample(1.0) == pytest.approx(10.0)
    assert lanes['y'].sample(1.9) == 0.0
    assert lanes['y'].sample(2.5) == pytest.approx(5.0)


def test_runaway_recursion_hits_engine_bound():
    node = Seg(0.1, LINEAR, {'x': 1.0})
    for _ in range(60):
        node = Seq(node)
    with pytest.raises(RecursionError):
        lower(node)


def test_loop_requires_horizon():
    with pytest.raises(ValueError):
        lower(Loop(0.5, command('tick')))
