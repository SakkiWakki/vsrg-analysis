"""Spec: the executor's settled-actor mirror never drifts from the sim.

The op-stream executor answers a property read from a mirror of `_current`
instead of crossing to the host - but only while the actor's tween queue is
idle, because a running tween means `SimActor.get` interpolates rather than
reading the settled value.

Mirroring is OPT-IN PER ACTOR: on gat, 147 actors have a mirrored property
written while the body reads only 6 of them, and feeding all of them cost more
than the reads it served. The executor opts an actor in the first time it has to
answer a read for it the hard way.

Three invariants, all silent if broken: a settled value the mirror reports must
equal what `get` would return; an actor with a busy queue must be flagged so its
reads cross instead; and an actor nobody opted in must not be fed at all. The
comparisons are against the sim, not against the mirror's own bookkeeping.
"""
import pytest

pytest.importorskip('lupa')

from analysis.games.notitg.guard_surface import PROP_GET_TARGETS, PROP_SLOTS
from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.xml_actors import parse_actor_xml


class _Recorder:
    """Stands in for the executor: records what the mirror is told."""

    def __init__(self):
        self.values: dict = {}
        self.tweening: dict = {}

    def on_prop(self, rec_id, prop, value):
        for slot in PROP_SLOTS.get(prop, ()):
            self.values[(rec_id, slot)] = float(value)

    def on_tween(self, rec_id, busy):
        self.tweening[rec_id] = bool(busy)


def _env(xml):
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(xml).root)
    return env


_CHART = ('<ActorFrame><children>'
          '<Quad Name="A"/><Quad Name="B"/>'
          '</children></ActorFrame>')


def _assert_mirror_matches(env, rec):
    """Every mirrored entry must equal what the sim would answer."""
    for (rec_id, slot), mirrored in rec.values.items():
        actor = env._actors[rec_id]
        if rec.tweening.get(rec_id):
            continue          # interpolating: the executor crosses instead
        expected = actor.get(PROP_GET_TARGETS[slot])
        assert mirrored == pytest.approx(float(expected)), (
            f'actor {rec_id} {PROP_GET_TARGETS[slot]}: '
            f'mirror {mirrored} vs sim {expected}')


def test_settled_pokes_reach_the_mirror():
    env = _env(_CHART)
    rec = _Recorder()
    env.install_prop_mirror(rec.on_prop, rec.on_tween)
    rec_id = next(iter(env._actors))
    env.mirror_actor(rec_id)
    env._actor_poke(rec_id, 'x', 120.0)
    env._actor_poke(rec_id, 'zoom', 2.0)
    assert rec.values[(rec_id, PROP_SLOTS['x'][0])] == 120.0
    for slot in PROP_SLOTS['scale_x']:
        assert rec.values[(rec_id, slot)] == 2.0
    _assert_mirror_matches(env, rec)


def test_opting_an_actor_in_seeds_its_current_state():
    """An actor is opted in mid-run, long after the load pass poked it - so
    `mirror_actor` must back-fill, not just subscribe."""
    env = _env(_CHART)
    rec_id = next(iter(env._actors))
    env._actor_poke(rec_id, 'x', 55.0)
    rec = _Recorder()
    env.install_prop_mirror(rec.on_prop, rec.on_tween)
    env.mirror_actor(rec_id)
    assert rec.values[(rec_id, PROP_SLOTS['x'][0])] == 55.0
    assert rec.tweening[rec_id] is False


def test_an_actor_nobody_opted_in_is_never_fed():
    """The targeting IS the optimisation: feeding every poked actor measured
    slower than the reads it served."""
    env = _env(_CHART)
    rec = _Recorder()
    env.install_prop_mirror(rec.on_prop, rec.on_tween)
    wanted, other = list(env._actors)[:2]
    env.mirror_actor(wanted)
    env._actor_poke(wanted, 'x', 10.0)
    env._actor_poke(other, 'x', 20.0)
    assert (wanted, PROP_SLOTS['x'][0]) in rec.values
    assert (other, PROP_SLOTS['x'][0]) not in rec.values


def test_a_queued_tween_flags_the_actor():
    """While the queue is occupied `get` interpolates, so the executor must be
    told to cross rather than trust the mirror."""
    env = _env(_CHART)
    rec = _Recorder()
    env.install_prop_mirror(rec.on_prop, rec.on_tween)
    rec_id = next(iter(env._actors))
    env.mirror_actor(rec_id)
    env._actor_poke(rec_id, 'linear', 1.0)
    env._actor_poke(rec_id, 'x', 300.0)
    assert rec.tweening[rec_id] is True


def test_the_flag_clears_when_the_queue_drains():
    env = _env(_CHART)
    rec = _Recorder()
    env.install_prop_mirror(rec.on_prop, rec.on_tween)
    rec_id = next(iter(env._actors))
    env.mirror_actor(rec_id)
    env._actor_poke(rec_id, 'linear', 0.5)
    env._actor_poke(rec_id, 'x', 300.0)
    assert rec.tweening[rec_id] is True
    env.set_time(2.0, 4.0)
    env.drain(2.0)
    assert rec.tweening[rec_id] is False
    _assert_mirror_matches(env, rec)


def test_finishtweening_lands_the_destination_in_the_mirror():
    """`finishtweening` jumps to the queued destination and clears - the mirror
    has to see BOTH the new value and the emptied queue."""
    env = _env(_CHART)
    rec = _Recorder()
    env.install_prop_mirror(rec.on_prop, rec.on_tween)
    rec_id = next(iter(env._actors))
    env.mirror_actor(rec_id)
    env._actor_poke(rec_id, 'linear', 5.0)
    env._actor_poke(rec_id, 'x', 400.0)
    env._actor_poke(rec_id, 'finishtweening')
    assert rec.tweening[rec_id] is False
    assert rec.values[(rec_id, PROP_SLOTS['x'][0])] == 400.0
    _assert_mirror_matches(env, rec)
