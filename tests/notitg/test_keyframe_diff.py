"""The keyframe-parity oracle: two sim runs of the same chart diverge nowhere
(reflexivity), and a real animation difference is detected AND attributed to
the actor/property/time - the "why does this look wrong" signal the AST
interpreter is graded against."""
import pytest

pytest.importorskip('lupa')

from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.sim.keyframe_diff import (
    Divergence, diff_runs, sample_grid)
from analysis.games.notitg.xml_actors import parse_actor_xml


def _run(init_x):
    xml = (f'<ActorFrame><children>'
           f'<Quad Name="Q" InitCommand="%function(self) self:x({init_x}) end"/>'
           f'</children></ActorFrame>')
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(xml).root)
    return env


def test_identical_runs_do_not_diverge():
    assert diff_runs(_run(50), _run(50), sample_grid(0.0, 2.0)) == []


def test_representation_difference_that_plays_the_same_is_not_a_divergence():
    # One run records the destination as many identical instant pokes (what a
    # per-frame driver holding a constant emits); the other records ONE. The
    # instant-collapse gives different keyframe SHAPES, but both hold the same
    # value at every time - the value-diff must call them the same animation.
    def run(n_pokes):
        env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
        env.load_actors(parse_actor_xml(
            '<ActorFrame><children><Quad Name="Q"/></children></ActorFrame>'
        ).root)
        actor = next(iter(env.actor_keyframes()), None)
        rec_id = next(iter(env.actors))
        for i in range(n_pokes):
            env.set_time(i * 0.1, 0.0)
            env._actor_poke(rec_id, 'x', 80.0)
        return env

    assert diff_runs(run(1), run(8), sample_grid(0.0, 1.0)) == []


def test_real_difference_is_detected_and_attributed():
    divs = diff_runs(_run(50), _run(200), sample_grid(0.0, 1.0))
    assert divs, 'a genuine value difference must surface'
    first = divs[0]
    assert isinstance(first, Divergence)
    assert first.prop == 'x'
    assert 'Q' in first.actor
    assert first.left == (50,) and first.right == (200,)


def test_property_on_one_side_only_diffs_against_rest():
    # One run pokes rotation, the other never touches it: the poked side must
    # diverge from the other's rest, not be silently dropped.
    poked = ('<Quad Name="Q" InitCommand="%function(self)'
             ' self:rotationz(90) end"/>')
    plain = '<Quad Name="Q"/>'

    def run(inner):
        env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
        env.load_actors(parse_actor_xml(
            f'<ActorFrame><children>{inner}</children></ActorFrame>').root)
        return env

    divs = diff_runs(run(poked), run(plain), sample_grid(0.0, 1.0))
    assert any(d.prop == 'rotation' for d in divs)
