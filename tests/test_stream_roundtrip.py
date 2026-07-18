"""Producer/consumer round-trip properties for compiled streams.

Every lossy stage between a recorded stream and its sampler must
reproduce the recording under the SAMPLER'S OWN semantics - not under
semantics the sampler doesn't have. Two shipped bugs motivated this
harness: the instant-run simplifier assumed linear interpolation where
EventTimeline step-holds (hidden flips migrated seconds late), and
driver mod windows resolved with rest gaps between per-frame applies
where the engine persists targets (a decaying spin never came to rest).
Each compress/resolve stage gets a property here; new stages register
alongside.
"""
import math

import pytest

from analysis.player.render.effects.timeline import (
    EventTimeline, Keyframe, SIMPLIFY_EPS, simplify_instants)

_TICK = 1.0 / 50.0  # the NotITG rig cadence the recorder sees


def _instants(points):
    return [Keyframe(t, (v,), 0.0, 0) for t, v in points]


def _raw_streams():
    """Representative recorded shapes: name -> raw instant keyframes."""
    ramp = [(1.0 + i * _TICK, 100.0 + 3.0 * i) for i in range(120)]
    sine = [(2.0 + i * _TICK, 50.0 * math.sin(i * 0.35)) for i in range(160)]
    toggle = [(0.5, 1.0), (10.0, 0.0)] + [(10.0 + i * _TICK, 0.0)
                                          for i in range(1, 400)]
    sawtooth = []
    v = 480.0
    for i in range(300):
        v += 3.0
        if v > 980.0:
            v -= 500.0
        sawtooth.append((3.0 + i * _TICK, v))
    return {'ramp': ramp, 'sine': sine, 'constant-after-flip': toggle,
            'sawtooth': sawtooth}


@pytest.mark.parametrize('name', sorted(_raw_streams()))
def test_simplified_stream_reproduces_recording(name):
    """simplify(raw) sampled through EventTimeline matches the raw
    recording at every recorded instant, to the simplifier's eps."""
    points = _raw_streams()[name]
    raw = _instants(points)
    tl = EventTimeline(simplify_instants(raw), rest=(0.0,))
    for t, v in points:
        assert tl.sample(t)[0] == pytest.approx(v, abs=SIMPLIFY_EPS * 4), \
            f'{name} diverges at t={t}'


def test_transition_stays_at_run_start():
    """A value that flips and then holds must flip AT the flip, not at
    the end of the constant run (the eroded-hidden regression: a field
    unhidden at t=10 stayed compiled-hidden until t=56)."""
    points = [(0.5, 1.0)] + [(10.0 + i * _TICK, 0.0) for i in range(500)]
    tl = EventTimeline(simplify_instants(_instants(points)), rest=(1.0,))
    assert tl.sample(9.99)[0] == 1.0
    assert tl.sample(10.0)[0] == 0.0
    assert tl.sample(15.0)[0] == 0.0


def test_driver_burst_tracks_and_rests():
    """Per-frame driver applies (chained spike windows, finite approach)
    must TRACK the decaying target while the burst runs and float back
    to rest after it ends - never plateau mid-decay (the endless
    receptor-spin regression)."""
    from analysis.games.notitg.mod_channels import compile_mod_channels

    rows = []
    n = 200
    for i in range(n):
        t = 10.0 + i * _TICK
        target = 10000.0 * (1.0 - i / n) ** 2  # decelerating decay to 0
        rows.append({
            'modstring': f'*10000 {target:.2f} confusionoffset',
            'player': 1,
            't_start': t,
            't_end': (10.0 + (i + 1) * _TICK) if i + 1 < n else t + 1 / 60,
            'time_based': True,
        })
    channels = compile_mod_channels(rows)

    # The channel chase currently lags the exact engine fapproach by up
    # to ~35% mid-transient (it still tracks the decay's shape and
    # settles right). These bounds pin the shape; tightening them to a
    # few percent is the open fapproach-exactness work in ModChannels.
    previous = float('inf')
    for i in (30, 60, 120):
        t = 10.0 + i * _TICK + 0.5
        target = 100.0 * (1.0 - i / n) ** 2
        got = channels.values_at(t, 0).get('confusionoffset', 0.0)
        assert 0.4 * target <= got <= 1.2 * target + 2.0, \
            f'chase lost the decay shape at t={t}: {got} vs {target}'
        assert got < previous, 'chase must decay monotonically'
        previous = got
    end = 10.0 + n * _TICK
    assert channels.values_at(end + 3.0, 0).get(
        'confusionoffset', 0.0) == pytest.approx(0.0, abs=0.5)


def test_staged_actions_park_the_template_cursor():
    """One owner for mod_actions: once the sweep stages them, the
    template's own `curaction` replay loop must be parked past the table
    end - both firing gave every action twice (18 proxies in a table the
    chart fills with 9)."""
    pytest.importorskip('lupa')
    from analysis.games.notitg.sim.env import SimEnvironment
    from analysis.games.notitg.xml_actors import parse_actor_xml

    chart = """
    <ActorFrame InitCommand="%function(self)
        curaction = 1
        mod_actions = {{1, function() end}, {2, 'Ping'}}
    end" />
    """
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(chart).root)
    staged = env.prepare_mod_actions()
    assert staged == 2
    assert env._host.env['curaction'] > 2
