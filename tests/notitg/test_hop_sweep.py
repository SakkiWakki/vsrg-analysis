"""Phase 4 specs: the bodyless hop sweep vs the tick grid.

A chart with no Update body sweeps by hopping between activity points
(action fires, queue boundaries). The gate: hop lanes must match
tick-grid lanes wherever both record - the fold's dt-independence made
real. Bodied charts must keep the tick grid untouched.
"""
import pytest

from analysis.games.notitg.sim.loop import LiveSim
from analysis.games.notitg.xml_actors import parse_actor_xml

_BODYLESS_CHART = """
<ActorFrame InitCommand="%function(self)
    spinner = self
end"
GoMessageCommand="%function(self)
    self:stoptweening(); self:linear(0.5); self:x(120);
    self:sleep(0.25); self:decelerate(1.0); self:x(-60)
end"
><children>
<Layer Type="Quad" OnCommand="sleep,0.1;linear,0.4;zoomx,3"
 LoopMessageCommand="%function(self)
    self:linear(0.2); self:y(GAMESTATE:GetSongBeat())
    self:sleep(0.3); self:queuecommand('Rearm')
 end"
 RearmCommand="%function(self) self:addx(5) end" />
</children></ActorFrame>
"""


def _sweep(hop, end=6.0):
    parsed = parse_actor_xml(_BODYLESS_CHART)
    sim = LiveSim(parsed.root, lambda b: b * 0.5, 0.0, end)
    sim.env._staged_actions = [
        (1.0, 2.0, 'Go'), (2.0, 4.0, 'Loop'), (3.5, 7.0, 'Loop')]
    sim.env._next_action = 0
    sim.hop_enabled = hop
    sim.advance_to(end)
    return sim


def _lanes(sim):
    out = {}
    for rec_id, actor in sim.env._actors.items():
        for prop, lanes in actor._seg.items():
            out[(rec_id, prop)] = lanes
    return out


def test_hop_sweep_matches_tick_sweep():
    hop = _lanes(_sweep(hop=True))
    tick = _lanes(_sweep(hop=False))
    assert set(hop) == set(tick)

    step = 1.0 / 60.0
    for key, lanes in tick.items():
        for lane_h, lane_t in zip(hop[key], lanes):
            for i in range(120):
                t = i * 0.05
                # Command-fire quantization differs (hop fires at exact
                # boundaries, ticks quantize up to one step), so values
                # may lead/lag by at most one tick of trajectory.
                a = lane_h.sample(t)
                window = (lane_t.sample(t - step), lane_t.sample(t),
                          lane_t.sample(t + step))
                lo, hi = min(window), max(window)
                assert lo - 1e-6 <= a <= hi + 1e-6, (key, t, a, window)


def test_hop_sweep_is_far_fewer_steps():
    sim = _sweep(hop=True)
    # 6s at 60Hz = 360 ticks; activity points are a couple dozen.
    assert sim.frontier == pytest.approx(6.0)


def test_bodied_charts_keep_the_tick_grid():
    parsed = parse_actor_xml(_BODYLESS_CHART)
    sim = LiveSim(parsed.root, lambda b: b * 0.5, 0.0, 2.0)
    sim._body = 'counter = 1'
    sim.hop_enabled = True
    sim.advance_to(1.0)
    assert sim.now >= 1.0 - 1.0 / 60.0
