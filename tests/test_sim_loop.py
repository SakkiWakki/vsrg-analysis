"""Engine-loop tests: the sim runs charts through their own Lua.

The synthetic chart uses the classic template's exact self-scheduling
idiom (InitCommand arms `queuecommand('Update')`; UpdateCommand re-arms
with `sleep(0.02); queuecommand('Update')`, gat default.xml:3838/4695) -
the loop itself knows nothing about per-frame rigs.
"""
from pathlib import Path

import pytest

pytest.importorskip('lupa')

from analysis.games.notitg.sim import (  # noqa: E402
    coalesce_applied, run_chart_sim, run_sim, summarize)
from analysis.games.notitg.xml_actors import parse_actor_xml  # noqa: E402

GAT_SM = Path('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/UKSRT8'
              '/5. gat/gat.sm')

_SELF_SCHEDULING_CHART = """
<ActorFrame InitCommand="%function(self)
    counter = 0
    self:queuecommand('Update');
end"
UpdateCommand="%function(self)
    counter = counter + 1
    self:x(counter);
    if GAMESTATE:GetSongBeat() >= 2 and GAMESTATE:GetSongBeat() < 4 then
        GAMESTATE:ApplyModifiers('*10 50 drunk');
    end
    if counter == 10 then
        MESSAGEMAN:Broadcast('Ping');
    end
    self:sleep(0.02);
    self:queuecommand('Update');
end"
><children>
<Layer Type="Quad" OnCommand="diffusealpha,0.5"
 PingMessageCommand="%function(self) self:y(77) end" />
</children></ActorFrame>
"""


def _run_synthetic(end_seconds=4.0):
    parsed = parse_actor_xml(_SELF_SCHEDULING_CHART)
    # 120 BPM: beat = 2 * seconds.
    return run_sim(parsed.root, lambda b: b * 0.5, 0.0, end_seconds)


def test_update_chain_self_schedules():
    result = _run_synthetic()
    root_actor = next(a for a in result.actors.values()
                      if 'x' in a.keyframes())
    fires = len(root_actor.keyframes()['x'])
    # ~4s at the chart's own 0.02s cadence = ~200 fires, quantized by
    # the first-tick arm. The loop imposed no rig knowledge.
    assert 180 <= fires <= 205
    assert result.faults == 0


def test_update_cadence_is_the_charts_not_the_ticks():
    result = _run_synthetic()
    root_actor = next(a for a in result.actors.values()
                      if 'x' in a.keyframes())
    kfs = root_actor.keyframes()['x']
    gaps = [b.t - a.t for a, b in zip(kfs[10:20], kfs[11:21])]
    for gap in gaps:
        assert gap == pytest.approx(0.02, abs=1e-6)


def test_per_frame_mods_coalesce_to_one_window_per_player():
    result = _run_synthetic()
    windows = coalesce_applied(result.applied_mods)
    # A playerless ApplyModifiers targets BOTH engine players; player
    # expansion happens at ingestion so per-player clearalls can meet
    # these windows on the same key.
    assert len(windows) == 2
    assert sorted(w.player for w in windows) == [0, 1]
    for w in windows:
        assert w.modstring == '*10 50 drunk'
        # Live while song beat in [2, 4) = seconds [1, 2).
        assert w.t_start == pytest.approx(1.0, abs=0.05)
        assert w.t_end == pytest.approx(2.0, abs=0.05)
        assert w.calls > 40


def test_broadcast_reaches_child_at_fire_time():
    result = _run_synthetic()
    child = next(a for a in result.actors.values() if 'y' in a.keyframes())
    (kf,) = child.keyframes()['y']
    assert kf.values == (77.0,)
    # Fire #10 of a 0.02s chain that armed on the first tick.
    assert kf.t == pytest.approx(10 * 0.02, abs=0.05)


def test_classic_oncommand_recorded_on_child():
    result = _run_synthetic()
    child = next(a for a in result.actors.values() if 'y' in a.keyframes())
    assert child.keyframes()['alpha'][0].values == (0.5,)


_PROXY_CHART = """
<ActorFrame InitCommand="%function(self)
    holder = self
    self:x(100);
    P1 = SCREENMAN:GetTopScreen():GetChild('PlayerP1')
end"><children>
<Layer Type="ActorFrame" InitCommand="%function(self) self:x(20); end"
><children>
<Layer Type="ActorProxy" OnCommand="%function(self)
    self:SetTarget(P1:GetChild('NoteField')); self:x(3)
end"/>
</children></Layer>
<Layer Type="ActorProxy" OnCommand="%function(self)
    self:SetTarget(P1:GetChild('Judgment'))
end"/>
</children></ActorFrame>
"""


def _proxy_instances():
    from types import SimpleNamespace

    from analysis.games.notitg.mod_channels import compile_mod_channels
    from analysis.games.notitg.sim.producers import _sim_field_instances
    parsed = parse_actor_xml(_PROXY_CHART)
    result = run_sim(parsed.root, lambda b: b * 0.5, 0.0, 1.0)
    env = result.env
    doc = SimpleNamespace(root=parsed.root)
    return _sim_field_instances(doc, env, env.actor_keyframes(), None,
                                env.named_actor_keyframes(), None,
                                compile_mod_channels([]),
                                t0=result.load_seconds)


def test_notefield_proxy_becomes_composed_field_instance():
    instances = _proxy_instances()
    # The Judgment proxy is NOT a field instance - GetChild hands back
    # real per-name child recorders, so its target never matches a
    # player notefield. No player-2 touch -> no player instances either.
    assert len(instances) == 1
    (inst,) = instances
    assert inst['kind'] == 'proxy' and inst['player'] == 1
    assert inst['name'].startswith('holder_')
    # Parent-chain composition in the transform channel: root x(100) +
    # inner frame x(20) + the proxy's own x(3) place the capture centre
    # at design (123, 0).
    H, alpha = inst['transform'].at(0.5)
    assert alpha == 1.0
    assert H[2, 0] == pytest.approx(123.0 - 320.0)
    assert H[2, 1] == pytest.approx(0.0 - 240.0)




