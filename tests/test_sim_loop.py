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


def test_per_frame_mods_coalesce_to_one_window():
    result = _run_synthetic()
    windows = coalesce_applied(result.applied_mods)
    assert len(windows) == 1
    (w,) = windows
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


@pytest.mark.skipif(not GAT_SM.exists(), reason='local gat chart absent')
def test_gat_compiles_via_sim_to_the_contract():
    from analysis.games.notitg.sim.producers import compile_via_sim
    compiled = compile_via_sim(GAT_SM, end_seconds=60.0)
    assert compiled is not None
    for key in ('mod_events', 'shader_flags', 'unsupported', 'elements',
                'tree', 'has_background', 'field_copies', 'screen_transform',
                'base_field_hidden', 'named_actors', 'recorded_keyframes',
                'warnings'):
        assert key in compiled, key
    assert not any('aborted' in w for w in compiled['warnings'])
    assert compiled['named_actors'] > 50
    assert len(compiled['tree']) > 0
    assert len(compiled['mod_events']) > 100
    rows = compiled['mod_events']
    assert all(r['t_end'] >= r['t_start'] for r in rows)


@pytest.mark.skipif(not GAT_SM.exists(), reason='local gat chart absent')
def test_gat_runs_under_the_loop():
    result = run_chart_sim(GAT_SM, end_seconds=45.0)
    assert result is not None
    stats = summarize(result)
    # The load pass alone binds hundreds of actors; the first 45s cover
    # the intro + the beat-128 proxy-wall section, so the chart's own
    # Update rig must have driven real recording.
    assert stats['actors'] > 300
    assert stats['recorded_keyframes'] > 500
    assert stats['applied_calls'] > 100
    assert stats['mod_windows'] > 5
    assert stats['ticks'] == pytest.approx(45.0 * 60, abs=61)
    print('gat sim stats:', stats)
