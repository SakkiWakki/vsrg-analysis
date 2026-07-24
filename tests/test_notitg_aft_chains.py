"""AFT render-target chain graph (games/notitg/aft_chains) + its wiring
through sim.producers into field-instance `capture_source` annotations.

gat 2 (more_afts.xml / Mdrqnxtagon.xml) chains AFTs: an upstream AFT
captures a proxy, a sprite draws that capture, and a downstream AFT
captures ONLY that sprite - a 2-stage chain whose consumer must blit the
isolated upstream content, not the finished frame. gat 1 uses a single
AFT capturing the whole screen (capture_source None), which must stay
byte-identical.

These cover the pure graph builder (unit) and the end-to-end harvest
through a synthetic chart, so no NotITG install is needed.
"""
import pytest

from analysis.games.notitg.aft_chains import build_chain_graph

pytest.importorskip('lupa')

from analysis.games.notitg.field_instances import NotitgFieldInstances
from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.sim.producers import _sim_field_instances
from analysis.games.notitg.xml_actors import parse_actor_xml
from analysis.player.render.mods.channels import ModChannels

# A lone base player: keeps the direct-draw path, so the harvest emits
# only the AFT/proxy copies these tests inspect (no player instances).
_ONE_PLAYER = ModChannels({}, {0})


# ---- pure graph builder ------------------------------------------------

def test_single_aft_captures_screen():
    # One AFT, no upstream blit before it: it captures the whole screen
    # (capture_of None) - the gat 1 path, the byte-identical no-op.
    graph = build_chain_graph(
        aft_nodes={10: 'aftA'}, blit_sources={}, draw_order=[10],
        screen_content_ids=set())
    assert graph.capture_of('aftA') is None


def test_two_stage_chain_isolates_upstream():
    # upstream AFT 'aftA' captured; sprite 20 blits it; AFT 'aftB'
    # captures only that sprite -> aftB isolates aftA.
    graph = build_chain_graph(
        aft_nodes={10: 'aftA', 30: 'aftB'},
        blit_sources={20: 'aftA'},
        draw_order=[10, 20, 30],
        screen_content_ids=set())
    assert graph.capture_of('aftA') is None
    assert graph.capture_of('aftB') == 'aftA'


def test_screen_content_before_aft_is_not_isolated():
    # A live proxy/player draw between the blit and the AFT makes the AFT
    # a whole-screen capture, never a single-source isolation.
    graph = build_chain_graph(
        aft_nodes={10: 'aftA', 40: 'aftB'},
        blit_sources={20: 'aftA'},
        draw_order=[10, 20, 30, 40],
        screen_content_ids={30})
    assert graph.capture_of('aftB') is None


def test_multiple_blits_before_aft_is_not_isolated():
    # Two blits captured into one AFT is not a clean single-source
    # isolation -> whole screen.
    graph = build_chain_graph(
        aft_nodes={10: 'aftA', 12: 'aftB', 30: 'aftC'},
        blit_sources={20: 'aftA', 21: 'aftB'},
        draw_order=[10, 12, 20, 21, 30],
        screen_content_ids=set())
    assert graph.capture_of('aftC') is None


def test_deep_chain_flagged_unresolved():
    # aftA -> sprite20 -> aftB -> sprite40 -> aftC: aftC isolates aftB,
    # and aftB is itself fed by a chain stage (sprite40 targets it),
    # so aftC is recorded as an unresolved deep chain.
    graph = build_chain_graph(
        aft_nodes={10: 'aftA', 30: 'aftB', 50: 'aftC'},
        blit_sources={20: 'aftA', 40: 'aftB'},
        draw_order=[10, 20, 30, 40, 50],
        screen_content_ids=set())
    assert graph.capture_of('aftB') == 'aftA'
    assert graph.capture_of('aftC') == 'aftB'
    assert 'aftC' in graph.unresolved_depth


# ---- end-to-end harvest through producers ------------------------------

def _instances(xml):
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    root = parse_actor_xml(xml).root
    env.load_actors(root)
    doc = type('Doc', (), {'root': root})()
    return _sim_field_instances(
        doc, env, env.actor_keyframes(), osc_context=None,
        named_keyframes={}, field_oscillators=None,
        mod_channels=_ONE_PLAYER, t0=0.0)


def _afts(instances):
    return [i for i in instances if i['kind'] == 'aft']


def test_frag_sampler_emits_no_plain_blit():
    # A Frag= capture sampler draws THROUGH its shader - that draw is
    # the chart_shaders fullscreen pass, so no plain aft instance:
    # emitting one papers the raw capture over the composite (gat 2's
    # horizon sampler erased the afthell rig). The plain sibling still
    # blits.
    afts = _afts(_instances(
        '<ActorFrame><children>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="capA"/>'
        '<Sprite Frag="shaders/post.frag" InitCommand="%function(self)'
        '  self:SetTexture(capA:GetTexture()) end" Var="shaded"/>'
        '<Sprite InitCommand="%function(self)'
        '  self:SetTexture(capA:GetTexture()) end" Var="plain"/>'
        '</children></ActorFrame>'))
    assert len(afts) == 1


def test_single_aft_consumer_has_no_capture_source():
    # One AFT + one consumer sprite: whole-screen capture, no chain
    # annotation (the gat 1 parity path).
    afts = _afts(_instances(
        '<ActorFrame><children>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="capA"/>'
        '<Sprite InitCommand="%function(self)'
        '  self:SetTexture(capA:GetTexture()) end" Var="s0"/>'
        '</children></ActorFrame>'))
    assert len(afts) == 1
    assert afts[0].get('capture_source') is None


def test_chain_consumer_carries_capture_source():
    # aftA captured; sprite draws it; aftB captures that sprite; a final
    # sprite draws aftB -> that final consumer isolates aftA (aftB's
    # captured upstream).
    afts = _afts(_instances(
        '<ActorFrame><children>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="capA"/>'
        '<Sprite InitCommand="%function(self)'
        '  self:SetTexture(capA:GetTexture()) end" Var="s0"/>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="capB"/>'
        '<Sprite InitCommand="%function(self)'
        '  self:SetTexture(capB:GetTexture()) end" Var="s1"/>'
        '</children></ActorFrame>'))
    consumers = {i['name']: i.get('capture_source') for i in afts}
    # s0 blits aftA (whole screen), s1 blits aftB (isolates aftA).
    src_a = _aft_node_name(afts, 'capA')
    assert consumers['s0'] is None
    assert consumers['s1'] == src_a


def test_chain_freeze_key_is_upstream_source():
    # A chain consumer keys its preserve-texture freeze on the shared
    # upstream node, so co-consumers of one chain node share its retained
    # isolated capture; a non-chain consumer keys on its own name.
    afts = _afts(_instances(
        '<ActorFrame><children>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="capA"/>'
        '<Sprite InitCommand="%function(self)'
        '  self:SetTexture(capA:GetTexture()) end" Var="s0"/>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="capB"/>'
        '<Sprite InitCommand="%function(self)'
        '  self:SetTexture(capB:GetTexture()) end" Var="s1"/>'
        '</children></ActorFrame>'))
    eff = NotitgFieldInstances(afts)
    keys = {i['name']: eff._extra(i, 0.0)[0] for i in afts}
    src_a = _aft_node_name(afts, 'capA')
    assert keys['s0'] == 's0'          # whole-screen: own name
    assert keys['s1'] == src_a         # chain: upstream source node


def _aft_node_name(afts, var_prefix):
    """The synthetic node name of the AFT a chain consumer isolates: the
    only capture_source present among the annotated consumers."""
    sources = {i['capture_source'] for i in afts if i.get('capture_source')}
    assert len(sources) == 1
    return next(iter(sources))
