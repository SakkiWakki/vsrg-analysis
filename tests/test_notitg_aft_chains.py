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


def test_deep_chain_resolves_transitively():
    # aftA -> sprite20 -> aftB -> sprite40 -> aftC: every link resolves
    # with its transitive depth; nothing is unresolved (the composed
    # evaluator walks the chain in document order).
    graph = build_chain_graph(
        aft_nodes={10: 'aftA', 30: 'aftB', 50: 'aftC'},
        blit_sources={20: 'aftA', 40: 'aftB'},
        draw_order=[10, 20, 30, 40, 50],
        screen_content_ids=set())
    assert graph.capture_of('aftB') == 'aftA'
    assert graph.capture_of('aftC') == 'aftB'
    assert (graph.depth_of('aftA'), graph.depth_of('aftB'),
            graph.depth_of('aftC')) == (0, 1, 2)
    assert graph.unresolved_depth == ()


def test_isolating_node_records_its_stage_sprite():
    graph = build_chain_graph(
        aft_nodes={10: 'aftA', 30: 'aftB'},
        blit_sources={20: 'aftA'},
        draw_order=[10, 20, 30],
        screen_content_ids=set())
    assert graph.stage_of('aftB') == 20
    assert graph.stage_of('aftA') is None


def test_self_capture_cycle_is_feedback():
    # A sprite drawing aftA's own texture captured back into aftA (the
    # cyriak recursion): previous-frame content re-entering the capture.
    # Classified feedback + demoted to whole-screen until the persistent
    # ping-pong targets consume it.
    graph = build_chain_graph(
        aft_nodes={30: 'aftA'},
        blit_sources={20: 'aftA'},
        draw_order=[20, 30],
        screen_content_ids=set())
    assert graph.feedback == {'aftA'}
    assert graph.capture_of('aftA') is None
    assert graph.stage_of('aftA') is None
    assert 'aftA' in graph.unresolved_depth


def test_two_node_capture_loop_is_feedback():
    # aftA captures a blit of aftB while aftB captures a blit of aftA -
    # a two-frame ping-pong loop; both are feedback.
    graph = build_chain_graph(
        aft_nodes={30: 'aftA', 50: 'aftB'},
        blit_sources={20: 'aftB', 40: 'aftA'},
        draw_order=[20, 30, 40, 50],
        screen_content_ids=set())
    assert graph.feedback == {'aftA', 'aftB'}
    assert graph.capture_of('aftA') is None
    assert graph.capture_of('aftB') is None


def test_chain_past_depth_cap_demotes_to_screen():
    # A 1+MAX_CHAIN_DEPTH+1 ladder: nodes at depth <= cap resolve, the
    # one past it demotes to whole-screen and is reported, never silent.
    from analysis.games.notitg.aft_chains import MAX_CHAIN_DEPTH

    n = MAX_CHAIN_DEPTH + 2
    aft_nodes = {i * 10: f'aft{i}' for i in range(n)}
    blit_sources = {i * 10 - 5: f'aft{i - 1}' for i in range(1, n)}
    draw_order = sorted(list(aft_nodes) + list(blit_sources))
    graph = build_chain_graph(aft_nodes, blit_sources, draw_order,
                              screen_content_ids=set())
    deepest, capped = f'aft{n - 2}', f'aft{n - 1}'
    assert graph.depth_of(deepest) == MAX_CHAIN_DEPTH
    assert graph.capture_of(deepest) == f'aft{n - 3}'
    assert graph.capture_of(capped) is None
    assert graph.depth_capped == (capped,)
    assert capped in graph.unresolved_depth


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


def test_chain_emits_capture_and_stage_instances():
    """A 2-stage chain compiles to the composed-capture instance set:
    the ROOT node materializes as an at-position 'capture' slot, the
    isolating node becomes a fold-at-consume 'stage' record carrying
    its captured sprite's transform, and every sampler is stamped with
    its direct source for the fold walk."""
    insts = _instances(
        '<ActorFrame><children>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="capA"/>'
        '<Sprite InitCommand="%function(self)'
        '  self:SetTexture(capA:GetTexture()) end" Var="s0"/>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="capB"/>'
        '<Sprite InitCommand="%function(self)'
        '  self:SetTexture(capB:GetTexture()) end" Var="s1"/>'
        '</children></ActorFrame>')
    (capture,) = [i for i in insts if i['kind'] == 'capture']
    (stage,) = [i for i in insts if i['kind'] == 'stage']
    consumers = [i for i in insts if i['kind'] == 'aft']
    assert stage['source'] == capture['name']
    assert {c['aft_node'] for c in consumers} \
        == {capture['name'], stage['name']}
    # Document order: the root's slot snapshots before its sampler
    # draws, and the stage record precedes the stage's consumers.
    kinds = [i['kind'] for i in insts]
    assert kinds == ['capture', 'aft', 'stage', 'aft']


def test_stage_chain_folds_into_consumer_entry():
    """The effect folds a consumer's stage chain at sample time: the
    entry blits the ROOT slot under the composed transform (stage first,
    then consumer - render-once/consume-many, no materialization), and
    its extra keys the root so the renderer serves the slot."""
    from types import SimpleNamespace

    from PySide6.QtCore import QPointF

    from analysis.games.notitg import field_compose

    def link(x):
        from analysis.player.render.effects.timeline import Keyframe
        return field_compose.link_timelines(
            {'x': [Keyframe(0.0, (x,), 0.0, 0)],
             'y': [Keyframe(0.0, (240.0,), 0.0, 0)]})

    capture = field_compose.instance('rootA', 'capture', 0, [link(320.0)])
    stage = field_compose.instance('nodeB', 'stage', 0, [link(330.0)])
    stage['source'] = 'rootA'
    consumer = field_compose.instance('s1', 'aft', 0, [link(325.0)])
    consumer['aft_node'] = 'nodeB'

    effect = NotitgFieldInstances([capture, stage, consumer])
    ctx = SimpleNamespace(t_now=1.0, chart_rect=(0, 0, 640, 480))
    frame = effect.at(ctx)
    by_scope = {entry[2]: entry for entry in frame.fields}
    assert by_scope['capture'][3] == 'rootA'
    entry = by_scope['screen']
    assert entry[3] == ('rootA', True)
    # Stage shifts +10, consumer +5: the folded blit lands +15.
    mapped = entry[0].map(QPointF(320.0, 240.0))
    assert (mapped.x(), mapped.y()) == pytest.approx((335.0, 240.0))


def test_chainless_aft_emits_no_capture_instance():
    # gat 1: one whole-screen AFT + consumer -> the legacy single-screen
    # path, no capture/stage instances at all.
    insts = _instances(
        '<ActorFrame><children>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="capA"/>'
        '<Sprite InitCommand="%function(self)'
        '  self:SetTexture(capA:GetTexture()) end" Var="s0"/>'
        '</children></ActorFrame>')
    assert {i['kind'] for i in insts} == {'aft'}


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
