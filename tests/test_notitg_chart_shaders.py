"""Harvest of map-supplied `Frag=` fragment shaders from the actor tree
(sim.producers._chart_shaders): the FULLSCREEN half of the tier-2 shader
path that shader_bridge.chart_shader_effect consumes.

Synthetic reductions of the getfucked2 pattern: an
`<... Type="ActorFrameTexture">` render target named via Create() (not
SetTextureName), a fullscreen sprite that draws its capture
(`SetTexture(aft:GetTexture())`) with a `Frag=` program, and per-frame
`GetShader():uniform1f(...)` pokes. Verifies the fullscreen/per-actor
gate, uniform-stream harvest, and visibility windows without needing the
NotITG install.
"""
import pytest

pytest.importorskip('lupa')

from analysis.games.notitg.sim.actor import SimActor
from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.sim.producers import _chart_shaders
from analysis.games.notitg.xml_actors import parse_actor_xml


def test_simactor_records_uniform_streams():
    a = SimActor(0.0)
    a.poke('GetShader', [])
    a.poke('uniform1f', ['phase', 0.5])
    assert a.keyframes()['uniform:phase'][0].values[0] == pytest.approx(0.5)
    # A uniform is ordinary tween state: the destination advances with an
    # add/get like any scalar, confirming it rides the same write path.
    a.poke('addaux', [])  # unrelated verb, must not disturb the channel
    a.poke('uniform1f', ['phase', 0.9])
    assert a.get('uniform:phase') == pytest.approx(0.9)


def test_getshader_uniform_chains_in_sim():
    # In the sim, GetShader() returns the recorder so :uniform1f lands on
    # the same actor - the whole reason the pokes are harvestable.
    env, _w = _load(
        '<ActorFrame InitCommand="%function(self)'
        '  self:GetShader():uniform1f(\'amt\', 3.0) end"/>')
    streams = {p: k for a in env.actors.values()
               for p, k in a.keyframes().items() if p.startswith('uniform:')}
    assert streams['uniform:amt'][0].values[0] == pytest.approx(3.0)


def test_actorframetexture_type_marks_aft():
    env, _w = _load(
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="cap"/>')
    cap = next(a for a in env.actors.values() if a.is_aft)
    assert cap.aft_source is None  # it IS the target, not a copy
    assert cap.read('GetTexture').marker.startswith('aft:')


def _load(xml):
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(xml).root)
    return env, None


def _chart_with_frag(tmp_path, frag_body='sampler0'):
    lua = tmp_path / 'lua'
    (lua / 'shaders').mkdir(parents=True)
    (lua / 'shaders' / 'post.frag').write_text(
        'uniform sampler2D sampler0;\n'
        'uniform float phase;\n'
        'varying vec2 imageCoord;\n'
        f'void main() {{ gl_FragColor = texture2D({frag_body}, imageCoord)'
        ' * phase; }\n')
    return lua


def _run(lua_dir, xml):
    """Parse `xml` as the chart's default.xml under `lua_dir`, load it,
    and harvest chart_shaders. Mirrors what producers does, minimally."""
    from analysis.games.notitg import modfile
    (lua_dir / 'default.xml').write_text(xml)
    root, _c, _cl = modfile._load_document(lua_dir)
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(root)
    doc = type('Doc', (), {'root': root})()
    return _chart_shaders(doc, env, env.actor_keyframes())


def test_fullscreen_frag_harvested(tmp_path):
    lua = _chart_with_frag(tmp_path)
    passes = _run(lua, (
        '<ActorFrame><children>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="cap"/>'
        '<Sprite Frag="shaders/post.frag" InitCommand="%function(self)'
        '  self:SetTexture(cap:GetTexture())'
        '  self:GetShader():uniform1f(\'phase\', 0.7)'
        ' end"/>'
        '</children></ActorFrame>'))
    assert len(passes) == 1
    entry = passes[0]
    assert entry['frag_path'].endswith('post.frag')
    assert 'phase' in entry['uniforms']
    assert entry['uniforms']['phase'][0]['strength'] == pytest.approx(0.7)


def test_per_actor_frag_skipped(tmp_path):
    # A Frag= sprite that draws its OWN small texture (no AFT capture)
    # is per-actor (Stage B): feeding it the fullscreen capture would
    # wrongly post-process the whole screen, so it is not harvested.
    lua = _chart_with_frag(tmp_path)
    passes = _run(lua, (
        '<Sprite Frag="shaders/post.frag" Texture="white"'
        ' InitCommand="%function(self)'
        '  self:GetShader():uniform1f(\'phase\', 0.5) end"/>'))
    assert passes == []


def test_hidden_channel_becomes_visibility_window(tmp_path):
    lua = _chart_with_frag(tmp_path)
    passes = _run(lua, (
        '<ActorFrame><children>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="cap"/>'
        '<Sprite Frag="shaders/post.frag" InitCommand="%function(self)'
        '  self:SetTexture(cap:GetTexture()); self:hidden(1) end"'
        ' OnCommand="hidden,0"/>'
        '</children></ActorFrame>'))
    assert len(passes) == 1
    windows = passes[0]['windows']
    # hidden,1 then shown -> a 0 (off) then 1 (live) strength window.
    assert [w['strength'] for w in windows] == [0.0, 1.0]
