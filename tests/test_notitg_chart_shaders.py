"""Map-supplied `Frag=` fragment shaders compile as SHADED FIELD-INSTANCE
BLITS of their source AFT's at-position capture slot - never as a
finished-frame fullscreen pass.

The pass model died of the AFT curtain idiom: charts draw a black quad
between the capture node and its Frag= sampler (the raw scene is covered,
the sampler redraws the pre-curtain capture shaded on top), so a pass
sampling the finished frame gets black in and emits black out (gat 2's
MonitorOn window, chart 440s -> end). The at-position slot is pre-curtain
by construction, which is exactly the engine's sampler0 = m_pTexture
contract (Sprite::DrawPrimitives).

Synthetic reductions of the getfucked2 pattern: an
`<... Type="ActorFrameTexture">` render target named via Create() (not
SetTextureName), a sprite that draws its capture
(`SetTexture(aft:GetTexture())`) with a `Frag=` program, and per-frame
`GetShader():uniform1f(...)` pokes. No NotITG install needed.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip('lupa')

from analysis.games.notitg.field_instances import NotitgFieldInstances
from analysis.games.notitg.sim.actor import SimActor
from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.sim.producers import _sim_field_instances
from analysis.games.notitg.xml_actors import parse_actor_xml
from analysis.player.render.mods.channels import ModChannels

# A lone base player keeps the direct-draw path, so the harvest emits
# only the AFT/proxy copies these tests inspect (no player instances).
_ONE_PLAYER = ModChannels({}, {0})


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
    env, _root = _load(
        '<ActorFrame InitCommand="%function(self)'
        '  self:GetShader():uniform1f(\'amt\', 3.0) end"/>')
    streams = {p: k for a in env.actors.values()
               for p, k in a.keyframes().items() if p.startswith('uniform:')}
    assert streams['uniform:amt'][0].values[0] == pytest.approx(3.0)


def test_actorframetexture_type_marks_aft():
    env, _root = _load(
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="cap"/>')
    cap = next(a for a in env.actors.values() if a.is_aft)
    assert cap.aft_source is None  # it IS the target, not a copy
    assert cap.read('GetTexture').marker.startswith('aft:')


def _load(xml):
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    root = parse_actor_xml(xml).root
    env.load_actors(root)
    return env, root


def _chart_with_frag(tmp_path):
    lua = tmp_path / 'lua'
    (lua / 'shaders').mkdir(parents=True)
    (lua / 'shaders' / 'post.frag').write_text(
        'uniform sampler2D sampler0;\n'
        'uniform float phase;\n'
        'varying vec2 imageCoord;\n'
        'void main() { gl_FragColor = texture2D(sampler0, imageCoord)'
        ' * phase; }\n')
    return lua


def _instances(lua_dir, xml):
    """Parse `xml` as the chart's default.xml under `lua_dir`, load it,
    and harvest the field instances. Mirrors what producers does,
    minimally (the modfile loader stamps `_base_dir` so Frag= paths
    resolve against the chart directory)."""
    from analysis.games.notitg import modfile
    (lua_dir / 'default.xml').write_text(xml)
    root, _c, _cl = modfile._load_document(lua_dir / 'default.xml')
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(root)
    doc = type('Doc', (), {'root': root})()
    return _sim_field_instances(
        doc, env, env.actor_keyframes(), osc_context=None,
        named_keyframes={}, field_oscillators=None,
        mod_channels=_ONE_PLAYER, t0=0.0)


def _shaded(instances):
    return [i for i in instances if i['kind'] == 'aft' and i.get('frag')]


def test_fullscreen_frag_sampler_compiles_as_shaded_blit(tmp_path):
    # The gat 2 monitor rig shape: an untransformed fullscreen sampler.
    # It must emit its own shaded blit (of the pre-curtain slot), not
    # vanish into a finished-frame pass.
    lua = _chart_with_frag(tmp_path)
    insts = _instances(lua, (
        '<ActorFrame><children>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="cap"/>'
        '<Sprite Frag="shaders/post.frag" InitCommand="%function(self)'
        '  self:SetTexture(cap:GetTexture())'
        '  self:GetShader():uniform1f(\'phase\', 0.7)'
        ' end"/>'
        '</children></ActorFrame>'))
    (inst,) = _shaded(insts)
    assert inst['frag'].endswith('post.frag')

    effect = NotitgFieldInstances(insts)
    frame = effect.at(SimpleNamespace(t_now=1.0, chart_rect=(0, 0, 640, 480)))
    entry = next(e for e in frame.fields
                 if e[2] == 'screen' and len(e[3] or ()) >= 3)
    _key, _live, (path, uniforms, _tint, _add, _samplers), _mesh = entry[3]
    assert path.endswith('post.frag')
    assert uniforms['phase'] == pytest.approx(0.7)


def test_per_actor_frag_sprite_is_no_field_instance(tmp_path):
    # A Frag= sprite that draws its OWN small texture (no AFT capture)
    # is not a capture consumer: no field instance, its shading is the
    # (deferred) storyboard-sprite shader tier.
    lua = _chart_with_frag(tmp_path)
    insts = _instances(lua, (
        '<Sprite Frag="shaders/post.frag" Texture="white"'
        ' InitCommand="%function(self)'
        '  self:GetShader():uniform1f(\'phase\', 0.5) end"/>'))
    assert insts == []


def test_hidden_channel_gates_the_blit(tmp_path):
    # These sprites sit hidden,1 until their section's show message; the
    # blit's visibility rides the instance transform (None while hidden),
    # replacing the old pass-tier `windows` stream.
    lua = _chart_with_frag(tmp_path)
    insts = _instances(lua, (
        '<ActorFrame><children>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="cap"/>'
        '<Sprite Frag="shaders/post.frag" InitCommand="%function(self)'
        '  self:SetTexture(cap:GetTexture()); self:hidden(1) end"'
        ' Var="curtained"/>'
        '<Sprite Frag="shaders/post.frag" InitCommand="%function(self)'
        '  self:SetTexture(cap:GetTexture()) end" Var="shown"/>'
        '</children></ActorFrame>'))
    hidden, shown = _shaded(insts)
    assert hidden['transform'].at(1.0) is None
    assert shown['transform'].at(1.0) is not None


def test_uniform_texture_file_bind_rides_the_blit_payload(tmp_path):
    # The ascii.frag idiom: the frag declares an extra sampler and the
    # chart binds it to another sprite's FILE texture. The instance
    # carries {sampler: absolute path} so the GL blit uploads it.
    lua = _chart_with_frag(tmp_path)
    (lua / 'asciitable.png').write_bytes(b'\x89PNG')
    insts = _instances(lua, (
        '<ActorFrame><children>'
        '<Sprite File="asciitable.png" Var="spriteAscii"/>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="cap"/>'
        '<Sprite Frag="shaders/post.frag" InitCommand="%function(self)'
        '  self:SetTexture(cap:GetTexture())'
        "  self:GetShader():uniformTexture('samplerAscii',"
        ' spriteAscii:GetTexture())'
        ' end"/>'
        '</children></ActorFrame>'))
    (inst,) = _shaded(insts)
    assert inst['frag_samplers'] \
        == {'samplerAscii': str(lua / 'asciitable.png')}


def test_aft_sampler_bind_is_not_a_file_bind(tmp_path):
    # A sampler bound to a CAPTURE texture has no file to upload; the
    # payload leaves it out, so a frag needing it fails to build on GL
    # and the blit falls back to its unshaded draw rather than running
    # black.
    lua = _chart_with_frag(tmp_path)
    insts = _instances(lua, (
        '<ActorFrame><children>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="cap"/>'
        '<ActorFrameTexture InitCommand="%function(self) self:Create() end"'
        ' Var="cap2"/>'
        '<Sprite Frag="shaders/post.frag" InitCommand="%function(self)'
        '  self:SetTexture(cap:GetTexture())'
        "  self:GetShader():uniformTexture('samplerFeedback',"
        ' cap2:GetTexture())'
        ' end"/>'
        '</children></ActorFrame>'))
    (inst,) = _shaded(insts)
    assert inst['frag_samplers'] == {}
