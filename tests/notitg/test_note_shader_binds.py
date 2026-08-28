"""Spec: per-player note-shader binds (LunaPlayer SetArrowShader
@0x00533740, SetHoldShader @0x00533aa0, SetReceptorShader @0x00535a40,
Clear* unbinds).

The chart passes a SOURCE actor's GetShader() handle; our GetShader
chains self, so the argument is the source actor's table. The env
resolves it to a recorder id, records it on the player's
`note_shader:{category}` step channel (rest -1 = unbound), and collects
the source id so the producers can export its Frag=/Vert= program.
"""
import pytest

pytest.importorskip('lupa')

from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.xml_actors import parse_actor_xml

_CHART = ('<ActorFrame><children>'
          '<Quad Name="Player"/><Actor Name="Source"/>'
          '</children></ActorFrame>')


@pytest.fixture
def env():
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(_CHART).root)
    return env


def _ids(env):
    labels = {label.split(':')[-1]: rec_id
              for rec_id, label in env._labels.items()}
    return labels['Player'], labels['Source']


def test_bind_records_the_source_recorder_id(env):
    player, source = _ids(env)
    handle = env._tables[source]
    for verb, category in (('SetArrowShader', 'arrow'),
                           ('SetHoldShader', 'hold'),
                           ('SetReceptorShader', 'receptor')):
        env._actor_poke(player, verb, handle)
        value = env._actors[player].get(f'note_shader:{category}')
        assert value == float(source), f'{verb} did not record'
    assert env.note_shader_sources == {source}


def test_clear_returns_the_channel_to_unbound(env):
    player, source = _ids(env)
    env._actor_poke(player, 'SetArrowShader', env._tables[source])
    env._actor_poke(player, 'ClearArrowShader')
    assert env._actors[player].get('note_shader:arrow') == -1.0


def test_unbound_rest_is_minus_one(env):
    player, _ = _ids(env)
    assert env._actors[player].get('note_shader:hold') == -1.0


def test_bind_without_a_resolvable_handle_is_dropped(env):
    player, _ = _ids(env)
    env._actor_poke(player, 'SetArrowShader', 'not-an-actor')
    assert env._actors[player].get('note_shader:arrow') == -1.0
    assert env.note_shader_sources == set()


def test_a_delayed_closure_still_means_its_own_actor():
    """A load body's `self` is a real PARAMETER, so a closure it
    registers for later holds its actor as an upvalue (engine
    semantics). The stripped-statement form resolved `self` through the
    globals at FIRE time: Government Knows' suzumebachi handler bound
    the master updater frame's (nonexistent) shader instead of the
    Frag= actor's, and the wasp section lost its program."""
    chart = '''<ActorFrame><children>
<Quad Name="Player"/>
<Actor Name="Source" OnCommand="%function( self )
  handlers = {}
  handlers[1] = function()
    P:SetArrowShader( self:GetShader() )
  end
end"/>
</children></ActorFrame>'''
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(parse_actor_xml(chart).root)
    player, source = _ids(env)
    env._host.env['P'] = env._tables[player]
    # Fire under a DIFFERENT global self, as the chart's updater would.
    env._host.env['self'] = env._tables[player]
    env._host.run('handlers[1]()')
    assert env.note_shader_sources == {source}
    assert env._actors[player].get('note_shader:arrow') == float(source)
