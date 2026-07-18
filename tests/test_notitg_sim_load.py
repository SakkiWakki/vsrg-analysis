"""Engine load-pass semantics of the NotITG sim: Condition gating,
load-captured expression commands, creation order, and the Lua 5.0
lexer compatibility rewrite. All cases are synthetic reductions of
shapes real charts exercise (the XGML template family, Snow Halation's
`485then`, GOODTEK's nested long comments)."""
import pytest

pytest.importorskip('lupa')

from analysis.games.notitg import xml_actors
from analysis.games.notitg.sim.env import SimEnvironment
from analysis.games.notitg.xml_actors import parse_actor_xml


def _load(xml):
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    warnings = env.load_actors(parse_actor_xml(xml).root)
    return env, warnings


def _recorded(env, prop):
    return [kf for frames in env.actor_keyframes().values()
            for name, kfs in frames.items() if name == prop for kf in kfs]


def test_condition_runs_setup_and_prunes_falsy_subtree():
    root = parse_actor_xml(
        '<ActorFrame><children>'
        '<Layer Condition="(function() helper = 7 return true end)()"/>'
        '<Layer Name="Dropped" Condition="false">'
        '<children><Quad Name="AlsoDropped"/></children></Layer>'
        '<Quad InitCommand="%function(self) got = helper end"/>'
        '</children></ActorFrame>').root
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.load_actors(root)
    assert env._host.env['got'] == 7
    names = [c.attrs.get('Name') for c in root.children]
    assert 'Dropped' not in names


def test_faulting_condition_keeps_actor():
    env, warnings = _load(
        '<ActorFrame><children>'
        '<Quad Name="Kept" Condition="no_such_fn()"/>'
        '</children></ActorFrame>')
    assert any('Condition' in w for w in warnings)
    assert 'Kept' in env._labels.values() or any(
        'Kept' in label for label in env._labels.values())


def test_child_condition_reads_parent_init_global():
    # Engine creation order: parent InitCommand runs BEFORE children
    # load, so a child's Condition sees what it bound (XGML `prefix`).
    env, _w = _load(
        '<ActorFrame InitCommand="%function(self) flag = true end">'
        '<children>'
        '<Quad Name="Gated" Condition="flag"/>'
        '</children></ActorFrame>')
    assert any('Gated' in label for label in env._labels.values())


def test_expression_command_captured_at_actor_load():
    # `%expr` commands evaluate ONCE at creation and hold the resulting
    # function - the XGML template nils its globals after init, and the
    # command must keep firing (UpdateCommand="%prefix.update").
    env, _w = _load(
        '<ActorFrame InitCommand="%function(self)'
        ' function ping(self) self:x(55) end end">'
        '<children>'
        '<Quad GoMessageCommand="%ping"'
        ' InitCommand="%function(self) end"/>'
        '<Quad InitCommand="%function(self) ping = nil end"/>'
        '</children></ActorFrame>')
    env._broadcast(None, 'Go')
    assert any(kf.value == 55.0 for kf in _recorded(env, 'x'))


def test_broken_update_body_degrades_to_warning():
    env = SimEnvironment(0.0, 0, to_seconds=lambda b: b * 0.5)
    env.run_update_body('this is not lua (', name='t.Update')
    env.run_update_body('this is not lua (', name='t.Update')
    assert len(env._warnings) == 1
    assert 't.Update' in env._warnings[0]


# -- Lua 5.0 lexer compatibility ------------------------------------------

def test_lua50_number_keyword_gets_space():
    src = xml_actors._lua50_compat('if beat > 453 and beat < 485then x=1 end')
    assert '485 then' in src


def test_lua50_number_inside_identifier_untouched():
    src = xml_actors._lua50_compat('x2then = 0x1F + 1e5')
    assert src == 'x2then = 0x1F + 1e5'


def test_lua50_unknown_escape_passes_char_through():
    src = xml_actors._lua50_compat(r"find(mod, '\+$') .. '\n'")
    assert r"'+$'" in src
    assert r"'\n'" in src


def test_lua50_nested_long_comment_blanked_to_matched_close():
    src = xml_actors._lua50_compat('--[[ a [[ b ]] c ]] live = 1')
    assert src.endswith('live = 1')
    assert ' b ' not in src


def test_lua50_number_in_string_or_comment_untouched():
    src = xml_actors._lua50_compat("s = '485then' -- 485then")
    assert src == "s = '485then' -- 485then"


def test_attrs_rewritten_only_for_lua_values():
    parsed = parse_actor_xml(
        '<Quad Condition="beat < 485then true" Texture="485then.png"'
        ' InitCommand="%function(self) if x < 485then end end"/>')
    attrs = parsed.root.attrs
    assert '485 then' in attrs['Condition']
    assert '485 then' in attrs['InitCommand']
    assert attrs['Texture'] == '485then.png'
