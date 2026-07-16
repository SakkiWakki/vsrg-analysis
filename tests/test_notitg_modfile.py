"""NotITG modfile compiler: actor XML parsing, CODE-chunk mod harvest,
classic command -> storyboard keyframes, and a guarded integration test
against the real gat pilot."""
from pathlib import Path

import pytest

pytest.importorskip('lupa')

from analysis.games.notitg import xml_actors
from analysis.games.notitg.mod_stubs import StubEnvironment
from analysis.games.notitg.modfile import (compile_modfile, parse_fgchanges)

_GAT_SM = Path('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/'
               'UKSRT8/5. gat/gat.sm')


# -- XML parser -----------------------------------------------------------

def test_type_attribute_drives_actor_kind():
    parsed = xml_actors.parse_actor_xml(
        '<LAER Type="Sprite" Texture="white" OnCommand="x,10"/>')
    assert parsed.root.kind == 'Sprite'
    assert parsed.root.attrs['Texture'] == 'white'


def test_lua_body_unwrapped_from_function_wrapper():
    xml = ('<CODE Type="Quad" InitCommand="%function(self)\n'
           '  mods = {}\nend"/>')
    parsed = xml_actors.parse_actor_xml(xml)
    assert len(parsed.lua_chunks) == 1
    body = parsed.lua_chunks[0].body
    assert 'function' not in body
    assert 'mods = {}' in body


def test_bare_percent_expression_kept():
    parsed = xml_actors.parse_actor_xml(
        '<Quad Condition="%FUCK_EXE"/>')
    assert parsed.lua_chunks[0].body == 'FUCK_EXE'


def test_classic_command_parsed_into_verbs():
    parsed = xml_actors.parse_actor_xml(
        '<Sprite Type="Sprite" '
        'OnCommand="x,100;zoom,2;linear,1;y,300;diffusealpha,0.5"/>')
    assert len(parsed.classic_commands) == 1
    verbs = [v for v, _ in parsed.classic_commands[0].commands]
    assert verbs == ['x', 'zoom', 'linear', 'y', 'diffusealpha']


def test_commas_inside_parens_do_not_split():
    parsed = xml_actors.parse_actor_xml(
        '<Quad Type="Quad" OnCommand="y,-22+(18*0);zoom,1"/>')
    commands = parsed.classic_commands[0].commands
    assert commands == [('y', ['-22+(18*0)']), ('zoom', ['1'])]


def test_gt_inside_lua_body_does_not_end_tag():
    xml = ('<CODE Type="Quad" InitCommand="%function(self)\n'
           '  if a > b and c < d then x = 1 end\nend" OnCommand="hidden,1"/>')
    parsed = xml_actors.parse_actor_xml(xml)
    assert parsed.root.kind == 'Quad'
    assert 'a > b and c < d' in parsed.lua_chunks[0].body
    assert parsed.classic_commands[0].commands == [('hidden', ['1'])]


def test_children_wrapper_flattens_into_parent():
    xml = ('<ActorFrame><children>'
           '<Quad Type="Quad" OnCommand="x,1"/>'
           '<Quad Type="Quad" OnCommand="x,2"/>'
           '</children></ActorFrame>')
    parsed = xml_actors.parse_actor_xml(xml)
    assert parsed.root.kind == 'ActorFrame'
    assert len(parsed.root.children) == 2


def test_duplicate_attribute_keeps_last_value():
    parsed = xml_actors.parse_actor_xml(
        '<Quad OnCommand="x,1" OnCommand="x,2"/>')
    assert parsed.root.attrs['OnCommand'] == 'x,2'


def test_malformed_unterminated_tag_survives():
    parsed = xml_actors.parse_actor_xml(
        '<ActorFrame><children><Quad Type="Quad" OnCommand="x,1"/>'
        '<Quad Type="Quad" ')
    assert parsed.root.kind == 'ActorFrame'
    kinds = [c.kind for c in parsed.root.children]
    assert 'Quad' in kinds


def test_etree_chokes_on_notitg_xml_documenting_the_reason():
    import xml.etree.ElementTree as ET

    with pytest.raises(ET.ParseError):
        ET.fromstring('<CODE InitCommand="%function(self) '
                      'if a < b then end end"/>')


# -- classic command -> storyboard elements -------------------------------

def _seconds(beat):
    return float(beat)


def test_classic_command_string_becomes_element_keyframes():
    from analysis.games.notitg import modfile

    parsed = xml_actors.parse_actor_xml(
        '<Sprite Type="Sprite" Texture="white" '
        'OnCommand="x,100;linear,1;y,300;diffusealpha,0.5"/>')
    elements = modfile._compile_elements(
        parsed.classic_commands, _seconds, start_beat=0.0)
    assert len(elements) == 1
    element = elements[0]

    assert element.kind == 'sprite'
    assert element.asset == 'white'
    # x is set immediately; y eases over 1s starting at t=0.
    assert element.sample('x', 0.0) == (100.0,)
    assert element.sample('y', 1.0) == (300.0,)
    assert element.sample('y', 0.0) != (300.0,)
    assert element.sample('alpha', 1.0) == (0.5,)


def test_sleep_advances_the_command_clock():
    from analysis.games.notitg import modfile

    parsed = xml_actors.parse_actor_xml(
        '<Quad Type="Quad" OnCommand="x,10;sleep,2;x,20"/>')
    elements = modfile._compile_elements(
        parsed.classic_commands, _seconds, start_beat=0.0)
    element = elements[0]
    assert element.sample('x', 0.0) == (10.0,)
    assert element.sample('x', 1.9) == (10.0,)
    assert element.sample('x', 2.0) == (20.0,)


def test_diffuse_sets_color_and_alpha():
    from analysis.games.notitg import modfile

    parsed = xml_actors.parse_actor_xml(
        '<Quad Type="Quad" OnCommand="diffuse,1,0,0,0.5"/>')
    element = modfile._compile_elements(
        parsed.classic_commands, _seconds, start_beat=0.0)[0]
    assert element.sample('color', 0.0) == (1.0, 0.0, 0.0)
    assert element.sample('alpha', 0.0) == (0.5,)


# -- CODE chunk mod harvest -----------------------------------------------

def test_stub_harvests_mods_table():
    env = StubEnvironment(start_beat=0.0)
    env.run("mods = {{4, 8, '*5 40 drunk', 'len'}, "
            "{16, 32, '2x', 'end', 1}}", name='t')
    rows = env.mods
    assert len(rows) == 2
    assert rows[0][3] == '*5 40 drunk'
    assert rows[0][4] == 'len'
    assert rows[1][5] == 1


def test_stub_harvests_mods2_and_mod_actions():
    env = StubEnvironment(start_beat=0.0)
    env.run("mods2 = {{1.5, 2.0, 'drunk', 'end'}}\n"
            "mod_actions = {{4, function() end}}", name='t')
    assert env.mods2[0][3] == 'drunk'
    assert callable(env.mod_actions[0][2])


def test_stub_provides_lua50_math_mod_and_gamestate():
    env = StubEnvironment(start_beat=0.0)
    env.run("result = math.mod(7, 3)\n"
            "b = GAMESTATE:GetSongBeat()", name='t')
    assert env._host.env['result'] == 1.0
    assert env._host.env['b'] == 0.0


def test_stub_records_shader_flags_at_load():
    env = StubEnvironment(start_beat=12.0)
    env.run("GAMESTATE:SetShaderFlag(55)\n"
            "GAMESTATE:SetShaderFlagNum(48, 2)", name='t')
    flags = env.shader_flags
    assert flags[0] == (12.0, 55, None)
    assert flags[1] == (12.0, 48, 2)


def test_stub_permissive_singleton_swallows_unknown_calls():
    env = StubEnvironment(start_beat=0.0)
    # An engine method we never stubbed must no-op, not fault.
    env.run("SCREENMAN:SomethingWeNeverModeled():AndChain()\n"
            "ok = true", name='t')
    assert env._host.env['ok'] is True


def test_mod_event_normalization_len_vs_end():
    from analysis.games.notitg import modfile

    env = StubEnvironment(start_beat=0.0)
    env.run("mods = {{4, 8, 'a', 'len'}, {4, 20, 'b', 'end'}}", name='t')
    events = modfile._normalize_mod_events(env, _seconds)
    by_string = {e['modstring']: e for e in events}
    assert by_string['a']['t_start'] == 4.0
    assert by_string['a']['t_end'] == 12.0    # len: start + length
    assert by_string['b']['t_end'] == 20.0    # end: the field itself


# -- resilience -----------------------------------------------------------

def test_compile_never_raises_on_missing_lua(tmp_path):
    sm = tmp_path / 'nolua.sm'
    sm.write_text('#TITLE:x;\n#BPMS:0.000=120.000;\n')
    assert compile_modfile(str(sm)) is None


def test_compile_survives_malformed_xml(tmp_path):
    song = tmp_path / 'song'
    lua = song / 'lua'
    lua.mkdir(parents=True)
    (lua / 'default.xml').write_text(
        '<ActorFrame><children><CODE Type="Quad" '
        'InitCommand="%function(self) mods = {{4,8,\'drunk\',\'len\'}} '
        'this is not valid lua $$$ end"/>'
        '<Quad Type="Quad" OnCommand="x,5;y,')  # truncated mid-attribute
    sm = song / 'chart.sm'
    sm.write_text('#TITLE:x;\n#OFFSET:0.000;\n#BPMS:0.000=120.000;\n'
                  '#FGCHANGES:0.000=lua=1.000=0=0=1=====,;\n')
    result = compile_modfile(str(sm))
    assert result is not None
    assert isinstance(result['mod_events'], list)
    assert isinstance(result['warnings'], list)


def test_parse_fgchanges_reads_beat_and_name(tmp_path):
    sm = tmp_path / 'chart.sm'
    sm.write_text('#FGCHANGES:0.500=lua=1.000=0=0=1=====,;\n'
                  '#BGCHANGES:0.000=bg=1.000=0=0=1=====,;\n')
    entries = parse_fgchanges(str(sm))
    fg = [e for e in entries if e[2] == 'FGCHANGES']
    assert fg == [(0.5, 'lua', 'FGCHANGES')]


# -- integration (guarded on the real pilot) ------------------------------

@pytest.mark.skipif(not _GAT_SM.exists(),
                    reason='NotITG gat pilot not present')
def test_gat_pilot_harvests_mods_and_elements():
    result = compile_modfile(str(_GAT_SM))
    assert result is not None
    assert len(result['mod_events']) > 100
    assert len(result['elements']) > 0
    assert any(e['time_based'] for e in result['mod_events'])
    assert any(e['player'] is not None for e in result['mod_events'])
    assert result['unsupported']['count'] > 0
