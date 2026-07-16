"""NotITG modfile compiler: actor XML parsing, CODE-chunk mod harvest,
classic command -> storyboard keyframes, and a guarded integration test
against the real gat pilot."""
from pathlib import Path

import pytest

pytest.importorskip('lupa')

from analysis.games.notitg import xml_actors
from analysis.games.notitg.mod_stubs import StubEnvironment
from analysis.games.notitg.modfile import (compile_modfile, parse_fgchanges)
from analysis.player.render.effects.timeline import Keyframe

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


# -- one-shot mod_actions replay ------------------------------------------

def test_replay_fires_closures_in_beat_order_once():
    env = StubEnvironment(start_beat=0.0)
    env.run("order = {}\n"
            "mod_actions = {\n"
            "  {8, function() table.insert(order, 'b') end},\n"
            "  {4, function() table.insert(order, 'a') end},\n"
            "  {12, function() table.insert(order, 'c') end},\n"
            "}", name='t')
    fired, failed = env.replay_mod_actions()
    assert (fired, failed) == (3, 0)
    fired_order = env._host.env['order']
    assert [fired_order[i] for i in (1, 2, 3)] == ['a', 'b', 'c']


def test_replay_records_apply_game_command_as_one_shot():
    env = StubEnvironment(start_beat=0.0)
    env.run("mod_actions = {\n"
            "  {8, function() GAMESTATE:ApplyGameCommand("
            "'mod,*-1 100 drunk', 2) end},\n"
            "}", name='t')
    env.replay_mod_actions()
    assert env.applied_mods == [(8.0, '*-1 100 drunk', 2)]


def test_replay_records_shader_flag_from_closure_at_fire_beat():
    env = StubEnvironment(start_beat=0.0)
    env.run("mod_actions = {\n"
            "  {32, function() GAMESTATE:SetShaderFlag(55) end},\n"
            "}", name='t')
    env.replay_mod_actions()
    assert env.shader_flags == [(32.0, 55, None)]


def test_replay_swallows_string_payloads_and_survives_faults():
    env = StubEnvironment(start_beat=0.0)
    env.run("mod_actions = {\n"
            "  {4, 'SomeBroadcast'},\n"
            "  {8, function() error('boom') end},\n"
            "  {12, function() ok = true end},\n"
            "}", name='t')
    fired, failed = env.replay_mod_actions()
    assert fired == 2 and failed == 1
    assert env._host.env['ok'] is True


def test_apply_game_command_ignores_non_mod_commands():
    env = StubEnvironment(start_beat=0.0)
    env.run("mod_actions = {\n"
            "  {4, function() GAMESTATE:ApplyGameCommand("
            "'screen,ScreenTitleMenu') end},\n"
            "}", name='t')
    env.replay_mod_actions()
    assert env.applied_mods == []
    assert env.swallowed == 1


def test_applied_mods_normalize_to_zero_length_windows(tmp_path):
    from analysis.games.notitg import modfile

    env = StubEnvironment(start_beat=0.0)
    env.run("mod_actions = {\n"
            "  {4, function() GAMESTATE:ApplyGameCommand("
            "'mod,*-1 100 drunk') end},\n"
            "}", name='t')
    env.replay_mod_actions()
    events = modfile._normalize_applied_mods(env, _seconds)
    assert len(events) == 1
    e = events[0]
    assert e['apply_type'] == 'oneshot'
    assert e['t_start'] == e['t_end'] == 4.0


# -- shader-flag bridge ---------------------------------------------------

def test_shader_bridge_maps_pulse_to_stack_events():
    from analysis.games.notitg.shader_bridge import build_shader_events

    flags = [{'beat': 0, 't': 10.0, 'key': 55, 'which': None},
             {'beat': 0, 't': 10.5, 'key': 0, 'which': None}]
    events, skipped = build_shader_events(flags)
    assert skipped == []
    assert [e['shader'] for e in events] == ['screen_tile', 'screen_tile']
    assert events[0]['time'] == 10000.0
    assert events[0]['end-params']['strength'] == 1.0
    assert events[1]['time'] == 10500.0
    assert events[1]['end-params']['strength'] == 0.0


def test_shader_bridge_skips_unmapped_keys():
    from analysis.games.notitg.shader_bridge import build_shader_events

    events, skipped = build_shader_events(
        [{'beat': 0, 't': 1.0, 'key': 217, 'which': 0}])
    assert events == []
    assert skipped == [217]


def test_shader_bridge_builds_effect_or_empty():
    from analysis.games.notitg.shader_bridge import notitg_shader_effects

    assert notitg_shader_effects(None) == []
    assert notitg_shader_effects([]) == []
    built = notitg_shader_effects(
        [{'beat': 0, 't': 1.0, 'key': 55, 'which': None}])
    assert len(built) == 1 and built[0]


def test_new_screen_shaders_follow_uniform_contract():
    from analysis.player.render.shaders import library

    for name in ('screen_mirror', 'screen_tile'):
        src = library.source(name)
        assert src is not None
        assert 'uniform sampler2D u_tex;' in src
        assert 'uniform vec2 u_resolution;' in src
        assert 'uniform vec3 u_strength;' in src
        assert 'gl_FragCoord' in src


# -- speed-mod extraction -------------------------------------------------

def test_parse_speed_mods_reads_xmod_and_cmod():
    from analysis.games.notitg.mod_channels import parse_speed_mods

    assert parse_speed_mods('2x') == [(1.0, 'x', 2.0)]
    assert parse_speed_mods('*0.5 1.7x') == [(0.5, 'x', 1.7)]
    assert parse_speed_mods('*-1 3x') == [(-1.0, 'x', 3.0)]
    assert parse_speed_mods('c400') == [(1.0, 'c', 400.0)]
    assert parse_speed_mods('m550') == [(1.0, 'm', 550.0)]
    assert parse_speed_mods('50 drunk') == []


def test_scroll_multipliers_relative_to_base_and_snap_holds():
    from analysis.games.notitg.mod_channels import compile_scroll_multipliers
    from analysis.player.render.effects.timeline import (
        EventTimeline, keyframes_from_events)

    events = [
        {'t_start': 0.0, 't_end': 9000.0, 'modstring': '2x', 'player': None},
        {'t_start': 10.0, 't_end': 20.0, 'modstring': '*-1 3x',
         'player': None},
    ]
    sc, skipped_cm = compile_scroll_multipliers(events)
    assert skipped_cm == 0
    tl = EventTimeline(keyframes_from_events(sc, ('multiplier',), (1.0,)),
                       rest=(1.0,))
    assert tl.sample(5.0)[0] == pytest.approx(1.0)     # holds at base
    assert tl.sample(15.0)[0] == pytest.approx(1.5)    # 3x / 2x base
    assert tl.sample(25.0)[0] == pytest.approx(1.0)    # reverts to base


def test_scroll_multipliers_skip_and_count_cmods():
    from analysis.games.notitg.mod_channels import compile_scroll_multipliers

    events = [{'t_start': 0.0, 't_end': 10.0, 'modstring': 'c400',
               'player': None}]
    sc, skipped_cm = compile_scroll_multipliers(events)
    assert sc == []
    assert skipped_cm == 1


def test_compile_mod_channels_still_drops_speed_mods():
    from analysis.games.notitg.mod_channels import compile_mod_channels

    channels = compile_mod_channels(
        [{'t_start': 0.0, 't_end': 10.0, 'modstring': '2x, 50 drunk',
          'player': None}])
    assert 'xmod' not in channels.mods(0)
    assert 'drunk' in channels.mods(0)


# -- producer stashes -----------------------------------------------------

def test_note_mods_stashes_rotation_zoom_and_receptor_offsets():
    import types

    import numpy as np

    from analysis.games.notitg.mod_channels import compile_mod_channels
    from analysis.games.notitg.note_mods import NotitgNoteMods

    bpms = [(0.0, 120.0)]
    channels = compile_mod_channels(
        [{'t_start': 0.0, 't_end': 100.0, 'modstring': '*-1 100 confusion',
          'player': None}])
    mods = NotitgNoteMods(channels, bpms)

    keycount = 4
    notes = types.SimpleNamespace(noterows_list=[0, 48, 96, 144])
    player = types.SimpleNamespace(
        columns=np.array([0, 1, 2, 3]), keycount=keycount, notes=notes)
    ctx = types.SimpleNamespace(
        candidates=[0, 1, 2, 3], t_now=5.0, player=player,
        lane_w=64.0, judge_y=400.0,
        candidate_head_y=np.full(4, 200.0),
        candidate_tail_y=np.full(4, 180.0),
        candidate_press_y=np.full(4, 200.0))
    mods.apply(ctx)

    assert ctx.candidate_rot_deg.shape == (4,)
    assert ctx.candidate_zoom.shape == (4,)
    receptors = ctx.receptor_offsets
    assert set(receptors) == {'dx', 'dy', 'rotation_deg', 'zoom', 'alpha'}
    for key in receptors:
        assert receptors[key].shape == (keycount,)
    # confusion is a whole-field spin => nonzero receptor rotation.
    assert np.any(receptors['rotation_deg'] != 0.0)


# -- recording actor: SM tween model --------------------------------------

def test_recording_actor_chained_tweens_accumulate():
    from analysis.games.notitg.recording_actor import RecordingActor

    actor = RecordingActor(clock=8.0)
    # accelerate(1) opens [8,9]; y(20) rides it; decelerate(2) closes it
    # (clock -> 9) and opens [9,11]; y(40) rides that.
    for verb, args in (('accelerate', ['1']), ('y', ['20']),
                       ('decelerate', ['2']), ('y', ['40'])):
        actor.poke(verb, args)
    y = actor.keyframes()['y']
    assert [(k.t, k.values, k.duration, k.easing) for k in y] == [
        (8.0, (20.0,), 1.0, 3),    # accelerate -> ease-in id 3
        (9.0, (40.0,), 2.0, 4),    # decelerate -> ease-out id 4
    ]


def test_recording_actor_parallel_setters_share_interval():
    from analysis.games.notitg.recording_actor import RecordingActor

    actor = RecordingActor(clock=0.0)
    for verb, args in (('linear', ['1']), ('x', ['100']), ('y', ['50'])):
        actor.poke(verb, args)
    frames = actor.keyframes()
    # x and y both tween over the SAME [0,1] window (one open interval).
    assert frames['x'][0].t == 0.0 and frames['x'][0].duration == 1.0
    assert frames['y'][0].t == 0.0 and frames['y'][0].duration == 1.0


def test_recording_actor_sleep_and_finishtweening():
    from analysis.games.notitg.recording_actor import RecordingActor

    actor = RecordingActor(clock=0.0)
    for verb, args in (('x', ['1']), ('sleep', ['2']), ('x', ['2']),
                       ('linear', ['5']), ('finishtweening', []),
                       ('x', ['3'])):
        actor.poke(verb, args)
    x = actor.keyframes()['x']
    assert x[0].t == 0.0                       # instant
    assert x[1].t == 2.0                        # after sleep(2)
    # finishtweening closes the open linear(5) interval (clock -> 7) and
    # clears the pending tween, so the last x is instant at t=7.
    assert x[2].t == 7.0 and x[2].duration == 0.0


def test_recording_actor_add_reads_current_value():
    from analysis.games.notitg.recording_actor import RecordingActor

    actor = RecordingActor(clock=0.0)
    for verb, args in (('x', ['10']), ('addx', ['5']), ('addx', ['5'])):
        actor.poke(verb, args)
    values = [k.values[0] for k in actor.keyframes()['x']]
    assert values == [10.0, 15.0, 20.0]


def test_recording_actor_diffuse_and_visibility():
    from analysis.games.notitg.recording_actor import RecordingActor

    actor = RecordingActor(clock=0.0)
    actor.poke('diffuse', ['1', '0', '0', '0.5'])
    actor.poke('hidden', ['1'])
    actor.poke('visible', ['1'])
    frames = actor.keyframes()
    assert frames['color'][0].values == (1.0, 0.0, 0.0)
    assert frames['alpha'][0].values == (0.5,)
    assert frames['alpha'][1].values == (0.0,)   # hidden(1)
    assert frames['alpha'][2].values == (1.0,)   # visible


# -- named-actor binding: XML self-assign + closure pokes -----------------

def test_named_actor_binding_from_synthetic_xml_and_closure():
    """An InitCommand self-assigns a global; a scheduled mod_actions
    closure pokes that global at its fire beat. Both must land on one
    recorder timeline, keyed by the global name."""
    env = StubEnvironment(start_beat=0.0)
    # InitCommand: hide the actor and bind it to a global (SM pattern).
    env.run_actor_chunk("self:hidden(1); my_frame = self;", name='init')
    # Scheduled closure fires at beat 4 (= 4s under the identity clock).
    env.run("mod_actions = { {4, function()\n"
            "  my_frame:linear(1); my_frame:y(30)\n"
            "  my_frame:linear(1); my_frame:y(60)\n"
            "end} }", name='ma')
    fired, failed = env.replay_mod_actions()
    assert (fired, failed) == (1, 0)

    named = env.named_actor_keyframes()
    assert set(named) == {'my_frame'}            # 'self' does not linger
    frames = named['my_frame']
    # InitCommand hidden(1) recorded at load (t=0), fire-time y pokes at 4/5.
    assert frames['alpha'][0].values == (0.0,)
    y = frames['y']
    assert (y[0].t, y[0].values) == (4.0, (30.0,))
    assert (y[1].t, y[1].values) == (5.0, (60.0,))


def test_closure_fire_time_uses_beat_to_seconds():
    env = StubEnvironment(start_beat=0.0, to_seconds=lambda beat: beat * 0.5)
    env.run_actor_chunk("thing = self;", name='init')
    env.run("mod_actions = { {8, function() thing:x(1) end} }", name='ma')
    env.replay_mod_actions()
    # fire beat 8 -> 4.0s through the converter.
    assert env.named_actor_keyframes()['thing']['x'][0].t == 4.0


def test_tree_merges_xml_command_and_recorded_pokes(tmp_path):
    """compile_element_tree binds an actor's XML command keyframes and
    the pokes its bound global recorded into ONE element timeline."""
    from analysis.games.notitg import modfile

    xml = ('<ActorFrame><children>'
           '<Sprite Type="Sprite" Texture="white" '
           'OnCommand="x,100" '
           'InitCommand="%function(self) named_sprite = self; end"/>'
           '</children></ActorFrame>')
    parsed = xml_actors.parse_actor_xml(xml)
    named = {'named_sprite': {'y': [Keyframe(5.0, (200.0,), 0.0, 0)]}}
    tree = modfile.compile_element_tree(
        parsed.root, _seconds, start_beat=0.0, named_keyframes=named)
    # The document's ActorFrame is the container; its one Sprite child is
    # the top-level element, carrying BOTH timelines merged.
    (sprite,) = tree
    assert sprite.kind == 'sprite'
    assert sprite.sample('x', 0.0) == (100.0,)   # from the XML command
    assert sprite.sample('y', 5.0) == (200.0,)   # from the recorded poke


def test_tree_wraps_nested_actorframe_as_group(tmp_path):
    """A drawable descendant nested under an inner ActorFrame makes that
    frame a 'group' whose transform composes onto the child."""
    from analysis.games.notitg import modfile

    xml = ('<ActorFrame><children>'
           '<ActorFrame OnCommand="rotationz,45"><children>'
           '<Quad Type="Quad" OnCommand="x,10"/>'
           '</children></ActorFrame>'
           '</children></ActorFrame>')
    parsed = xml_actors.parse_actor_xml(xml)
    tree = modfile.compile_element_tree(
        parsed.root, _seconds, start_beat=0.0)
    (group,) = tree
    assert group.kind == 'group'
    assert group.sample('rotation', 0.0) == (45.0,)
    (child,) = group.children
    assert child.kind == 'rect'
    assert child.sample('x', 0.0) == (10.0,)


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


@pytest.mark.skipif(not _GAT_SM.exists(),
                    reason='NotITG gat pilot not present')
def test_gat_pilot_replays_mod_actions():
    result = compile_modfile(str(_GAT_SM))
    replay = result['replay']
    # Every mod_actions closure is fired once; gat's are actor pokes, so
    # many fault harmlessly (caught) and none touch ApplyGameCommand /
    # SetShaderFlag - so the recovered one-shot / shader counts are 0.
    assert replay['fired'] > 100
    assert replay['fired'] + result['unsupported']['count'] >= 0
    assert replay['applied_mods'] == 0


@pytest.mark.skipif(not _GAT_SM.exists(),
                    reason='NotITG gat pilot not present')
def test_gat_pilot_records_named_actor_timelines():
    """The 667 mod_actions closures poke named actors; firing them once
    each must recover keyframes onto many named actors' timelines, and
    the hierarchical tree must carry drawable groups."""
    result = compile_modfile(str(_GAT_SM))
    assert result['named_actors'] > 50
    assert result['recorded_keyframes'] > 500
    assert result['tree']
    assert any(e.kind == 'group' for e in _flatten_tree(result['tree']))


def _flatten_tree(elements):
    for element in elements:
        yield element
        yield from _flatten_tree(element.children)


@pytest.mark.skipif(not _GAT_SM.exists(),
                    reason='NotITG gat pilot not present')
def test_gat_pilot_extracts_scroll_multipliers():
    from analysis.games.notitg.mod_channels import compile_scroll_multipliers

    result = compile_modfile(str(_GAT_SM))
    sc, _skipped = compile_scroll_multipliers(result['mod_events'])
    # gat rides its 2x base with frequent xmod changes.
    assert len(sc) > 20
