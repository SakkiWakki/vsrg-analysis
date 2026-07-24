"""NotITG modfile compiler: actor XML parsing, recording-actor tween and
oscillator state, classic command -> storyboard keyframes, scroll/mod
channels, message dispatch, and a guarded integration test against the
real gat pilot compiled through the engine-loop sim."""
from pathlib import Path

import pytest

pytest.importorskip('lupa')

from analysis.games.notitg import xml_actors
from analysis.games.notitg.modfile import parse_fgchanges
from analysis.games.notitg.sim.producers import compile_via_sim
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


def test_bare_percent_expression_becomes_call_statement():
    # `%expr` is an expression whose value is the command: the runnable
    # chunk evaluates it and calls the result when it is a function
    # (the XGML template's `UpdateCommand="%prefix.update"`).
    parsed = xml_actors.parse_actor_xml(
        '<Quad Condition="%FUCK_EXE"/>')
    body = parsed.lua_chunks[0].body
    assert '(FUCK_EXE)' in body
    assert '__cmd(self)' in body


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


def test_quad_zoomto_size_feeds_renderer_wh():
    # A fullscreen flash quad sizes via zoomto -> size_x/size_y; the
    # renderer sizes fill kinds from w/h, so a Quad must expose both or it
    # renders at zero size (the gat ZZZZZFLASHES / TargOn quads did).
    from analysis.games.notitg import modfile

    parsed = xml_actors.parse_actor_xml(
        '<Quad Type="Quad" OnCommand="zoomto,640,480;diffuse,1,0,0,1"/>')
    element = modfile._compile_elements(
        parsed.classic_commands, _seconds, start_beat=0.0)[0]
    assert element.kind == 'rect'
    assert element.sample('w', 0.0) == (640.0,)
    assert element.sample('h', 0.0) == (480.0,)
    # The absolute-size override is preserved too (draw path reads it).
    assert element.sample('size_x', 0.0) == (640.0,)


def test_sprite_zoomto_does_not_hijack_wh():
    # Sprites size from their pixmap; only fill kinds borrow size_x/y.
    from analysis.games.notitg import modfile

    parsed = xml_actors.parse_actor_xml(
        '<Sprite Type="Sprite" Texture="white" OnCommand="zoomto,32,32"/>')
    element = modfile._compile_elements(
        parsed.classic_commands, _seconds, start_beat=0.0)[0]
    assert element.kind == 'sprite'
    assert element.sample('w', 0.0) == (0.0,)
    assert element.sample('size_x', 0.0) == (32.0,)


def test_blend_add_marks_element_additive():
    # gat's split judgment lines blend add so the red bars glow.
    from analysis.games.notitg import modfile

    parsed = xml_actors.parse_actor_xml(
        '<Quad Type="Quad" OnCommand="zoomto,640,4;blend,add"/>')
    element = modfile._compile_elements(
        parsed.classic_commands, _seconds, start_beat=0.0)[0]
    assert element.additive is True

    plain = xml_actors.parse_actor_xml(
        '<Quad Type="Quad" OnCommand="zoomto,640,4"/>')
    plain_el = modfile._compile_elements(
        plain.classic_commands, _seconds, start_beat=0.0)[0]
    assert plain_el.additive is False


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


def test_scroll_base_inferred_from_widest_xmod_window():
    """The persistent baseline is the chart's own, not a fixed 2x: a chart
    whose widest xmod window is 2.5x rests the field at the user's speed
    (mult 1.0) and scales bursts relative to 2.5, not to a hardcoded 2.0
    (which would rest 25% too fast). The approach-prefixed `*2 2.1x` baseline
    form is read too."""
    from analysis.games.notitg.mod_channels import (
        _infer_base_xmod, compile_scroll_multipliers)
    from analysis.player.render.effects.timeline import (
        EventTimeline, keyframes_from_events)

    events = [
        {'t_start': 0.0, 't_end': 9999.0, 'modstring': '*2 2.5x, *-1 overhead',
         'player': None},
        {'t_start': 30.0, 't_end': 34.0, 'modstring': '5x', 'player': None},
    ]
    assert _infer_base_xmod(events) == pytest.approx(2.5)
    sc, _skipped = compile_scroll_multipliers(events)
    tl = EventTimeline(keyframes_from_events(sc, ('multiplier',), (1.0,)),
                       rest=(1.0,))
    # baseline holds at rest (1.0); the 5x burst is 5/2.5 = 2.0.
    assert tl.sample(10.0)[0] == pytest.approx(1.0)
    assert tl.sample(32.0)[0] == pytest.approx(2.0)


def test_scroll_base_falls_back_to_engine_default_without_xmod():
    """A chart that authors no xmod window resolves against the engine
    default (1x), not a phantom 2x base: a bare `drunk` window and an empty
    chart both infer base 1.0, so any later burst is a genuine multiple of
    the user's speed rather than being halved."""
    from analysis.games.notitg.mod_channels import _infer_base_xmod

    assert _infer_base_xmod([]) == pytest.approx(1.0)
    no_xmod = [{'t_start': 5.0, 't_end': 9.0, 'modstring': 'drunk',
                'player': None}]
    assert _infer_base_xmod(no_xmod) == pytest.approx(1.0)


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


def test_scroll_multipliers_persistent_window_holds_flat_over_bursts():
    """A persistent xmod window (a `{0, 9999, '2.5x'}` baseline the reader
    re-applies as per-frame bursts) holds a FLAT rate: the overlapped
    burst windows resolve away instead of sawtoothing to base between
    them (the Crazy Shuffle scroll-stutter regression)."""
    from analysis.games.notitg.mod_channels import compile_scroll_multipliers
    from analysis.player.render.effects.timeline import (
        EventTimeline, keyframes_from_events)

    # The chart's baseline (widest window) is 2x; the persistent burst the
    # reader re-applies every ~0.1s is 2.5x, so the held rate is 2.5/2.
    events = [{'t_start': 0.0, 't_end': 9000.0, 'modstring': '2x',
               'player': None},
              {'t_start': 0.0, 't_end': 100.0, 'modstring': '*-1 2.5x',
               'player': None}]
    t = 0.1
    while t < 5.0:
        events.append({'t_start': t, 't_end': t + 0.017,
                       'modstring': '2.5x', 'player': None})
        t += 0.1
    sc, _skipped = compile_scroll_multipliers(events)
    tl = EventTimeline(keyframes_from_events(sc, ('multiplier',), (1.0,)),
                       rest=(1.0,))
    # 2.5x / 2.0 base = 1.25, held flat across every burst gap (a sawtooth
    # would dip toward 1.0 at each 0.083s gap between reapplies).
    for t in (1.0, 2.0, 3.0, 4.0):
        assert tl.sample(t)[0] == pytest.approx(1.25)


def test_scroll_multipliers_fast_toggle_stays_monotonic():
    """A scroll xmod toggled faster than its chase completes keeps the
    breakpoints time-ordered (the _xmod_breakpoints overrun regression):
    an unclamped ramp arriving past the next event made durations go
    negative and the mult jitter."""
    from analysis.games.notitg.mod_channels import compile_scroll_multipliers

    events = []
    t = 0.0
    while t < 3.0:
        events.append({'t_start': t, 't_end': t + 0.05,
                       'modstring': '*1 4x', 'player': None})
        t += 0.1
    sc, _skipped = compile_scroll_multipliers(events)
    assert all(e['duration'] >= 0.0 for e in sc)


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
        lane_w=64.0, judge_y=400.0, chart_rect=(0.0, 0.0, 400.0, 800.0),
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


def _receptor_alpha_for(modstring):
    import types

    import numpy as np

    from analysis.games.notitg.mod_channels import compile_mod_channels
    from analysis.games.notitg.note_mods import NotitgNoteMods

    channels = compile_mod_channels(
        [{'t_start': 0.0, 't_end': 100.0, 'modstring': modstring,
          'player': None}])
    mods = NotitgNoteMods(channels, [(0.0, 120.0)])
    player = types.SimpleNamespace(
        columns=np.array([], dtype=np.int64), keycount=4,
        notes=types.SimpleNamespace(noterows_list=[]))
    ctx = types.SimpleNamespace(
        candidates=[], t_now=5.0, player=player,
        lane_w=64.0, judge_y=400.0, chart_rect=(0.0, 0.0, 400.0, 800.0),
        candidate_head_y=np.zeros(0), candidate_tail_y=np.zeros(0),
        candidate_press_y=np.zeros(0))
    mods.apply(ctx)
    return ctx.receptor_offsets['alpha']


def test_receptor_alpha_ignores_stealth_family():
    # Engine truth (refs/notitg/decompile ReceptorArrowRow.c @0053b390):
    # receptor base alpha = clamp01((1 - dark - dark_col) *
    # (1 - fadeToFail)). The stealth/stealthglow appearance path never
    # reaches receptors, so a fully stealthed field keeps them visible.
    import numpy as np

    alpha = _receptor_alpha_for('*-1 100 stealth, *-1 100 stealthglow')
    np.testing.assert_allclose(alpha, 1.0)


def test_receptor_alpha_darkens_per_column():
    import numpy as np

    alpha = _receptor_alpha_for('*-1 100 dark0, *-1 50 dark2')
    np.testing.assert_allclose(alpha, [0.0, 1.0, 0.5, 1.0])


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


def test_recording_actor_oscillator_span_open_replace_stop():
    """A kind verb opens a span; a DIFFERENT kind closes the open one at
    the current clock and starts fresh; stopeffect closes the last. Each
    span carries its magnitude/period/clock."""
    from analysis.games.notitg.recording_actor import RecordingActor

    actor = RecordingActor(clock=0.0)
    # bounce at clock 0, then advance the clock (a sleep) and replace with
    # bob, then advance and stop.
    actor.poke('bounce', [])
    actor.poke('effectperiod', ['1'])
    actor.poke('effectclock', ['bgm'])
    actor.poke('effectmagnitude', ['0', '-100', '0'])
    actor.poke('sleep', ['4'])          # clock -> 4
    actor.poke('bob', [])                # closes bounce at 4, opens bob
    actor.poke('effectmagnitude', ['100', '0', '0'])
    actor.poke('sleep', ['4'])          # clock -> 8
    actor.poke('stopeffect', [])         # closes bob at 8

    spans = actor.oscillator_spans()
    assert [(s.kind, s.start, s.end) for s in spans] == [
        ('bounce', 0.0, 4.0), ('bob', 4.0, 8.0)]
    assert spans[0].clock == 'bgm' and spans[0].period == 1.0
    assert spans[0].magnitude_at(0.0) == (0.0, -100.0, 0.0)
    assert spans[1].magnitude_at(4.0) == (100.0, 0.0, 0.0)


def test_recording_actor_oscillator_repoke_same_kind_is_one_span():
    """Re-poking the same kind (gat's per-frame `a:vibrate()`) continues
    the open span rather than starting a new one each tick; magnitude
    pokes accumulate into that one span's envelope."""
    from analysis.games.notitg.recording_actor import RecordingActor

    actor = RecordingActor(clock=0.0)
    for clock, mag in ((0.0, '10'), (1.0, '20'), (2.0, '30')):
        actor.reset_clock(clock)
        actor.poke('vibrate', [])
        actor.poke('effectmagnitude', [mag, '0', '0'])
    spans = actor.oscillator_spans()
    assert len(spans) == 1
    assert spans[0].kind == 'vibrate' and spans[0].start == 0.0
    assert spans[0].end == 2.0
    # The magnitude in force steps up with each poke.
    assert spans[0].magnitude_at(0.5) == (10.0, 0.0, 0.0)
    assert spans[0].magnitude_at(1.5) == (20.0, 0.0, 0.0)
    assert spans[0].magnitude_at(2.5) == (30.0, 0.0, 0.0)


def test_oscillator_span_magnitude_at_step_holds():
    """`magnitude_at` step-holds the last sample at or before a clock, and
    returns the first sample for a clock before them all."""
    from analysis.games.notitg.recording_actor import _OscSpan

    span = _OscSpan('vibrate', 0.0, 1.0, 0.0, 'bgm')
    span.set_magnitude(1.0, (10.0, 0.0, 0.0))
    span.set_magnitude(2.0, (20.0, 0.0, 0.0))
    span.set_magnitude(3.0, (30.0, 0.0, 0.0))
    assert span.magnitude_at(0.5) == (10.0, 0.0, 0.0)   # before first
    assert span.magnitude_at(1.5) == (10.0, 0.0, 0.0)   # holds 1.0
    assert span.magnitude_at(2.0) == (20.0, 0.0, 0.0)   # exact
    assert span.magnitude_at(9.0) == (30.0, 0.0, 0.0)   # holds last


def test_bare_kind_verbs_record_engine_default_magnitudes():
    """A kind setter overwrites the magnitude with its engine default
    (Actor.h): a bare `vibrate()` records (10,10,10) - the +-10px
    per-frame shake charts lean on for the receptor mirage - and a bare
    `bob()` records (0,0,20) (z-only: no 2D motion, but the span is
    real and kept)."""
    from analysis.games.notitg.recording_actor import RecordingActor

    actor = RecordingActor(clock=0.0)
    actor.poke('vibrate', [])
    actor.poke('stopeffect', [])
    actor.poke('bob', [])
    actor.poke('stopeffect', [])
    vib, bob = actor.oscillator_spans()
    assert vib.magnitude_at(0.0) == (10.0, 10.0, 10.0)
    assert bob.magnitude_at(0.0) == (0.0, 0.0, 20.0)
    assert bob.period == 2.0
    assert vib.explicit_end and bob.explicit_end


def test_open_span_is_not_explicitly_ended():
    """A span still running when recording ends is marked non-explicit,
    so the synthesis extends it to the compile end (the engine keeps an
    un-stopped effect going)."""
    from analysis.games.notitg.recording_actor import RecordingActor

    actor = RecordingActor(clock=0.0)
    actor.poke('vibrate', [])
    (span,) = actor.oscillator_spans()
    assert not span.explicit_end


def test_recording_actor_set_vanish_point_records_channels():
    """`SetVanishPoint(x, y)` records onto the vanish_x/vanish_y channels
    (rest = screen centre); per-frame pokes build a stream."""
    from analysis.games.notitg.recording_actor import RecordingActor

    actor = RecordingActor(clock=0.0)
    actor.poke('SetVanishPoint', ['400', '100'])
    actor.reset_clock(1.0)
    actor.poke('SetVanishPoint', ['200', '300'])
    frames = actor.keyframes()
    assert [k.values for k in frames['vanish_x']] == [(400.0,), (200.0,)]
    assert [k.values for k in frames['vanish_y']] == [(100.0,), (300.0,)]


def test_recording_actor_diffuse_and_visibility():
    """`diffuse`/`diffusealpha` feed the `alpha` channel; `hidden`/`visible`
    feed a SEPARATE `hidden` channel (SM's hard visibility bit is
    independent of diffusealpha), so a crossfade can ride an actor a
    `hidden,1` currently gates off."""
    from analysis.games.notitg.recording_actor import RecordingActor

    actor = RecordingActor(clock=0.0)
    actor.poke('diffuse', ['1', '0', '0', '0.5'])
    actor.poke('hidden', ['1'])
    actor.poke('visible', ['1'])
    frames = actor.keyframes()
    assert frames['color'][0].values == (1.0, 0.0, 0.0)
    # diffusealpha never gets clobbered by the visibility pokes.
    assert frames['alpha'][0].values == (0.5,)
    assert len(frames['alpha']) == 1
    # hidden(1) then visible(1): the hidden bit goes on, then off.
    assert frames['hidden'][0].values == (1.0,)   # hidden(1)
    assert frames['hidden'][1].values == (0.0,)   # visible(1) -> shown


# -- named-actor binding: XML self-assign + closure pokes -----------------

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


def test_tree_splits_below_and_above_first_proxy():
    """Content preceding the first ActorProxy in document order compiles
    into the pre-field below band; content after it stays above (z 0).
    The renderer composites every notefield copy between those bands, so
    an opaque backdrop early in the tree draws under the copies while
    later overlays still cover them (engine document order)."""
    from analysis.games.notitg import modfile

    xml = ('<ActorFrame><children>'
           '<Quad Type="Quad" OnCommand="x,1"/>'
           '<Layer Type="ActorProxy"/>'
           '<Quad Type="Quad" OnCommand="x,2"/>'
           '</children></ActorFrame>')
    parsed = xml_actors.parse_actor_xml(xml)
    tree = modfile.compile_element_tree(parsed.root, _seconds,
                                        start_beat=0.0)
    backdrop, overlay = tree
    assert backdrop.z == modfile._PRE_FIELD_Z
    assert backdrop.sample('x', 0.0) == (1.0,)
    assert overlay.z == 0
    assert overlay.sample('x', 0.0) == (2.0,)


def test_tree_split_keeps_straddling_group_on_both_sides():
    """A group holding content on both sides of the first proxy compiles
    once per side, each half carrying the group's own transform so the
    children keep their screen placement across the split."""
    from analysis.games.notitg import modfile

    xml = ('<ActorFrame><children>'
           '<ActorFrame OnCommand="rotationz,45"><children>'
           '<Quad Type="Quad" OnCommand="x,1"/>'
           '<ActorProxy/>'
           '<Quad Type="Quad" OnCommand="x,2"/>'
           '</children></ActorFrame>'
           '</children></ActorFrame>')
    parsed = xml_actors.parse_actor_xml(xml)
    tree = modfile.compile_element_tree(parsed.root, _seconds,
                                        start_beat=0.0)
    under, over = tree
    assert under.z == modfile._PRE_FIELD_Z
    assert over.z == 0
    for group, x in ((under, 1.0), (over, 2.0)):
        assert group.kind == 'group'
        assert group.sample('rotation', 0.0) == (45.0,)
        (child,) = group.children
        assert child.sample('x', 0.0) == (x,)


def test_tree_split_partitions_each_proxy_subtree():
    """A later section splits around ITS OWN first proxy, not just the
    document's: section-internal art preceding the section's copies
    compiles into the below band even though an earlier section already
    holds the document's first ActorProxy (multi-song documents)."""
    from analysis.games.notitg import modfile

    xml = ('<ActorFrame><children>'
           '<ActorFrame><children>'
           '<Quad Type="Quad" OnCommand="x,1"/>'
           '<ActorProxy/>'
           '</children></ActorFrame>'
           '<ActorFrame><children>'
           '<Quad Type="Quad" OnCommand="x,2"/>'
           '<ActorProxy/>'
           '<Quad Type="Quad" OnCommand="x,3"/>'
           '</children></ActorFrame>'
           '</children></ActorFrame>')
    parsed = xml_actors.parse_actor_xml(xml)
    tree = modfile.compile_element_tree(parsed.root, _seconds,
                                        start_beat=0.0)
    unders = [e for e in tree if e.z == modfile._PRE_FIELD_Z]
    overs = [e for e in tree if e.z == 0]
    assert sorted(g.children[0].sample('x', 0.0) for g in unders) \
        == [(1.0,), (2.0,)]
    (over,) = overs
    assert over.children[0].sample('x', 0.0) == (3.0,)


# -- resilience -----------------------------------------------------------

def test_compile_never_raises_on_missing_lua(tmp_path):
    sm = tmp_path / 'nolua.sm'
    sm.write_text('#TITLE:x;\n#BPMS:0.000=120.000;\n')
    assert compile_via_sim(str(sm)) is None


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
    result = compile_via_sim(str(sm))
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







def _flatten_tree(elements):
    for element in elements:
        yield element
        yield from _flatten_tree(element.children)




# -- effect oscillators ---------------------------------------------------

def _identity_osc_clock():
    """An oscillator clock over an identity beat<->second mapping, so a
    span's second clock reads straight through as its beat/phase source
    (unit tests fix the phase directly, no chart timing involved)."""
    from analysis.games.notitg import modfile
    return modfile._OscillatorClock(lambda beat: float(beat), (0.0, 64.0))


def _make_span(kind, start, end, mag, period=1.0, offset=0.0, clock='bgm'):
    from analysis.games.notitg.recording_actor import _OscSpan
    span = _OscSpan(kind, start, period, offset, clock)
    span.end = end
    span.set_magnitude(start, mag)
    return span


def _no_rng():
    """An RNG for non-vibrate spans, which never draw from it."""
    import random
    return random.Random(0)


def test_effect_pct_matches_engine_scale_and_wrap():
    """SM's pct = clamp(fmod(phase + offset, period) / period, 0, 1)
    (Actor.cpp:273-278). Wraps every period, honours offset, clamps."""
    from analysis.games.notitg.modfile import _effect_pct

    assert _effect_pct(0.0, 1.0, 0.0) == 0.0
    assert _effect_pct(0.25, 1.0, 0.0) == pytest.approx(0.25)
    assert _effect_pct(1.25, 1.0, 0.0) == pytest.approx(0.25)   # wraps
    assert _effect_pct(0.0, 1.0, 0.5) == pytest.approx(0.5)     # offset
    assert _effect_pct(2.0, 2.0, 0.0) == 0.0                    # period 2


def test_bounce_span_synthesises_abs_sine_on_y():
    """bounce: pos += mag * sin(pct*pi) (Actor.cpp:344). Magnitude
    (0,-100,0) drives y by -100*sin(pct*pi); the step keyframe at a sample
    time holds exactly that value."""
    import math
    from analysis.games.notitg.modfile import _span_keyframes

    span = _make_span('bounce', 0.0, 1.0, (0.0, -100.0, 0.0))
    frames = _span_keyframes(span, _identity_osc_clock(), _no_rng())
    assert set(frames) == {'x', 'y'}   # x delta is 0 but present
    from analysis.player.render.effects.timeline import EventTimeline
    y = EventTimeline(frames['y'], rest=(0.0,))
    for t in (0.25, 0.5, 0.75):
        assert y.sample(t)[0] == pytest.approx(-100.0 * math.sin(t * math.pi),
                                               abs=1.0)


def test_bob_span_synthesises_full_sine():
    """bob: pos += mag * sin(pct*2pi) (Actor.cpp:353) - a full sine, so it
    swings both directions unlike bounce."""
    from analysis.games.notitg.modfile import _span_keyframes
    from analysis.player.render.effects.timeline import EventTimeline

    span = _make_span('bob', 0.0, 1.0, (100.0, 0.0, 0.0))
    x = EventTimeline(_span_keyframes(span, _identity_osc_clock(),
                                      _no_rng())['x'], rest=(0.0,))
    assert x.sample(0.25)[0] == pytest.approx(100.0, abs=2.0)   # sin(pi/2)
    assert x.sample(0.75)[0] == pytest.approx(-100.0, abs=2.0)  # sin(3pi/2)


def test_wag_span_drives_rotation_by_z_magnitude():
    """wag: rotation += mag * sin(pct*2pi) (Actor.cpp:332). The 2D rotation
    is the z magnitude; x/y magnitudes are 3D rotations we drop."""
    from analysis.games.notitg.modfile import _span_keyframes
    from analysis.player.render.effects.timeline import EventTimeline

    span = _make_span('wag', 0.0, 1.0, (0.0, 0.0, 30.0))
    frames = _span_keyframes(span, _identity_osc_clock(), _no_rng())
    assert set(frames) == {'rotation'}
    r = EventTimeline(frames['rotation'], rest=(0.0,))
    assert r.sample(0.25)[0] == pytest.approx(30.0, abs=1.0)


def test_spin_span_accumulates_rotation_over_elapsed():
    """spin: rotation += effectDelta * mag every frame (Actor.cpp:599),
    i.e. rotation(t) = mag.z * (phase - phase_start). Linear, unbounded."""
    from analysis.games.notitg.modfile import _span_keyframes
    from analysis.player.render.effects.timeline import EventTimeline

    span = _make_span('spin', 0.0, 4.0, (0.0, 0.0, 90.0))
    r = EventTimeline(_span_keyframes(span, _identity_osc_clock(),
                                      _no_rng())['rotation'], rest=(0.0,))
    # phase advances 1:1 with time under the identity clock, so at t=2 the
    # accumulated rotation is 90 * 2 = 180.
    assert r.sample(2.0)[0] == pytest.approx(180.0, abs=1.0)
    assert r.sample(4.0)[0] == pytest.approx(360.0, abs=1.0)


def test_vibrate_span_is_seeded_and_reproducible():
    """vibrate: pos += mag * randomf(-1,1) (Actor.cpp:338). Seeded, so the
    same span + seed compiles the identical jitter twice."""
    import random
    from analysis.games.notitg.modfile import _span_keyframes

    span = _make_span('vibrate', 0.0, 0.5, (10.0, 10.0, 0.0))
    clock = _identity_osc_clock()
    a = _span_keyframes(span, clock, random.Random('seed'))
    b = _span_keyframes(_make_span('vibrate', 0.0, 0.5, (10.0, 10.0, 0.0)),
                        clock, random.Random('seed'))
    assert [k.values for k in a['x']] == [k.values for k in b['x']]
    # The jitter stays within the magnitude bound.
    assert all(abs(k.values[0]) <= 10.0 + 1e-6 for k in a['x'])


def test_span_returns_to_rest_after_end():
    """A synthesised span appends a trailing rest keyframe, so the delta
    is 0 once the effect stops rather than holding its last sample."""
    from analysis.games.notitg.modfile import _span_keyframes
    from analysis.player.render.effects.timeline import EventTimeline

    span = _make_span('bounce', 0.0, 1.0, (0.0, -100.0, 0.0))
    y = EventTimeline(_span_keyframes(span, _identity_osc_clock(),
                                      _no_rng())['y'], rest=(0.0,))
    assert y.sample(5.0)[0] == pytest.approx(0.0)


def test_message_command_helpers_split_by_name():
    parsed = xml_actors.parse_actor_xml(
        '<LAER Type="Sprite" InitCommand="%function(self) end"'
        ' OnCommand="x,1" HideCommand="hidden,1"'
        ' SetupFUCKMessageCommand="%function(self) end"/>')
    actor = parsed.root
    assert set(actor.message_commands()) == {'SetupFUCK'}
    assert set(actor.named_commands()) == {'Hide'}


# -- asset resolution -----------------------------------------------------

def test_sprite_texture_resolves_against_actor_base_dir(tmp_path):
    from analysis.games.notitg import modfile

    (tmp_path / 'hold.png').write_bytes(b'x')
    parsed = xml_actors.parse_actor_xml(
        '<LAER Type="Sprite" Texture="hold" OnCommand="diffusealpha,1"/>')
    parsed.root._base_dir = tmp_path
    element = modfile._leaf_element(parsed.root, 0.0, {})
    assert element.asset == str(tmp_path / 'hold.png')


def test_untyped_actor_with_image_file_becomes_sprite(tmp_path):
    from analysis.games.notitg import modfile

    (tmp_path / 'darkcircle.png').write_bytes(b'x')
    parsed = xml_actors.parse_actor_xml(
        '<Layer File="darkcircle" OnCommand="diffusealpha,1"/>')
    parsed.root._base_dir = tmp_path
    element = modfile._leaf_element(parsed.root, 0.0, {})
    assert element is not None and element.kind == 'sprite'
    assert element.asset == str(tmp_path / 'darkcircle.png')


def test_sprite_manifest_yields_inner_texture(tmp_path):
    from analysis.games.notitg import modfile

    (tmp_path / 'shame_idle.png').write_bytes(b'x')
    (tmp_path / 'idle.sprite').write_text(
        '[Sprite]\nTexture=shame_idle.png\nFrame0000=0\n')
    resolved = modfile._resolve_texture_path('idle.sprite', tmp_path)
    assert resolved == str(tmp_path / 'shame_idle.png')


def test_white_texture_stays_a_name(tmp_path):
    from analysis.games.notitg import modfile

    parsed = xml_actors.parse_actor_xml('<Sprite Type="Sprite" Texture="white"'
                                        ' OnCommand="diffusealpha,1"/>')
    parsed.root._base_dir = tmp_path
    assert modfile._resolve_asset(parsed.root) == 'white'
