"""Lua host sandbox + the fluXis storyboard-script recorder."""
import json

import pytest

pytest.importorskip('lupa')

from analysis.games.fluxis.fsb_storyboard import parse_fsb
from analysis.games.fluxis.lua_storyboard import (ScriptRecorder,
                                                  record_script_elements)
from analysis.player.render.lua import LuaHost
from analysis.player.render.lua.host import LuaScriptError


def _chart(**overrides):
    chart = {
        'hitobjects': [
            {'time': 1000.0, 'column': 0, 'is_hold': False,
             'end_time': None, 'type': 0},
            {'time': 1500.0, 'column': 2, 'is_hold': True,
             'end_time': 2000.0, 'type': 0},
            {'time': 3000.0, 'column': 1, 'is_hold': False,
             'end_time': None, 'type': 1},
        ],
        'timing_points': [(0.0, 120.0), (2000.0, 180.0)],
        'scroll_velocities': [(500.0, 2.0)],
        'effect_streams': {'flash': [{'time': 1000.0, 'duration': 100.0}]},
        'meta': {'title': 'T', 'artist': 'A', 'mapper': 'M',
                 'difficulty': 'D', 'background': '', 'cover': ''},
    }
    chart.update(overrides)
    return chart


def _recorder(seed='test.lua'):
    return ScriptRecorder(_chart(), (1920.0, 1080.0), '/tmp', seed=seed)


# ── sandbox --------------------------------------------------------------

def test_sandbox_blocks_dangerous_globals():
    host = LuaHost()
    for name in ('io', 'os', 'require', 'dofile', 'load', 'loadstring',
                 'loadfile', 'debug', 'package', 'python'):
        assert host.run(f'return {name} == nil'), name


def test_sandbox_has_safe_stdlib():
    host = LuaHost()
    assert host.run('return math.floor(3.7)') == 3
    assert host.run("return string.upper('ab')") == 'AB'
    assert host.run('return unpack({1, 2})') == (1, 2)


def test_run_raises_script_error_not_crash():
    host = LuaHost()
    with pytest.raises(LuaScriptError):
        host.run('this is not lua')
    with pytest.raises(LuaScriptError):
        host.run("error('boom')")


# ── element recording ------------------------------------------------------

def test_script_records_box_with_animation():
    recorder = _recorder()
    recorder.run("""
        local box = StoryboardBox()
        box.layer = Layer('Overlay')
        box.time = 1000
        box.endtime = 3000
        box.anchor = Anchor('Centre')
        box.origin = Anchor('Centre')
        box.width = 100
        box.height = 50
        box:animate('Fade', 1000, 500, 0, 1, Easing('OutQuint'))
        Add(box)
    """)
    (el,) = recorder.recorded
    assert el['type'] == 0
    assert el['layer'] == 2
    assert el['anchor'] == 18
    (anim,) = el['animations']
    assert anim == {'start': 1000.0, 'duration': 500.0, 'easing': 13,
                    'type': 7, 'start-value': '0', 'end-value': '1',
                    'use-start': True}


def test_setversion_2_times_become_absolute():
    recorder = _recorder()
    recorder.run("""
        SetVersion(2)
        local box = StoryboardBox()
        box.time = 5000
        box.endtime = 6000
        box:animate('Fade', 0, 500, 0, 1, 0)
        Add(box)
    """)
    (el,) = recorder.recorded
    assert el['animations'][0]['start'] == 5000.0


def test_sprite_text_and_color4_fields():
    recorder = _recorder()
    recorder.run("""
        local spr = StoryboardSprite()
        spr.texture = 'img.png'
        spr.color = Color4(1, 0, 0, 1)
        Add(spr)
        local txt = StoryboardText()
        txt.text = 'hello'
        txt.size = 40
        Add(txt)
    """)
    sprite, text = recorder.recorded
    assert sprite['parameters'] == {'file': 'img.png'}
    assert sprite['color'] == 0xFF0000FF
    assert text['parameters'] == {'text': 'hello', 'size': 40.0}


# ── host API ---------------------------------------------------------------

def test_map_queries_and_bpm():
    recorder = _recorder()
    recorder.run("""
        notes = map:NotesInRange(0, 2000)
        ticks = map:NotesInRange(0, 5000, HitObjectType('Tick'))
        bpm_early = BPMAtTime(100)
        bpm_late = BPMAtTime(2500)
        events = map:EventsInRange(0, 5000, 'Flash')
    """)
    env = recorder._host.env
    assert len(env['notes']) == 2
    assert env['notes'][2]['lane'] == 3
    assert env['notes'][2]['holdTime'] == 500.0
    assert len(env['ticks']) == 1
    assert env['bpm_early'] == 120.0
    assert env['bpm_late'] == 180.0
    assert env['events'][1]['duration'] == 100.0


def test_randomrange_is_deterministic_per_seed():
    values = []
    for _ in range(2):
        recorder = _recorder(seed='same.lua')
        recorder.run('v = RandomRange(0, 100)')
        values.append(recorder._host.env['v'])
    assert values[0] == values[1]


def test_process_reads_element_params_and_defaults():
    recorder = _recorder()
    recorder.run("""
        DefineParameter('speed', 'Speed', 'float', 7)
        function process(parent)
            local box = StoryboardBox()
            box.time = parent.time
            box.x = parent:param('offset', 1)
            box.y = parent:param('speed', 0)
            box.width = 10
            box.height = 10
            Add(box)
        end
    """)
    recorder.process({'time': 250.0, 'endtime': 500.0,
                      'parameters': {'offset': 42}})
    (el,) = recorder.recorded
    assert el['start'] == 250.0
    assert el['x'] == 42.0     # element parameter wins
    assert el['y'] == 7.0      # DefineParameter fallback


def test_fft_frames_are_silent_without_audio():
    recorder = _recorder()
    recorder.run("""
        frames = AudioAnalyzer:AmplitudesInRange(0, 100, 50, 16)
        first_total = frames[1].bands.total
        silent = frames[1]:IsSilent(0.001)
    """)
    env = recorder._host.env
    assert len(env['frames']) == 3
    assert env['first_total'] == 0.0
    assert env['silent'] is True


# ── end-to-end through parse_fsb -------------------------------------------

def test_parse_fsb_runs_scripts_with_chart_context(tmp_path):
    (tmp_path / 'gen.lua').write_text("""
        function process(parent)
            local box = StoryboardBox()
            box.layer = Layer('Overlay')
            box.time = 1000
            box.endtime = 2000
            box.width = 10
            box.height = 10
            box:animate('Fade', 1000, 1000, 1, 0, 0)
            Add(box)
        end
    """, encoding='utf-8')
    fsb = tmp_path / 'sb.fsb'
    fsb.write_text(json.dumps({'elements': [
        {'type': 3, 'layer': 0, 'start': 0.0, 'end': 0.0,
         'parameters': {'path': 'gen.lua'}},
    ]}), encoding='utf-8')

    assert parse_fsb(fsb) is None                    # no chart context
    sb = parse_fsb(fsb, chart=_chart())
    (el,) = sb.elements
    assert el.kind == 'rect'
    assert el.z == 700
    assert el.sample('alpha', 1.5) == pytest.approx((0.5,))


def test_broken_script_skips_without_raising(tmp_path):
    fsb_raw = {'elements': [
        {'type': 3, 'parameters': {'path': 'broken.lua'}},
        {'type': 3, 'parameters': {'path': 'missing.lua'}},
    ]}
    (tmp_path / 'broken.lua').write_text('this is not lua',
                                         encoding='utf-8')
    assert record_script_elements(fsb_raw, tmp_path, _chart()) == []
