"""Storyboard IR, renderer, and the fluXis/osu compilers."""
from types import SimpleNamespace

import pytest

from analysis.games.fluxis.fsb_storyboard import parse_fsb
from analysis.games.osu.storyboard_osb import (_parse_objects,
                                               parse_osu_storyboard)
from analysis.player.render.storyboard import Storyboard, StoryboardEffect
from analysis.player.render.storyboard.model import build_timelines
from analysis.player.render.storyboard.render import _design_transform


def _ctx(t=0.0, rect=(0, 0, 1000, 700)):
    return SimpleNamespace(t_now=t, chart_rect=rect)


# ── model ---------------------------------------------------------------

def test_build_timelines_rest_defaults_and_overrides():
    timelines = build_timelines({'x': 5.0, 'color': (0.5, 0.25, 0.0)})
    assert timelines['x'].sample(0.0) == (5.0,)
    assert timelines['alpha'].sample(0.0) == (1.0,)
    assert timelines['scale_x'].sample(0.0) == (1.0,)
    assert timelines['color'].sample(0.0) == (0.5, 0.25, 0.0)


# ── design-space mapping -------------------------------------------------

def test_design_transform_min_fit_letterboxes():
    sb = Storyboard(design_w=1920, design_h=1080, fit='min')
    k, ox, oy = _design_transform(sb, (0, 0, 960, 1080))
    assert k == pytest.approx(0.5)                # width-bound
    assert ox == pytest.approx(0.0)
    assert oy == pytest.approx((1080 - 540) / 2)  # vertically centered


def test_design_transform_height_fit_extends_sideways():
    sb = Storyboard(design_w=640, design_h=480, fit='height')
    k, ox, oy = _design_transform(sb, (0, 0, 1000, 480))
    assert k == pytest.approx(1.0)
    assert ox == pytest.approx((1000 - 640) / 2)
    assert oy == pytest.approx(0.0)


# ── fluXis .fsb compiler --------------------------------------------------

def _fsb(tmp_path, payload):
    import json
    path = tmp_path / 'sb.fsb'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def _fsb_element(**overrides):
    element = {
        'type': 0, 'layer': 2, 'z-index': 0,
        'start': 1000.0, 'end': 3000.0,
        'anchor': 18, 'origin': 18, 'x': 10.0, 'y': -20.0,
        'width': 100.0, 'height': 50.0,
        'color': 0xFF0000FF, 'parameters': {}, 'animations': [],
    }
    element.update(overrides)
    return element


def test_fsb_rect_element_basics(tmp_path):
    sb = parse_fsb(_fsb(tmp_path, {
        'resolution': {'x': 2560.0, 'y': 1440.0},
        'elements': [_fsb_element()],
    }))
    (el,) = sb.elements
    assert (sb.design_w, sb.design_h, sb.fit) == (2560.0, 1440.0, 'min')
    assert el.kind == 'rect'
    assert el.z == 700                       # Overlay
    assert el.anchor == (0.5, 0.5)           # bitmask 18 = Centre
    assert (el.t_start, el.t_end) == (1.0, 3.0)
    assert el.sample('x', 0.0) == (10.0,)
    assert el.sample('color', 0.0) == pytest.approx((1.0, 0.0, 0.0))
    assert el.sample('alpha', 0.0) == (1.0,)  # packed alpha byte 0xFF


def test_fsb_v1_animation_times_are_absolute(tmp_path):
    fade = {'start': 2000.0, 'duration': 1000.0, 'easing': 0,
            'type': 7, 'start-value': '0', 'end-value': '1'}
    v1 = parse_fsb(_fsb(tmp_path, {
        'elements': [_fsb_element(animations=[fade])]}))
    v2 = parse_fsb(_fsb(tmp_path, {
        'version': 2, 'elements': [_fsb_element(animations=[fade])]}))
    assert v1.elements[0].sample('alpha', 2.5) == pytest.approx((0.5,))
    # v2: same animation starts at element start (1s) + 2s = 3s.
    assert v2.elements[0].sample('alpha', 3.5) == pytest.approx((0.5,))


def test_fsb_scalar_scale_feeds_both_axes(tmp_path):
    grow = {'start': 1000.0, 'duration': 1000.0, 'easing': 0,
            'type': 2, 'start-value': '1', 'end-value': '3'}
    sb = parse_fsb(_fsb(tmp_path, {
        'elements': [_fsb_element(animations=[grow])]}))
    (el,) = sb.elements
    assert el.sample('scale_x', 1.5) == pytest.approx((2.0,))
    assert el.sample('scale_y', 1.5) == pytest.approx((2.0,))


def test_fsb_skips_unsupported_kinds_and_empty(tmp_path):
    assert parse_fsb(_fsb(tmp_path, {'elements': [
        _fsb_element(type=3), _fsb_element(type=9)]})) is None
    assert parse_fsb(tmp_path / 'missing.fsb') is None


def test_fsb_sprite_resolves_asset_beside_file(tmp_path):
    sb = parse_fsb(_fsb(tmp_path, {'elements': [
        _fsb_element(type=1, parameters={'file': 'img.png'})]}))
    (el,) = sb.elements
    assert el.kind == 'sprite'
    assert el.asset == str(tmp_path / 'img.png')


# ── osu .osb compiler ------------------------------------------------------

def _osu_chart(tmp_path, events, osb=None):
    osu = tmp_path / 'map.osu'
    osu.write_text('[General]\nMode: 3\n[Events]\n' + events
                   + '\n[TimingPoints]\n', encoding='utf-8')
    if osb is not None:
        (tmp_path / 'extra.osb').write_text(osb, encoding='utf-8')
    return osu


def test_osb_sprite_lifetime_and_rest_values(tmp_path):
    sb = parse_osu_storyboard(_osu_chart(tmp_path, '\n'.join((
        'Sprite,Foreground,Centre,"pic.png",320,240',
        ' F,0,1000,2000,0,1',
        ' M,0,1500,2500,100,50,200,80',
    ))))
    (el,) = sb.elements
    assert (el.t_start, el.t_end) == (1.0, 2.5)
    assert el.origin == (0.5, 0.5)
    assert el.z == -820
    # rest = first command's start values; x/y overridden by first M.
    assert el.sample('alpha', 0.0) == (0.0,)
    assert el.sample('x', 0.0) == (100.0,)
    assert el.sample('x', 2.0) == pytest.approx((150.0,))
    assert el.sample('alpha', 1.5) == pytest.approx((0.5,))


def test_osb_short_command_holds_value(tmp_path):
    sb = parse_osu_storyboard(_osu_chart(tmp_path, '\n'.join((
        'Sprite,Background,TopLeft,"a.png",10,20',
        ' M,0,0,1000,192,188',
    ))))
    (el,) = sb.elements
    assert el.sample('x', 0.5) == (192.0,)
    assert el.sample('y', 5.0) == (188.0,)


def test_osb_loop_expansion(tmp_path):
    lines = '\n'.join((
        'Sprite,Foreground,TopLeft,"a.png",0,0',
        ' L,1000,3',
        '  F,0,0,100,0,1',
    ))
    (obj,) = _parse_objects(lines.splitlines())
    starts = sorted(st for _t, _e, st, _et, _s, _v in obj.commands)
    assert starts == [1000.0, 1100.0, 1200.0]


def test_osb_fail_layer_dropped_and_flags(tmp_path):
    sb = parse_osu_storyboard(_osu_chart(tmp_path, '\n'.join((
        'Sprite,Fail,TopLeft,"f.png",0,0',
        ' F,0,0,1000,1',
        'Sprite,Overlay,TopLeft,"o.png",0,0',
        ' F,0,0,1000,1',
        ' P,0,0,1000,A',
    ))))
    (el,) = sb.elements
    assert el.z == 700
    assert el.additive


def test_osb_variables_substituted(tmp_path):
    sb = parse_osu_storyboard(_osu_chart(
        tmp_path, '',
        osb='\n'.join((
            '[Variables]',
            '$alpha=0.25',
            '[Events]',
            'Sprite,Pass,TopLeft,"v.png",0,0',
            ' F,0,0,1000,$alpha',
        ))))
    (el,) = sb.elements
    assert el.sample('alpha', 0.5) == (0.25,)


def test_osb_rotation_converts_radians(tmp_path):
    import math
    sb = parse_osu_storyboard(_osu_chart(tmp_path, '\n'.join((
        'Sprite,Background,TopLeft,"a.png",0,0',
        f' R,0,0,1000,0,{math.pi}',
    ))))
    (el,) = sb.elements
    assert el.sample('rotation', 0.5)[0] == pytest.approx(90.0)


def test_osb_no_objects_returns_none(tmp_path):
    assert parse_osu_storyboard(_osu_chart(
        tmp_path, '0,0,"bg.jpg",0,0')) is None


# ── renderer ---------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, args))
        return record

    def named(self, name):
        return [args for n, args in self.calls if n == name]


def _rect_storyboard(tmp_path, **element_overrides):
    fade = {'start': 1000.0, 'duration': 1000.0, 'easing': 0,
            'type': 7, 'start-value': '1', 'end-value': '0'}
    return parse_fsb(_fsb(tmp_path, {'elements': [
        _fsb_element(animations=[fade], **element_overrides)]}))


def test_effect_inactive_outside_element_window(tmp_path):
    eff = StoryboardEffect(_rect_storyboard(tmp_path))
    assert eff.at(_ctx(t=0.5)) is None
    assert eff.at(_ctx(t=3.5)) is None
    assert eff


def test_effect_draws_at_layer_z_and_paints_rect(tmp_path):
    eff = StoryboardEffect(_rect_storyboard(tmp_path))
    frame = eff.at(_ctx(t=1.2))
    ((z, draw),) = frame.draws
    assert z == 700

    recorder = _Recorder()
    draw(_ctx(t=1.2), recorder)
    assert recorder.named('fillRect')
    assert recorder.named('save') and recorder.named('restore')
    # fade 1 -> 0 over [1s, 2s]: at 1.2s alpha is 0.8.
    assert recorder.named('setOpacity')[0][0] == pytest.approx(0.8)


def test_effect_skips_invisible_elements(tmp_path):
    eff = StoryboardEffect(_rect_storyboard(tmp_path))
    frame = eff.at(_ctx(t=2.999))
    recorder = _Recorder()
    frame.draws[0][1](_ctx(t=2.999), recorder)
    assert not recorder.named('fillRect')   # alpha eased to ~0
