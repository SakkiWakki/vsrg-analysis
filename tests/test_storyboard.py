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


# ── hierarchical groups ---------------------------------------------------

from analysis.player.render.storyboard.model import Element  # noqa: E402
from analysis.player.render.storyboard.render import (  # noqa: E402
    _is_white_texture, _white_pixmap)


def _leaf(kind, **overrides):
    fields = dict(
        kind=kind, z=0, z_index=0, t_start=0.0, t_end=float('inf'),
        anchor=(0.0, 0.0), origin=(0.0, 0.0), timelines=build_timelines())
    fields.update(overrides)
    return Element(**fields)


def _group(children, **overrides):
    return _leaf('group', children=tuple(children), **overrides)


def _rendered_bbox(storyboard, t, size=200):
    """Render one frame into an offscreen ARGB image and return the
    bounding box (x0, y0, x1, y1) of the non-transparent pixels, or None
    when nothing drew. Design space maps 1:1 into the image."""
    from PySide6.QtGui import QImage, QPainter

    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(0)
    eff = StoryboardEffect(storyboard)
    frame = eff.at(_ctx(t=t, rect=(0, 0, size, size)))
    if frame is None:
        return None
    painter = QPainter(image)
    for _z, draw in frame.draws:
        draw(_ctx(t=t, rect=(0, 0, size, size)), painter)
    painter.end()

    xs, ys = [], []
    for py in range(size):
        for px in range(size):
            if image.pixelColor(px, py).alpha() > 0:
                xs.append(px)
                ys.append(py)
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def _center_group(children, **overrides):
    """A group hung at the design-rect center (anchor 0.5,0.5), so a
    rotated child stays on-canvas around the midpoint."""
    return _leaf('group', anchor=(0.5, 0.5), children=tuple(children),
                 **overrides)


def test_group_rotation_moves_child_position():
    # A child rect sits to the RIGHT of the group's center origin.
    # Rotating the group 90deg clockwise (Qt y-down => +x maps to -y)
    # swings the child ABOVE the center.
    child = _leaf('rect', timelines=build_timelines({'x': 60.0, 'w': 8.0,
                                                     'h': 8.0}))
    flat_sb = Storyboard(200, 200, 'height',
                         (_center_group([child],
                                        timelines=build_timelines()),))
    rot_sb = Storyboard(200, 200, 'height',
                        (_center_group([child],
                                       timelines=build_timelines(
                                           {'rotation': 90.0})),))

    flat = _rendered_bbox(flat_sb, t=0.5)
    rotated = _rendered_bbox(rot_sb, t=0.5)
    assert flat is not None and rotated is not None
    # Unrotated: child sits to the RIGHT of center (x ~160), y ~100.
    assert flat[0] > 140 and 90 < flat[1] < 110
    # Rotated 90deg about the center (Qt clockwise, y-down): the +x
    # offset maps to +y, so the child swings BELOW center (y ~160) at
    # x ~100 - a position the flat layout never occupies.
    assert rotated[1] > 140 and 90 < rotated[0] < 110


def test_group_window_culls_whole_subtree():
    child = _leaf('rect', timelines=build_timelines({'w': 20.0, 'h': 20.0}))
    sb = Storyboard(200, 200, 'height',
                    (_group([child], t_start=5.0, t_end=10.0,
                            timelines=build_timelines()),))
    # Group window is [5, 10): nothing draws before it opens.
    assert _rendered_bbox(sb, t=1.0) is None
    assert _rendered_bbox(sb, t=7.0) is not None


def test_child_window_culls_while_siblings_draw():
    early = _leaf('rect', t_start=0.0, t_end=2.0,
                  timelines=build_timelines({'x': 10.0, 'w': 8.0, 'h': 8.0}))
    late = _leaf('rect', t_start=5.0, t_end=9.0,
                 timelines=build_timelines({'x': 120.0, 'w': 8.0, 'h': 8.0}))
    sb = Storyboard(200, 200, 'height',
                    (_group([early, late], timelines=build_timelines()),))
    at_1 = _rendered_bbox(sb, t=1.0)
    at_6 = _rendered_bbox(sb, t=6.0)
    assert at_1 is not None and at_1[0] < 40      # only the early child
    assert at_6 is not None and at_6[0] > 100     # only the late child


def test_group_alpha_multiplies_onto_children():
    child = _leaf('rect', timelines=build_timelines(
        {'w': 20.0, 'h': 20.0, 'alpha': 0.5}))
    sb = Storyboard(200, 200, 'height',
                    (_group([child],
                            timelines=build_timelines({'alpha': 0.4})),))
    from PySide6.QtGui import QImage, QPainter
    image = QImage(200, 200, QImage.Format.Format_ARGB32)
    image.fill(0)
    eff = StoryboardEffect(sb)
    painter = QPainter(image)
    for _z, draw in eff.at(_ctx(t=0.5, rect=(0, 0, 200, 200))).draws:
        draw(_ctx(t=0.5, rect=(0, 0, 200, 200)), painter)
    painter.end()
    # composed opacity = 0.4 * 0.5 = 0.2 over white => alpha ~= 51/255.
    alpha = image.pixelColor(5, 5).alpha()
    assert 40 < alpha < 65


# ── hidden vs alpha split -------------------------------------------------

def test_hidden_bit_gates_draw_independently_of_alpha():
    """SM's `hidden` bit hard-gates the draw even at full alpha: a
    diffusealpha crossfade can ride an actor a `hidden,1` currently
    hides."""
    shown = _leaf('rect', timelines=build_timelines(
        {'w': 20.0, 'h': 20.0, 'alpha': 1.0, 'hidden': 0.0}))
    hidden = _leaf('rect', timelines=build_timelines(
        {'w': 20.0, 'h': 20.0, 'alpha': 1.0, 'hidden': 1.0}))
    shown_sb = Storyboard(200, 200, 'height', (shown,))
    hidden_sb = Storyboard(200, 200, 'height', (hidden,))
    assert _rendered_bbox(shown_sb, t=0.5) is not None
    assert _rendered_bbox(hidden_sb, t=0.5) is None   # gated off at alpha 1


def test_hidden_group_hides_whole_subtree():
    child = _leaf('rect', timelines=build_timelines({'w': 20.0, 'h': 20.0}))
    sb = Storyboard(200, 200, 'height',
                    (_group([child],
                            timelines=build_timelines({'hidden': 1.0})),))
    assert _rendered_bbox(sb, t=0.5) is None


# ── SM built-in 'white' texture -------------------------------------------

def test_white_texture_recognized_and_synthesized():
    assert _is_white_texture('white')
    assert _is_white_texture('  White ')
    assert not _is_white_texture('/path/to/white.png')
    assert not _is_white_texture(None)
    pm = _white_pixmap()
    assert not pm.isNull()
    assert pm.toImage().pixelColor(0, 0).alpha() == 255


def test_sprite_referencing_white_draws_without_missing_warning(capsys):
    sprite = _leaf('sprite', asset='white',
                   timelines=build_timelines({'alpha': 1.0}))
    sb = Storyboard(200, 200, 'height', (sprite,))
    bbox = _rendered_bbox(sb, t=0.5)
    assert bbox is not None                       # the white pixmap drew
    assert 'missing' not in capsys.readouterr().out
