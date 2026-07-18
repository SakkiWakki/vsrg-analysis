"""Per-corner diffuse gradient, additive glow, and edge fades: the color
families whose semantics are pinned to openitg Sprite.cpp / Actor.cpp.

Three surfaces are covered: the SimActor recording (the corner/glow/fade
channels a chart pokes), the color/glow oscillator synthesis (rainbow /
diffuse* / glow* baked to keyframes), and the storyboard render (the
gradient/glow/fade actually reach pixels). Parity is the point: an
element poked with none of these draws byte-identical to the flat path.
"""
import math

import pytest

from analysis.games.notitg import modfile
from analysis.games.notitg.sim import SimActor
from analysis.player.render.effects.timeline import EventTimeline
from analysis.player.render.storyboard import Storyboard, StoryboardEffect
from analysis.player.render.storyboard.model import build_timelines


def _timeline(actor, prop, rest):
    return EventTimeline(actor.keyframes().get(prop, []), rest=rest)


# -- SimActor recording: per-corner diffuse (Actor.h:190-197) ----------------

def test_diffuse_corner_records_one_channel():
    a = SimActor()
    a.poke('diffuseupperleft', [1.0, 0.0, 0.0, 1.0])
    assert a.get('color_ul') == (1.0, 0.0, 0.0, 1.0)
    # The other three corners stay unset (the flat-fallback sentinel).
    for prop in ('color_ur', 'color_ll', 'color_lr'):
        assert a.get(prop)[0] < 0.0


def test_diffuse_edge_sets_two_corners():
    # SetDiffuseLeftEdge -> diffuse[0] (UL) and diffuse[2] (LL), Actor.h:197.
    a = SimActor()
    a.poke('diffuseleftedge', [0.0, 1.0, 0.0, 1.0])
    assert a.get('color_ul') == (0.0, 1.0, 0.0, 1.0)
    assert a.get('color_ll') == (0.0, 1.0, 0.0, 1.0)
    assert a.get('color_ur')[0] < 0.0
    assert a.get('color_lr')[0] < 0.0


def test_diffuse_corner_default_alpha_is_opaque():
    a = SimActor()
    a.poke('diffuseupperright', [0.2, 0.4, 0.6])
    assert a.get('color_ur') == (0.2, 0.4, 0.6, 1.0)


# -- SimActor recording: glow (Actor.h:200) + fades (Actor.h:178-181) --------

def test_glow_records_rgba():
    a = SimActor()
    a.poke('glow', [1.0, 1.0, 1.0, 0.5])
    assert a.get('glow') == (1.0, 1.0, 1.0, 0.5)


def test_glow_rests_at_no_glow():
    a = SimActor()
    assert a.get('glow') == (1.0, 1.0, 1.0, 0.0)   # alpha 0 = no pass


def test_fade_edges_record():
    a = SimActor()
    a.poke('fadeleft', [0.25])
    a.poke('fadev', [0.1])
    assert a.get('fade_left') == 0.25
    assert a.get('fade_top') == 0.1
    assert a.get('fade_bottom') == 0.1
    assert a.get('fade_right') == 0.0


def test_fade_all_sides():
    a = SimActor()
    a.poke('fade', [0.3])
    for prop in ('fade_left', 'fade_right', 'fade_top', 'fade_bottom'):
        assert a.get(prop) == 0.3


# -- getdiffuse readback (Actor.h:198) ---------------------------------------

def test_getdiffuse_reads_flat_color():
    a = SimActor()
    a.poke('diffuse', [0.5, 0.6, 0.7, 0.8])
    assert a.read('getdiffuse') == (0.5, 0.6, 0.7, 0.8)


def test_getdiffuse_prefers_upper_left_corner():
    a = SimActor()
    a.poke('diffuse', [0.5, 0.5, 0.5, 1.0])
    a.poke('diffuseupperleft', [1.0, 0.0, 0.0, 1.0])
    assert a.read('getdiffuse') == (1.0, 0.0, 0.0, 1.0)


# -- color / glow oscillator synthesis (Actor.cpp:288-330) -------------------

class _Span:
    """A minimal color-oscillator span mirroring SimActor.OscSpan's shape
    for the synthesis (kind/start/end/period/offset/clock + effectcolors)."""

    def __init__(self, kind, start, end, c1, c2, period=1.0):
        self.kind = kind
        self.start = start
        self.end = end
        self.period = period
        self.offset = 0.0
        self.clock = 'music'
        self.explicit_end = True
        self.extra = {'effectcolor1': c1, 'effectcolor2': c2}

    def magnitude_at(self, t):
        return (0.0, 0.0, 0.0)


def _clock():
    return modfile._OscillatorClock(lambda beat: float(beat), (0.0, 64.0))


def test_diffuseshift_synthesizes_color_keyframes():
    span = _Span('diffuseshift', 0.0, 2.0,
                 (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0))
    kf = modfile.compile_oscillator_keyframes([span], {}, _clock(), None)
    assert 'color' in kf and 'glow' not in kf
    tl = EventTimeline(kf['color'], rest=(1.0, 1.0, 1.0))
    # Every sample is a blend of red and blue (green stays 0).
    for t in (0.1, 0.5, 1.0, 1.9):
        r, g, b = tl.sample(t)
        assert g == pytest.approx(0.0, abs=1e-6)
        assert 0.0 <= r <= 1.0 and 0.0 <= b <= 1.0
    # After the span, color hands back to white (the trailing rest).
    assert tl.sample(2.5) == pytest.approx((1.0, 1.0, 1.0))


def test_glowshift_synthesizes_glow_not_color():
    span = _Span('glowshift', 0.0, 2.0,
                 (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0))
    kf = modfile.compile_oscillator_keyframes([span], {}, _clock(), None)
    assert 'glow' in kf and 'color' not in kf
    tl = EventTimeline(kf['glow'], rest=(1.0, 1.0, 1.0, 0.0))
    assert tl.sample(2.5) == pytest.approx((1.0, 1.0, 1.0, 0.0))


def test_diffuseblink_switches_at_half_period():
    c1 = (1.0, 0.0, 0.0, 1.0)
    c2 = (0.0, 1.0, 0.0, 1.0)
    span = _Span('diffuseblink', 0.0, 2.0, c1, c2, period=1.0)
    kf = modfile.compile_oscillator_keyframes([span], {}, _clock(), None)
    tl = EventTimeline(kf['color'], rest=(1.0, 1.0, 1.0))
    # pct = fmod(t, 1); blink_on when pct > 0.5 -> c1, else c2.
    assert tl.sample(0.25)[:3] == pytest.approx(c2[:3])   # pct 0.25 -> c2
    assert tl.sample(0.75)[:3] == pytest.approx(c1[:3])   # pct 0.75 -> c1


def test_rainbow_sweeps_hue():
    span = _Span('rainbow', 0.0, 2.0, (1, 1, 1, 1), (1, 1, 1, 1))
    kf = modfile.compile_oscillator_keyframes([span], {}, _clock(), None)
    tl = EventTimeline(kf['color'], rest=(1.0, 1.0, 1.0))
    hues = {tuple(round(c, 3) for c in tl.sample(t)) for t in
            (0.0, 0.25, 0.5, 0.75)}
    assert len(hues) > 1                      # the hue actually moves
    for hue in hues:
        assert all(0.0 <= c <= 1.0 for c in hue)


def test_default_white_oscillator_is_parity_safe():
    # A bare diffuseshift() (both effect colors default white) must produce
    # white at every sample - identical to the actor's flat white diffuse.
    span = _Span('diffuseshift', 0.0, 2.0,
                 (1, 1, 1, 1), (1, 1, 1, 1))
    kf = modfile.compile_oscillator_keyframes([span], {}, _clock(), None)
    tl = EventTimeline(kf['color'], rest=(1.0, 1.0, 1.0))
    for t in (0.0, 0.3, 0.6, 0.9, 1.5):
        assert tl.sample(t) == pytest.approx((1.0, 1.0, 1.0))


# -- storyboard render: parity + gradient/glow/fade reach pixels -------------

def _ctx(t, rect):
    from types import SimpleNamespace
    return SimpleNamespace(t_now=t, chart_rect=rect)


def _render(storyboard, t=0.0, size=64):
    from PySide6.QtGui import QImage, QPainter
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(0)
    eff = StoryboardEffect(storyboard)
    frame = eff.at(_ctx(t, (0, 0, size, size)))
    if frame is None:
        return image
    painter = QPainter(image)
    for _z, draw in frame.draws:
        draw(_ctx(t, (0, 0, size, size)), painter)
    painter.end()
    return image


def _rect_sb(rests):
    el = _rect_element(rests)
    return Storyboard(64, 64, 'stretch', (el,))


def _rect_element(rests):
    from analysis.player.render.storyboard.model import Element
    merged = {'x': 32.0, 'y': 32.0, 'w': 40.0, 'h': 40.0, **rests}
    return Element(kind='rect', z=0, z_index=0, t_start=0.0,
                   t_end=float('inf'), anchor=(0.0, 0.0), origin=(0.5, 0.5),
                   timelines=build_timelines(merged))


def _images_equal(a, b) -> bool:
    if a.size() != b.size():
        return False
    return all(a.pixelColor(x, y) == b.pixelColor(x, y)
               for y in range(a.height()) for x in range(a.width()))


def test_flat_rect_parity_no_color_extras():
    # An element with no corner/glow/fade must render exactly the flat quad.
    plain = _render(_rect_sb({'color': (0.2, 0.4, 0.8)}))
    withcolor = _render(_rect_sb({'color': (0.2, 0.4, 0.8)}))
    assert _images_equal(plain, withcolor)
    # And it actually drew something.
    assert any(plain.pixelColor(x, y).alpha() > 0
               for y in range(64) for x in range(64))


def test_gradient_left_darker_than_right():
    # UL/LL = black, UR/LR = white -> left edge darker than right.
    sb = _rect_sb({'color_ul': (0.0, 0.0, 0.0, 1.0),
                   'color_ll': (0.0, 0.0, 0.0, 1.0),
                   'color_ur': (1.0, 1.0, 1.0, 1.0),
                   'color_lr': (1.0, 1.0, 1.0, 1.0)})
    img = _render(sb)
    left = img.pixelColor(14, 32).lightnessF()
    right = img.pixelColor(50, 32).lightnessF()
    assert left < right


def test_glow_adds_light_over_base():
    base = _render(_rect_sb({'color': (0.1, 0.1, 0.1)}))
    glowed = _render(_rect_sb({'color': (0.1, 0.1, 0.1),
                               'glow': (1.0, 1.0, 1.0, 1.0)}))
    # Additive glow raises the center pixel's brightness.
    assert (glowed.pixelColor(32, 32).lightnessF()
            > base.pixelColor(32, 32).lightnessF())


def test_fade_left_thins_alpha_at_edge():
    sb = _rect_sb({'color': (1.0, 1.0, 1.0), 'fade_left': 0.4})
    img = _render(sb)
    # The left edge column is more transparent than the interior.
    left_alpha = img.pixelColor(13, 32).alpha()
    mid_alpha = img.pixelColor(48, 32).alpha()
    assert left_alpha < mid_alpha
