"""Effects framework: compositor, timeline sampling, transform math."""
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QTransform

from analysis.player.render.effects import composite
from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.effects.timeline import EventTimeline, Keyframe
from analysis.player.render.effects.playfield_transform import (
    PlayfieldTransformEffect)


class _Effect:
    def __init__(self, frame):
        self._frame = frame

    def at(self, ctx):
        return self._frame


def _ctx(t=0.0):
    player = SimpleNamespace(keycount=4)
    return SimpleNamespace(t_now=t, x0=100.0, lane_w=40.0, judge_y=600,
                           player=player)


def test_composite_identity_when_no_effects():
    assert composite([], _ctx()).is_identity


def test_composite_skips_inactive_effects():
    frame = composite([_Effect(None), _Effect(None)], _ctx())
    assert frame.is_identity


def test_begin_effect_transform_confined_to_chart_rect():
    # Opacity + transform apply only inside the effect bracket; the HUD
    # paints outside it, so an invisibility effect can never hide the
    # sidebar. Assert the bracket clips to chart_rect and is restorable.
    from analysis.player.render.qt_renderer import QtPlayerRenderer
    from analysis.player.render.effects.base import CompositeFrame

    class FakePainter:
        def __init__(self):
            self.saved = 0
            self.clip = None
            self.opacity = 1.0
        def save(self): self.saved += 1
        def restore(self): self.saved -= 1
        def setClipRect(self, r): self.clip = r
        def setOpacity(self, o): self.opacity = o
        def setTransform(self, t, combine): pass

    ctx = _ctx_win()
    p = FakePainter()
    frame = CompositeFrame(transform=None, opacity=0.0)
    wrapped = QtPlayerRenderer._begin_effect_transform(frame, p, ctx)
    assert wrapped and p.saved == 1
    assert p.opacity == 0.0
    assert (p.clip.width(), p.clip.height()) == pytest.approx(
        (ctx.chart_rect[2], ctx.chart_rect[3]))
    QtPlayerRenderer._end_effect_transform(p)
    assert p.saved == 0   # opacity/clip fully unwound before HUD paints


def test_composite_multiplies_opacity_and_composes_transforms():
    a = QTransform().translate(10, 0)
    b = QTransform().scale(2, 2)
    frame = composite([_Effect(EffectFrame(transform=a, opacity=0.5)),
                       _Effect(EffectFrame(transform=b, opacity=0.5))],
                      _ctx())
    assert frame.opacity == pytest.approx(0.25)
    assert frame.transform is not None
    assert not frame.is_identity


def test_composite_splits_draws_by_z():
    below = (-1, lambda c, p: None)
    above = (2, lambda c, p: None)
    mid = (0, lambda c, p: None)
    frame = composite([_Effect(EffectFrame(draws=(above, below, mid)))],
                      _ctx())
    assert [z for z, _ in frame.below] == [-1]
    assert [z for z, _ in frame.above] == [0, 2]


# ── timeline ----------------------------------------------------------

def test_timeline_rest_before_first_keyframe():
    tl = EventTimeline([Keyframe(5.0, (10.0,), 0.0, 0)], rest=(0.0,))
    assert tl.sample(1.0) == (0.0,)


def test_timeline_eases_then_holds():
    # linear (easing id 0 = None) ease from 0 -> 10 over 2s at t=5.
    tl = EventTimeline([Keyframe(5.0, (10.0,), 2.0, 0)], rest=(0.0,))
    assert tl.sample(6.0) == pytest.approx((5.0,))    # halfway
    assert tl.sample(8.0) == pytest.approx((10.0,))   # settled
    assert tl.sample(99.0) == pytest.approx((10.0,))  # holds


# ── playfield transform ----------------------------------------------

def _ctx_win(t=0.0, W=1366, H=768):
    player = SimpleNamespace(keycount=4, W=W, H=H)
    return SimpleNamespace(t_now=t, x0=100.0, lane_w=40.0, judge_y=600,
                           player=player, chart_rect=(0, 0, W, H))


def test_playfield_rotate_is_about_field_center():
    eff = PlayfieldTransformEffect(
        rotate=[{'time': 0, 'roll': 90.0, 'duration': 0, 'ease': 0}])
    ctx = _ctx_win(t=1.0)
    frame = eff.at(ctx)
    cx = ctx.x0 + ctx.player.keycount * ctx.lane_w / 2   # 180
    cy = ctx.judge_y                                     # 600
    mapped = frame.transform.map(cx, cy)
    assert mapped[0] == pytest.approx(cx)   # receptor is fixed under rotation
    assert mapped[1] == pytest.approx(cy)


def test_playfield_z_depth_zooms_about_screen_center():
    # z = +100 halves the perspective scale (100/(100+100)); the field
    # shrinks about screen center.
    eff = PlayfieldTransformEffect(
        move=[{'time': 0, 'x': 0, 'y': 0, 'z': 100.0,
               'duration': 0, 'ease': 0}])
    ctx = _ctx_win(t=1.0)
    frame = eff.at(ctx)
    scx, scy = ctx.player.W / 2, ctx.player.H / 2
    assert frame.transform.map(scx, scy) == pytest.approx((scx, scy))
    m11 = frame.transform.m11()
    assert m11 == pytest.approx(0.5)


def test_playfield_move_scales_to_window():
    # x = ref width -> full-window shift, clamped to _MAX_OFFSET_FRAC.
    eff = PlayfieldTransformEffect(
        move=[{'time': 0, 'x': 1366.0, 'y': 0, 'z': 0,
               'duration': 0, 'ease': 0}])
    ctx = _ctx_win(t=1.0, W=1366, H=768)
    frame = eff.at(ctx)
    # raw dx would be 1366px; clamped to 0.5 * W = 683.
    assert frame.transform.dx() == pytest.approx(683.0)


def test_playfield_inactive_returns_none():
    eff = PlayfieldTransformEffect(move=[], scale=[], rotate=[])
    assert eff.at(_ctx()) is None
    assert not eff


def test_playfield_move_scales_with_chart_rect_width():
    # The same authored move shifts proportionally more in a wider
    # viewport -- effects reference the playfield/chart region size.
    eff = PlayfieldTransformEffect(
        move=[{'time': 0, 'x': 100.0, 'y': 0, 'z': 0,
               'duration': 0, 'ease': 0}])
    narrow = _ctx_win(t=1.0, W=800, H=600)
    wide = _ctx_win(t=1.0, W=1600, H=600)
    assert (eff.at(wide).transform.dx()
            > eff.at(narrow).transform.dx())


# ── camera / scene bracket ----------------------------------------------

def test_camera_move_is_negated_and_screen_proportional():
    from analysis.player.render.effects.camera import CameraEffect
    eff = CameraEffect(move=[{'time': 0, 'x': 1366.0, 'y': 0,
                              'duration': 0, 'ease': 0}])
    ctx = _ctx_win(t=1.0, W=1366, H=768)
    frame = eff.at(ctx)
    assert frame.transform is None
    # camera right by one ref-width -> scene slides one chart-width left.
    assert frame.scene_transform.dx() == pytest.approx(-1366.0)


def test_camera_scale_pivots_on_chart_center():
    from analysis.player.render.effects.camera import CameraEffect
    eff = CameraEffect(scale=[{'time': 0, 'scale': 2.0,
                               'duration': 0, 'ease': 0}])
    ctx = _ctx_win(t=1.0, W=1000, H=800)
    scene = eff.at(ctx).scene_transform
    assert scene.map(500.0, 400.0) == pytest.approx((500.0, 400.0))
    assert scene.m11() == pytest.approx(2.0)


def test_camera_inactive_returns_none():
    from analysis.player.render.effects.camera import CameraEffect
    assert CameraEffect().at(_ctx_win(t=1.0)) is None
    assert not CameraEffect()


def test_composite_scene_channel_and_top_split():
    from analysis.player.render.effects.base import SCENE_TOP_Z
    scene = QTransform().translate(5, 0)
    frame = composite([
        _Effect(EffectFrame(scene_transform=scene)),
        _Effect(EffectFrame(draws=((SCENE_TOP_Z + 100, lambda c, p: None),
                                   (2, lambda c, p: None),
                                   (-1, lambda c, p: None)))),
    ], _ctx())
    assert frame.scene_transform is not None
    assert not frame.is_identity
    assert [z for z, _ in frame.below] == [-1]
    assert [z for z, _ in frame.above] == [2]
    assert [z for z, _ in frame.top] == [SCENE_TOP_Z + 100]


def test_begin_scene_transform_clips_to_chart_rect():
    from analysis.player.render.qt_renderer import QtPlayerRenderer
    from analysis.player.render.effects.base import CompositeFrame

    class FakePainter:
        def __init__(self):
            self.saved = 0
            self.clip = None
        def save(self): self.saved += 1
        def restore(self): self.saved -= 1
        def setClipRect(self, r): self.clip = r
        def setTransform(self, t, combine): pass

    ctx = _ctx_win()
    p = FakePainter()
    frame = CompositeFrame(scene_transform=QTransform().translate(9, 9))
    assert QtPlayerRenderer._begin_scene_transform(frame, p, ctx)
    assert p.saved == 1 and p.clip is not None
    QtPlayerRenderer._end_effect_transform(p)
    assert p.saved == 0
    assert not QtPlayerRenderer._begin_scene_transform(
        CompositeFrame(), p, ctx)


def test_upscroll_mirrors_about_chart_center():
    from analysis.player.render.effects.upscroll import UpscrollEffect
    ctx = _ctx_win(W=1000, H=800)
    frame = UpscrollEffect().at(ctx)
    # judge line at 0.8H lands at 0.2H; center is fixed.
    assert frame.transform.map(500.0, 640.0) == pytest.approx((500.0, 160.0))
    assert frame.transform.map(500.0, 400.0) == pytest.approx((500.0, 400.0))


def test_composite_concatenates_field_instances():
    a = _Effect(EffectFrame(fields=((None, 1.0),)))
    b = _Effect(EffectFrame(
        fields=((QTransform().translate(50, 0), 0.5),)))
    frame = composite([a, b], _ctx())
    assert len(frame.fields) == 2
    assert frame.fields[0] == (None, 1.0)
    assert frame.fields[1][1] == 0.5
    assert not frame.is_identity


def test_fluxis_adapter_wires_camera():
    from analysis.games.fluxis.adapter import FluxisAdapter
    from analysis.player.render.effects.camera import CameraEffect
    replay = {'_fluxis_effect_streams': {
        'camera-move': [{'time': 0, 'x': 100.0, 'y': 0,
                         'duration': 0, 'ease': 0}]}}
    effects = FluxisAdapter().effects(replay)
    assert any(isinstance(e, CameraEffect) for e in effects)


# ── flash --------------------------------------------------------------

def _flash_event(time_ms, duration_ms, *, start=(1, 1, 1, 1), start_a=1.0,
                 end=(1, 1, 1, 1), end_a=0.0, background=False, ease=0):
    def color(c):
        return dict(zip('RGBA', (float(v) for v in c)))
    return {'time': time_ms, 'duration': duration_ms, 'ease': ease,
            'background': background,
            'start-color': color(start), 'start-alpha': start_a,
            'end-color': color(end), 'end-alpha': end_a}


class _FillRecorder:
    def __init__(self):
        self.fills = []

    def fillRect(self, rect, color):
        self.fills.append((rect, color))


def _flash_fill(eff, ctx):
    frame = eff.at(ctx)
    if frame is None:
        return None
    recorder = _FillRecorder()
    for _z, fn in frame.draws:
        fn(ctx, recorder)
    (_, color), = recorder.fills
    return color


def test_flash_invisible_before_first_event_and_after_fade_out():
    from analysis.player.render.effects.flash import FlashEffect
    eff = FlashEffect([_flash_event(1000, 500)])
    assert eff.at(_ctx_win(t=0.5)) is None
    assert eff.at(_ctx_win(t=2.0)) is None


def test_flash_snaps_to_start_then_fades_to_end():
    from analysis.player.render.effects.flash import FlashEffect
    eff = FlashEffect([_flash_event(1000, 1000, start=(1, 0, 0, 1),
                                    start_a=1.0, end=(1, 0, 0, 1),
                                    end_a=0.0)])
    at_start = _flash_fill(eff, _ctx_win(t=1.0))
    assert (at_start.redF(), at_start.alphaF()) == pytest.approx((1.0, 1.0))
    halfway = _flash_fill(eff, _ctx_win(t=1.5))
    assert halfway.alphaF() == pytest.approx(0.5, abs=0.01)


def test_flash_background_flag_selects_below_z():
    from analysis.player.render.effects.flash import FlashEffect
    front = FlashEffect([_flash_event(0, 1000)]).at(_ctx_win(t=0.2))
    back = FlashEffect([_flash_event(0, 1000, background=True)]).at(
        _ctx_win(t=0.2))
    assert front.draws[0][0] > 0
    assert back.draws[0][0] < 0


def test_flash_later_event_overrides_mid_fade():
    from analysis.player.render.effects.flash import FlashEffect
    eff = FlashEffect([
        _flash_event(0, 10000, start=(1, 0, 0, 1), end=(1, 0, 0, 1),
                     end_a=1.0),
        _flash_event(1000, 1000, start=(0, 0, 1, 1), end=(0, 0, 1, 1),
                     end_a=1.0),
    ])
    color = _flash_fill(eff, _ctx_win(t=1.0))
    assert (color.redF(), color.blueF()) == pytest.approx((0.0, 1.0))


# ── shake --------------------------------------------------------------

def _shake_events():
    return [{'time': 1000, 'duration': 1000, 'magnitude': 15.0}]


def test_shake_inactive_outside_window():
    from analysis.player.render.effects.shake import ShakeEffect
    eff = ShakeEffect(_shake_events())
    assert eff.at(_ctx_win(t=0.5)) is None
    assert eff.at(_ctx_win(t=2.5)) is None


def test_shake_offsets_are_deterministic_and_bounded():
    from analysis.player.render.effects.shake import ShakeEffect
    a = ShakeEffect(_shake_events())
    b = ShakeEffect(_shake_events())
    ctx = _ctx_win(t=1.3, W=1366, H=768)
    ta, tb = a.at(ctx).transform, b.at(ctx).transform
    assert (ta.dx(), ta.dy()) == (tb.dx(), tb.dy())
    assert ta.dx() != 0 or ta.dy() != 0
    # Magnitude is in the 1366x768 reference space; at that window size
    # offsets stay within +-magnitude on each axis.
    assert abs(ta.dx()) <= 15.0 and abs(ta.dy()) <= 15.0


def test_shake_returns_to_rest_at_end():
    from analysis.player.render.effects.shake import ShakeEffect
    eff = ShakeEffect(_shake_events())
    assert eff.at(_ctx_win(t=1.9999)) is None or (
        abs(eff.at(_ctx_win(t=1.9999)).transform.dx()) < 1.0)


# ── scroll multiplier --------------------------------------------------

def test_effective_scroll_speed_samples_chart_multiplier():
    from analysis.player.sv.render import SvRenderController
    from analysis.player.render.effects.timeline import (
        EventTimeline, keyframes_from_events)
    timeline = EventTimeline(
        keyframes_from_events(
            [{'time': 1000, 'duration': 1000, 'multiplier': 0.5, 'ease': 0}],
            ('multiplier',), (1.0,)),
        rest=(1.0,))
    player = SimpleNamespace(scroll_speed=100.0,
                             _scroll_mult_timeline=timeline)
    sv = SvRenderController(player)
    assert sv.effective_scroll_speed(0.0) == pytest.approx(100.0)
    assert sv.effective_scroll_speed(1.5) == pytest.approx(75.0)
    assert sv.effective_scroll_speed(5.0) == pytest.approx(50.0)

    player._scroll_mult_timeline = None
    assert sv.effective_scroll_speed(1.5) == pytest.approx(100.0)


def test_fluxis_adapter_exposes_scroll_multipliers_and_new_effects():
    from analysis.games.fluxis.adapter import FluxisAdapter
    from analysis.player.render.effects.flash import FlashEffect
    from analysis.player.render.effects.shake import ShakeEffect
    events = {'scroll-multiply': [{'time': 0, 'duration': 0,
                                   'multiplier': 0.75, 'ease': 0}],
              'flash': [_flash_event(0, 100)],
              'shake': _shake_events()}
    replay = {'_fluxis_effect_streams': events}
    adapter = FluxisAdapter()
    assert adapter.scroll_multipliers(replay) == events['scroll-multiply']
    built = adapter.effects(replay)
    assert any(isinstance(e, FlashEffect) for e in built)
    assert any(isinstance(e, ShakeEffect) for e in built)


# ── loops expansion ---------------------------------------------------

def test_loops_clones_group_per_count_offset_by_distance():
    from analysis.games.fluxis.fsc_chart import _expand_loops
    streams = {
        'flash': [{'time': 1000, 'group': 'a', 'duration': 0},
                  {'time': 1200, 'group': 'a', 'duration': 0}],
        'loops': [{'time': 5000, 'target': 'a', 'distance': 100, 'count': 3}],
    }
    out = _expand_loops(streams)
    times = sorted(e['time'] for e in out['flash'])
    # Originals kept (1000, 1200) plus 3 clones: each clone preserves the
    # 0/200ms delta above the group's lowest time (1000), i starting at 0.
    assert times == [1000, 1200, 5000, 5100, 5200, 5200, 5300, 5400]
    assert 'loops' not in out


def test_loops_drops_stream_and_is_noop_without_matching_group():
    from analysis.games.fluxis.fsc_chart import _expand_loops
    streams = {
        'flash': [{'time': 0, 'group': 'b', 'duration': 0}],
        'loops': [{'time': 5000, 'target': 'missing', 'distance': 100,
                   'count': 3}],
    }
    out = _expand_loops(streams)
    assert [e['time'] for e in out['flash']] == [0]
    assert 'loops' not in out


def test_loops_ignores_ungrouped_events():
    from analysis.games.fluxis.fsc_chart import _expand_loops
    streams = {
        'flash': [{'time': 0, 'group': 'a', 'duration': 0},
                  {'time': 999, 'duration': 0}],
        'loops': [{'time': 4000, 'target': 'a', 'distance': 0, 'count': 2}],
    }
    out = _expand_loops(streams)
    times = sorted(e['time'] for e in out['flash'])
    # The ungrouped 999 event is untouched; only the grouped one clones.
    assert times == [0, 999, 4000, 4000]


# ── layer fade --------------------------------------------------------

def _fade_ctx(t):
    return SimpleNamespace(t_now=t, layer_opacities=None)


def test_layer_fade_maps_hitobjects_to_note_layers():
    from analysis.player.render.effects.layer_fade import (
        FADE_HITOBJECTS, LayerFadeEffect)
    eff = LayerFadeEffect({'layerfade': [
        {'time': 0, 'duration': 0, 'alpha': 0.0, 'layer': FADE_HITOBJECTS}]})
    ctx = _fade_ctx(1.0)
    eff.at(ctx)
    assert ctx.layer_opacities['taps'] == pytest.approx(0.0)
    assert ctx.layer_opacities['lns'] == pytest.approx(0.0)
    assert 'lanes' not in ctx.layer_opacities


def test_layer_fade_playfield_covers_stage_and_receptors():
    from analysis.player.render.effects.layer_fade import (
        FADE_PLAYFIELD, LayerFadeEffect)
    eff = LayerFadeEffect({'layerfade': [
        {'time': 0, 'duration': 0, 'alpha': 0.2, 'layer': FADE_PLAYFIELD}]})
    ctx = _fade_ctx(1.0)
    eff.at(ctx)
    for name in ('taps', 'lanes', 'judgment'):
        assert ctx.layer_opacities[name] == pytest.approx(0.2)


def test_layer_fade_eases_over_duration_then_holds():
    from analysis.player.render.effects.layer_fade import (
        FADE_STAGE, LayerFadeEffect)
    eff = LayerFadeEffect({'layerfade': [
        {'time': 1000, 'duration': 1000, 'alpha': 0.0, 'ease': 0,
         'layer': FADE_STAGE}]})
    before = _fade_ctx(0.5)
    eff.at(before)
    assert before.layer_opacities['lanes'] == pytest.approx(1.0)
    mid = _fade_ctx(1.5)
    eff.at(mid)
    assert mid.layer_opacities['lanes'] == pytest.approx(0.5)
    after = _fade_ctx(3.0)
    eff.at(after)
    assert after.layer_opacities['lanes'] == pytest.approx(0.0)


def test_layer_fade_merges_legacy_hitfade_and_playfieldfade():
    from analysis.player.render.effects.layer_fade import LayerFadeEffect
    eff = LayerFadeEffect({
        'hitfade': [{'time': 0, 'duration': 0, 'alpha': 0.5}],
        'playfieldfade': [{'time': 0, 'duration': 0, 'alpha': 0.3}]})
    ctx = _fade_ctx(1.0)
    eff.at(ctx)
    # hitfade defaults to HitObjects (taps 0.5); playfieldfade forces
    # Playfield, which also covers lanes at 0.3. taps takes the min.
    assert ctx.layer_opacities['taps'] == pytest.approx(0.3)
    assert ctx.layer_opacities['lanes'] == pytest.approx(0.3)


def test_layer_fade_hud_layer_has_no_field_effect():
    from analysis.player.render.effects.layer_fade import (
        FADE_HUD, LayerFadeEffect)
    eff = LayerFadeEffect({'layerfade': [
        {'time': 0, 'duration': 0, 'alpha': 0.0, 'layer': FADE_HUD}]})
    assert not eff


# ── pulse -------------------------------------------------------------

class _PenRecorder:
    def __init__(self):
        self.pen_width = None

    def save(self): pass
    def restore(self): pass
    def setBrush(self, *_): pass
    def drawRect(self, *_): pass

    def setPen(self, pen):
        self.pen_width = pen.widthF()


def _pulse_width(eff, t):
    frame = eff.at(_ctx_win(t=t))
    if frame is None:
        return 0.0
    rec = _PenRecorder()
    frame.draws[0][1](_ctx_win(t=t), rec)
    return rec.pen_width


def test_pulse_inactive_before_and_after_event():
    from analysis.player.render.effects.pulse import PulseEffect
    eff = PulseEffect([{'time': 1000, 'duration': 1000, 'width': 40,
                        'in-percent': 0.5, 'easing': 0}])
    assert eff.at(_ctx_win(t=0.5)) is None
    assert eff.at(_ctx_win(t=2.5)) is None


def test_pulse_grows_in_then_shrinks_out():
    from analysis.player.render.effects.pulse import PulseEffect
    eff = PulseEffect([{'time': 1000, 'duration': 1000, 'width': 40,
                        'in-percent': 0.5, 'easing': 0}])
    peak = _pulse_width(eff, 1.5)     # end of the grow phase -> full width
    assert peak == pytest.approx(40.0)
    growing = _pulse_width(eff, 1.25)
    shrinking = _pulse_width(eff, 1.75)
    assert 0.0 < growing < 40.0
    assert 0.0 < shrinking < 40.0


# ── beat pulse --------------------------------------------------------

def _bp_timing(bpm=120.0):
    return [(0.0, bpm)]


def test_beat_pulse_zooms_about_chart_center():
    from analysis.player.render.effects.beat_pulse import BeatPulseEffect
    eff = BeatPulseEffect([{'time': 0, 'strength': 1.2, 'zoom': 0.25,
                            'interval': 1}], _bp_timing(120.0), end_ms=2000)
    ctx = _ctx_win(t=0.05)   # inside the first beat's zoom-in phase
    frame = eff.at(ctx)
    rx, ry, w, h = ctx.chart_rect
    cx, cy = rx + w / 2, ry + h / 2
    assert frame.transform.map(cx, cy) == pytest.approx((cx, cy))
    assert frame.transform.m11() > 1.0


def test_beat_pulse_repeats_per_beat_until_end():
    from analysis.player.render.effects.beat_pulse import BeatPulseEffect
    # 120 BPM -> 500ms/beat; two beats fit before end_ms=1000.
    eff = BeatPulseEffect([{'time': 0, 'strength': 1.5, 'zoom': 0.5,
                            'interval': 1}], _bp_timing(120.0), end_ms=1000)
    assert len(eff._beats) == 2
    # Peak (end of zoom-in) at 0.25s into the first beat = strength.
    frame = eff.at(_ctx_win(t=0.25))
    assert frame.transform.m11() == pytest.approx(1.5)


def test_beat_pulse_skips_unit_strength_and_tiny_interval():
    from analysis.player.render.effects.beat_pulse import BeatPulseEffect
    flat = BeatPulseEffect([{'time': 0, 'strength': 1.0, 'interval': 1}],
                           _bp_timing(), end_ms=2000)
    tiny = BeatPulseEffect([{'time': 0, 'strength': 1.5, 'interval': 0.0}],
                           _bp_timing(), end_ms=2000)
    assert not flat
    assert not tiny


def test_fluxis_adapter_builds_layerfade_pulse_beatpulse():
    from analysis.games.fluxis.adapter import FluxisAdapter
    from analysis.player.render.effects.beat_pulse import BeatPulseEffect
    from analysis.player.render.effects.layer_fade import LayerFadeEffect
    from analysis.player.render.effects.pulse import PulseEffect
    events = {
        'layerfade': [{'time': 0, 'duration': 0, 'alpha': 0.5}],
        'pulse': [{'time': 0, 'duration': 500, 'width': 32,
                   'in-percent': 0.5}],
        'beatpulse': [{'time': 0, 'strength': 1.2, 'interval': 1}],
    }
    replay = {'_fluxis_effect_streams': events,
              '_fluxis_timing_points': [(0.0, 120.0)],
              '_fluxis_end_time': 2000.0}
    built = FluxisAdapter().effects(replay)
    assert any(isinstance(e, LayerFadeEffect) for e in built)
    assert any(isinstance(e, PulseEffect) for e in built)
    assert any(isinstance(e, BeatPulseEffect) for e in built)
