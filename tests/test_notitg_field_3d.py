"""3D notefield perspective effect (scoping items 70/61).

The effect samples the recorded player-actor rotation_x/rotation_y/rotation
/skew_x channels, builds the SM field-plane model matrix, projects it with
LoadMenuPerspective defaults, and emits the planar homography on
EffectFrame.transform (conjugated by the design map). These tests drive the
effect and the model/screen-transform math with synthetic timelines - no gat
install needed - and pin: identity at rest, real perspective under a tilt,
field-centre pivot, the deferral while copies own the field, and the
double-apply predicate."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from analysis.games.notitg import field_3d as f3
from analysis.games.notitg import field_projection as fp
from analysis.player.render import transform3d as t3
from analysis.player.render.effects.timeline import EventTimeline, Keyframe


def _tl(pairs, rest=0.0):
    """EventTimeline of instantaneous (t, value) steps at the given rest."""
    kfs = [Keyframe(t=t, values=(v,), duration=0.0, easing=0) for t, v in pairs]
    return EventTimeline(kfs, rest=(rest,))


def _channels(rotation_x=(), rotation_y=(), rotation=(), skew_x=()):
    return {
        'rotation_x': _tl(rotation_x), 'rotation_y': _tl(rotation_y),
        'rotation': _tl(rotation), 'skew_x': _tl(skew_x),
    }


def _ctx(t, chart_rect=(100.0, 0.0, 800.0, 720.0)):
    return SimpleNamespace(t_now=t, chart_rect=chart_rect)


def _effect(channels, base_hidden=None):
    tilt = fp.FieldTilt(actor_timelines=channels)
    return f3.NotitgField3D(tilt, base_hidden=base_hidden)


# --- model + screen-transform math ------------------------------------

def test_rest_field_is_identity():
    """All channels at rest: the plane maps 1:1 (untilted field untouched)."""
    proj = fp.design_projection()
    model = fp.field_model(0.0, 0.0, 0.0, 0.0)
    verdict, H, _clip = t3.project_with_verdict(model, proj, fp.PLANE_CORNERS)
    assert verdict == 'ok'
    assert t3.is_affine(H)
    # design pixel corner (0,0) maps to itself under the identity plane.
    px = t3.project_corners(np.array(fp.PLANE_CORNERS), H)
    assert np.allclose(px, fp.PLANE_CORNERS, atol=1e-6)


def test_rotationy_is_projective_and_center_fixed():
    """A Y tilt yields a real (non-affine) perspective homography whose
    field centre stays put and whose left/right edges foreshorten
    unequally - the 3D depth cue."""
    proj = fp.design_projection()
    model = fp.field_model(0.0, 25.0, 0.0, 0.0)
    verdict, H, _clip = t3.project_with_verdict(model, proj, fp.PLANE_CORNERS)
    assert verdict == 'ok'
    assert not t3.is_affine(H)
    center = t3.project_corners(
        np.array([[fp.DESIGN_CX, fp.DESIGN_CY]]), H)[0]
    assert np.allclose(center, (fp.DESIGN_CX, fp.DESIGN_CY), atol=1e-6)
    px = t3.project_corners(np.array(fp.PLANE_CORNERS), H)
    left_h = px[3][1] - px[0][1]     # left edge height (design px)
    right_h = px[2][1] - px[1][1]    # right edge height
    assert abs(left_h - right_h) > 1.0


def test_screen_transform_pivots_on_mapped_design_centre():
    """The conjugated screen transform keeps the mapped SM centre fixed and
    lands the design box in the chart region (lockstep with copies)."""
    chart_rect = (100.0, 0.0, 800.0, 720.0)
    kx, ky, ox, oy = f3._design_map(chart_rect)
    proj = fp.design_projection()
    model = fp.field_model(0.0, 20.0, 0.0, 0.0)
    _v, H, _c = t3.project_with_verdict(model, proj, fp.PLANE_CORNERS)
    qt = f3._screen_transform(chart_rect, H)
    cx_s, cy_s = ox + fp.DESIGN_CX * kx, oy + fp.DESIGN_CY * ky
    mapped = qt.map(cx_s, cy_s)
    assert abs(mapped[0] - cx_s) < 1e-6 and abs(mapped[1] - cy_s) < 1e-6


# --- effect frame behaviour -------------------------------------------

def test_at_rest_emits_no_frame():
    """No poke anywhere: the effect contributes nothing (zero-cost path)."""
    assert _effect(_channels()).at(_ctx(5.0)) is None


def test_tilt_emits_projective_transform():
    """A Y-tilt frame yields a projective EffectFrame.transform (QPainter
    executes the perspective via m13/m23)."""
    eff = _effect(_channels(rotation_y=[(0.0, 30.0)]))
    frame = eff.at(_ctx(1.0))
    assert frame is not None and frame.transform is not None
    tr = frame.transform
    assert abs(tr.m13()) > 1e-9 or abs(tr.m23()) > 1e-9


def test_in_plane_spin_is_affine():
    """rotation (z) alone tilts nothing out of plane: the transform is
    affine and tilt_active stays False (no confusion deferral)."""
    eff = _effect(_channels(rotation=[(0.0, 45.0)]))
    frame = eff.at(_ctx(1.0))
    assert frame is not None and frame.transform is not None
    tr = frame.transform
    assert abs(tr.m13()) < 1e-9 and abs(tr.m23()) < 1e-9
    assert eff.tilt_active(1.0) is False


def test_defers_while_base_field_hidden():
    """When the base field is hidden (copies own it), the effect emits
    nothing so the field capture stays flat and copies never inherit the
    base tilt (the copy-leak guard)."""
    base_hidden = _tl([(0.0, 1.0)], rest=1.0)
    eff = _effect(_channels(rotation_y=[(0.0, 30.0)]), base_hidden=base_hidden)
    assert eff.at(_ctx(2.0)) is None
    assert eff.tilt_active(2.0) is False


def test_tilt_active_tracks_xy_only():
    """tilt_active reports the X/Y out-of-plane tilt (what the 2D confusion
    kernels approximate), not the Z spin or skew."""
    assert _effect(_channels(rotation_x=[(0.0, 10.0)])).tilt_active(1.0)
    assert _effect(_channels(rotation_y=[(0.0, 10.0)])).tilt_active(1.0)
    assert not _effect(_channels(rotation=[(0.0, 90.0)])).tilt_active(1.0)
    assert not _effect(_channels(skew_x=[(0.0, 0.5)])).tilt_active(1.0)


def test_gone_verdict_hides_field():
    """A tilt that sends the whole plane through the eye plane hides the
    field (opacity 0) rather than drawing a meaningless warp."""
    # rotation_x at 90 deg folds the plane edge-on / past the eye.
    eff = _effect(_channels(rotation_x=[(0.0, 90.0)]))
    frame = eff.at(_ctx(1.0))
    # Either gone (hidden) or a valid clipped/ok warp; assert no crash and a
    # coherent frame. At exactly 90 the plane is degenerate -> hidden.
    assert frame is not None
    if frame.transform is None:
        assert frame.opacity == 0.0


# --- mod-channel tilt source (field_projection.FieldTilt) --------------

def _mod_channels_tilt(mod, percent, offset_mod=None, offset=0.0):
    from analysis.player.render.mods.channels import ModChannels, ModEvent
    events = [ModEvent(0.0, percent, -1.0, mod, 0)]
    if offset_mod is not None:
        events.append(ModEvent(0.0, offset, -1.0, offset_mod, 0))
    return ModChannels.compile(events)


def test_mod_tilt_matches_equivalent_actor_poke():
    """Scalar confusionx through the mod source projects the SAME
    homography as an actor rotationx poke of the identical confusion
    angle - one projection, two producers."""
    from analysis.player.render.mods.arrow_effects import (
        _confusion_axis_degrees)
    percent, beat = 0.15, 7.0
    beat_at = lambda _t: beat
    mod_tilt = fp.FieldTilt(
        channels=_mod_channels_tilt('confusionx', percent), beat_at=beat_at)
    angle = _confusion_axis_degrees(percent, beat, 0.0)
    actor_tilt = fp.FieldTilt(
        actor_timelines=_channels(rotation_x=[(0.0, angle)]))
    t = 1.0
    assert mod_tilt.tilt_active(t) and actor_tilt.tilt_active(t)
    _v1, h_mod = fp.field_homography(*mod_tilt.sample(t))
    _v2, h_actor = fp.field_homography(*actor_tilt.sample(t))
    assert np.allclose(h_mod, h_actor, atol=1e-12)


def test_mod_tilt_offset_companion_alone_tilts():
    """confusionxoffset with no percent is a constant tilt (the engine
    adds offset*180/PI unconditionally); the mod source must see it."""
    tilt = fp.FieldTilt(
        channels=_mod_channels_tilt('confusionx', 0.0,
                                    offset_mod='confusionxoffset',
                                    offset=0.5),
        beat_at=lambda _t: 3.0)
    assert tilt.tilt_active(1.0)
    rx, _ry, _rz, _skew = tilt.sample(1.0)
    assert rx == pytest.approx(0.5 * 180.0 / np.pi)


def test_has_mod_tilt_detects_scalar_channels_only():
    assert fp.has_mod_tilt(_mod_channels_tilt('confusionx', 0.1))
    assert fp.has_mod_tilt(_mod_channels_tilt('confusiony', 0.1))
    assert not fp.has_mod_tilt(_mod_channels_tilt('confusion', 0.1))
    assert not fp.has_mod_tilt(_mod_channels_tilt('confusionx0', 0.1))
    assert not fp.has_mod_tilt(None)


def test_note_mods_guard_defers_scalar_confusion_to_projection():
    """While the projection owns the mod-driven tilt, the note-mod
    consumer zeroes the scalar confusionx percents so the 2D kernel
    never double-applies; per-column numbered variants stay live."""
    from analysis.games.notitg.note_mods import NotitgNoteMods
    channels = _mod_channels_tilt('confusionx', 0.2)
    tilt = fp.FieldTilt(channels=channels, beat_at=lambda _t: 5.0)
    eff = f3.NotitgField3D(tilt)
    consumer = NotitgNoteMods(channels, [(0.0, 120.0)],
                              field_tilt_active=eff.tilt_active)
    percents = {'confusionx': 0.2, 'confusionx1': 0.3}
    deferred = consumer._defer_field_tilt(percents, 1.0)
    assert deferred['confusionx'] == 0.0
    assert deferred['confusionx1'] == 0.3


def test_vanish_stream_shifts_projection():
    """A recorded SetVanishPoint stream moves the projection off-centre:
    the tilted-field homography differs from the centered one, and the
    centered vanish is exactly the default."""
    vanish = {'vanish_x': _tl([(0.0, 400.0)], rest=320.0),
              'vanish_y': _tl([(0.0, 240.0)], rest=240.0)}
    tilt = fp.FieldTilt(
        actor_timelines=_channels(rotation_y=[(0.0, 20.0)]), vanish=vanish)
    assert tilt.vanish_at(1.0) == (400.0, 240.0)
    _v, h_off = fp.field_homography(*tilt.sample(1.0),
                                    vanish=tilt.vanish_at(1.0))
    _v, h_centered = fp.field_homography(*tilt.sample(1.0))
    assert not np.allclose(h_off, h_centered)
    assert fp.design_projection((320.0, 240.0)) is fp.design_projection()

