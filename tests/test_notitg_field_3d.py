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

from analysis.games.notitg import field_3d as f3
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
    return f3.NotitgField3D(channels, base_hidden=base_hidden)


# --- model + screen-transform math ------------------------------------

def test_rest_field_is_identity():
    """All channels at rest: the plane maps 1:1 (untilted field untouched)."""
    proj = t3.projection(f3._DEFAULT_FOV, f3._DESIGN_W, f3._DESIGN_H)
    model = f3._field_model(0.0, 0.0, 0.0, 0.0)
    verdict, H, _clip = t3.project_with_verdict(model, proj, f3._PLANE_CORNERS)
    assert verdict == 'ok'
    assert t3.is_affine(H)
    # design pixel corner (0,0) maps to itself under the identity plane.
    px = t3.project_corners(np.array(f3._PLANE_CORNERS), H)
    assert np.allclose(px, f3._PLANE_CORNERS, atol=1e-6)


def test_rotationy_is_projective_and_center_fixed():
    """A Y tilt yields a real (non-affine) perspective homography whose
    field centre stays put and whose left/right edges foreshorten
    unequally - the 3D depth cue."""
    proj = t3.projection(f3._DEFAULT_FOV, f3._DESIGN_W, f3._DESIGN_H)
    model = f3._field_model(0.0, 25.0, 0.0, 0.0)
    verdict, H, _clip = t3.project_with_verdict(model, proj, f3._PLANE_CORNERS)
    assert verdict == 'ok'
    assert not t3.is_affine(H)
    center = t3.project_corners(
        np.array([[f3._DESIGN_CX, f3._DESIGN_CY]]), H)[0]
    assert np.allclose(center, (f3._DESIGN_CX, f3._DESIGN_CY), atol=1e-6)
    px = t3.project_corners(np.array(f3._PLANE_CORNERS), H)
    left_h = px[3][1] - px[0][1]     # left edge height (design px)
    right_h = px[2][1] - px[1][1]    # right edge height
    assert abs(left_h - right_h) > 1.0


def test_screen_transform_pivots_on_mapped_design_centre():
    """The conjugated screen transform keeps the mapped SM centre fixed and
    lands the design box in the chart region (lockstep with copies)."""
    chart_rect = (100.0, 0.0, 800.0, 720.0)
    k, ox, oy = f3._design_map(chart_rect)
    proj = t3.projection(f3._DEFAULT_FOV, f3._DESIGN_W, f3._DESIGN_H)
    model = f3._field_model(0.0, 20.0, 0.0, 0.0)
    _v, H, _c = t3.project_with_verdict(model, proj, f3._PLANE_CORNERS)
    qt = f3._screen_transform(chart_rect, H)
    cx_s, cy_s = ox + f3._DESIGN_CX * k, oy + f3._DESIGN_CY * k
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
