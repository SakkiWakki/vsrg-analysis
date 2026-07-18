"""Prototype: the note-scale ZOOM family expressed as spatial curves.

The zoom mods reproject a note's SCALE (not position): mini/tiny shrink
every note by a flat scalar, pulse pulses the scale with y_offset, shrink
tapers approaching arrows by distance, and confusionx foreshortens the
whole field to a cos(angle) uniform zoom. Unlike the dx/dy/rot families
this axis is MULTIPLICATIVE: `ArrowEffects._zoom` folds the kernels as a
running product (with one mult-then-add step for shrink), not a sum. Each
builder here returns an axis Curve (see `curves.Curve`) reproducing the
matching kernel; `zoom_curve` folds them in the engine's exact order,
validated byte-equal in tests/test_mod_curves_zoom.py.

  mini        -> flat scalar   1 - mini*0.5
  tiny        -> flat scalar   pow(0.5, tiny)
  pulse       -> per-note      sine of y_offset about a rest scale
  shrink      -> per-note      (mult, add) taper, gated on y_offset >= 0
  confusionx  -> flat scalar   abs(cos(radians(confusionx_rot)))  [z_push gate]
  waveform_z  -> per-note      perspective reprojection of the +z push

Fold (arrow_effects._zoom):
    base = zoom_from_mini(mini) * tiny_zoom(tiny)          # scalar
    zoom = base * pulse                                    # multiply
    zoom = zoom * shrink_mult + shrink_add                 # affine
    if z_push is None:                                     # 2D fallback only
        zoom = zoom * waveform_z_zoom(z_push_accumulated)
    if confusionx active:
        zoom = zoom * confusionx_zoom(...)

The z_push gate mirrors the projected-note path: when the caller passes a
real per-note z push (project_3d), the camera applies the perspective
divide, so this curve does NOT fold `waveform_z_zoom` (that would double-
count depth). `z_push=None` is the flat 2D fallback that reprojects the
summed +z push to zoom via the same perspective scale.

confusionx is the derived read-off of the ported rot_x tilt
(`mod_curves.confusionx_rot`): the identical `_confusion_axis_degrees`
angle, turned to a zoom by abs(cos(radians(angle))) exactly as
`arrow_effects.confusionx_zoom` does.

New combinators used (defined LOCALLY, proposed for curves.py -- see the
port report): `mul` (product of curves, the multiplicative sibling of
`curves.add`) and `affine` (base*mult + add, the mult-then-add shrink
needs). Neither is expressible with the additive algebra's `add`/`scale`.
"""
from __future__ import annotations

import numpy as np

from analysis.player.render.mods import curves as cv
from analysis.player.render.mods import mod_curves as mc
from analysis.player.render.mods.arrow_effects import (
    ARROW_SIZE, PI, perspective_z_scale, _confusion_offset)


# ---------------------------------------------------------------------------
# Local combinators (multiplicative axis) -- proposed for curves.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Kernel curves (each byte-equal to its arrow_effects kernel)
# ---------------------------------------------------------------------------

def mini_tiny_base(mini_percent, tiny_percent) -> cv.Curve:
    """The flat base scale: zoom_from_mini(mini) * tiny_zoom(tiny)
    (ArrowEffects.cpp:389 / 1582). A y-independent scalar broadcast to
    every note: (1 - mini*0.5) * pow(0.5, tiny)."""
    base = (1.0 - mini_percent * 0.5) * np.power(0.5, tiny_percent)
    return cv.const(float(base))


def pulse_zoom(inner, outer, offset=0.0, period=0.0,
               arrow_size=ARROW_SIZE) -> cv.Curve:
    """pulse GetZoomVariable (ArrowEffects.cpp:1596-1610 + GetPulseInner
    :1630-1641): a per-note zoom pulsing with y_offset. Off (flat 1.0) when
    inner and outer are both 0.
        height = 0.4*(AS + period*AS)
        sine   = sin((yoff + 100*offset) / height)
        inner_rest = inner*0.5 + 1   (nudged to 0.01 if exactly 0)
        zoom   = sine*(outer*0.5) + inner_rest"""
    if inner == 0.0 and outer == 0.0:
        return cv.const(1.0)
    height = 0.4 * (arrow_size + period * arrow_size)
    inner_rest = inner * 0.5 + 1.0
    if inner_rest == 0.0:
        inner_rest = 0.01
    phase = cv.affine_phase(1.0 / height, terms=(100.0 * offset / height,))
    return cv.add(cv.scale(outer * 0.5, cv.sine(phase)), cv.const(inner_rest))


def shrink_curves(shrink_mult, shrink_linear, arrow_size=ARROW_SIZE):
    """shrink family (ArrowEffects.cpp:1611-1626), applied only to arrows
    still approaching (y_offset >= 0). Returns a (mult_curve, add_curve)
    pair for the `affine` fold, matching `arrow_effects.shrink_zoom`:
      shrinkmult   -> mult = 1/(1 + yoff*(mult/100))  where approaching
      shrinklinear -> add  = yoff*(0.5*linear/AS)      where approaching
    Notes with y_offset < 0 keep mult 1 / add 0."""
    def mult(y, c):
        if shrink_mult == 0.0:
            return np.ones(np.shape(y), dtype=np.float64)
        yv = np.asarray(y, dtype=np.float64)
        return np.where(yv >= 0.0, 1.0 / (1.0 + yv * (shrink_mult / 100.0)), 1.0)

    def add(y, c):
        if shrink_linear == 0.0:
            return np.zeros(np.shape(y), dtype=np.float64)
        yv = np.asarray(y, dtype=np.float64)
        return np.where(yv >= 0.0, yv * (0.5 * shrink_linear / arrow_size), 0.0)

    return mult, add


def confusionx_zoom(percent, offset=0.0) -> cv.Curve:
    """confusionx reprojected to a uniform zoom: abs(cos(radians(angle)))
    where `angle` is the ported rot_x tilt (mod_curves.confusionx_rot, in
    degrees), read from ctx.beat. The derived read-off of the tilt, exactly
    as `arrow_effects.confusionx_zoom` computes it. `offset` is a scalar or
    a per-note confusion offset array (aligned with the batch)."""
    rot = mc.confusionx_rot(percent, offset=offset)
    return cv.chain(np.abs, cv.chain(np.cos, cv.scale(PI / 180.0, rot)))


def waveform_z_zoom(z_push) -> cv.Curve:
    """The 2D-fallback reprojection of an accumulated engine +z push to a
    zoom multiplier (center-plane perspective scale, ArrowEffects
    bumpy_zoom style). `z_push` is the summed per-note +z contribution
    (engine px) the caller already accumulated on the z axis."""
    scale = perspective_z_scale(np.asarray(z_push, dtype=np.float64))
    return cv.const(np.asarray(scale, dtype=np.float64))


# ---------------------------------------------------------------------------
# Assembled fold (byte-equal to arrow_effects._zoom)
# ---------------------------------------------------------------------------

def zoom_curve(percents, cols, keycount, arrow_size=ARROW_SIZE,
               beat_now=0.0, z_push=None, waveform_push=None) -> cv.Curve:
    """The per-note zoom multiplier, folding the family in the engine's
    exact order (see module header). Returns a Curve `f(y_offset, ctx)`.

    Params ride here (not in Ctx) because they are the parsed modstring
    percents, fixed per frame. confusionx reads the beat from ctx.beat (the
    curve pattern), so `beat_now` is the contract's beat handle: the
    integrator sets ctx.beat == beat_now, exactly the one beat
    `arrow_effects._zoom` takes as its beat_now argument.

    The +z push is a separately-ported family (z axis), so the two engine
    branches split into two explicit arguments here instead of one dual-use
    one (`arrow_effects._zoom` recomputes the fallback push inline; the
    curve cannot):
      - `z_push` NOT None -> project_3d path: the camera owns the perspective
        divide, so the waveform reprojection is SKIPPED (whatever its value).
      - `z_push` is None -> 2D fallback: reproject `waveform_push` (the
        upstream-accumulated +z push) via `waveform_z_zoom` when nonzero
        (engine gate `if np.any(...)`). None == no push == no reprojection.
    This is exactly `_zoom`'s `z_push is None` test, with the fallback push
    named so the integrator hands in the z-family sum it already computes."""
    base = mini_tiny_base(_get(percents, 'mini'), _get(percents, 'tiny'))
    pulse = pulse_zoom(_get(percents, 'pulseinner'), _get(percents, 'pulseouter'),
                       _get(percents, 'pulseoffset'), _get(percents, 'pulseperiod'),
                       arrow_size)
    shrink_mult, shrink_add = shrink_curves(_get(percents, 'shrinkmult'),
                                            _get(percents, 'shrinklinear'), arrow_size)

    folded = cv.affine(cv.mul(base, pulse), shrink_mult, shrink_add)

    reproject = z_push is None and waveform_push is not None and np.any(waveform_push)
    if reproject:
        folded = cv.mul(folded, waveform_z_zoom(waveform_push))

    # confusionx activates on its scalar, its offset companion, OR any
    # numbered per-column variant (confusionx0..); the per-note offset is
    # the confusionxoffset plus those per-column values, exactly as
    # arrow_effects._zoom gates + builds it via _confusion_offset.
    if (_get(percents, 'confusionx') or _get(percents, 'confusionxoffset')
            or _confusionx_active(percents, keycount)):
        offset = _confusion_offset(percents, 'confusionx', cols, keycount)
        folded = cv.mul(folded, confusionx_zoom(_get(percents, 'confusionx'), offset))

    return folded


def _get(p, name):
    return p.get(name, 0.0)


def _confusionx_active(percents, keycount):
    return any(f'confusionx{c}' in percents for c in range(keycount))
