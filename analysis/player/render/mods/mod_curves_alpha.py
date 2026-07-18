"""Alpha / visibility + glow mods expressed as spatial curves.

This family is the visibility sibling of the position curves: instead of a
per-note displacement, each builder returns a Curve over the note's
VISIBILITY y-position (`vis_y = y_offset + tipsy dy`, the coordinate
`arrow_effects._alpha` already consumes) producing the per-note ALPHA
MULTIPLIER (and, for stealthglow, the per-note GLOW). Validated byte-equal
in tests/test_mod_curves_alpha.py.

  hidden / sudden / mini  -> alpha : the center-line visibility windows.
                                     mini sets the center line (via
                                     `_center_line`); hidden fades a note as
                                     it nears the receptor, sudden as it
                                     recedes. Each is a linear fade WINDOW
                                     in vis_y, clamped to [-1, 0] and summed
                                     into the visibility adjust.
  blink                   -> alpha : a per-frame flicker scalar (quantized
                                     sine of t), y-independent, added to the
                                     same adjust.
  stealth / stealthglow   -> alpha : a flat per-frame subtraction from the
                                     adjust (the note fill fades out).
  boomerang               -> alpha : a fade WINDOW past the fold, multiplied
                                     onto the assembled alpha.
  stealthglow             -> glow  : the GLOW channel: flat `percent`,
                                     gated off for past-receptor notes.

Why this family needs its own primitives (window_remap / clamp01 / cross the
past-note gate) rather than the position algebra's affine-phase-through-
periodic-kernel shape: visibility is built from CLIPPED LINEAR RAMPS over
vis_y (fade windows) summed into an adjust, then clamped -- a window/clamp
shape, not a trig one. The per-frame scalars (hidden/sudden/mini/blink/
stealth) are all y-independent and close over the curve at build time, so
evaluation stays one vectorized pass over vis_y. The visibility -> alpha map
itself is the identity (`alpha_from_visible`), pinned by the port-parity
contract to the smooth visible fraction.

The engine's hard GetAlpha 0/1 cut and the GetGlow ramp (`display_alpha`)
are a SEPARATE draw-boundary mapping, not part of this multiplier; they are
left to the compositor exactly as `_alpha` leaves them.
"""
from __future__ import annotations

import numpy as np

from analysis.player.render.mods import curves as cv
from analysis.player.render.mods.arrow_effects import (
    ARROW_SIZE, BOOMERANG_PEAK_PERCENTAGE, CENTER_LINE_Y, FADE_DIST_Y,
    SCREEN_HEIGHT, _center_line, _hidden_sudden, _quantize, _scale)


# ---------------------------------------------------------------------------
# Local primitives (proposed for curves.py -- the window/clamp corner)
# ---------------------------------------------------------------------------

def window_remap(l1, h1, l2, h2, lo, hi) -> cv.Curve:
    """A linear fade WINDOW over the input coordinate: `_scale` remap from
    the input range [l1, h1] onto [l2, h2], clamped to [lo, hi]. The shape
    of every hidden/sudden fade edge and the boomerang fade -- a clipped
    ramp, the visibility family's analogue of the position algebra's
    periodic kernel. Endpoints are scalars closed over at build."""
    return lambda y, c: np.clip(_scale(np.asarray(y, dtype=np.float64),
                                        l1, h1, l2, h2), lo, hi)


def past_gate(rest, curve: cv.Curve) -> cv.Curve:
    """Every appearance term exempts past-receptor notes (vis_y < 0), which
    take the constant `rest` (1.0 for a visibility fraction, 0.0 for glow).
    `np.where(vis_y < 0, rest, curve)` -- the visibility family's boundary
    guard, matching ArrowGetPercentVisible's y_pos<0 early-out."""
    return lambda y, c: np.where(np.asarray(y, dtype=np.float64) < 0.0,
                                 float(rest), curve(y, c))


# ---------------------------------------------------------------------------
# Curve builders per _alpha branch. Each mirrors arrow_effects exactly.
# ---------------------------------------------------------------------------

def _blink_adjust_scalar(percent, t_now) -> float:
    """blink_adjust (ArrowEffects.cpp:470-475), reproduced as the frame
    scalar it is: a quantized sine flicker of wallclock t, scaled by percent,
    <= 0. y-independent, so it enters the visibility adjust as a constant."""
    if percent == 0.0:
        return 0.0
    f = np.sin(t_now * 10.0)
    f = _quantize(f, 0.3333)
    return percent * _scale(f, 0, 1, -1.0, 0.0)


def _visibility_adjust(percents, t_now) -> cv.Curve:
    """The additive visibility ADJUST curve (ArrowGetPercentVisible body,
    ArrowEffects.cpp:441-484, minus the final `1 + adjust` clamp and the
    y<0 early-out): hidden + sudden fade windows + a flat stealth subtract +
    the blink flicker scalar. Reads vis_y; returns adjust (<= 0 typically).

    Branch-for-branch with `percent_visible`: the fade windows only build
    when the center line is finite (an extreme mini drives it to +inf,
    collapsing the field to a point and skipping the windows to avoid
    inf-inf NaN); stealth folds stealth + stealthglow (both hide the fill)."""
    hidden = percents.get('hidden', 0.0)
    sudden = percents.get('sudden', 0.0)
    stealth = percents.get('stealth', 0.0) + percents.get('stealthglow', 0.0)
    hidden_off = percents.get('hiddenoffset', 0.0)
    sudden_off = percents.get('suddenoffset', 0.0)
    mini = percents.get('mini', 0.0)
    blink = _blink_adjust_scalar(percents.get('blink', 0.0), t_now)

    hs = _hidden_sudden(hidden, sudden)
    center = _center_line(mini)

    terms = []
    if np.isfinite(center):
        hidden_end = center + FADE_DIST_Y * _scale(hs, 0, 1, -1.0, -1.25) + center * hidden_off
        hidden_start = center + FADE_DIST_Y * _scale(hs, 0, 1, 0.0, -0.25) + center * hidden_off
        sudden_end = center + FADE_DIST_Y * _scale(hs, 0, 1, -0.0, 0.25) + center * sudden_off
        sudden_start = center + FADE_DIST_Y * _scale(hs, 0, 1, 1.0, 1.25) + center * sudden_off
        if hidden != 0.0:
            terms.append(cv.scale(hidden, window_remap(hidden_start, hidden_end,
                                                       0.0, -1.0, -1.0, 0.0)))
        if sudden != 0.0:
            terms.append(cv.scale(sudden, window_remap(sudden_start, sudden_end,
                                                       -1.0, 0.0, -1.0, 0.0)))
    if stealth != 0.0:
        terms.append(cv.const(-stealth))
    if blink != 0.0:
        terms.append(cv.const(blink))

    return cv.add(*terms) if terms else cv.const(0.0)


def percent_visible_curve(percents, t_now) -> cv.Curve:
    """ArrowGetPercentVisible (ArrowEffects.cpp:441-484) as a curve over
    vis_y: `clip(1 + adjust, 0, 1)`, with past-receptor notes (vis_y < 0)
    pinned to full visibility. Returns the per-note visible fraction."""
    visible = cv.clamp01(cv.add(cv.const(1.0), _visibility_adjust(percents, t_now)))
    return past_gate(1.0, visible)


def alpha_curve(percents, t_now) -> cv.Curve:
    """`_alpha` (ArrowEffects.cpp) as a curve over vis_y: the visible
    fraction (identity `alpha_from_visible`), times the boomerang fade
    window when boomerang is on. Returns the per-note alpha multiplier.

    Byte-equal drop-in for `arrow_effects._alpha(percents, cols, vis_y,
    t_now)` -- cols is unused by the alpha math (the fade windows key off
    vis_y alone), so the curve takes only (percents, t_now)."""
    visible = percent_visible_curve(percents, t_now)
    boomerang = percents.get('boomerang', 0.0)
    if not boomerang:
        return visible
    return cv.mul(visible, boomerang_visibility_curve(boomerang))


def boomerang_visibility_curve(percent) -> cv.Curve:
    """boomerang_visibility (ArrowEffects.cpp) as a curve over vis_y: a note
    beyond the fold `p = H * 0.75` fades to 0 over one ARROW_SIZE (the
    engine's bIsPastPeakOut + cull, the closest expressible behavior),
    gated by percent. Returns `1 - percent * (1 - fade)` in [0, 1]."""
    p = SCREEN_HEIGHT * BOOMERANG_PEAK_PERCENTAGE
    fade = window_remap(p, p + ARROW_SIZE, 1.0, 0.0, 0.0, 1.0)
    return cv.add(cv.const(1.0 - percent), cv.scale(percent, fade))


def glow_curve(percent) -> cv.Curve:
    """stealthglow_amount (ArrowEffects.cpp) as a curve over vis_y: a flat
    `clip(percent, 0, 1)` glow, gated off (0.0) for past-receptor notes
    (vis_y < 0). Returns the per-note glow multiplier -- the GLOW field of
    NoteOffsets. rest (percent 0) = 0 = no glow."""
    return past_gate(0.0, cv.const(float(np.clip(percent, 0.0, 1.0))))
