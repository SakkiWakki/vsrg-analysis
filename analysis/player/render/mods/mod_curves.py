"""Prototype: three arrow-effect mods expressed as spatial curves.

Each builder returns an axis Curve (see `curves.Curve`) reproducing the
corresponding hardcoded kernel in `arrow_effects`, but as a composition
of curve primitives instead of transcribed math. Validated byte-equal in
tests/test_mod_curves.py.

  drunk       -> x axis  : cos of an affine phase in y_offset
  bumpy       -> z axis  : sin of an affine phase in y_offset
  confusionx  -> rot_x   : a y-independent whole-field tilt (degrees)

These span the model: a horizontal displacement, the depth axis, and an
out-of-plane rotation -- the last is the degenerate zero-y-slope,
identity-kernel case, proving the same shape covers 2D warps and 3D tilt.
"""
from __future__ import annotations

import numpy as np

from analysis.player.render.mods import curves as cv
from analysis.player.render.mods.arrow_effects import (
    BUMPY_HEIGHT, DRUNK_ARROW_MAGNITUDE, DRUNK_COLUMN_FREQUENCY,
    DRUNK_OFFSET_FREQUENCY, PI, SCREEN_HEIGHT)


def drunk_x(percent, speed=0.0, offset=0.0, period=0.0, is_tan=False,
            cosecant=False) -> cv.Curve:
    """drunk GetXPos: percent * arrow_size * 0.5 * cos(drunk_angle), where
    drunk_angle = t*(1+speed) + col*(offset*0.2 + 0.2)
                  + y_offset*(period*10 + 10)/SCREEN_HEIGHT.
    is_tan swaps cos for the tan/cosecant kernel (tandrunk)."""
    col_term = lambda y, c: c.cols.astype(np.float64) * (
        offset * DRUNK_COLUMN_FREQUENCY + DRUNK_COLUMN_FREQUENCY)
    time_term = lambda y, c: c.t * (1.0 + speed)
    y_coeff = (period * DRUNK_OFFSET_FREQUENCY + DRUNK_OFFSET_FREQUENCY) / SCREEN_HEIGHT

    phase = cv.affine_phase(y_coeff, terms=(time_term, col_term))
    kernel = cv.tangent(phase, cosecant) if is_tan else cv.cosine(phase)
    amp = lambda c: percent * c.arrow_size * DRUNK_ARROW_MAGNITUDE
    return cv.scale(amp, kernel)


def bumpy_z(percent, offset=0.0, period=0.0, is_tan=False,
            cosecant=False) -> cv.Curve:
    """bumpy GetZPos: percent * 40 * sin(bumpy_angle), where
    bumpy_angle = (y_offset + 100*offset) / (period*16 + 16).
    is_tan swaps sin for the tan/cosecant kernel (tanbumpy)."""
    denom = period * BUMPY_HEIGHT + BUMPY_HEIGHT
    phase = cv.affine_phase(1.0 / denom, terms=(100.0 * offset / denom,))
    kernel = cv.tangent(phase, cosecant) if is_tan else cv.sine(phase)
    return cv.scale(percent * 40.0, kernel)


def confusionx_rot(percent, offset=0.0) -> cv.Curve:
    """confusionx ReceptorGetRotationX, in DEGREES (the real out-of-plane
    tilt, not the zoom reprojection): a whole-field angle broadcast to
    every note, independent of y_offset.
        (beat*percent mod 2pi) * -180/pi + offset*180/pi"""
    def angle(c):
        spin = np.mod(c.beat * percent, 2.0 * PI) * -180.0 / PI
        return spin + offset * 180.0 / PI
    return cv.const(angle)
