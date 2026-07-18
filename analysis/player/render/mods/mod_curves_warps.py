"""Prototype: the sinusoidal-warp arrow-effect mods as spatial curves.

Each builder returns an axis Curve (see `curves.Curve`) reproducing the
corresponding hardcoded kernel in `arrow_effects`, but as a composition
of curve primitives instead of transcribed math. Validated byte-equal in
tests/test_mod_curves_warps.py.

  tipsy    -> y axis  : cos of an affine phase in time+column (y-flat)
  bumpy_x  -> x axis  : sin of an affine phase in y_offset (bumpy on X)
  tornado  -> x axis  : cos of an arccos-windowed phase, remapped to a
                        windowed x displacement

All three are the family shape the algebra was built for -- an affine
phase through a periodic kernel, scaled by an amplitude. tipsy is the
y-flat degenerate (its phase has no y_offset term, only time+column);
bumpy_x is the plain twin of the reference bumpy_z on the X axis; tornado
adds a per-note arccos WINDOW: the phase carries a per-column arccos bias,
and the kernel output is affinely remapped back into the column's
[min_x, max_x] travel and shifted off the column's real x. That remap is
a per-note (scale, add) pair the combinators already provide -- the only
new piece is a local primitive that precomputes the window scalars.
"""
from __future__ import annotations

import numpy as np

from analysis.player.render.mods import curves as cv
from analysis.player.render.mods.arrow_effects import (
    ARROW_SIZE, BUMPY_HEIGHT, SCREEN_HEIGHT, TIPSY_ARROW_MAGNITUDE,
    TIPSY_COLUMN_FREQUENCY, TIPSY_TIMER_FREQUENCY, TORNADO_OFFSET_FREQUENCY,
    column_offsets)


def tipsy_y(percent, speed=0.0, offset=0.0, is_tan=False,
            cosecant=False) -> cv.Curve:
    """tipsy GetYPos: percent * arrow_size * 0.4 * cos(tipsy_angle), where
    tipsy_angle = t*(speed*1.2 + 1.2) + col*(offset*1.8 + 1.8).
    The phase has NO y_offset term (columns bob in place along the scroll
    axis), so this is the y-flat degenerate of the drunk shape. is_tan
    swaps cos for the tan/cosecant kernel (tantipsy)."""
    time_term = lambda y, c: c.t * (speed * TIPSY_TIMER_FREQUENCY + TIPSY_TIMER_FREQUENCY)
    col_term = lambda y, c: c.cols.astype(np.float64) * (
        offset * TIPSY_COLUMN_FREQUENCY + TIPSY_COLUMN_FREQUENCY)

    phase = cv.affine_phase(0.0, terms=(time_term, col_term))
    kernel = cv.tangent(phase, cosecant) if is_tan else cv.cosine(phase)
    amp = lambda c: percent * c.arrow_size * TIPSY_ARROW_MAGNITUDE
    return cv.scale(amp, kernel)


def bumpy_x(percent, offset=0.0, period=0.0, is_tan=False,
            cosecant=False) -> cv.Curve:
    """bumpyx GetXPos: percent * 40 * sin(bumpy_angle) on the X axis (the
    exact bumpy_z shove applied sideways instead of into z), where
    bumpy_angle = (y_offset + 100*offset) / (period*16 + 16).
    is_tan swaps sin for the tan/cosecant kernel (tanbumpyx)."""
    denom = period * BUMPY_HEIGHT + BUMPY_HEIGHT
    phase = cv.affine_phase(1.0 / denom, terms=(100.0 * offset / denom,))
    kernel = cv.tangent(phase, cosecant) if is_tan else cv.sine(phase)
    return cv.scale(percent * 40.0, kernel)


def arccos_window(keycount, arrow_size, dimension=0):
    """Precompute the per-column tornado arccos-window scalars, keyed by a
    note's column index (CalculateTornadoOffsetFromMagnitude,
    ArrowEffects.cpp:217). For each column: `real` its field x, `min_x`
    the low edge of the x-span of the columns within `width` of it,
    `half_span` half that span's width, and `bias` the y-independent
    arccos phase term -- arccos of the column's real x mapped to [-1, 1]
    within its window. Window half-width narrows 3 -> 2 in wide fields
    (>4 cols) for dimension 0 (X) only.

    Returns a `lambda(cols) -> (real, min_x, half_span, bias)` where all
    four are per-note arrays gathered by column. `half_span` is
    (max_x - min_x)/2, the amplitude of the kernel-output remap; `min_x`
    plus `half_span` recovers the window center for the additive term.

    This is the arccos-windowed sibling of the plain affine phase -- a
    candidate curves.py primitive (see report)."""
    xoffsets = column_offsets(keycount, arrow_size)
    width = 2 if (dimension == 0 and keycount > 4) else 3
    idx = np.arange(keycount)
    start = np.clip(idx - width, 0, keycount - 1)
    end = np.clip(idx + width, 0, keycount - 1)

    col_min = np.array([xoffsets[start[i]:end[i] + 1].min() for i in idx])
    col_max = np.array([xoffsets[start[i]:end[i] + 1].max() for i in idx])
    col_span = np.where(col_max == col_min, 1.0, col_max - col_min)
    between = np.clip((xoffsets - col_min) * 2.0 / col_span - 1.0, -1.0, 1.0)
    col_bias = np.arccos(between)
    col_half = (col_max - col_min) / 2.0

    def gather(cols):
        c = cols.astype(np.int64)
        return xoffsets[c], col_min[c], col_half[c], col_bias[c]
    return gather


def tornado_x(percent, keycount, arrow_size=ARROW_SIZE, offset=0.0,
              period=0.0, is_tan=False, cosecant=False,
              dimension=0) -> cv.Curve:
    """tornado GetXPos (ArrowEffects.cpp:820-834). The column's real x maps
    into [-1, 1] within its arccos window; (y_offset + offset) advances the
    phase; cos (or tan/cosec for tantornado) maps back to a windowed x:
        rads     = arccos(between) + (y_offset + offset)*((period*6)+6)/H
        adjusted = _scale(cos(rads), -1, 1, min_x, max_x)
        dx       = (adjusted - real) * percent
    The arccos bias and the [min_x, max_x] remap are per-column constants
    (arccos_window); the (y_offset + offset) slope is the affine phase; the
    remap-and-shift is a per-note (scale, add) on the kernel output.
    `dimension` (0 = X, 2 = Z) selects the window width."""
    window = arccos_window(keycount, arrow_size, dimension)
    y_coeff = (period * TORNADO_OFFSET_FREQUENCY + TORNADO_OFFSET_FREQUENCY) / SCREEN_HEIGHT
    bias = lambda y, c: window(c.cols)[3]
    offset_term = offset * y_coeff

    phase = cv.affine_phase(y_coeff, terms=(bias, offset_term))
    kernel = cv.tangent(phase, cosecant) if is_tan else cv.cosine(phase)

    # _scale(k, -1, 1, min_x, max_x) = k*half_span + (min_x + half_span);
    # dx = adjusted - real, then scaled by percent.
    half_span = lambda c: window(c.cols)[2]

    def center_shift(c):
        real, min_x, half, _bias = window(c.cols)
        return min_x + half - real

    remap = cv.add(cv.scale(half_span, kernel), cv.const(center_shift))
    return cv.scale(percent, remap)
