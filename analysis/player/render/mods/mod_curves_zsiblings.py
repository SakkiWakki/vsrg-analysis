"""Z-axis siblings of the warp / waveform mod families as spatial curves.

Every builder here returns an axis Curve (see `curves.Curve`) reproducing
the +z contribution its `<name>z` channel accumulates in `arrow_effects.
_z_push` (GetZPos, ArrowEffects.cpp:1371-1538). The engine reuses the SAME
kernel function for a mod's X and Z siblings -- `drunk_x`, `_tornado_offset`,
`digital_x`, `zigzag_x`, `sawtooth_x`, `square_x`, `bounce_x` each take an
axis-agnostic set of parameters, and the only thing the Z path changes is
which channel set feeds them (the 'z'-suffixed companions, resolved up in
`_warp_family_sum` / `_drunk_pair` / `_tornado_pair`). So a Z sibling's math
is bit-for-bit its already-ported X sibling; these builders delegate to the
exact same curve compositions.

  drunkz    -> z axis : cos of an affine phase in y_offset (drunk on Z)
  tornadoz  -> z axis : cos of an arccos-windowed phase, WINDOW WIDTH 3
  digitalz  -> z axis : quantized sine of an affine phase
  zigzagz   -> z axis : triangle wave of an affine phase
  sawtoothz -> z axis : fractional-part ramp of an affine phase
  squarez   -> z axis : square wave of an affine phase
  bouncez   -> z axis : rectified sine (abs) of an affine phase

The ONE per-axis constant is tornado's arccos-window width: the engine
narrows it 3 -> 2 in wide (>4-col) fields for dimension 0 (X) ONLY; the Z
window (dimension 2) keeps width 3 (ArrowEffects::Init :358). So `tornadoz`
mirrors `mod_curves_warps.arccos_window` at dimension=2, and everything else
reuses the X builders unchanged.

Validated byte-equal (rtol=1e-12) in tests/test_mod_curves_zsiblings.py,
including points at and astride the discontinuous kernels' breaks.
"""
from __future__ import annotations

import numpy as np

from analysis.player.render.mods import curves as cv
from analysis.player.render.mods.arrow_effects import (
    ARROW_SIZE, BUMPY_HEIGHT, DRUNK_ARROW_MAGNITUDE, DRUNK_COLUMN_FREQUENCY,
    DRUNK_OFFSET_FREQUENCY, PI, SCREEN_HEIGHT, TORNADO_OFFSET_FREQUENCY)
from analysis.player.render.mods.mod_curves_warps import arccos_window


TORNADO_Z_DIMENSION = 2


def drunkz_z(percent, speed=0.0, offset=0.0, period=0.0, is_tan=False,
             cosecant=False) -> cv.Curve:
    """drunkz GetZPos (the drunk cos-warp pushed into z instead of x,
    _drunk_pair suffix='z' -> drunk_x with dimension-agnostic params). Same
    phase and amplitude as `mod_curves.drunk_x`; only the target axis differs.
        percent * arrow_size * 0.5 * cos(drunk_angle), where
        drunk_angle = t*(1+speed) + col*(offset*0.2 + 0.2)
                      + y_offset*(period*10 + 10)/SCREEN_HEIGHT.
    is_tan swaps cos for the tan/cosecant kernel (tandrunkz)."""
    col_term = lambda y, c: c.cols.astype(np.float64) * (
        offset * DRUNK_COLUMN_FREQUENCY + DRUNK_COLUMN_FREQUENCY)
    time_term = lambda y, c: c.t * (1.0 + speed)
    y_coeff = (period * DRUNK_OFFSET_FREQUENCY + DRUNK_OFFSET_FREQUENCY) / SCREEN_HEIGHT

    phase = cv.affine_phase(y_coeff, terms=(time_term, col_term))
    kernel = cv.tangent(phase, cosecant) if is_tan else cv.cosine(phase)
    amp = lambda c: percent * c.arrow_size * DRUNK_ARROW_MAGNITUDE
    return cv.scale(amp, kernel)


def tornadoz_z(percent, keycount, arrow_size=ARROW_SIZE, offset=0.0,
               period=0.0, is_tan=False, cosecant=False) -> cv.Curve:
    """tornadoz GetZPos (_tornado_pair suffix='z' -> _tornado_offset with
    dimension=2). The exact tornado remap of `mod_curves_warps.tornado_x`,
    but the arccos window uses dimension 2, so its half-width STAYS 3 in a
    wide (>4-col) field where the X window narrows to 2 (ArrowEffects::Init
    :358 gates the narrowing on dimension == 0). is_tan -> tantornadoz."""
    window = arccos_window(keycount, arrow_size, TORNADO_Z_DIMENSION)
    y_coeff = (period * TORNADO_OFFSET_FREQUENCY + TORNADO_OFFSET_FREQUENCY) / SCREEN_HEIGHT
    bias = lambda y, c: window(c.cols)[3]
    offset_term = offset * y_coeff

    phase = cv.affine_phase(y_coeff, terms=(bias, offset_term))
    kernel = cv.tangent(phase, cosecant) if is_tan else cv.cosine(phase)

    half_span = lambda c: window(c.cols)[2]

    def center_shift(c):
        real, min_x, half, _bias = window(c.cols)
        return min_x + half - real

    remap = cv.add(cv.scale(half_span, kernel), cv.const(center_shift))
    return cv.scale(percent, remap)


def _digital_phase(offset, period, arrow_size) -> cv.Curve:
    """CalculateDigitalAngle: PI*(y_offset + offset)/(AS + period*AS), the
    sine phase shared by digitalz and squarez. Grouped bit-exactly to the
    kernel so the discontinuous square wave agrees on its sign-flip breaks."""
    denom = arrow_size + period * arrow_size
    return cv.grouped_phase(PI, offset, denom)


def digitalz_z(percent, offset, period, steps, arrow_size=ARROW_SIZE,
               is_tan=False, cosecant=False) -> cv.Curve:
    """digitalz GetZPos (digital_x on z, ArrowEffects.cpp:943-952): a sine
    shove of the note's y_offset quantized to `steps + 1` discrete levels,
    scaled to +/- ARROW_SIZE/2. is_tan swaps sin for the tan/cosec kernel
    (tandigitalz, ArrowEffects.cpp:954-966).
        percent * AS * 0.5 * round(levels * sin(angle)) / levels"""
    phase = _digital_phase(offset, period, arrow_size)
    kernel = cv.tangent(phase, cosecant) if is_tan else cv.sine(phase)
    stepped = cv.quantize(kernel, steps + 1.0)
    return cv.scale(percent * arrow_size * 0.5, stepped)


def zigzagz_z(percent, offset, period, arrow_size=ARROW_SIZE) -> cv.Curve:
    """zigzagz GetZPos (zigzag_x on z, ArrowEffects.cpp:908-917): a triangle
    wave of the note's y_offset, scaled to +/- ARROW_SIZE/2. `offset` shifts
    by 100 engine px per percent; `period` stretches the wave.
        angle = PI*(1/(period+1))*((y_offset + 100*offset)/AS)"""
    outer = PI * (1.0 / (period + 1.0))
    phase = lambda y, c: outer * (
        (np.asarray(y, dtype=np.float64) + 100.0 * offset) / arrow_size)
    return cv.scale(percent * (arrow_size / 2.0), cv.triangle(phase))


def sawtoothz_z(percent, period, arrow_size=ARROW_SIZE) -> cv.Curve:
    """sawtoothz GetZPos (sawtooth_x on z, ArrowEffects.cpp:919-928): a
    rising sawtooth, the fractional part of a y_offset ramp, scaled to
    ARROW_SIZE. `period` stretches the ramp; no offset companion in the
    engine formula.
        ramp = (0.5/(period+1) * y_offset)/AS ; return percent*AS*frac(ramp)"""
    ramp = cv.grouped_phase(0.5 / (period + 1.0), 0.0, arrow_size)
    return cv.scale(percent * arrow_size, cv.sawtooth_wave(ramp))


def squarez_z(percent, offset, period, arrow_size=ARROW_SIZE) -> cv.Curve:
    """squarez GetZPos (square_x on z, ArrowEffects.cpp:970-981): a square
    wave of the note's y_offset, scaled to +/- ARROW_SIZE/2. Shares the
    digital phase.
        percent * AS * 0.5 * rage_square(angle)"""
    phase = _digital_phase(offset, period, arrow_size)
    return cv.scale(percent * arrow_size * 0.5, cv.square_wave(phase))


def bouncez_z(percent, offset, period, arrow_size=ARROW_SIZE) -> cv.Curve:
    """bouncez GetZPos (bounce_x on z, ArrowEffects.cpp:983-993): a rectified
    sine (abs(sin)) of the note's y_offset, scaled to ARROW_SIZE/2. Base
    period 60 engine px, stretched by `period`; `offset` shifts by 1 engine
    px per percent.
        percent * AS * 0.5 * abs(sin((y_offset + offset)/(60 + period*60)))"""
    denom = 60.0 + period * 60.0
    phase = cv.grouped_phase(1.0, offset, denom)
    rectified = cv.chain(np.abs, cv.sine(phase))
    return cv.scale(percent * arrow_size * 0.5, rectified)
