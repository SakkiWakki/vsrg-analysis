"""Prototype: the DIGITAL / WAVEFORM WARP mods as spatial curves.

These are the DISCONTINUOUS members of the family (digital, zigzag,
sawtooth, square, bounce): a periodic sideways (X) shove of the note's
y_offset, but through a NON-SMOOTH kernel - a stepped sine, a triangle, a
sawtooth ramp, a square wave, a rectified sine. They are the proof that
the curve algebra (affine phase -> periodic kernel -> scale) is not
special to sin/cos/tan: swapping in a discontinuous kernel reproduces the
warp exactly, with no change to the phase or scaling machinery.

  digital   -> x axis : quantized sine of an affine phase in y_offset
  zigzag    -> x axis : triangle wave of an affine phase
  sawtooth  -> x axis : fractional-part ramp of an affine phase
  square    -> x axis : square wave of an affine phase
  bounce    -> x axis : rectified sine (abs) of an affine phase

Only the KERNEL differs from the smooth ports in `mod_curves`; the phase
is `curves.affine_phase` and the amplitude is `curves.scale` throughout.
The discontinuous kernels (triangle / sawtooth_wave / square_wave /
quantize) are defined LOCALLY here as candidate curve primitives; see the
migration report for the ones proposed for `curves.py`. Validated
byte-equal in tests/test_mod_curves_waveform.py, including points at and
astride the kernel discontinuities.
"""
from __future__ import annotations

import numpy as np

from analysis.player.render.mods import curves as cv
from analysis.player.render.mods.arrow_effects import ARROW_SIZE, PI


# ---------------------------------------------------------------------------
# Waveform-warp mod builders (each returns an x-axis Curve)
# ---------------------------------------------------------------------------

def _digital_phase(offset, period, arrow_size) -> cv.Curve:
    """CalculateDigitalAngle: PI*(y_offset + offset)/(AS + period*AS), the
    sine phase shared by digital and square. Grouped bit-exactly to the
    kernel so the discontinuous square wave agrees on its sign-flip breaks."""
    denom = arrow_size + period * arrow_size
    return cv.grouped_phase(PI, offset, denom)


def digital_x(percent, offset, period, steps, arrow_size=ARROW_SIZE,
              is_tan=False, cosecant=False) -> cv.Curve:
    """digital (ITGmania ArrowEffects.cpp:943-952): a sine shove of the
    note's y_offset quantized to `steps + 1` discrete levels, scaled to
    +/- ARROW_SIZE/2. is_tan swaps sin for the tan/cosec kernel
    (tandigital, ArrowEffects.cpp:954-966).
        percent * AS * 0.5 * round(levels * sin(angle)) / levels"""
    phase = _digital_phase(offset, period, arrow_size)
    kernel = cv.tangent(phase, cosecant) if is_tan else cv.sine(phase)
    stepped = cv.quantize(kernel, steps + 1.0)
    return cv.scale(percent * arrow_size * 0.5, stepped)


def zigzag_x(percent, offset, period, arrow_size=ARROW_SIZE) -> cv.Curve:
    """zigzag (ITGmania ArrowEffects.cpp:908-917): a triangle wave of the
    note's y_offset, scaled to +/- ARROW_SIZE/2. `offset` shifts by 100
    engine px per percent; `period` stretches the wave.
        angle = PI*(1/(period+1))*((y_offset + 100*offset)/AS)"""
    outer = PI * (1.0 / (period + 1.0))
    phase = lambda y, c: outer * (
        (np.asarray(y, dtype=np.float64) + 100.0 * offset) / arrow_size)
    return cv.scale(percent * (arrow_size / 2.0), cv.triangle(phase))


def sawtooth_x(percent, period, arrow_size=ARROW_SIZE) -> cv.Curve:
    """sawtooth (ITGmania ArrowEffects.cpp:919-928): a rising sawtooth, the
    fractional part of a y_offset ramp, scaled to ARROW_SIZE. `period`
    stretches the ramp; no offset companion in the engine formula.
        ramp = (0.5/(period+1) * y_offset)/AS ; return percent*AS*frac(ramp)"""
    ramp = cv.grouped_phase(0.5 / (period + 1.0), 0.0, arrow_size)
    return cv.scale(percent * arrow_size, cv.sawtooth_wave(ramp))


def square_x(percent, offset, period, arrow_size=ARROW_SIZE) -> cv.Curve:
    """square (ITGmania ArrowEffects.cpp:970-981): a square wave of the
    note's y_offset, scaled to +/- ARROW_SIZE/2. Shares the digital phase.
        percent * AS * 0.5 * rage_square(angle)"""
    phase = _digital_phase(offset, period, arrow_size)
    return cv.scale(percent * arrow_size * 0.5, cv.square_wave(phase))


def bounce_x(percent, offset, period, arrow_size=ARROW_SIZE) -> cv.Curve:
    """bounce (ITGmania ArrowEffects.cpp:983-993): a rectified sine
    (abs(sin)) of the note's y_offset, scaled to ARROW_SIZE/2. Base period
    60 engine px, stretched by `period`; `offset` shifts by 1 engine px per
    percent.
        percent * AS * 0.5 * abs(sin((y_offset + offset)/(60 + period*60)))"""
    denom = 60.0 + period * 60.0
    phase = cv.grouped_phase(1.0, offset, denom)
    rectified = cv.chain(np.abs, cv.sine(phase))
    return cv.scale(percent * arrow_size * 0.5, rectified)
