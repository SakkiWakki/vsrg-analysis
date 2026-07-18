"""Beat + linear-move mods expressed as spatial curves.

Each builder returns an axis Curve (see `curves.Curve`) reproducing the
corresponding hardcoded kernel in `arrow_effects`, but as a composition
of curve primitives instead of transcribed math. Validated byte-equal in
tests/test_mod_curves_beatmove.py.

  beat / beaty / beatz  -> x / y / z : a per-frame beat pulse (scalar
                                       amplitude) through sin of an affine
                                       phase in y_offset
  movex / movey / movez -> x / y / z : a flat per-note displacement, one
                                       arrow width at 100%
  xmode                 -> x axis     : percent * y_offset, the identity
                                       affine phase (no periodic kernel)
  parabola              -> any axis   : percent * (y_offset/AS)^2, a pure
                                       quadratic in y_offset
  attenuate             -> any axis   : parabola scaled by the column's
                                       signed x-offset

The move/xmode family is the flat / affine-only corner of the algebra (a
const, or an identity phase). beat contributes the per-frame SCALAR
amplitude pattern: its shape lives entirely in a frame scalar (beat_factor
is y-independent), leaving the y-dependence to the shared sin kernel.
parabola/attenuate are the polynomial corner: a power kernel (`np.square`
through `chain`) rather than a periodic one.
"""
from __future__ import annotations

import numpy as np

from analysis.player.render.mods import curves as cv
from analysis.player.render.mods.arrow_effects import (
    ARROW_SIZE, BEAT_OFFSET_HEIGHT, BEAT_PI_HEIGHT, PI, beat_factor,
    column_offsets)


def _beat_shift_phase(period) -> cv.Curve:
    """The affine phase inside the beat sin term (_beat_shift): the shared
    yoff/((period*15)+15) + PI/2 across beat / beaty / beatz."""
    height = (period * BEAT_OFFSET_HEIGHT) + BEAT_OFFSET_HEIGHT
    return cv.affine_phase(1.0 / height, terms=(PI / BEAT_PI_HEIGHT,))


def _beat_axis(percent, beat_now, offset, period, mult) -> cv.Curve:
    """Shared body of beat_x / beat_y / beat_z (only the target axis differs
    in arrow_effects). percent * beat_factor(...) * sin(phase), where
    beat_factor is a per-frame scalar keyed to the song beat."""
    amp = percent * beat_factor(beat_now, offset, mult)
    return cv.scale(amp, cv.sine(_beat_shift_phase(period)))


def beat_x(percent, beat_now, offset=0.0, period=0.0, mult=0.0) -> cv.Curve:
    """GetXPos beat (ArrowEffects.cpp:897-906): a periodic sideways shove
    keyed to the song beat. Companions beatoffset / beatperiod / beatmult."""
    return _beat_axis(percent, beat_now, offset, period, mult)


def beat_y(percent, beat_now, offset=0.0, period=0.0, mult=0.0) -> cv.Curve:
    """GetYPos beaty (ArrowEffects.cpp:762-771): the beat pulse on the Y
    (scroll) axis. Companions beatyoffset / beatyperiod / beatymult."""
    return _beat_axis(percent, beat_now, offset, period, mult)


def beat_z(percent, beat_now, offset=0.0, period=0.0, mult=0.0) -> cv.Curve:
    """GetZPos beatz (ArrowEffects.cpp:1481-1489): the beat pulse on Z, a
    +z push in engine px. Companions beatzoffset / beatzperiod / beatzmult."""
    return _beat_axis(percent, beat_now, offset, period, mult)


def movex_x(percent, arrow_size=ARROW_SIZE) -> cv.Curve:
    """NotITG movex: 100% = one arrow width along x, flat in y_offset."""
    return cv.const(percent * arrow_size)


def movey_y(percent, arrow_size=ARROW_SIZE) -> cv.Curve:
    """NotITG movey: 100% = one arrow width along y, flat in y_offset."""
    return cv.const(percent * arrow_size)


def movez_z(percent, arrow_size=ARROW_SIZE) -> cv.Curve:
    """NotITG movez: 100% = one arrow width along +z, flat in y_offset."""
    return cv.const(percent * arrow_size)


def xmode_x(percent) -> cv.Curve:
    """GetXPos xmode (ArrowEffects.cpp:990-1019), single-side field: simply
    percent * y_offset, the identity affine phase with no periodic kernel
    -- the further a note is from the receptor, the more it is shoved
    sideways, turning the vertical scroll into a diagonal."""
    return cv.scale(percent, cv.affine_phase(1.0))


def parabola(percent, arrow_size=ARROW_SIZE) -> cv.Curve:
    """parabolax/y/z (ArrowEffects.cpp:931-934, 638-641, 1445-1448):
    percent * (y_offset/AS)^2. A quadratic push whose axis is chosen by the
    caller. No column term (unlike attenuate)."""
    return cv.scale(percent, cv.power(2, cv.affine_phase(1.0 / arrow_size)))


def attenuate(percent, keycount, arrow_size=ARROW_SIZE) -> cv.Curve:
    """attenuatex/y/z (ArrowEffects.cpp:936-940, 756-759, 1450-1454):
    percent * (y_offset/AS)^2 * (xoff/AS). Like parabola but scaled by the
    column's signed x-offset, so the push grows with distance from field
    center and flips sign across the center column."""
    offsets = column_offsets(keycount, arrow_size) / arrow_size
    col_amp = lambda c: percent * offsets[c.cols.astype(np.int64)]
    return cv.scale(col_amp, cv.power(2, cv.affine_phase(1.0 / arrow_size)))
