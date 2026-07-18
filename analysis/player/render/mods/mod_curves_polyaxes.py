"""Parabola + attenuate on the Y and Z axes, as spatial curves.

The X siblings (parabola / attenuate) already live in
`mod_curves_beatmove.py` as axis-agnostic builders: the arrow_effects
`parabola` and `attenuate` kernels take no axis argument, so parabolax /
parabolay / parabolaz are literally the SAME function summed into a
different aggregator (dx / dy / z-push). These builders are the Y- and
Z-named entry points of that one shape, matching the beat_x/beat_y/beat_z
companion naming, so a modstring channel maps 1:1 to a builder.

  parabolay / parabolaz -> y / z : percent * (y_offset/AS)^2, a pure
                                   quadratic in y_offset (no column term).
  attenuatey / attenuatez -> y / z : parabola scaled by the column's
                                   signed x-offset, so the push grows with
                                   distance from field center and flips
                                   sign across the center column.

Both are the polynomial corner of the algebra: a power(2, affine_phase)
kernel rather than a periodic one. Validated byte-equal in
tests/test_mod_curves_polyaxes.py.
"""
from __future__ import annotations

from analysis.player.render.mods import curves as cv
from analysis.player.render.mods.arrow_effects import ARROW_SIZE
from analysis.player.render.mods.mod_curves_beatmove import attenuate, parabola


def parabolay_y(percent, arrow_size=ARROW_SIZE) -> cv.Curve:
    """parabolay (ArrowEffects.cpp:638-641): the quadratic push on the Y
    (scroll) axis. Same shape as parabolax/parabolaz, summed into dy."""
    return parabola(percent, arrow_size)


def parabolaz_z(percent, arrow_size=ARROW_SIZE) -> cv.Curve:
    """parabolaz (ArrowEffects.cpp:1445-1448): the quadratic push on Z, a
    +z push in engine px. Same shape as parabolax/parabolay."""
    return parabola(percent, arrow_size)


def attenuatey_y(percent, keycount, arrow_size=ARROW_SIZE) -> cv.Curve:
    """attenuatey (ArrowEffects.cpp:756-759): the column-scaled quadratic on
    the Y axis. Same shape as attenuatex/attenuatez, summed into dy."""
    return attenuate(percent, keycount, arrow_size)


def attenuatez_z(percent, keycount, arrow_size=ARROW_SIZE) -> cv.Curve:
    """attenuatez (ArrowEffects.cpp:1450-1454): the column-scaled quadratic
    on Z, a +z push. Same shape as attenuatex/attenuatey."""
    return attenuate(percent, keycount, arrow_size)
