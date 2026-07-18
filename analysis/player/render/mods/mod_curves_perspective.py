"""Perspective-reprojection mods expressed as spatial curves.

Each builder returns an X-axis Curve (see `curves.Curve`) reproducing the
corresponding hardcoded kernel in `arrow_effects`, but as a composition of
curve primitives instead of transcribed math. Validated byte-equal in
tests/test_mod_curves_perspective.py.

  hallway     -> x axis : per-note recede toward the field-center vanishing
                          line with DEPTH (a pinhole 1/(H+depth) contraction)
  confusiony  -> x axis : horizontal foreshortening from the Y-axis
                          confusion tilt, cos(angle)-1 of the column x-offset

Both are 2D reprojections of an out-of-plane tilt and both land on the SAME
additive per-note dx field the position mods feed (xoff * <contraction>).
They differ in what drives the contraction:

  hallway is DEPTH-keyed: the contraction factor H/(H+max(y_offset,0)) is a
  rational function of y_offset with a max(.,0) clamp -- neither an affine
  phase nor a periodic kernel, so it needs a dedicated depth-factor curve
  (`hallway_factor`, a candidate primitive; see report).

  confusiony is BEAT-keyed and y-INDEPENDENT: the tilt angle is a per-frame
  scalar (a beat spin plus a per-column constant offset), so its cos rides
  through the existing `cosine` kernel over a flat (zero-slope) phase, and
  the whole term broadcasts flat over y_offset -- the confusion-tilt sibling
  of the flat-broadcast column mods.

The shared piece is a per-column xoff gather (`column_xoff`): the note's
signed field x-offset, gathered by column index, the same per-column
constant `column_shift` / `attenuate` already build locally.
"""
from __future__ import annotations

import numpy as np

from analysis.player.render.mods import curves as cv
from analysis.player.render.mods.arrow_effects import (
    ARROW_SIZE, PI, SCREEN_HEIGHT, column_offsets)


def column_xoff(keycount, arrow_size=ARROW_SIZE):
    """The note's signed field x-offset gathered by column index: a
    `lambda(cols) -> xoff` reading `column_offsets(keycount)[cols]` (a
    per-note array aligned with `cols`). The per-column-constant gather
    shared by hallway / confusiony (and the same shape `column_shift` /
    `attenuate` build); a candidate curves.py primitive (see report)."""
    xoffsets = column_offsets(keycount, arrow_size)

    def gather(cols):
        return xoffsets[cols.astype(np.int64)]
    return gather


def hallway_factor(field_height=SCREEN_HEIGHT) -> cv.Curve:
    """The pinhole depth contraction of `hallway_x` as a curve in y_offset:
    `H/(H+max(y_offset,0)) - 1`, in (-1, 0]. A rational function of the
    note's depth with a max(.,0) clamp (notes at/through the receptor, where
    y_offset <= 0, get factor 1 => contribution 0; approaching notes recede).
    Neither affine nor periodic, so it is its own kernel over y_offset -- a
    candidate curves.py primitive (see report)."""
    def curve(y, c):
        depth = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
        return field_height / (field_height + depth) - 1.0
    return curve


def hallway_x(percent, keycount, arrow_size=ARROW_SIZE,
              field_height=SCREEN_HEIGHT) -> cv.Curve:
    """hallway (arrow_effects.hallway_x): notes recede toward the field-center
    vanishing line with depth. `dx = xoff * (H/(H+max(y,0)) - 1) * percent` --
    the per-column x-offset scaled by the depth contraction (`hallway_factor`)
    and by percent. y-DEPENDENT (the only perspective mod that reads depth)."""
    xoff = column_xoff(keycount, arrow_size)
    scaled = cv.scale(lambda c: percent * xoff(c.cols), hallway_factor(field_height))
    return scaled


def _confusiony_angle_rad(percent, offset):
    """The Y-axis confusion tilt angle in radians as a per-frame scalar
    (`_confusion_axis_degrees * PI/180`, folded): the beat spin
    `mod(beat*percent, 2*PI) * -1` plus the per-column constant `offset`
    (already in radians-equivalent: `_confusion_offset` returns degrees that
    `confusiony_dx` scales by PI/180, which cancels the `*180/PI` inside
    `_confusion_axis_degrees`). Returns a `lambda(ctx) -> angle` so it stays
    a whole-field per-frame value broadcast to every note; `offset` may be a
    scalar or a per-column array."""
    return lambda c: np.mod(c.beat * percent, 2.0 * PI) * -1.0 + offset


def confusiony_dx(percent, offset, keycount, arrow_size=ARROW_SIZE) -> cv.Curve:
    """confusiony (arrow_effects.confusiony_dx): horizontal foreshortening
    from a whole-field tilt about the vertical axis. `dx = xoff *
    (cos(angle) - 1)`, the per-column x-offset contracted toward center as
    the field tilts. y-INDEPENDENT: the tilt angle is a per-frame scalar
    (beat spin + per-column offset), so cos rides the `cosine` kernel over a
    flat zero-slope phase and the term broadcasts flat over y_offset.
    `offset` is the per-note `_confusion_offset` (scalar or per-column)."""
    xoff = column_xoff(keycount, arrow_size)
    angle = cv.const(_confusiony_angle_rad(percent, offset))
    cos_minus_one = cv.add(cv.chain(np.cos, angle), cv.const(-1.0))
    return cv.scale(lambda c: xoff(c.cols), cos_minus_one)
