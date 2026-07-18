"""Column-permutation mods (flip / invert) expressed as spatial curves.

Each builder returns an axis Curve (see `curves.Curve`) reproducing the
corresponding hardcoded kernel in `arrow_effects`, but as a composition
of curve primitives instead of transcribed math. Validated byte-equal in
tests/test_mod_curves_columns.py.

  flip   -> x axis : mirror the WHOLE field, col -> (N-1) - col
  invert -> x axis : mirror WITHIN each half of the field

Unlike the drunk/tornado family, these are y-INDEPENDENT: a note's
displacement depends only on which column it is in. The kernel remaps a
column through a fixed permutation and moves the note by the x-offset
DIFFERENCE between its remapped and real column (times percent). That
difference is a per-column constant gathered by the note's column index
and broadcast flat over y_offset -- the column-remap sibling of tornado's
arccos_window. The only new piece is `column_shift`, a local primitive
that precomputes the per-column dx and gathers it by column; the curve is
then that gather broadcast (via `const`) and scaled by percent.
"""
from __future__ import annotations

import numpy as np

from analysis.player.render.mods import curves as cv
from analysis.player.render.mods.arrow_effects import (
    ARROW_SIZE, column_offsets, flip_permutation, invert_permutation)


def column_shift(keycount, permutation, arrow_size=ARROW_SIZE):
    """Precompute the per-column x displacement of a column-permutation mod
    (_mirror_column_shift, ArrowEffects.cpp GetXPos flip/invert): for each
    column, `xoffsets[permutation[col]] - xoffsets[col]`, the distance from
    a column's real field x to its permuted column's field x.

    Returns a `lambda(cols) -> dx` gathering that per-column displacement by
    the note's column index (a per-note array aligned with `cols`). The
    result is y-independent -- the caller broadcasts it flat over y_offset.

    This is the column-remap sibling of `arccos_window`: a per-column
    constant keyed by column index, a candidate curves.py primitive (see
    report)."""
    xoffsets = column_offsets(keycount, arrow_size)
    col_dx = xoffsets[permutation] - xoffsets

    def gather(cols):
        return col_dx[cols.astype(np.int64)]
    return gather


def _mirror_x(percent, keycount, permutation, arrow_size) -> cv.Curve:
    """Shared body of flip_x / invert_x (only the permutation differs). The
    per-column dx (column_shift) broadcast flat over y_offset, scaled by
    percent."""
    shift = column_shift(keycount, permutation, arrow_size)
    return cv.scale(percent, cv.const(lambda c: shift(c.cols)))


def flip_x(percent, keycount, arrow_size=ARROW_SIZE) -> cv.Curve:
    """GetXPos flip (ArrowEffects.cpp:240-253): mirror the whole field
    (col -> (N-1) - col), move by the x-offset difference. percent gates
    the shove; at 100% the field is fully mirrored."""
    return _mirror_x(percent, keycount, flip_permutation(keycount), arrow_size)


def invert_x(percent, keycount, arrow_size=ARROW_SIZE) -> cv.Curve:
    """GetXPos invert (ArrowEffects.cpp:254-294): mirror within each half of
    the field, move by the x-offset difference. percent gates the shove."""
    return _mirror_x(percent, keycount, invert_permutation(keycount),
                     arrow_size)
