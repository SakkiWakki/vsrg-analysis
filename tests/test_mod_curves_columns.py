"""Validation: the column-permutation mods (mod_curves_columns) reproduce
the hardcoded arrow_effects flip/invert kernels, over a cols + y_offset
sweep on 4-key and 8-key fields (the permutation differs by keycount).

The kernel is y-independent (a per-column x shift), so the y_offset sweep
proves the curve broadcasts that shift flat over every note; the keycount
sweep proves the permutation is gathered per-field. Equality is to a few
ULP (`RTOL`) -- the curve form only reassociates the percent scale, which
is intrinsic to the port."""
import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import curves as cv
from analysis.player.render.mods import mod_curves_columns as mcc

# A handful of ULP: reassociation-only error, nothing structural.
RTOL = 1e-12

# y_offsets spanning above/below the receptor; the kernel ignores them,
# so this proves the flat broadcast holds at every note.
Y = np.linspace(-800.0, 800.0, 33)


def _cols(keycount):
    """Cycle every column of the field across the note batch."""
    return np.arange(Y.shape[0]) % keycount


def _ctx(keycount):
    return cv.Ctx(t=12.34, beat=40.5, cols=_cols(keycount),
                  arrow_size=ae.ARROW_SIZE)


@pytest.mark.parametrize('keycount', [4, 8])
@pytest.mark.parametrize('percent', [1.0, 0.5, 0.3, -0.4, 1.25])
def test_flip_curve_equals_kernel(keycount, percent):
    curve = mcc.flip_x(percent, keycount)
    got = curve(Y, _ctx(keycount))
    want = ae.flip_x(percent, _cols(keycount), keycount)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('keycount', [4, 8])
@pytest.mark.parametrize('percent', [1.0, 0.5, 0.3, -0.4, 1.25])
def test_invert_curve_equals_kernel(keycount, percent):
    curve = mcc.invert_x(percent, keycount)
    got = curve(Y, _ctx(keycount))
    want = ae.invert_x(percent, _cols(keycount), keycount)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('keycount', [4, 8])
def test_flip_curve_is_y_flat(keycount):
    """The displacement depends only on column, never on y_offset: a note
    in a given column gets the same shift at every scroll position."""
    curve = mcc.flip_x(1.0, keycount)
    ctx = cv.Ctx(cols=np.full(Y.shape, 1), arrow_size=ae.ARROW_SIZE)
    got = curve(Y, ctx)
    np.testing.assert_allclose(got, got[0], rtol=RTOL)
