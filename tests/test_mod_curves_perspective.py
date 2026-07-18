"""Validation: the perspective-reprojection mods (mod_curves_perspective)
reproduce the hardcoded arrow_effects hallway / confusiony kernels, over a
cols + y_offset + beat sweep on 4-key and 8-key fields.

hallway is depth-keyed (a per-note dx that varies with y_offset), so the
y_offset sweep proves the depth contraction; confusiony is y-independent
but beat-keyed, so the beat sweep proves the tilt angle and the y_offset
sweep proves the flat broadcast. The cols/keycount sweep proves the
per-column xoff gather. Equality is to a few ULP (`RTOL`) -- the curve form
only reassociates the scale/offset, intrinsic to the port."""
import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import curves as cv
from analysis.player.render.mods import mod_curves_perspective as mcp

# A handful of ULP: reassociation-only error, nothing structural.
RTOL = 1e-12

# y_offsets spanning above and below the receptor (hallway clamps y<=0,
# confusiony ignores y entirely).
Y = np.linspace(-800.0, 800.0, 33)


def _cols(keycount):
    """Cycle every column of the field across the note batch."""
    return np.arange(Y.shape[0]) % keycount


def _ctx(keycount, beat=40.5):
    return cv.Ctx(t=12.34, beat=beat, cols=_cols(keycount),
                  arrow_size=ae.ARROW_SIZE)


# ---------------------------------------------------------------------------
# hallway
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('keycount', [4, 8])
@pytest.mark.parametrize('percent', [1.0, 0.5, 0.3, -0.4, 1.25])
def test_hallway_curve_equals_kernel(keycount, percent):
    curve = mcp.hallway_x(percent, keycount)
    got = curve(Y, _ctx(keycount))
    want = ae.hallway_x(percent, _cols(keycount), Y, keycount)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('keycount', [4, 8])
def test_hallway_below_receptor_unscaled(keycount):
    """Notes at or through the receptor (y_offset <= 0) get factor 1 => 0
    contribution: the field only recedes for approaching notes."""
    curve = mcp.hallway_x(1.0, keycount)
    y_below = np.full(keycount, -50.0)
    ctx = cv.Ctx(cols=np.arange(keycount), arrow_size=ae.ARROW_SIZE)
    got = curve(y_below, ctx)
    np.testing.assert_allclose(got, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# confusiony
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('keycount', [4, 8])
@pytest.mark.parametrize('beat', [0.0, 40.5, 123.75, -7.25])
@pytest.mark.parametrize('percent,offset',
                         [(1.0, 0.0), (0.5, 0.0), (0.3, 0.2),
                          (-0.4, -0.1), (0.0, 0.35)])
def test_confusiony_curve_equals_kernel(keycount, beat, percent, offset):
    cols = _cols(keycount)
    curve = mcp.confusiony_dx(percent, offset, keycount)
    got = curve(Y, _ctx(keycount, beat=beat))
    want = ae.confusiony_dx(percent, cols, beat, keycount, offset)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=1e-13)


@pytest.mark.parametrize('keycount', [4, 8])
def test_confusiony_per_column_offset(keycount):
    """The tilt offset may be a per-note array (numbered confusiony0..
    companions): the curve must honor the per-column angle."""
    cols = _cols(keycount)
    offset = (cols.astype(np.float64) + 1.0) * 0.05
    curve = mcp.confusiony_dx(0.7, offset, keycount)
    got = curve(Y, _ctx(keycount, beat=17.0))
    want = ae.confusiony_dx(0.7, cols, 17.0, keycount, offset)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=1e-13)


@pytest.mark.parametrize('keycount', [4, 8])
def test_confusiony_is_y_flat(keycount):
    """Displacement depends only on column + beat, never y_offset: a note in
    a given column gets the same dx at every scroll position."""
    curve = mcp.confusiony_dx(0.6, 0.1, keycount)
    ctx = cv.Ctx(beat=9.0, cols=np.full(Y.shape, 1), arrow_size=ae.ARROW_SIZE)
    got = curve(Y, ctx)
    np.testing.assert_allclose(got, got[0], rtol=RTOL)
