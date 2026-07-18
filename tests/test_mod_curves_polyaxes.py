"""Validation: the curve-composed Y/Z parabola + attenuate mods
(mod_curves_polyaxes) reproduce the hardcoded arrow_effects kernels, over
a y-offset + percent sweep. Same contract as tests/test_mod_curves.py:
byte-equal to a few ULP (reassociation only, nothing structural).

parabola / attenuate are continuous polynomials (power(2, affine_phase)),
so the distributive affine_phase is sub-ULP-safe -- no grouped_phase or
break-astride cases are needed here (there are no discontinuities). The
Y and Z builders share arrow_effects' single axis-agnostic kernel with
the X siblings, so the same kernel is the oracle for every axis."""
import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import curves as cv
from analysis.player.render.mods import mod_curves_polyaxes as mc

# A handful of ULP: reassociation-only error, nothing structural.
RTOL = 1e-12


# A representative visible-note batch: y_offsets spanning above/below the
# receptor, columns cycling a 4-key field.
Y = np.linspace(-800.0, 800.0, 33)
COLS = np.arange(Y.shape[0]) % 4
KC = 4


def _ctx():
    return cv.Ctx(cols=COLS, arrow_size=ae.ARROW_SIZE)


@pytest.mark.parametrize('percent', [1.0, 0.5, -0.3, 2.0])
@pytest.mark.parametrize('builder', [mc.parabolay_y, mc.parabolaz_z])
def test_parabola_axis_curve_equals_kernel(builder, percent):
    curve = builder(percent)
    got = curve(Y, _ctx())
    want = ae.parabola(percent, Y)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('percent', [1.0, 0.5, -0.3, 2.0])
@pytest.mark.parametrize('builder', [mc.attenuatey_y, mc.attenuatez_z])
def test_attenuate_axis_curve_equals_kernel(builder, percent):
    curve = builder(percent, KC)
    got = curve(Y, _ctx())
    want = ae.attenuate(percent, COLS, Y, KC)
    np.testing.assert_allclose(got, want, rtol=RTOL)
