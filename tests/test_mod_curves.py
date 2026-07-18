"""Validation: the curve-composed mods (mod_curves) reproduce the
hardcoded arrow_effects kernels, over a y-offset + parameter sweep. This
is the proof that porting a mod to the curve algebra changes nothing
observable -- the precondition for migrating the rest and deleting the
kernels.

Equality is to a few ULP (`RTOL`), not bit-identity: the curve form
reassociates the arithmetic (distributes the amplitude, splits the
affine phase), which is intrinsic to expressing the mod as composable
primitives. A ~1e-15 relative difference is far below any geometric
significance (sub-femtopixel), so this is the honest contract for a port
that intentionally re-shapes the math."""
import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import curves as cv
from analysis.player.render.mods import mod_curves as mc

# A handful of ULP: reassociation-only error, nothing structural.
RTOL = 1e-12


# A representative visible-note batch: y_offsets spanning above/below the
# receptor, columns cycling a 4-key field.
Y = np.linspace(-800.0, 800.0, 33)
COLS = np.arange(Y.shape[0]) % 4
T = 12.34
BEAT = 40.5
KC = 4


def _ctx():
    return cv.Ctx(t=T, beat=BEAT, cols=COLS, arrow_size=ae.ARROW_SIZE)


@pytest.mark.parametrize('percent,speed,offset,period,is_tan', [
    (1.0, 0.0, 0.0, 0.0, False),
    (0.5, 0.3, 0.0, 0.0, False),
    (1.0, 0.0, 0.7, 0.0, False),
    (1.0, 0.0, 0.0, 0.9, False),
    (0.8, 0.2, 0.4, 0.6, False),
    (1.0, 0.0, 0.0, 0.0, True),
])
def test_drunk_curve_equals_kernel(percent, speed, offset, period, is_tan):
    curve = mc.drunk_x(percent, speed=speed, offset=offset, period=period,
                       is_tan=is_tan)
    got = curve(Y, _ctx())
    want = ae.drunk_x(percent, COLS, Y, T, KC, speed=speed, offset=offset,
                      period=period, is_tan=is_tan)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('percent,offset,period,is_tan', [
    (1.0, 0.0, 0.0, False),
    (0.5, 0.6, 0.0, False),
    (1.0, 0.0, 0.8, False),
    (0.7, 0.3, 0.4, False),
    (1.0, 0.0, 0.0, True),
])
def test_bumpy_curve_equals_kernel(percent, offset, period, is_tan):
    curve = mc.bumpy_z(percent, offset=offset, period=period, is_tan=is_tan)
    got = curve(Y, _ctx())
    want = ae.bumpy_z(percent, Y, offset=offset, period=period, is_tan=is_tan)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('percent,offset', [
    (1.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.25),
    (0.3, -0.4),
])
def test_confusionx_curve_equals_kernel_degrees(percent, offset):
    """The curve emits the real rotation in degrees (matching
    _confusion_axis_degrees, the engine's ReceptorGetRotationX)."""
    curve = mc.confusionx_rot(percent, offset=offset)
    got = curve(Y, _ctx())
    want = np.broadcast_to(
        ae._confusion_axis_degrees(percent, BEAT, offset), Y.shape)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('percent,offset', [
    (1.0, 0.0),
    (0.5, 0.25),
    (0.3, -0.4),
])
def test_confusionx_zoom_reprojection_matches(percent, offset):
    """Chaining abs(cos(radians)) onto the degree curve reproduces the 2D
    zoom proxy (confusionx_zoom), showing the old zoom fake is a derived
    read-off of the real rotation curve, not a separate formula."""
    deg = mc.confusionx_rot(percent, offset=offset)
    zoom = cv.chain(np.abs, cv.chain(np.cos, cv.scale(ae.PI / 180.0, deg)))
    got = zoom(Y, _ctx())
    want = np.broadcast_to(
        ae.confusionx_zoom(percent, BEAT, offset), Y.shape)
    np.testing.assert_allclose(got, want, rtol=RTOL)
