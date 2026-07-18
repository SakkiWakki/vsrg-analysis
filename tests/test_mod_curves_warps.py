"""Validation: the sinusoidal-warp curve mods (mod_curves_warps) reproduce
the hardcoded arrow_effects kernels, over a y-offset + parameter sweep.
This is the proof that porting tipsy / bumpy_x / tornado to the curve
algebra changes nothing observable -- the precondition for deleting the
kernels.

Equality is to a few ULP (`RTOL`), not bit-identity: the curve form
reassociates the arithmetic (distributes the amplitude, splits the affine
phase, expands the tornado remap into scale + add), which is intrinsic to
expressing the mod as composable primitives. A ~1e-15 relative difference
is far below any geometric significance, so this is the honest contract
for a port that intentionally re-shapes the math."""
import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import curves as cv
from analysis.player.render.mods import mod_curves_warps as mcw

# A handful of ULP: reassociation-only error, nothing structural.
RTOL = 1e-12


# A representative visible-note batch: y_offsets spanning above/below the
# receptor, columns cycling a 4-key field.
Y = np.linspace(-800.0, 800.0, 33)
COLS = np.arange(Y.shape[0]) % 4
T = 12.34
BEAT = 40.5
KC = 4

# A wide field too, to exercise the tornado window-width narrowing (3->2)
# that only fires for >4 columns on the X dimension.
COLS_WIDE = np.arange(Y.shape[0]) % 8
KC_WIDE = 8


def _ctx(cols=COLS):
    return cv.Ctx(t=T, beat=BEAT, cols=cols, arrow_size=ae.ARROW_SIZE)


@pytest.mark.parametrize('percent,speed,offset,is_tan', [
    (1.0, 0.0, 0.0, False),
    (0.5, 0.3, 0.0, False),
    (1.0, 0.0, 0.7, False),
    (0.8, 0.2, 0.4, False),
    (1.0, 0.0, 0.0, True),
    (0.6, 0.5, 0.9, True),
])
def test_tipsy_curve_equals_kernel(percent, speed, offset, is_tan):
    curve = mcw.tipsy_y(percent, speed=speed, offset=offset, is_tan=is_tan)
    got = curve(Y, _ctx())
    want = ae.tipsy_y(percent, COLS, T, ae.ARROW_SIZE, speed=speed,
                      offset=offset, is_tan=is_tan)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('percent,offset,period,is_tan', [
    (1.0, 0.0, 0.0, False),
    (0.5, 0.6, 0.0, False),
    (1.0, 0.0, 0.8, False),
    (0.7, 0.3, 0.4, False),
    (1.0, 0.0, 0.0, True),
    (0.9, 0.2, 0.5, True),
])
def test_bumpy_x_curve_equals_kernel(percent, offset, period, is_tan):
    curve = mcw.bumpy_x(percent, offset=offset, period=period, is_tan=is_tan)
    got = curve(Y, _ctx())
    want = ae.bumpy_x(percent, Y, offset=offset, period=period, is_tan=is_tan)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('percent,offset,period', [
    (1.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (1.0, 30.0, 0.0),
    (1.0, 0.0, 0.7),
    (0.8, -25.0, 0.4),
])
@pytest.mark.parametrize('cols,keycount', [(COLS, KC), (COLS_WIDE, KC_WIDE)])
def test_tornado_curve_equals_kernel(percent, offset, period, cols, keycount):
    curve = mcw.tornado_x(percent, keycount, offset=offset, period=period)
    got = curve(Y, _ctx(cols))
    want = ae.tornado_x(percent, cols, Y, keycount, offset=offset,
                        period=period)
    np.testing.assert_allclose(got, want, rtol=RTOL)


# tan-tornado's phase = arccos(bias) + y*coeff; near a tan pole a ~1e-16
# reassociation perturbation is amplified past rtol on the huge value the
# tan reaches. These param sets keep every sample off the asymptote so the
# comparison stays in the well-conditioned range (values near a pole are
# geometrically meaningless / clamped in-engine anyway).
@pytest.mark.parametrize('percent,offset,period', [
    (1.0, 0.0, 0.0),
    (0.5, 15.0, 0.25),
    (0.8, -25.0, 0.4),
])
@pytest.mark.parametrize('cols,keycount', [(COLS, KC), (COLS_WIDE, KC_WIDE)])
def test_tan_tornado_curve_equals_kernel(percent, offset, period, cols,
                                         keycount):
    curve = mcw.tornado_x(percent, keycount, offset=offset, period=period,
                          is_tan=True)
    got = curve(Y, _ctx(cols))
    want = ae.tan_tornado_x(percent, cols, Y, keycount, offset=offset,
                            period=period)
    np.testing.assert_allclose(got, want, rtol=RTOL)


# The tipsy/bumpy_x/tornado kernels in arrow_effects hardcode
# cosecant=False; the curve builders expose the Cosecant flag (the tan*
# family's alternate 1/sin kernel). Validate that path against the shared
# _select_tan(cosecant=True), the same swap the algebra performs.
@pytest.mark.parametrize('percent,speed,offset', [(1.0, 0.0, 0.0),
                                                  (0.6, 0.4, 0.5)])
def test_tipsy_cosecant_matches_select_tan(percent, speed, offset):
    curve = mcw.tipsy_y(percent, speed=speed, offset=offset, is_tan=True,
                        cosecant=True)
    got = curve(Y, _ctx())
    angle = ae._tipsy_angle(COLS, T, speed, offset)
    want = percent * (ae._select_tan(angle, cosecant=True)
                      * ae.ARROW_SIZE * ae.TIPSY_ARROW_MAGNITUDE)
    np.testing.assert_allclose(got, want, rtol=RTOL)


# offset shifts the bumpy phase off y=0 so the swept y_offset=0 sample
# doesn't land on sin=0 (1/sin -> inf, matched on both sides but noisy).
@pytest.mark.parametrize('percent,offset,period', [(1.0, 0.2, 0.0),
                                                   (0.7, 0.3, 0.4)])
def test_bumpy_x_cosecant_matches_select_tan(percent, offset, period):
    curve = mcw.bumpy_x(percent, offset=offset, period=period, is_tan=True,
                        cosecant=True)
    got = curve(Y, _ctx())
    angle = ae._bumpy_angle(Y, offset, period)
    want = percent * 40.0 * ae._select_tan(angle, cosecant=True)
    np.testing.assert_allclose(got, want, rtol=RTOL)
