"""Validation: the curve-composed beat + linear-move mods
(mod_curves_beatmove) reproduce the hardcoded arrow_effects kernels, over
a y-offset + parameter + beat sweep. Same contract as
tests/test_mod_curves.py: byte-equal to a few ULP (reassociation only,
nothing structural), the proof that porting these mods to the curve
algebra changes nothing observable.

The beat sweep matters here: beat_factor is a piecewise, sign-flipping,
per-frame scalar (accel window / decay window / zeroed dead zone / even-
beat negation), so several beat values are exercised to cover every branch
of the pulse, not just one phase."""
import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import mod_curves_beatmove as mc
from analysis.player.render.mods import curves as cv

# A handful of ULP: reassociation-only error, nothing structural.
RTOL = 1e-12


# A representative visible-note batch: y_offsets spanning above/below the
# receptor, columns cycling a 4-key field.
Y = np.linspace(-800.0, 800.0, 33)
COLS = np.arange(Y.shape[0]) % 4
T = 12.34
KC = 4

# Beat values chosen to hit each branch of beat_factor: the accel window
# (amount = scaled^2), the decay window (1-(1-a)^2), the >= total_time
# dead zone (returns 0), an even vs odd integer beat (sign flip), and a
# negative beat (early return 0).
BEATS = [40.5, 40.05, 40.35, 40.9, 41.05, 0.0, -3.2]


def _ctx(beat=0.0):
    return cv.Ctx(t=T, beat=beat, cols=COLS, arrow_size=ae.ARROW_SIZE)


@pytest.mark.parametrize('beat', BEATS)
@pytest.mark.parametrize('percent,offset,period,mult', [
    (1.0, 0.0, 0.0, 0.0),
    (0.5, 0.3, 0.0, 0.0),
    (1.0, 0.0, 0.7, 0.0),
    (1.0, 0.0, 0.0, 0.9),
    (0.8, 0.2, 0.4, 0.6),
])
@pytest.mark.parametrize('builder,kernel', [
    (mc.beat_x, ae.beat_x),
    (mc.beat_y, ae.beat_y),
    (mc.beat_z, ae.beat_z),
])
def test_beat_curve_equals_kernel(builder, kernel, percent, offset, period,
                                  mult, beat):
    curve = builder(percent, beat, offset=offset, period=period, mult=mult)
    got = curve(Y, _ctx(beat))
    want = kernel(percent, Y, beat, offset=offset, period=period, mult=mult)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('percent', [1.0, 0.5, -0.3, 2.0])
@pytest.mark.parametrize('builder,kernel', [
    (mc.movex_x, ae.movex_x),
    (mc.movey_y, ae.movey_y),
    (mc.movez_z, ae.movez_z),
])
def test_move_curve_equals_kernel(builder, kernel, percent):
    curve = builder(percent)
    got = curve(Y, _ctx())
    want = np.broadcast_to(kernel(percent), Y.shape)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('percent', [1.0, 0.5, -0.3, 2.0])
def test_xmode_curve_equals_kernel(percent):
    curve = mc.xmode_x(percent)
    got = curve(Y, _ctx())
    want = ae.xmode_x(percent, Y)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('percent', [1.0, 0.5, -0.3, 2.0])
def test_parabola_curve_equals_kernel(percent):
    curve = mc.parabola(percent)
    got = curve(Y, _ctx())
    want = ae.parabola(percent, Y)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('percent', [1.0, 0.5, -0.3, 2.0])
def test_attenuate_curve_equals_kernel(percent):
    curve = mc.attenuate(percent, KC)
    got = curve(Y, _ctx())
    want = ae.attenuate(percent, COLS, Y, KC)
    np.testing.assert_allclose(got, want, rtol=RTOL)
