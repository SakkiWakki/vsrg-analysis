"""Validation: the curve-composed alpha / visibility + glow mods
(mod_curves_alpha) reproduce the hardcoded arrow_effects `_alpha` and
`stealthglow_amount` kernels, over a vis_y + parameter + time sweep. Same
contract as the other mod_curves tests: byte-equal to a few ULP
(reassociation only, nothing structural), the proof that porting the alpha
family to the curve algebra changes nothing observable.

The sweep exercises every _alpha branch: the hidden / sudden fade windows
(with offsets), mini setting the center line (incl. the >=200% +inf collapse
that skips the windows), the blink flicker over several t (its quantized
sine sign-flips), stealth / stealthglow flat subtracts, the boomerang fade
window, and combos. vis_y spans above/below the receptor so the y<0
early-out (visibility pinned to 1, glow to 0) is covered on every case."""
import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import mod_curves_alpha as mca

# A handful of ULP: reassociation-only error, nothing structural.
RTOL = 1e-12

# A representative visible-note batch: vis_y spanning above/below the
# receptor (so the y<0 exemption is hit) and around the center line (160)
# where the hidden/sudden windows live, columns cycling a 4-key field.
VISY = np.linspace(-400.0, 600.0, 51)
COLS = np.arange(VISY.shape[0]) % 4

# t values chosen to move blink's quantized sine across its levels (sign
# flips + the zeroed step), so the flicker scalar is exercised, not frozen.
TIMES = [0.0, 0.11, 0.157, 0.31, 12.34, -2.5]


def _ctx():
    return mca.cv.Ctx(cols=COLS, arrow_size=ae.ARROW_SIZE)


# Each dict is a full percents bundle; the curve and the kernel must agree
# on vis_y for it. Covers each mod alone and the meaningful combos.
CASES = [
    {},
    {'hidden': 1.0},
    {'hidden': 0.6, 'hiddenoffset': 0.4},
    {'sudden': 1.0},
    {'sudden': 0.7, 'suddenoffset': -0.3},
    {'hidden': 1.0, 'sudden': 1.0},
    {'hidden': 0.8, 'sudden': 0.5, 'hiddenoffset': 0.2, 'suddenoffset': 0.1},
    {'mini': 0.5, 'hidden': 1.0},
    {'mini': 1.0, 'hidden': 1.0, 'sudden': 1.0},
    {'mini': 2.0, 'hidden': 1.0, 'sudden': 1.0},  # center -> +inf, windows skip
    {'mini': 2.5, 'sudden': 1.0},                 # center -> -inf regime
    {'stealth': 0.5},
    {'stealthglow': 0.7},
    {'stealth': 0.3, 'stealthglow': 0.4},
    {'blink': 1.0},
    {'blink': 0.5, 'hidden': 1.0},
    {'boomerang': 1.0},
    {'boomerang': 0.6, 'hidden': 1.0},
    {'boomerang': 1.0, 'sudden': 1.0, 'blink': 1.0, 'stealth': 0.2},
    {'hidden': 1.0, 'sudden': 1.0, 'mini': 0.5, 'blink': 0.5,
     'stealthglow': 0.3, 'boomerang': 0.8, 'hiddenoffset': 0.1,
     'suddenoffset': -0.1},
]


@pytest.mark.parametrize('t_now', TIMES)
@pytest.mark.parametrize('percents', CASES)
def test_alpha_curve_equals_kernel(percents, t_now):
    curve = mca.alpha_curve(percents, t_now)
    got = curve(VISY, _ctx())
    want = ae._alpha(percents, COLS, VISY, t_now)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)


@pytest.mark.parametrize('t_now', TIMES)
@pytest.mark.parametrize('percents', CASES)
def test_percent_visible_curve_equals_kernel(percents, t_now):
    # Mirror _alpha's blink injection: percent_visible reads _blink_adjust.
    vis = dict(percents)
    vis['_blink_adjust'] = ae.blink_adjust(percents.get('blink', 0.0), t_now)
    want = ae.percent_visible(vis, COLS, VISY)
    got = mca.percent_visible_curve(percents, t_now)(VISY, _ctx())
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)


@pytest.mark.parametrize('percent', [0.0, 0.25, 0.5, 1.0, 1.5])
def test_glow_curve_equals_kernel(percent):
    got = mca.glow_curve(percent)(VISY, _ctx())
    want = ae.stealthglow_amount(percent, VISY)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)


@pytest.mark.parametrize('percent', [0.25, 0.5, 1.0])
def test_boomerang_visibility_curve_equals_kernel(percent):
    got = mca.boomerang_visibility_curve(percent)(VISY, _ctx())
    want = ae.boomerang_visibility(percent, VISY)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)


def test_blink_scalar_equals_kernel():
    for t_now in TIMES:
        for percent in (0.0, 0.5, 1.0):
            got = mca._blink_adjust_scalar(percent, t_now)
            want = ae.blink_adjust(percent, t_now)
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)
