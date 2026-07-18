"""Validation: the curve-composed ZOOM family (mod_curves_zoom) reproduces
the multiplicative fold of arrow_effects._zoom, over a y_offset + parameter
sweep, byte-equal (rtol=1e-12).

The zoom axis is a running PRODUCT (with one mult-then-add for shrink), so
this harness folds `zoom_curve` and compares against `ae._zoom` directly --
covering each kernel alone (mini, tiny, pulse, shrink, confusionx), the
z_push gate (None fallback that reprojects the +z push vs a supplied 3D
array that skips it), and stacked combos.

The z axis is a separately-ported family, so `zoom_curve` splits _zoom's
one dual-use `z_push` arg into two: `z_push` (the 3D camera push -> SKIP
reprojection) and `waveform_push` (the fallback push -> reproject). This
harness maps each engine call to the matching curve call:
  - engine `z_push=None`  -> curve `waveform_push=ae._z_push(...)` (reproject)
  - engine `z_push=<arr>`  -> curve `z_push=<arr>` (skip)"""
import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import curves as cv
from analysis.player.render.mods import mod_curves_zoom as mz

RTOL = 1e-12

# y_offsets straddling the receptor (shrink gates on y_offset >= 0), a
# 4-key column cycle, and a beat that puts confusionx off a wrap boundary.
Y = np.linspace(-800.0, 800.0, 33)
COLS = np.arange(Y.shape[0]) % 4
KEYCOUNT = 4
BEAT = 40.5
T = 12.34
AS = ae.ARROW_SIZE


def _ctx(beat=BEAT):
    return cv.Ctx(t=T, beat=beat, cols=COLS, arrow_size=AS)


def _want(percents, z_push=None):
    return ae._zoom(percents, COLS, Y, T, BEAT, KEYCOUNT, AS, Y.shape[0],
                    z_push=z_push)


def _got(percents, beat=BEAT, z_push=None, waveform_push=None):
    curve = mz.zoom_curve(percents, COLS, KEYCOUNT, AS, beat_now=beat,
                          z_push=z_push, waveform_push=waveform_push)
    return curve(Y, _ctx(beat))


def _assert(percents, z_push=None):
    """Compare the curve to ae._zoom for one percents dict. The engine's
    2D fallback (z_push=None) computes its own +z push; the curve is fed
    that same ae._z_push array as `waveform_push` to reproject like for
    like. A supplied 3D `z_push` array maps straight through (skip)."""
    if z_push is None:
        push = ae._z_push(percents, COLS, Y, T, BEAT, KEYCOUNT, AS)
        got = _got(percents, waveform_push=push)
    else:
        got = _got(percents, z_push=z_push)
    np.testing.assert_allclose(got, _want(percents, z_push=z_push), rtol=RTOL)


# --- individual kernels ----------------------------------------------------

@pytest.mark.parametrize('mini', [0.0, 0.5, 1.0, 1.5, 2.0, -1.0])
def test_mini(mini):
    _assert({'mini': mini})


@pytest.mark.parametrize('tiny', [0.0, 0.5, 1.0, 2.0, -1.0, 0.37])
def test_tiny(tiny):
    _assert({'tiny': tiny})


@pytest.mark.parametrize('inner,outer,offset,period', [
    (1.0, 1.0, 0.0, 0.0),
    (0.5, 2.0, 0.25, 0.0),
    (2.0, 0.5, -0.4, 1.5),
    (-2.0, 1.0, 0.0, 0.0),   # inner_rest hits exactly 0 -> 0.01 nudge
    (0.0, 0.0, 0.0, 0.0),    # off -> flat 1.0
])
def test_pulse(inner, outer, offset, period):
    _assert({'pulseinner': inner, 'pulseouter': outer,
             'pulseoffset': offset, 'pulseperiod': period})


@pytest.mark.parametrize('mult,linear', [
    (100.0, 0.0),
    (0.0, 100.0),
    (50.0, 200.0),
    (-30.0, -50.0),
    (0.0, 0.0),
])
def test_shrink(mult, linear):
    _assert({'shrinkmult': mult, 'shrinklinear': linear})


@pytest.mark.parametrize('percent,offset', [
    (1.0, 0.0),
    (0.5, 0.25),
    (2.0, -0.4),
    (0.3, 0.7),
])
def test_confusionx(percent, offset):
    _assert({'confusionx': percent, 'confusionxoffset': offset})


def test_confusionx_offset_only():
    _assert({'confusionxoffset': 0.5})


# --- z_push gate -----------------------------------------------------------

def test_z_push_none_no_z_family():
    # No +z channels: ae computes an all-zero push, folds nothing.
    _assert({'mini': 0.5, 'pulseinner': 1.0})


def test_z_push_none_reprojects():
    # A live +z family (bumpy) accumulates a nonzero push; the fallback
    # branch folds waveform_z_zoom. _assert feeds the same ae._z_push array.
    _assert({'bumpy': 1.0, 'mini': 0.5})


def test_z_push_array_skips_reprojection():
    # A supplied 3D push array: both paths SKIP waveform_z_zoom (camera owns
    # the divide), so zoom is mini/pulse only regardless of the push values.
    percents = {'mini': 0.5, 'pulseinner': 1.0, 'pulseouter': 1.0}
    z = np.linspace(0.0, 500.0, Y.shape[0])
    _assert(percents, z_push=z)


# --- stacked combos --------------------------------------------------------

@pytest.mark.parametrize('percents', [
    {'mini': 0.5, 'tiny': 0.5},
    {'mini': 1.0, 'tiny': -1.0, 'pulseinner': 1.0, 'pulseouter': 2.0},
    {'tiny': 0.5, 'shrinkmult': 50.0, 'shrinklinear': 100.0},
    {'mini': 0.5, 'pulseinner': 1.0, 'pulseouter': 1.0,
     'shrinkmult': 100.0, 'shrinklinear': 50.0, 'confusionx': 1.0},
    {'mini': 0.5, 'tiny': 0.5, 'pulseinner': 0.5, 'pulseouter': 2.0,
     'pulseoffset': 0.3, 'pulseperiod': 1.0, 'shrinkmult': 30.0,
     'shrinklinear': 80.0, 'confusionx': 0.7, 'confusionxoffset': 0.2},
])
def test_combos(percents):
    _assert(percents)


def test_full_stack_with_z_push_none():
    # Every kernel + a live +z family reprojected through the fallback.
    _assert({'mini': 0.5, 'tiny': 0.3, 'pulseinner': 1.0, 'pulseouter': 1.0,
             'shrinkmult': 50.0, 'shrinklinear': 100.0, 'confusionx': 1.0,
             'bumpy': 1.0})
