"""Validation: the DISCONTINUOUS waveform-warp mods (mod_curves_waveform)
reproduce the hardcoded arrow_effects kernels, over a y-offset + parameter
sweep that INCLUDES points at and astride each kernel's discontinuities.

This is the harder half of the curve-port proof: digital / zigzag /
sawtooth / square / bounce are non-smooth (a quantized sine staircase, a
triangle's kinks, a sawtooth's integer jumps, a square's sign flips, a
rectified-sine cusp). A curve port that only matched on the smooth
interior would be worthless - the whole claim is that the affine-phase ->
kernel -> scale algebra reproduces the wave EXACTLY, including at the
breakpoints. So the sweep is seeded with y_offsets that drive each phase
onto its discontinuity and to the values just below/above it.

Equality is to a few ULP (`RTOL`): the curve form reassociates the affine
phase and distributes the amplitude, a ~1e-15 relative difference far
below any geometric significance. The step/where/floor selection itself is
bit-identical (same operands, same order), so points ON a break agree too.
"""
import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import mod_curves_waveform as wf
from analysis.player.render.mods import curves as cv

# A handful of ULP: reassociation-only error, nothing structural.
RTOL = 1e-12

AS = ae.ARROW_SIZE
PI = ae.PI


def _ctx():
    # Waveform warps are y-only (no t / beat / cols); ctx carries only the
    # column pitch the amplitude scales against.
    return cv.Ctx(arrow_size=AS)


def _sweep_ys(landmarks):
    """A broad smooth y_offset linspace UNION each landmark y (a phase
    discontinuity) taken exactly and a hair below/above, so the sweep
    straddles every break. The exactly-on-break points are the ones that
    catch a phase built with the wrong arithmetic grouping."""
    smooth = np.linspace(-900.0, 900.0, 121)
    eps = np.array([-1e-6, 0.0, 1e-6])
    astride = np.asarray(landmarks, dtype=np.float64)[:, None] + eps
    return np.unique(np.concatenate([smooth, np.ravel(astride)]))


# ---------------------------------------------------------------------------
# digital: quantized sine. Discontinuities where round(levels*sin) steps.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('percent,offset,period,steps,is_tan', [
    (1.0, 0.0, 0.0, 0, False),
    (0.5, 0.7, 0.0, 0, False),
    (1.0, 0.0, 0.9, 0, False),
    (0.8, 0.4, 0.6, 3, False),
    (1.0, 0.0, 0.0, 7, False),
    (1.0, 0.0, 0.0, 2, True),
])
def test_digital_curve_equals_kernel(percent, offset, period, steps, is_tan):
    # Land y so sin(angle) hits the round() half-way steps k/levels + 0.5,
    # where the staircase jumps, plus a broad sweep.
    denom = AS + period * AS
    levels = steps + 1.0
    targets = []
    for m in np.arange(-2 * levels, 2 * levels + 1):
        s = (m + 0.5) / levels
        if -1.0 <= s <= 1.0:
            ang = np.arcsin(s)
            for base in (ang, PI - ang):
                targets.append(base * denom / PI - offset)
    y = _sweep_ys(targets)

    got = wf.digital_x(percent, offset, period, steps, is_tan=is_tan)(y, _ctx())
    want = ae.digital_x(percent, y, offset, period, steps, is_tan=is_tan)
    np.testing.assert_allclose(got, want, rtol=RTOL)


# ---------------------------------------------------------------------------
# zigzag: triangle wave. Kinks at u = 0.5, 1.5 (phase = PI/2, 3PI/2 mod 2PI).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('percent,offset,period', [
    (1.0, 0.0, 0.0),
    (0.5, 0.6, 0.0),
    (1.0, 0.0, 0.8),
    (0.7, 0.3, 0.4),
])
def test_zigzag_curve_equals_kernel(percent, offset, period):
    freq = PI * (1.0 / (period + 1.0)) / AS
    off = 100.0 * offset
    # phase = freq*(y + off); kinks where phase = n*PI/2 for odd/half n.
    targets = [(k * (PI / 2.0)) / freq - off for k in range(-6, 7)]
    y = _sweep_ys(targets)

    got = wf.zigzag_x(percent, offset, period)(y, _ctx())
    want = ae.zigzag_x(percent, y, offset, period)
    np.testing.assert_allclose(got, want, rtol=RTOL)


# ---------------------------------------------------------------------------
# sawtooth: frac ramp. Jumps at integer ramp values (frac 1 -> 0).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('percent,period', [
    (1.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.9),
    (0.7, 0.4),
])
def test_sawtooth_curve_equals_kernel(percent, period):
    slope = (0.5 / (period + 1.0)) / AS
    # ramp = slope*y; jumps where slope*y is an integer.
    targets = [n / slope for n in range(-8, 9)]
    y = _sweep_ys(targets)

    got = wf.sawtooth_x(percent, period)(y, _ctx())
    want = ae.sawtooth_x(percent, y, period)
    np.testing.assert_allclose(got, want, rtol=RTOL)


# ---------------------------------------------------------------------------
# square: rage_square. Sign flips at phase = PI mod 2PI, plus the <0.01
# near-zero guard band that flips a small positive phase.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('percent,offset,period', [
    (1.0, 0.0, 0.0),
    (0.5, 0.7, 0.0),
    (1.0, 0.0, 0.8),
    (0.7, 0.4, 0.6),
])
def test_square_curve_equals_kernel(percent, offset, period):
    denom = AS + period * AS
    # phase = PI*(y+offset)/denom; flips where phase = n*PI, i.e. y+offset =
    # n*denom. Guard band <0.01: land phase just inside [0, 0.01).
    targets = [n * denom - offset for n in range(-8, 9)]
    guard = [(0.005) * denom / PI - offset, (0.009) * denom / PI - offset]
    y = _sweep_ys(targets + guard)

    got = wf.square_x(percent, offset, period)(y, _ctx())
    want = ae.square_x(percent, y, offset, period)
    np.testing.assert_allclose(got, want, rtol=RTOL)


def test_square_guard_band_flips_sign():
    """The <0.01 wrapped-angle guard is NOT a no-op: a note whose phase
    lands in [0, 0.01) is pushed past PI and flips from +1 to -1. Confirm
    the curve reproduces that engine hold-flicker quirk, not just a plain
    square wave."""
    denom = AS + 0.0 * AS
    y_in_band = np.array([0.005 * denom / PI, 0.009 * denom / PI])
    got = wf.square_x(1.0, 0.0, 0.0)(y_in_band, _ctx())
    # A naive square wave would give +AS*0.5 here; the guard makes it -AS*0.5.
    assert np.all(got < 0.0)
    want = ae.square_x(1.0, y_in_band, 0.0, 0.0)
    np.testing.assert_allclose(got, want, rtol=RTOL)


# ---------------------------------------------------------------------------
# bounce: rectified sine. Cusp where sin crosses zero (phase = n*PI).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('percent,offset,period', [
    (1.0, 0.0, 0.0),
    (0.5, 0.6, 0.0),
    (1.0, 0.0, 0.8),
    (0.7, 0.3, 0.4),
])
def test_bounce_curve_equals_kernel(percent, offset, period):
    denom = 60.0 + period * 60.0
    # phase = (y+offset)/denom; abs-cusp where phase = n*PI.
    targets = [n * PI * denom - offset for n in range(-8, 9)]
    y = _sweep_ys(targets)

    got = wf.bounce_x(percent, offset, period)(y, _ctx())
    want = ae.bounce_x(percent, y, offset, period)
    np.testing.assert_allclose(got, want, rtol=RTOL)
