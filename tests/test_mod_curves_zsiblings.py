"""Validation: the Z-axis sibling curve mods (mod_curves_zsiblings) reproduce
the hardcoded arrow_effects kernels, over a y-offset + parameter sweep. This
is the proof that porting the drunk / tornado / waveform Z siblings to the
curve algebra changes nothing observable -- the precondition for swapping
`_z_push` to curves and deleting the kernels.

Equality is to a few ULP (`RTOL`), not bit-identity, for the continuous
kernels (drunk / tornado): the curve form reassociates the arithmetic
(distributes the amplitude, splits the affine phase, expands the tornado
remap into scale + add). The DISCONTINUOUS waveform kernels (digital /
zigzag / sawtooth / square / bounce) instead ride `grouped_phase` -- the
engine's exact operation order -- so their break-branch selection agrees
bit-for-bit and their error is EXACTLY zero even at points on a break;
those are asserted at rtol 0 with a matching atol.

The Z siblings share their formula shape and companion naming with their
already-ported X twins; the only per-axis difference is tornado's arccos
window, which stays width 3 on Z (dimension 2) where X narrows to 2 in a
wide field. The wide-field cases below exercise that."""
import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import curves as cv
from analysis.player.render.mods import mod_curves_zsiblings as mcz

# A handful of ULP: reassociation-only error, nothing structural.
RTOL = 1e-12


# A representative visible-note batch: y_offsets spanning above/below the
# receptor, columns cycling a 4-key field. Break-astride y_offsets are added
# per-kernel for the discontinuous waveforms.
Y = np.linspace(-800.0, 800.0, 33)
COLS = np.arange(Y.shape[0]) % 4
T = 12.34
BEAT = 40.5
KC = 4

# A wide field too, to exercise the tornado Z window keeping width 3 (the X
# window narrows 3->2 for >4 columns, but only on dimension 0).
COLS_WIDE = np.arange(Y.shape[0]) % 8
KC_WIDE = 8


def _ctx(cols=COLS, y=Y):
    return cv.Ctx(t=T, beat=BEAT, cols=cols, arrow_size=ae.ARROW_SIZE)


# ---------------------------------------------------------------------------
# drunkz (continuous cos/tan warp on z)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('percent,speed,offset,period,is_tan', [
    (1.0, 0.0, 0.0, 0.0, False),
    (0.5, 0.3, 0.0, 0.0, False),
    (1.0, 0.0, 0.7, 0.0, False),
    (1.0, 0.0, 0.0, 0.9, False),
    (0.8, 0.2, 0.4, 0.6, False),
    (1.0, 0.0, 0.0, 0.0, True),
])
def test_drunkz_curve_equals_kernel(percent, speed, offset, period, is_tan):
    curve = mcz.drunkz_z(percent, speed=speed, offset=offset, period=period,
                         is_tan=is_tan)
    got = curve(Y, _ctx())
    want = ae.drunk_x(percent, COLS, Y, T, KC, speed=speed, offset=offset,
                      period=period, is_tan=is_tan)
    np.testing.assert_allclose(got, want, rtol=RTOL)


# ---------------------------------------------------------------------------
# tornadoz (arccos-windowed warp on z; window width STAYS 3 in wide fields)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('percent,offset,period', [
    (1.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (1.0, 30.0, 0.0),
    (1.0, 0.0, 0.7),
    (0.8, -25.0, 0.4),
])
@pytest.mark.parametrize('cols,keycount', [(COLS, KC), (COLS_WIDE, KC_WIDE)])
def test_tornadoz_curve_equals_kernel(percent, offset, period, cols, keycount):
    curve = mcz.tornadoz_z(percent, keycount, offset=offset, period=period)
    got = curve(Y, _ctx(cols))
    want = ae._tornado_offset(percent, cols, Y, keycount, ae.ARROW_SIZE,
                              offset, period, is_tan=False,
                              dimension=mcz.TORNADO_Z_DIMENSION)
    np.testing.assert_allclose(got, want, rtol=RTOL)


# tan-tornadoz near a tan pole amplifies a ~1e-16 reassociation error past
# rtol; keep every sample off the asymptote (values near a pole are
# geometrically meaningless / clamped in-engine anyway).
@pytest.mark.parametrize('percent,offset,period', [
    (1.0, 0.0, 0.0),
    (0.5, 15.0, 0.25),
    (0.8, -25.0, 0.4),
])
@pytest.mark.parametrize('cols,keycount', [(COLS, KC), (COLS_WIDE, KC_WIDE)])
def test_tan_tornadoz_curve_equals_kernel(percent, offset, period, cols,
                                          keycount):
    curve = mcz.tornadoz_z(percent, keycount, offset=offset, period=period,
                           is_tan=True)
    got = curve(Y, _ctx(cols))
    want = ae._tornado_offset(percent, cols, Y, keycount, ae.ARROW_SIZE,
                              offset, period, is_tan=True,
                              dimension=mcz.TORNADO_Z_DIMENSION)
    np.testing.assert_allclose(got, want, rtol=RTOL)


def test_tornadoz_window_wider_than_tornadox_in_wide_field():
    """The load-bearing per-axis difference: on an 8-col field the Z window
    keeps width 3, so tornadoz differs from the X-dimension tornado. If the
    curve wrongly used dimension 0, the two would coincide."""
    z = mcz.tornadoz_z(1.0, KC_WIDE)(Y, _ctx(COLS_WIDE))
    x = ae._tornado_offset(1.0, COLS_WIDE, Y, KC_WIDE, ae.ARROW_SIZE,
                           0.0, 0.0, is_tan=False, dimension=0)
    assert not np.allclose(z, x)


# ---------------------------------------------------------------------------
# Discontinuous waveform siblings on z: exact (rtol 0) via grouped_phase,
# tested on y_offsets landing ON and astride each kernel's breaks.
# ---------------------------------------------------------------------------

# digital/square share PI*(y+offset)/(AS + period*AS): breaks where the sine
# wraps and where round() steps. Land y_offsets on integer multiples of AS
# (angle multiples of PI) and just astride them.
AS = ae.ARROW_SIZE
Y_BREAKS = np.concatenate([
    Y,
    np.array([-2 * AS, -AS, 0.0, AS, 2 * AS, 3 * AS]),
    np.array([-AS, 0.0, AS]) + 1e-9,
    np.array([-AS, 0.0, AS]) - 1e-9,
])
COLS_BREAKS = np.arange(Y_BREAKS.shape[0]) % KC


def _ctx_breaks():
    return cv.Ctx(t=T, beat=BEAT, cols=COLS_BREAKS, arrow_size=AS)


@pytest.mark.parametrize('percent,offset,period,steps,is_tan', [
    (1.0, 0.0, 0.0, 0.0, False),
    (0.7, 1.0, 0.0, 2.0, False),
    (1.0, 0.0, 0.5, 4.0, False),
    (0.5, 2.0, 0.3, 1.0, False),
    (1.0, 0.0, 0.0, 3.0, True),
])
def test_digitalz_curve_equals_kernel(percent, offset, period, steps, is_tan):
    curve = mcz.digitalz_z(percent, offset, period, steps, is_tan=is_tan)
    got = curve(Y_BREAKS, _ctx_breaks())
    want = ae.digital_x(percent, Y_BREAKS, offset, period, steps,
                        is_tan=is_tan)
    # tan kernel is continuous-ish; the sin/round path is exact. Both agree
    # bit-for-bit here because grouped_phase mirrors the engine order.
    np.testing.assert_allclose(got, want, rtol=0.0, atol=0.0)


@pytest.mark.parametrize('percent,offset,period', [
    (1.0, 0.0, 0.0),
    (0.7, 1.0, 0.0),
    (1.0, 0.0, 0.5),
    (0.5, -2.0, 0.3),
])
def test_zigzagz_curve_equals_kernel(percent, offset, period):
    curve = mcz.zigzagz_z(percent, offset, period)
    got = curve(Y_BREAKS, _ctx_breaks())
    want = ae.zigzag_x(percent, Y_BREAKS, offset, period)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=0.0)


@pytest.mark.parametrize('percent,period', [
    (1.0, 0.0),
    (0.7, 0.5),
    (0.5, 1.5),
])
def test_sawtoothz_curve_equals_kernel(percent, period):
    curve = mcz.sawtoothz_z(percent, period)
    got = curve(Y_BREAKS, _ctx_breaks())
    want = ae.sawtooth_x(percent, Y_BREAKS, period)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=0.0)


@pytest.mark.parametrize('percent,offset,period', [
    (1.0, 0.0, 0.0),
    (0.7, 1.0, 0.0),
    (1.0, 0.0, 0.5),
    (0.5, -2.0, 0.3),
])
def test_squarez_curve_equals_kernel(percent, offset, period):
    curve = mcz.squarez_z(percent, offset, period)
    got = curve(Y_BREAKS, _ctx_breaks())
    want = ae.square_x(percent, Y_BREAKS, offset, period)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=0.0)


@pytest.mark.parametrize('percent,offset,period', [
    (1.0, 0.0, 0.0),
    (0.7, 1.0, 0.0),
    (1.0, 0.0, 0.5),
    (0.5, -2.0, 0.3),
])
def test_bouncez_curve_equals_kernel(percent, offset, period):
    curve = mcz.bouncez_z(percent, offset, period)
    got = curve(Y_BREAKS, _ctx_breaks())
    want = ae.bounce_x(percent, Y_BREAKS, offset, period)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=0.0)
