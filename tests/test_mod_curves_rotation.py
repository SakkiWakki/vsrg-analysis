"""Validation: the curve-composed rotation mods (mod_curves_rotation)
reproduce the hardcoded arrow_effects rot_z kernels, over a beat +
parameter (+ per-note beat for dizzy) sweep. Same contract as
tests/test_mod_curves: equality to a few ULP, the reassociation-only
error intrinsic to expressing a mod as composable primitives.

confusion is y-independent (one scalar broadcast to every note); dizzy
is y-independent too but per-note through the note's OWN beat, so the
sweep varies note_beats across the batch. The angle wraps modulo a full
turn, so the note_beat sweep is chosen to straddle wrap points."""
import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import curves as cv
from analysis.player.render.mods import mod_curves_rotation as mr

RTOL = 1e-12


# A representative visible-note batch: y_offsets spanning above/below the
# receptor, columns cycling a 4-key field, note beats spread across the
# field so the dizzy (note_beat - beat) angle sweeps past wrap points.
Y = np.linspace(-800.0, 800.0, 33)
COLS = np.arange(Y.shape[0]) % 4
T = 12.34
BEAT = 40.5
NOTE_BEATS = np.linspace(-8.0, 96.0, Y.shape[0])


def _ctx():
    return cv.Ctx(t=T, beat=BEAT, cols=COLS, note_beats=NOTE_BEATS,
                  arrow_size=ae.ARROW_SIZE)


@pytest.mark.parametrize('percent,offset', [
    (1.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.25),
    (0.3, -0.4),
    (2.0, 0.7),
])
def test_confusion_rot_equals_kernel_degrees(percent, offset):
    curve = mr.confusion_rot(percent, offset=offset)
    got = curve(Y, _ctx())
    want = np.broadcast_to(
        ae.confusion_rotation(percent, BEAT, offset), Y.shape)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('percent', [1.0, 0.5, 0.3, 2.0, -1.0])
def test_dizzy_rot_equals_kernel_degrees(percent):
    curve = mr.dizzy_rot(percent)
    got = curve(Y, _ctx())
    want = ae.dizzy_rotation(percent, NOTE_BEATS, BEAT)
    np.testing.assert_allclose(got, want, rtol=RTOL)
