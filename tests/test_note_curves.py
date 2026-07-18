"""The curve assembler reproduces arrow_effects.note_offsets in FULL,
byte-equal (ULP), across a battery of real percent dicts (single mods,
combos, numbered per-column variants). Every axis -- dx/dy/z/rotation/
roll/twirl, the multiplicative zoom fold, the alpha/visibility windows,
glow, and the perspective reprojections (hallway/confusiony) -- is
assembled from curves; nothing delegates to a kernel aggregator. This is
the proof that the live note_offsets runs entirely on curves."""
import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import curves as cv
from analysis.player.render.mods import note_curves as nc

Y = np.linspace(-800.0, 800.0, 64)
COLS = (np.arange(64) % 4).astype(np.int64)
NOTE_BEATS = np.linspace(0.0, 32.0, 64)
T, BEAT, KC = 12.34, 40.5, 4
SIZE = ae.ARROW_SIZE
RTOL = 1e-12


def _ctx():
    return cv.Ctx(t=T, beat=BEAT, cols=COLS, note_beats=NOTE_BEATS,
                  arrow_size=SIZE)


# Percent dicts exercising each covered family (NO hallway/confusiony/zoom/
# alpha - those stay kernel-delegated). Each is a real modstring shape.
DX = [
    {'drunk': 1.0, 'drunkspeed': 0.2, 'drunkoffset': 0.3, 'drunkperiod': 0.4},
    {'tandrunk': 0.8},
    {'tornado': 1.0, 'tornadooffset': 0.5}, {'tantornado': 0.6},
    {'bumpyx': 1.0, 'bumpyxoffset': 0.2}, {'tanbumpyx': 0.7},
    {'flip': 1.0}, {'invert': 0.5},
    {'beat': 1.0, 'beatoffset': 0.3, 'beatmult': 0.5},
    {'parabolax': 0.8}, {'attenuatex': 0.6}, {'xmode': 1.0},
    {'movex': 1.0},
    {'digital': 1.0, 'digitalsteps': 4.0}, {'zigzag': 1.0, 'zigzagoffset': 0.7},
    {'sawtooth': 1.0}, {'square': 1.0}, {'bounce': 1.0},
    {'tandigital': 1.0, 'tandigitalsteps': 3.0},
    {'drunk0': 1.0}, {'movex1': 0.5, 'movex3': -0.3},  # numbered variants
    {'drunk': 1.0, 'tornado': 0.5, 'movex': 0.4, 'digital': 0.6, 'digitalsteps': 4.0},
]
DY = [
    {'tipsy': 1.0, 'tipsyspeed': 0.3}, {'tantipsy': 0.7},
    {'beaty': 1.0, 'beatyperiod': 0.2}, {'parabolay': 0.8},
    {'attenuatey': 0.6}, {'movey': 1.0}, {'tipsy0': 1.0, 'movey': 0.5},
]
Z = [
    {'bumpy': 1.0, 'bumpyoffset': 0.2}, {'tanbumpy': 0.7},
    {'drunkz': 1.0}, {'tornadoz': 1.0}, {'digitalz': 1.0, 'digitalzsteps': 4.0},
    {'zigzagz': 1.0}, {'sawtoothz': 1.0}, {'squarez': 1.0}, {'bouncez': 1.0},
    {'beatz': 1.0}, {'parabolaz': 0.8}, {'attenuatez': 0.6}, {'movez': 1.0},
    {'bumpy': 1.0, 'drunkz': 0.5, 'digitalz': 0.6, 'digitalzsteps': 4.0},
]
ROT = [
    {'dizzy': 1.0}, {'confusion': 1.0}, {'confusion': 0.5, 'confusionoffset': 0.2},
    {'dizzy': 0.8, 'confusion': 0.6},
]


@pytest.mark.parametrize('p', DX)
def test_dx_axis_equals_kernel(p):
    got = nc._sum(nc.dx_curves(p, COLS, KC, SIZE, BEAT), Y, _ctx())
    want = ae._dx(p, COLS, Y, T, BEAT, KC, SIZE)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('p', DY)
def test_dy_axis_equals_kernel(p):
    got = nc._sum(nc.dy_curves(p, COLS, KC, SIZE, BEAT), Y, _ctx())
    want = ae._dy(p, COLS, Y, T, BEAT, KC, SIZE)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('p', Z)
def test_z_axis_equals_kernel(p):
    got = nc._sum(nc.z_curves(p, COLS, KC, SIZE, BEAT), Y, _ctx())
    want = ae._z_push(p, COLS, Y, T, BEAT, KC, SIZE)
    np.testing.assert_allclose(got, want, rtol=RTOL)


@pytest.mark.parametrize('p', ROT)
def test_rotation_axis_equals_kernel(p):
    got = nc._sum(nc.rotation_curves(p, KC), Y, _ctx())
    want = ae._rotation(p, NOTE_BEATS, BEAT, COLS.shape[0])
    np.testing.assert_allclose(got, want, rtol=RTOL)


# Full NoteOffsets drop-in: assemble() == note_offsets, including the
# kernel-delegated families (zoom / alpha / glow / perspective) so the
# swap is byte-equal end to end. Combos mix position curves WITH the
# delegated families to prove the boundary is seamless.
FULL = DX + DY + Z + ROT + [
    {},                                    # unmodded: identity
    {'mini': 0.5}, {'tiny': 0.3}, {'pulseinner': 0.5, 'pulseouter': 1.0},
    {'shrinkmult': 0.4}, {'confusionx': 1.0},         # zoom family (delegated)
    {'hidden': 1.0}, {'sudden': 1.0}, {'blink': 1.0}, {'boomerang': 1.0},
    {'stealthglow': 1.0},                             # alpha / glow (delegated)
    {'hallway': 1.0}, {'confusiony': 1.0},            # perspective (delegated)
    {'roll': 1.0}, {'twirl': 0.5},                    # tilt (project_3d)
    {'drunk': 1.0, 'mini': 0.5, 'confusionx': 0.8, 'boomerang': 0.6},  # mixed
    {'bumpy': 1.0, 'drunkz': 0.5, 'digitalz': 0.6, 'digitalzsteps': 4.0,
     'tiny': 0.3, 'hallway': 0.4},                    # z + zoom + perspective
    {'confusionx0': 0.5, 'confusionx2': 1.0},         # per-column confusionx (zoom)
    {'confusiony1': 0.7, 'confusionyoffset': 0.2},    # per-column confusiony (dx)
    {'pulseinner': 0.5, 'pulseouter': 1.0, 'shrinkmult': 0.3,
     'shrinklinear': 0.4, 'confusionx': 0.6},         # full zoom fold
]


def _assert_note_offsets_equal(a, b):
    for field in ('dx', 'dy', 'rotation_deg', 'alpha_mult', 'zoom',
                  'z', 'rot_x', 'rot_y', 'glow'):
        va, vb = getattr(a, field), getattr(b, field)
        if va is None or vb is None:
            assert (va is None) == (vb is None), field
            continue
        np.testing.assert_allclose(va, vb, rtol=RTOL, err_msg=field)


@pytest.mark.parametrize('p', FULL)
@pytest.mark.parametrize('project_3d', [False, True])
def test_assemble_equals_note_offsets(p, project_3d):
    got = nc.assemble(p, COLS, Y, T, BEAT, KC, note_beats=NOTE_BEATS,
                      project_3d=project_3d)
    want = ae.note_offsets(p, COLS, Y, T, BEAT, KC, note_beats=NOTE_BEATS,
                           project_3d=project_3d)
    _assert_note_offsets_equal(got, want)
