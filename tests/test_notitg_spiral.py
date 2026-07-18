"""Tests for the note-mod tail: movez, stealthglow, and the spiral deferral.

Every new channel must rest at IDENTITY (percent 0 = pixel-exact no-op),
per the swarm parity discipline.
"""
import numpy as np

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import note_offsets


def _cols(n):
    return np.arange(n, dtype=np.int64)


def test_movez_is_one_arrow_at_full():
    # NotITG: 100% movez = one ARROW_SIZE along z, same shape as movex/movey.
    out = ae.movez_z(np.array([1.0, 0.5, -1.0]))
    np.testing.assert_allclose(out, [64.0, 32.0, -64.0])


def test_movez_rest_is_zero():
    out = ae.movez_z(np.array([0.0, 0.0]))
    np.testing.assert_array_equal(out, [0.0, 0.0])


def test_movez_feeds_z_channel_in_project_3d():
    # movez pushes the engine +z channel; with project_3d it appears in z
    # (one arrow width at 100%), leaving dx/dy untouched.
    cols = _cols(4)
    y = np.array([50.0, 50.0, 50.0, 50.0])
    r = note_offsets({'movez': 1.0}, cols, y, t_now=0.0, beat_now=0.0,
                     keycount=4, project_3d=True)
    np.testing.assert_allclose(r.z, [64.0, 64.0, 64.0, 64.0])
    np.testing.assert_allclose(r.dx, np.zeros(4))
    np.testing.assert_allclose(r.dy, np.zeros(4))


def test_movez_per_column_numbered_variant():
    # movez1 overrides only column 1; the global movez rests at 0.
    cols = _cols(4)
    y = np.full(4, 30.0)
    r = note_offsets({'movez1': 0.5}, cols, y, t_now=0.0, beat_now=0.0,
                     keycount=4, project_3d=True)
    np.testing.assert_allclose(r.z, [0.0, 32.0, 0.0, 0.0])


def test_movez_2d_fallback_reprojects_to_zoom():
    # Without project_3d, movez reprojects to a zoom multiplier via the
    # perspective divide d/(d-z), exactly like bumpy/waveform z siblings.
    cols = _cols(2)
    y = np.array([20.0, 20.0])
    r = note_offsets({'movez': 1.0}, cols, y, t_now=0.0, beat_now=0.0,
                     keycount=2, project_3d=False)
    d = ae.EYE_DISTANCE
    np.testing.assert_allclose(r.zoom, np.full(2, d / (d - 64.0)))


def test_movez_rest_noop_through_note_offsets():
    cols = _cols(4)
    y = np.array([10.0, -5.0, 100.0, 0.0])
    base = note_offsets({}, cols, y, t_now=0.0, beat_now=0.0, keycount=4,
                        project_3d=True)
    rest = note_offsets({'movez': 0.0}, cols, y, t_now=0.0, beat_now=0.0,
                        keycount=4, project_3d=True)
    np.testing.assert_array_equal(base.z, rest.z)
    np.testing.assert_array_equal(base.zoom, rest.zoom)


def test_stealthglow_amount_full_is_glow_one():
    out = ae.stealthglow_amount(1.0, np.array([50.0, 50.0]))
    np.testing.assert_allclose(out, [1.0, 1.0])


def test_stealthglow_amount_rest_is_zero():
    out = ae.stealthglow_amount(0.0, np.array([50.0, -5.0]))
    np.testing.assert_array_equal(out, [0.0, 0.0])


def test_stealthglow_amount_exempts_past_receptor():
    # Past-receptor notes (y < 0) are exempt, mirroring the visibility
    # early-out in ArrowGetPercentVisible.
    out = ae.stealthglow_amount(1.0, np.array([50.0, -1.0]))
    np.testing.assert_allclose(out, [1.0, 0.0])


def test_stealthglow_hides_fill_like_stealth():
    # stealthglow subtracts from visibility exactly like stealth: an
    # approaching note's alpha drops to 0 (fully hidden fill).
    cols = _cols(2)
    y = np.array([50.0, 50.0])
    glowed = note_offsets({'stealthglow': 1.0}, cols, y, t_now=0.0,
                          beat_now=0.0, keycount=2)
    stealthed = note_offsets({'stealth': 1.0}, cols, y, t_now=0.0,
                             beat_now=0.0, keycount=2)
    np.testing.assert_allclose(glowed.alpha_mult, stealthed.alpha_mult)
    np.testing.assert_allclose(glowed.alpha_mult, np.zeros(2))


def test_stealthglow_sets_glow_channel():
    cols = _cols(2)
    y = np.array([50.0, 50.0])
    r = note_offsets({'stealthglow': 1.0}, cols, y, t_now=0.0, beat_now=0.0,
                     keycount=2)
    assert r.glow is not None
    np.testing.assert_allclose(r.glow, [1.0, 1.0])


def test_stealthglow_rest_leaves_glow_none():
    cols = _cols(2)
    y = np.array([50.0, 50.0])
    r = note_offsets({}, cols, y, t_now=0.0, beat_now=0.0, keycount=2)
    assert r.glow is None
    plain_stealth = note_offsets({'stealth': 1.0}, cols, y, t_now=0.0,
                                 beat_now=0.0, keycount=2)
    assert plain_stealth.glow is None


def test_note_offsets_default_no_glow_field():
    # A fully unmodded call leaves glow at its rest (None), so unmodded
    # notes pay nothing.
    cols = _cols(4)
    y = np.array([0.0, 25.0, 50.0, 75.0])
    r = note_offsets({}, cols, y, t_now=1.0, beat_now=2.0, keycount=4)
    assert r.glow is None
