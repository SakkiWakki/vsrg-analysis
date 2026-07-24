"""Per-column stealth family + stealthpastreceptors in the note kernels.

NotITG stores the numbered column variants alongside the global
(PlayerOptions Stealth / StealthCol, StealthGlow / StealthGlowCol) and
sums them per note - the standard un-hide idiom applies `stealth` 100%
then drives `stealth<i>` NEGATIVE per column to reveal it. stealthglow
hides the fill the same way but keeps the note as an additive glow;
stealthpastreceptors keeps the stealth subtraction live for past-receptor
notes (which the stock appearance exemption would pin fully visible).
"""
import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import mod_curves_alpha as mca

COLS = np.arange(8) % 4
Y = np.full(8, 100.0)


def test_per_column_stealth_adds_to_the_global():
    percents = {'stealth': 1.0, 'stealth0': -1.0}
    vis = ae.percent_visible(percents, COLS, Y)
    assert np.allclose(vis[COLS == 0], 1.0)
    assert np.allclose(vis[COLS != 0], 0.0)


def test_per_column_stealth_partial_reveal():
    percents = {'stealth': 1.0, 'stealth2': -0.4}
    vis = ae.percent_visible(percents, COLS, Y)
    assert np.allclose(vis[COLS == 2], 0.4)
    assert np.allclose(vis[COLS != 2], 0.0)


def test_per_column_stealthglow_hides_the_fill():
    percents = {'stealthglow1': 1.0}
    vis = ae.percent_visible(percents, COLS, Y)
    assert np.allclose(vis[COLS == 1], 0.0)
    assert np.allclose(vis[COLS != 1], 1.0)


def test_stealth_past_receptors_drops_the_exemption():
    y = np.array([-100.0, 100.0])
    cols = np.array([0, 0])
    exempt = ae.percent_visible({'stealth': 1.0}, cols, y)
    assert exempt[0] == pytest.approx(1.0)
    gated = ae.percent_visible(
        {'stealth': 1.0, 'stealthpastreceptors': 1.0}, cols, y)
    assert np.allclose(gated, 0.0)


def test_glow_amount_takes_per_note_arrays():
    y = np.array([-50.0, 50.0, 50.0])
    amount = ae.stealthglow_amount(np.array([0.8, 0.8, 0.0]), y)
    assert np.allclose(amount, [0.0, 0.8, 0.0])


def test_glow_amount_past_receptors():
    y = np.array([-50.0, 50.0])
    amount = ae.stealthglow_amount(0.7, y, past_receptors=True)
    assert np.allclose(amount, [0.7, 0.7])


def test_note_offsets_emits_glow_for_column_only_stealthglow():
    offs = ae.note_offsets({'stealthglow0': 1.0}, COLS, Y, t_now=0.0,
                           beat_now=0.0, keycount=4)
    assert offs.glow is not None
    assert np.allclose(offs.glow[COLS == 0], 1.0)
    assert np.allclose(offs.glow[COLS != 0], 0.0)


@pytest.mark.parametrize('percents', [
    {'stealth': 1.0, 'stealth0': -1.0, 'stealth2': -0.5},
    {'stealthglow': 0.4, 'stealthglow1': 0.6},
    {'stealth': 1.0, 'stealthpastreceptors': 1.0},
    {'stealth': 0.8, 'stealthglow2': 1.0, 'stealthpastreceptors': 1.0,
     'hidden': 0.5},
])
@pytest.mark.parametrize('t_now', [0.0, 0.31])
def test_alpha_curve_parity_with_column_stealth(percents, t_now):
    vis_y = np.linspace(-400.0, 600.0, 51)
    cols = np.arange(51) % 4
    got = mca.alpha_curve(percents, t_now)(vis_y, mca.cv.Ctx(
        cols=cols, arrow_size=ae.ARROW_SIZE))
    want = ae._alpha(percents, cols, vis_y, t_now)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)
