"""Rolling std of offsets (overall + per hand)."""
from analysis.viz.plots import plot_rolling
from ._common import clean_arrays, new_fig


def build(replay, game='etterna', **_):
    _, offs, cols = clean_arrays(replay)
    fig, ax = new_fig(10, 4.5)
    plot_rolling(offs, cols, ax=ax)
    return fig


def register(add):
    add('Rolling stability', build, category='chart')
