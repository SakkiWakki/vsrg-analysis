"""Per-hand timing drift across chart segments."""
from analysis.viz.plots import plot_drift
from ._common import clean_arrays, new_fig


def build(replay, game='etterna', **_):
    rows, offs, cols = clean_arrays(replay)
    fig, ax = new_fig(10, 5)
    plot_drift(rows, offs, cols, ax=ax)
    return fig


def register(add):
    add('Drift (hands × time)', build, category='chart')
