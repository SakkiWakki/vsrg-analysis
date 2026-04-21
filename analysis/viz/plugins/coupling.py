"""Solo vs same-hand-paired column timing (any keycount)."""
from analysis.viz.plots import plot_coupling
from ._common import clean_arrays, new_fig


def build(replay, game='etterna', **_):
    rows, offs, cols = clean_arrays(replay)
    fig, ax = new_fig(9, 5)
    plot_coupling(rows, offs, cols, ax=ax)
    return fig


def register(add):
    add('Coupling (solo vs paired)', build, category='chart')
