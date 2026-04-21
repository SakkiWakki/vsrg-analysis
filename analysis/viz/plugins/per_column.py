"""Per-column mean offset bars (works for any keycount)."""
from analysis.viz.plots import plot_per_column
from ._common import clean_arrays, new_fig


def build(replay, game='etterna', **_):
    _, offs, cols = clean_arrays(replay)
    fig, ax = new_fig(8, 5)
    plot_per_column(cols, offs, ax=ax)
    return fig


def register(add):
    add('Per-column mean offset', build, category='chart')
