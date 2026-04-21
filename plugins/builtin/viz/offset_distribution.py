"""Per-hand histogram of timing offsets."""
from analysis.viz.plots import plot_offset_distribution
from ._common import clean_arrays, new_fig


def build(replay, game='etterna', **_):
    _, offs, cols = clean_arrays(replay)
    fig, ax = new_fig(9, 5)
    plot_offset_distribution(offs, cols, ax=ax)
    return fig


def register(add):
    add('Offset distribution', build, category='chart')
