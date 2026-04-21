"""Single/jump/hand/quad timing. Chord counts scale with keycount (quad is capped at 4)."""
from analysis.viz.plots import plot_chord_sizes
from ._common import clean_arrays, new_fig


def build(replay, game='etterna', **_):
    rows, offs, cols = clean_arrays(replay)
    fig, ax = new_fig(8, 5)
    plot_chord_sizes(rows, offs, cols, ax=ax)
    return fig


def register(add):
    add('Chord sizes', build, category='chart')
