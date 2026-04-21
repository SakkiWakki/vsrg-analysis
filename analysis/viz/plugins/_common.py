"""Shared helpers for visualization plugins. Keycount-agnostic."""
import numpy as np
import matplotlib.pyplot as plt

from analysis.etterna.replay import clean_offsets
from analysis.core.timing import default_hands
from analysis.viz.plots import col_colors, LEFT_CLR, RIGHT_CLR, MS


plt.style.use('dark_background')


def keycount_of(replay):
    kc = replay.get('keycount')
    if kc:
        return int(kc)
    cols = replay.get('columns')
    if cols is not None and len(cols):
        return int(cols.max()) + 1
    return 4


def hand_masks(replay):
    cols = replay['columns'][~replay['misses']] if 'misses' in replay else replay['columns']
    left, right = default_hands(keycount_of(replay))
    return np.isin(cols, left), np.isin(cols, right)


def clean_arrays(replay):
    c = clean_offsets(replay)
    return c['noterows'], c['offsets'], c['columns']


def new_fig(w=10, h=5):
    # Use Figure() directly (not plt.subplots) so figures created on a
    # background thread don't try to spawn a new pyplot-managed window.
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h))
    fig.patch.set_facecolor('#121212')
    ax = fig.add_subplot(111)
    return fig, ax
