"""2D heatmap of (column × offset) hit density. Works for any keycount."""
import numpy as np

from ._common import clean_arrays, keycount_of, new_fig


def build(replay, game='etterna', **_):
    _, offs, cols = clean_arrays(replay)
    kc = keycount_of(replay)
    fig, ax = new_fig(9, 5)
    if len(offs) == 0:
        ax.text(0.5, 0.5, 'no hits', ha='center', va='center',
                transform=ax.transAxes)
        return fig
    off_ms = offs * 1000.0
    ybins = np.linspace(-120, 120, 61)
    xbins = np.arange(kc + 1) - 0.5
    h, _, _ = np.histogram2d(cols, off_ms, bins=[xbins, ybins])
    im = ax.imshow(h.T, origin='lower', aspect='auto',
                   extent=[-0.5, kc - 0.5, ybins[0], ybins[-1]],
                   cmap='inferno')
    ax.axhline(0, color='#80deea', lw=0.8, alpha=0.6)
    ax.set_xticks(range(kc))
    ax.set_xlabel('column')
    ax.set_ylabel('offset (ms)')
    ax.set_title(f'Column × offset density ({kc}K)')
    fig.colorbar(im, ax=ax, label='hits')
    return fig


def register(add):
    add('Column × offset heatmap', build, category='chart')
