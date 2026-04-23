"""2D heatmap of column by offset density."""
import numpy as np

from analysis.components import Manifest, SURFACE_VIZ, VizFields
from analysis.components.viz_backend import VIZ_CATEGORY_CHART, new_figure


MANIFEST = Manifest(
    key='builtin:viz:column_heatmap',
    name='Column × offset heatmap',
    supported_surfaces={SURFACE_VIZ},
    plugin_fields={'viz': VizFields(category=VIZ_CATEGORY_CHART)},
)


def _draw(ctx):
    offs = ctx.replay.offsets_clean()
    cols = ctx.replay.columns_clean()
    keycount = ctx.replay.keycount()
    fig, ax = new_figure(9, 5)
    if len(offs) == 0:
        ax.text(0.5, 0.5, 'no hits', ha='center', va='center',
                transform=ax.transAxes)
        ctx.figure(fig)
        return
    off_ms = offs * 1000.0
    ybins = np.linspace(-120, 120, 61)
    xbins = np.arange(keycount + 1) - 0.5
    heatmap, _, _ = np.histogram2d(cols, off_ms, bins=[xbins, ybins])
    image = ax.imshow(
        heatmap.T,
        origin='lower',
        aspect='auto',
        extent=[-0.5, keycount - 0.5, ybins[0], ybins[-1]],
        cmap='inferno',
    )
    ax.axhline(0, color='#80deea', lw=0.8, alpha=0.6)
    ax.set_xticks(range(keycount))
    ax.set_xlabel('column')
    ax.set_ylabel('offset (ms)')
    ax.set_title(f'Column × offset density ({keycount}K)')
    fig.colorbar(image, ax=ax, label='hits')
    ctx.figure(fig)


def register_components(add):
    add(MANIFEST, _draw)
