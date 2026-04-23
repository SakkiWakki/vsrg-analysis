"""Per-hand histogram of timing offsets."""
from analysis.components import Manifest, SURFACE_VIZ, VizFields
from analysis.components.viz_backend import (
    LEFT_CLR,
    RIGHT_CLR,
    VIZ_CATEGORY_CHART,
    MS,
    new_figure,
)


MANIFEST = Manifest(
    key='builtin:viz:offset_distribution',
    name='Offset distribution',
    supported_surfaces={SURFACE_VIZ},
    plugin_fields={'viz': VizFields(category=VIZ_CATEGORY_CHART)},
)


def _draw(ctx):
    import numpy as np

    offs = ctx.replay.offsets_clean()
    cols = ctx.replay.columns_clean()
    fig, ax = new_figure(9, 5)
    keycount = ctx.replay.keycount()
    left_cols, right_cols = ctx.analysis.default_hands(keycount)
    left = np.isin(cols, left_cols)
    right = np.isin(cols, right_cols)
    bins = np.linspace(-80, 80, 81)
    ax.hist(offs[left] * MS, bins=bins, alpha=0.65,
            color=LEFT_CLR, label='Left')
    ax.hist(offs[right] * MS, bins=bins, alpha=0.65,
            color=RIGHT_CLR, label='Right')
    ax.axvline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_xlabel('offset (ms)  (negative=early, positive=late)')
    ax.set_ylabel('count')
    ax.set_title('Offset distribution by hand')
    ax.legend()
    ctx.figure(fig)


def register_components(add):
    add(MANIFEST, _draw)
