"""Timing offsets scattered over chart progression."""
from analysis.components import Manifest, SURFACE_VIZ, VizFields
from analysis.components.viz_backend import (
    LEFT_CLR,
    RIGHT_CLR,
    VIZ_CATEGORY_CHART,
    MS,
    new_figure,
)


MANIFEST = Manifest(
    key='builtin:viz:scatter_timeline',
    name='Timing scatter',
    supported_surfaces={SURFACE_VIZ},
    plugin_fields={'viz': VizFields(category=VIZ_CATEGORY_CHART)},
)


def _draw(ctx):
    import numpy as np

    rows = ctx.replay.noterows_clean()
    offs = ctx.replay.offsets_clean()
    cols = ctx.replay.columns_clean()
    time_sec = ctx.noterows_to_seconds(rows)
    fig, ax = new_figure(12, 5)
    keycount = ctx.replay.keycount()
    left_cols, right_cols = ctx.analysis.default_hands(keycount)
    left = np.isin(cols, left_cols)
    right = np.isin(cols, right_cols)
    ax.scatter(time_sec[left], offs[left] * MS, s=4, c=LEFT_CLR,
               alpha=0.45, label='Left')
    ax.scatter(time_sec[right], offs[right] * MS, s=4, c=RIGHT_CLR,
               alpha=0.45, label='Right')
    ax.axhline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('offset (ms)')
    ax.set_title('Timing offsets over chart')
    ax.set_ylim(-80, 80)
    ax.legend(markerscale=3)
    ctx.figure(fig)


def register_components(add):
    add(MANIFEST, _draw)
