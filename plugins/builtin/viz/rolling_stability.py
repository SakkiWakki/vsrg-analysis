"""Rolling std of offsets."""
from analysis.components import Manifest, SURFACE_VIZ, VizFields
from analysis.components.viz_backend import (
    LEFT_CLR,
    RIGHT_CLR,
    VIZ_CATEGORY_CHART,
    MS,
    new_figure,
)


MANIFEST = Manifest(
    key='builtin:viz:rolling_stability',
    name='Rolling stability',
    supported_surfaces={SURFACE_VIZ},
    plugin_fields={'viz': VizFields(category=VIZ_CATEGORY_CHART)},
)


def _draw(ctx):
    import numpy as np

    offs = ctx.replay.offsets_clean()
    cols = ctx.replay.columns_clean()
    fig, ax = new_figure(10, 4.5)
    rolling = ctx.analysis.rolling_stability(offs, cols, window=200)
    xs = rolling['centers']
    ax.plot(xs, np.array(rolling['std_all']) * MS, color='#eceff1',
            lw=1.4, label='all')
    ax.plot(xs, np.array(rolling['std_left']) * MS, color=LEFT_CLR,
            lw=1.4, label='left')
    ax.plot(xs, np.array(rolling['std_right']) * MS, color=RIGHT_CLR,
            lw=1.4, label='right')
    ax.set_xlabel('note index (chart progression)')
    ax.set_ylabel('rolling std (ms)')
    ax.set_title(f'Rolling timing stability (window={rolling["window"]})')
    ax.legend()
    ctx.figure(fig)


def register_components(add):
    add(MANIFEST, _draw)
