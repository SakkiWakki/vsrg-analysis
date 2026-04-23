"""Per-column mean offset bars."""
from analysis.components import Manifest, SURFACE_VIZ, VizFields
from analysis.components.viz_backend import (
    VIZ_CATEGORY_CHART,
    MS,
    col_colors,
    new_figure,
)


MANIFEST = Manifest(
    key='builtin:viz:per_column',
    name='Per-column mean offset',
    supported_surfaces={SURFACE_VIZ},
    plugin_fields={'viz': VizFields(category=VIZ_CATEGORY_CHART)},
)


def _draw(ctx):
    offs = ctx.replay.offsets_clean()
    cols = ctx.replay.columns_clean()
    fig, ax = new_figure(8, 5)
    stats = ctx.analysis.per_column_stats(cols, offs)
    col_ids = sorted(stats.keys())
    means = [stats[col]['mean'] * MS for col in col_ids]
    stds = [stats[col]['std'] * MS for col in col_ids]
    palette = col_colors((max(col_ids) + 1) if col_ids else 4)
    bars = ax.bar(
        [str(col) for col in col_ids],
        means,
        yerr=stds,
        capsize=6,
        color=[palette[col] for col in col_ids],
        edgecolor='w',
        lw=0.5,
    )
    ax.axhline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_ylabel('mean offset (ms)  ±std')
    ax.set_xlabel('column')
    ax.set_title('Per-column mean offset')
    fontsize = 9 if len(col_ids) <= 6 else 7
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + (bar.get_width() / 2), mean,
                f'{mean:+.1f}', ha='center',
                va='bottom' if mean > 0 else 'top',
                fontsize=fontsize, color='w')
    ctx.figure(fig)


def register_components(add):
    add(MANIFEST, _draw)
