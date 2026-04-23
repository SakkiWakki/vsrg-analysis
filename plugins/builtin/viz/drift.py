"""Per-hand timing drift across chart segments."""
from analysis.components import Manifest, SURFACE_VIZ, VizFields
from analysis.components.viz_backend import (
    LEFT_CLR,
    RIGHT_CLR,
    VIZ_CATEGORY_CHART,
    MS,
    new_figure,
)


MANIFEST = Manifest(
    key='builtin:viz:drift',
    name='Drift (hands × time)',
    supported_surfaces={SURFACE_VIZ},
    plugin_fields={'viz': VizFields(category=VIZ_CATEGORY_CHART)},
)


def _draw(ctx):
    rows = ctx.replay.noterows_clean()
    offs = ctx.replay.offsets_clean()
    cols = ctx.replay.columns_clean()
    fig, ax = new_figure(10, 5)
    drift = ctx.analysis.timing_drift(rows, offs, cols, n_segments=8)
    xs = [
        (segment['noterow_lo'] + segment['noterow_hi']) / 2
        for segment in drift['segments']
    ]
    left_means = [segment['left']['mean'] * MS for segment in drift['segments']]
    right_means = [segment['right']['mean'] * MS for segment in drift['segments']]
    left_std = [segment['left']['std'] * MS for segment in drift['segments']]
    right_std = [segment['right']['std'] * MS for segment in drift['segments']]
    ax.errorbar(xs, left_means, yerr=left_std, color=LEFT_CLR, lw=2,
                marker='o', capsize=4, label='Left')
    ax.errorbar(xs, right_means, yerr=right_std, color=RIGHT_CLR, lw=2,
                marker='s', capsize=4, label='Right')
    ax.axhline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_xlabel('noterow (chart segments)')
    ax.set_ylabel('mean offset (ms) ± std')
    ax.set_title('Per-hand drift across chart (8 segments)')
    ax.legend()
    ctx.figure(fig)


def register_components(add):
    add(MANIFEST, _draw)
