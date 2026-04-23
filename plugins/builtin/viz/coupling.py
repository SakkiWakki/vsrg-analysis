"""Solo vs same-hand-paired column timing."""
from analysis.components import Manifest, SURFACE_VIZ, VizFields
from analysis.components.viz_backend import VIZ_CATEGORY_CHART, MS, new_figure


MANIFEST = Manifest(
    key='builtin:viz:coupling',
    name='Coupling (solo vs paired)',
    supported_surfaces={SURFACE_VIZ},
    plugin_fields={'viz': VizFields(category=VIZ_CATEGORY_CHART)},
)


def _draw(ctx):
    import numpy as np

    rows = ctx.replay.noterows_clean()
    offs = ctx.replay.offsets_clean()
    cols = ctx.replay.columns_clean()
    fig, ax = new_figure(9, 5)
    coupling = ctx.analysis.coupling_analysis(rows, offs, cols)
    col_ids = sorted(coupling.keys())
    width = 0.38
    xs = np.arange(len(col_ids))
    solo_means = [coupling[col]['solo']['mean'] * MS for col in col_ids]
    pair_means = [coupling[col]['paired']['mean'] * MS for col in col_ids]
    solo_std = [coupling[col]['solo']['std'] * MS for col in col_ids]
    pair_std = [coupling[col]['paired']['std'] * MS for col in col_ids]
    ax.bar(xs - (width / 2), solo_means, width, yerr=solo_std, capsize=4,
           color='#9ccc65', edgecolor='w', lw=0.5, label='solo')
    ax.bar(xs + (width / 2), pair_means, width, yerr=pair_std, capsize=4,
           color='#ba68c8', edgecolor='w', lw=0.5,
           label='paired (same-hand partner)')
    ax.set_xticks(xs)
    labels = [f'c{col}' for col in col_ids] if len(col_ids) > 6 else [
        f'col {col}' for col in col_ids
    ]
    ax.set_xticklabels(labels, rotation=0 if len(col_ids) <= 8 else 45,
                       ha='center')
    ax.axhline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_ylabel('mean offset (ms) ± std')
    ax.set_title('Coupling: solo vs paired with same-hand neighbor')
    ax.legend()
    ctx.figure(fig)


def register_components(add):
    add(MANIFEST, _draw)
