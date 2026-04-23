"""Single/jump/hand/quad timing."""
from analysis.components import Manifest, SURFACE_VIZ, VizFields
from analysis.components.viz_backend import VIZ_CATEGORY_CHART, MS, new_figure


MANIFEST = Manifest(
    key='builtin:viz:chord_sizes',
    name='Chord sizes',
    supported_surfaces={SURFACE_VIZ},
    plugin_fields={'viz': VizFields(category=VIZ_CATEGORY_CHART)},
)


def _draw(ctx):
    rows = ctx.replay.noterows_clean()
    offs = ctx.replay.offsets_clean()
    cols = ctx.replay.columns_clean()
    fig, ax = new_figure(8, 5)
    chord = ctx.analysis.chord_vs_single(rows, offs, cols)
    names = ['single', 'jump', 'hand', 'quad']
    means = [chord[name]['mean'] * MS for name in names]
    stds = [chord[name]['std'] * MS for name in names]
    counts = [chord[name]['n'] for name in names]
    ax.bar(names, means, yerr=stds, capsize=6,
           color='#80cbc4', edgecolor='w', lw=0.5)
    for index, (mean, count) in enumerate(zip(means, counts)):
        ax.text(index, mean, f'{mean:+.1f}\nn={count}',
                ha='center',
                va='bottom' if mean > 0 else 'top',
                fontsize=9,
                color='w')
    ax.axhline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_ylabel('mean offset (ms) ± std')
    ax.set_title('Timing by chord size')
    ctx.figure(fig)


def register_components(add):
    add(MANIFEST, _draw)
