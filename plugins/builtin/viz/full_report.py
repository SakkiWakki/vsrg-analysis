"""Combined summary grid with plot selection."""
import math

import numpy as np

from analysis.components import Manifest, SURFACE_VIZ, VizFields
from analysis.components.viz_backend import (
    LEFT_CLR,
    RIGHT_CLR,
    VIZ_CATEGORY_WIDGET,
    MS,
    col_colors,
    new_figure,
)


MANIFEST = Manifest(
    key='builtin:viz:full_report',
    name='Full report (all plots)',
    supported_surfaces={SURFACE_VIZ},
    plugin_fields={'viz': VizFields(category=VIZ_CATEGORY_WIDGET)},
)

_PLOTS = [
    ('scatter', 'Timing scatter'),
    ('dist', 'Offset distribution'),
    ('per_col', 'Per-column mean'),
    ('chord', 'Chord sizes'),
    ('drift', 'Per-hand drift'),
    ('coupling', 'Coupling'),
    ('rolling', 'Rolling stability'),
]
_DEFAULT_SELECTION = [key for key, _ in _PLOTS]


def _draw(ctx):
    replay = ctx.replay
    analysis = ctx.analysis
    rows = replay.noterows_clean()
    offs = replay.offsets_clean()
    cols = replay.columns_clean()
    keycount = replay.keycount()

    def build_figure(selection):
        chosen = [key for key in selection if key in {key for key, _ in _PLOTS}]
        if not chosen:
            fig, _ax = new_figure(8, 4)
            fig.text(0.5, 0.5, 'no plots selected', ha='center', va='center')
            return fig

        has_scatter = 'scatter' in chosen
        others = [key for key in chosen if key != 'scatter']
        ncols = 3
        rows_other = int(math.ceil(len(others) / ncols))
        total_rows = (1 if has_scatter else 0) + rows_other
        fig_height = (3.0 if has_scatter else 0.0) + (2.2 * rows_other)
        from matplotlib.figure import Figure

        fig = Figure(figsize=(16, max(4, fig_height)), constrained_layout=True)
        fig.patch.set_facecolor('#121212')
        grid = fig.add_gridspec(max(1, total_rows), ncols)

        row_cursor = 0
        if has_scatter:
            _plot_scatter(fig.add_subplot(grid[row_cursor, :]),
                          rows, offs, cols, keycount)
            row_cursor += 1
        for index, key in enumerate(others):
            row = row_cursor + (index // ncols)
            col = index % ncols
            ax = fig.add_subplot(grid[row, col])
            _dispatch_plot(key, ax, rows, offs, cols, keycount, analysis)
        return fig

    ctx.widget(ctx.build_selectable_figure_widget(
        options=_PLOTS,
        default_selection=_DEFAULT_SELECTION,
        build_figure=build_figure,
    ))


def _dispatch_plot(key, ax, rows, offs, cols, keycount, analysis):
    match key:
        case 'scatter':
            _plot_scatter(ax, rows, offs, cols, keycount)
        case 'dist':
            _plot_distribution(ax, offs, cols, keycount, analysis)
        case 'per_col':
            _plot_per_column(ax, cols, offs, analysis)
        case 'chord':
            _plot_chord_sizes(ax, rows, offs, cols, analysis)
        case 'drift':
            _plot_drift(ax, rows, offs, cols, analysis)
        case 'coupling':
            _plot_coupling(ax, rows, offs, cols, analysis)
        case 'rolling':
            _plot_rolling(ax, offs, cols, analysis)


def _plot_distribution(ax, offs, cols, keycount, analysis):
    left_cols, right_cols = analysis.default_hands(keycount)
    left = np.isin(cols, left_cols)
    right = np.isin(cols, right_cols)
    bins = np.linspace(-80, 80, 81)
    ax.hist(offs[left] * MS, bins=bins, alpha=0.65, color=LEFT_CLR, label='Left')
    ax.hist(offs[right] * MS, bins=bins, alpha=0.65, color=RIGHT_CLR, label='Right')
    ax.axvline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_xlabel('offset (ms)  (negative=early, positive=late)')
    ax.set_ylabel('count')
    ax.set_title('Offset distribution by hand')
    ax.legend()


def _plot_scatter(ax, rows, offs, cols, keycount):
    half = keycount // 2
    left_cols = tuple(range(half))
    right_cols = tuple(range(half, keycount))
    left = np.isin(cols, left_cols)
    right = np.isin(cols, right_cols)
    ax.scatter(rows[left], offs[left] * MS, s=4, c=LEFT_CLR,
               alpha=0.45, label='Left')
    ax.scatter(rows[right], offs[right] * MS, s=4, c=RIGHT_CLR,
               alpha=0.45, label='Right')
    ax.axhline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_xlabel('noterow (chart progression)')
    ax.set_ylabel('offset (ms)')
    ax.set_title('Timing offsets over chart')
    ax.set_ylim(-80, 80)
    ax.legend(markerscale=3)


def _plot_per_column(ax, cols, offs, analysis):
    stats = analysis.per_column_stats(cols, offs)
    col_ids = sorted(stats.keys())
    means = [stats[col]['mean'] * MS for col in col_ids]
    stds = [stats[col]['std'] * MS for col in col_ids]
    palette = col_colors((max(col_ids) + 1) if col_ids else 4)
    bars = ax.bar([str(col) for col in col_ids], means, yerr=stds,
                  capsize=6, color=[palette[col] for col in col_ids],
                  edgecolor='w', lw=0.5)
    ax.axhline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_ylabel('mean offset (ms)  ±std')
    ax.set_xlabel('column')
    ax.set_title('Per-column mean offset')
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + (bar.get_width() / 2), mean,
                f'{mean:+.1f}', ha='center',
                va='bottom' if mean > 0 else 'top',
                fontsize=9 if len(col_ids) <= 6 else 7, color='w')


def _plot_chord_sizes(ax, rows, offs, cols, analysis):
    chord = analysis.chord_vs_single(rows, offs, cols)
    names = ['single', 'jump', 'hand', 'quad']
    means = [chord[name]['mean'] * MS for name in names]
    stds = [chord[name]['std'] * MS for name in names]
    counts = [chord[name]['n'] for name in names]
    ax.bar(names, means, yerr=stds, capsize=6, color='#80cbc4',
           edgecolor='w', lw=0.5)
    for index, (mean, count) in enumerate(zip(means, counts)):
        ax.text(index, mean, f'{mean:+.1f}\nn={count}', ha='center',
                va='bottom' if mean > 0 else 'top', fontsize=9, color='w')
    ax.axhline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_ylabel('mean offset (ms) ± std')
    ax.set_title('Timing by chord size')


def _plot_drift(ax, rows, offs, cols, analysis):
    drift = analysis.timing_drift(rows, offs, cols, n_segments=8)
    xs = [(segment['noterow_lo'] + segment['noterow_hi']) / 2
          for segment in drift['segments']]
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


def _plot_coupling(ax, rows, offs, cols, analysis):
    coupling = analysis.coupling_analysis(rows, offs, cols)
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


def _plot_rolling(ax, offs, cols, analysis):
    rolling = analysis.rolling_stability(offs, cols, window=200)
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


def register_components(add):
    add(MANIFEST, _draw)
