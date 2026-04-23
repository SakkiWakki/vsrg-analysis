"""Matplotlib visualizations for Etterna replay analysis."""
import sys
import json
import base64
import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from analysis.games.etterna.replay import parse_replay, clean_offsets, summary
from analysis.core.timing import (
    hand_split, per_column_stats, timing_drift,
    coupling_analysis, chord_vs_single, rolling_stability, full_analysis,
    default_hands,
)

plt.style.use('dark_background')

LEFT_CLR = '#4fc3f7'
RIGHT_CLR = '#ff8a65'
MS = 1000.0


def _new_fig(figsize):
    """Thread-safe figure factory. Returns (fig, ax). Uses Figure() directly
    instead of plt.subplots so calls from background threads don't get a new
    OS-level window from pyplot's figure manager."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize)
    fig.patch.set_facecolor('#121212')
    ax = fig.add_subplot(111)
    return fig, ax


def col_colors(keycount):
    """Gradient from left-blue to right-orange, per column."""
    if keycount <= 1:
        return ['#80deea']
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('hand', [LEFT_CLR, '#ffffff', RIGHT_CLR])
    return [matplotlib.colors.to_hex(cmap(i / (keycount - 1))) for i in range(keycount)]


def _hand_masks(columns, left_cols=None, right_cols=None, keycount=None):
    if left_cols is None or right_cols is None:
        if keycount is None:
            keycount = int(columns.max()) + 1 if len(columns) else 4
        left_cols, right_cols = default_hands(keycount)
    return np.isin(columns, left_cols), np.isin(columns, right_cols)


def plot_offset_distribution(offsets, columns, ax=None, save_path=None):
    close = ax is None
    if ax is None:
        fig, ax = _new_fig((9, 5))
    left, right = _hand_masks(columns)
    bins = np.linspace(-0.08, 0.08, 81)
    ax.hist(offsets[left] * MS, bins=bins * MS, alpha=0.65, color=LEFT_CLR, label='Left')
    ax.hist(offsets[right] * MS, bins=bins * MS, alpha=0.65, color=RIGHT_CLR, label='Right')
    ax.axvline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_xlabel('offset (ms)  (negative=early, positive=late)')
    ax.set_ylabel('count')
    ax.set_title('Offset distribution by hand')
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    if close:
        plt.show()
    return ax


def plot_scatter_timeline(noterows, offsets, columns, ax=None, save_path=None,
                          xlabel='noterow (chart progression)'):
    close = ax is None
    if ax is None:
        fig, ax = _new_fig((11, 4.5))
    left, right = _hand_masks(columns)
    ax.scatter(noterows[left], offsets[left] * MS, s=4, c=LEFT_CLR, alpha=0.45, label='Left')
    ax.scatter(noterows[right], offsets[right] * MS, s=4, c=RIGHT_CLR, alpha=0.45, label='Right')
    ax.axhline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('offset (ms)')
    ax.set_title('Timing offsets over chart')
    ax.set_ylim(-80, 80)
    ax.legend(markerscale=3)
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    if close:
        plt.show()
    return ax


def plot_per_column(columns, offsets, ax=None, save_path=None):
    close = ax is None
    if ax is None:
        fig, ax = _new_fig((7, 4.5))
    stats = per_column_stats(columns, offsets)
    cols = sorted(stats.keys())
    keycount = (max(cols) + 1) if cols else 4
    means = [stats[c]['mean'] * MS for c in cols]
    stds = [stats[c]['std'] * MS for c in cols]
    palette = col_colors(keycount)
    clrs = [palette[c] for c in cols]
    bars = ax.bar([str(c) for c in cols], means, yerr=stds, capsize=6,
                  color=clrs, edgecolor='w', lw=0.5)
    ax.axhline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_ylabel('mean offset (ms)  ±std')
    ax.set_xlabel('column')
    ax.set_title('Per-column mean offset')
    fontsize = 9 if len(cols) <= 6 else 7
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m,
                f'{m:+.1f}', ha='center', va='bottom' if m > 0 else 'top',
                fontsize=fontsize, color='w')
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    if close:
        plt.show()
    return ax


def plot_drift(noterows, offsets, columns, ax=None, save_path=None, n_segments=8):
    close = ax is None
    if ax is None:
        fig, ax = _new_fig((9, 4.5))
    drift = timing_drift(noterows, offsets, columns, n_segments=n_segments)
    xs = [(s['noterow_lo'] + s['noterow_hi']) / 2 for s in drift['segments']]
    l_means = [s['left']['mean'] * MS for s in drift['segments']]
    r_means = [s['right']['mean'] * MS for s in drift['segments']]
    l_std = [s['left']['std'] * MS for s in drift['segments']]
    r_std = [s['right']['std'] * MS for s in drift['segments']]
    ax.errorbar(xs, l_means, yerr=l_std, color=LEFT_CLR, lw=2,
                marker='o', capsize=4, label='Left')
    ax.errorbar(xs, r_means, yerr=r_std, color=RIGHT_CLR, lw=2,
                marker='s', capsize=4, label='Right')
    ax.axhline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_xlabel('noterow (chart segments)')
    ax.set_ylabel('mean offset (ms) ± std')
    ax.set_title(f'Per-hand drift across chart ({n_segments} segments)')
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    if close:
        plt.show()
    return ax


def plot_coupling(noterows, offsets, columns, ax=None, save_path=None):
    close = ax is None
    if ax is None:
        fig, ax = _new_fig((8, 4.5))
    cpl = coupling_analysis(noterows, offsets, columns)
    cols = sorted(cpl.keys())
    width = 0.38
    x = np.arange(len(cols))
    solo_m = [cpl[c]['solo']['mean'] * MS for c in cols]
    pair_m = [cpl[c]['paired']['mean'] * MS for c in cols]
    solo_s = [cpl[c]['solo']['std'] * MS for c in cols]
    pair_s = [cpl[c]['paired']['std'] * MS for c in cols]
    ax.bar(x - width / 2, solo_m, width, yerr=solo_s, capsize=4,
           color='#9ccc65', edgecolor='w', lw=0.5, label='solo')
    ax.bar(x + width / 2, pair_m, width, yerr=pair_s, capsize=4,
           color='#ba68c8', edgecolor='w', lw=0.5, label='paired (same-hand partner)')
    ax.set_xticks(x)
    labels = [f'c{c}' for c in cols] if len(cols) > 6 else [f'col {c}' for c in cols]
    ax.set_xticklabels(labels, rotation=0 if len(cols) <= 8 else 45, ha='center')
    ax.axhline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_ylabel('mean offset (ms) ± std')
    ax.set_title('Coupling: solo vs paired with same-hand neighbor')
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    if close:
        plt.show()
    return ax


def plot_rolling(offsets, columns, ax=None, save_path=None, window=200):
    close = ax is None
    if ax is None:
        fig, ax = _new_fig((9, 4))
    r = rolling_stability(offsets, columns, window=window)
    xs = r['centers']
    ax.plot(xs, np.array(r['std_all']) * MS, color='#eceff1', lw=1.4, label='all')
    ax.plot(xs, np.array(r['std_left']) * MS, color=LEFT_CLR, lw=1.4, label='left')
    ax.plot(xs, np.array(r['std_right']) * MS, color=RIGHT_CLR, lw=1.4, label='right')
    ax.set_xlabel('note index (chart progression)')
    ax.set_ylabel('rolling std (ms)')
    ax.set_title(f'Rolling timing stability (window={r["window"]})')
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    if close:
        plt.show()
    return ax


def plot_chord_sizes(noterows, offsets, columns, ax=None, save_path=None):
    close = ax is None
    if ax is None:
        fig, ax = _new_fig((7, 4.5))
    cvs = chord_vs_single(noterows, offsets, columns)
    names = ['single', 'jump', 'hand', 'quad']
    means = [cvs[n]['mean'] * MS for n in names]
    stds = [cvs[n]['std'] * MS for n in names]
    ns = [cvs[n]['n'] for n in names]
    ax.bar(names, means, yerr=stds, capsize=6, color='#80cbc4',
           edgecolor='w', lw=0.5)
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m, f'{m:+.1f}\nn={n}', ha='center',
                va='bottom' if m > 0 else 'top', fontsize=9, color='w')
    ax.axhline(0, color='w', lw=0.5, alpha=0.4)
    ax.set_ylabel('mean offset (ms) ± std')
    ax.set_title('Timing by chord size')
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    if close:
        plt.show()
    return ax


FULL_REPORT_PLOTS = [
    ('scatter',  'Timing scatter'),
    ('dist',     'Offset distribution'),
    ('per_col',  'Per-column mean'),
    ('chord',    'Chord sizes'),
    ('drift',    'Per-hand drift'),
    ('coupling', 'Coupling'),
    ('rolling',  'Rolling stability'),
]
FULL_REPORT_DEFAULT_SELECTION = [k for k, _ in FULL_REPORT_PLOTS]


def _plot_full_dispatch(key, ax, rows, offs, cols):
    if key == 'scatter':  plot_scatter_timeline(rows, offs, cols, ax=ax)
    elif key == 'dist':     plot_offset_distribution(offs, cols, ax=ax)
    elif key == 'per_col':  plot_per_column(cols, offs, ax=ax)
    elif key == 'chord':    plot_chord_sizes(rows, offs, cols, ax=ax)
    elif key == 'drift':    plot_drift(rows, offs, cols, ax=ax)
    elif key == 'coupling': plot_coupling(rows, offs, cols, ax=ax)
    elif key == 'rolling':  plot_rolling(offs, cols, ax=ax)


def plot_full_report(replay_data, save_path=None, show=True, title=None,
                     selection=None):
    """Grid of plots. `selection` is a list of keys from FULL_REPORT_PLOTS;
    layout adapts: if 'scatter' is selected it spans the full top row, and the
    remaining plots pack into a 3-wide grid below it."""
    clean = clean_offsets(replay_data)
    offs = clean['offsets']
    cols = clean['columns']
    rows = clean['noterows']

    if selection is None:
        selection = list(FULL_REPORT_DEFAULT_SELECTION)
    selection = [k for k in selection if k in {k for k, _ in FULL_REPORT_PLOTS}]
    if not selection:
        from matplotlib.figure import Figure; fig = Figure(figsize=(8, 4)); fig.patch.set_facecolor("#121212")
        fig.text(0.5, 0.5, 'no plots selected', ha='center', va='center')
        return fig

    has_scatter = 'scatter' in selection
    others = [k for k in selection if k != 'scatter']
    ncols = 3
    nrows_others = (len(others) + ncols - 1) // ncols
    total_rows = (1 if has_scatter else 0) + nrows_others

    # Keep the default (scatter + 6 others = 3 rows) at ~7.5in tall so it fits
    # a typical viewport without scrolling. Scatter row gets a bit more height
    # since it spans full width.
    row_h_others = 2.2
    row_h_scatter = 3.0
    fig_h = row_h_others * nrows_others + (row_h_scatter if has_scatter else 0)
    from matplotlib.figure import Figure
    fig = Figure(figsize=(16, max(4, fig_h)), constrained_layout=True)
    fig.patch.set_facecolor("#121212")
    gs = fig.add_gridspec(max(1, total_rows), ncols)

    row_cursor = 0
    if has_scatter:
        _plot_full_dispatch('scatter', fig.add_subplot(gs[row_cursor, :]),
                            rows, offs, cols)
        row_cursor += 1
    for i, key in enumerate(others):
        r = row_cursor + i // ncols
        c = i % ncols
        _plot_full_dispatch(key, fig.add_subplot(gs[r, c]), rows, offs, cols)

    if title:
        fig.suptitle(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def generate_html_report(replay_data, score_meta=None, output_path='report.html'):
    clean = clean_offsets(replay_data)
    offs, cols, rows = clean['offsets'], clean['columns'], clean['noterows']
    res = full_analysis(replay_data)
    summ = summary(replay_data)

    imgs = {}
    for name, func in (
        ('scatter', lambda a: plot_scatter_timeline(rows, offs, cols, ax=a)),
        ('dist', lambda a: plot_offset_distribution(offs, cols, ax=a)),
        ('col', lambda a: plot_per_column(cols, offs, ax=a)),
        ('drift', lambda a: plot_drift(rows, offs, cols, ax=a)),
        ('coupling', lambda a: plot_coupling(rows, offs, cols, ax=a)),
        ('rolling', lambda a: plot_rolling(offs, cols, ax=a)),
        ('chord', lambda a: plot_chord_sizes(rows, offs, cols, ax=a)),
    ):
        fig, ax = _new_fig((10, 4.5))
        func(ax)
        imgs[name] = _fig_to_base64(fig)

    meta_rows = ''
    if score_meta:
        for k in ('song', 'pack', 'steps', 'rate', 'grade',
                  'ssrnormpercent', 'wifescore', 'maxcombo', 'datetime'):
            if k in score_meta:
                meta_rows += f"<tr><td>{k}</td><td>{score_meta[k]}</td></tr>"

    hand = res['hand_split']
    stat_rows = f"""
      <tr><td>total notes</td><td>{summ['total_notes']}</td></tr>
      <tr><td>hits / misses</td><td>{summ['hits']} / {summ['misses']}</td></tr>
      <tr><td>mean offset</td><td>{summ['mean_offset_ms']:+.2f} ms</td></tr>
      <tr><td>median offset</td><td>{summ['median_offset_ms']:+.2f} ms</td></tr>
      <tr><td>std</td><td>{summ['std_offset_ms']:.2f} ms</td></tr>
      <tr><td>|mean|</td><td>{summ['abs_mean_ms']:.2f} ms</td></tr>
      <tr><td>Left hand</td><td>mean {hand['left']['mean']*MS:+.2f} ms, std {hand['left']['std']*MS:.2f} ms, n={hand['left']['n']}</td></tr>
      <tr><td>Right hand</td><td>mean {hand['right']['mean']*MS:+.2f} ms, std {hand['right']['std']*MS:.2f} ms, n={hand['right']['n']}</td></tr>
    """

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Etterna Replay Report</title>
<style>
body {{ background: #121212; color: #e0e0e0; font: 14px/1.4 -apple-system,sans-serif; margin: 30px; }}
h1, h2 {{ color: #ffab91; }}
table {{ border-collapse: collapse; margin: 10px 0 20px; }}
td {{ padding: 4px 12px; border-bottom: 1px solid #333; }}
td:first-child {{ color: #80deea; font-weight: bold; }}
img {{ max-width: 100%; margin: 10px 0; border-radius: 4px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
</style></head><body>
<h1>Etterna Replay Report</h1>
<h2>Score</h2>
<table>{meta_rows}</table>
<h2>Overall</h2>
<table>{stat_rows}</table>
<h2>Timing over chart</h2>
<img src="data:image/png;base64,{imgs['scatter']}">
<div class="grid">
  <div><h2>Offset distribution</h2><img src="data:image/png;base64,{imgs['dist']}"></div>
  <div><h2>Per-column</h2><img src="data:image/png;base64,{imgs['col']}"></div>
  <div><h2>Drift</h2><img src="data:image/png;base64,{imgs['drift']}"></div>
  <div><h2>Coupling</h2><img src="data:image/png;base64,{imgs['coupling']}"></div>
  <div><h2>Rolling stability</h2><img src="data:image/png;base64,{imgs['rolling']}"></div>
  <div><h2>Chord sizes</h2><img src="data:image/png;base64,{imgs['chord']}"></div>
</div>
</body></html>"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: etterna_viz.py <replay_file> [--html out.html]")
        sys.exit(1)
    rep = parse_replay(sys.argv[1])
    if '--html' in sys.argv:
        i = sys.argv.index('--html')
        out = sys.argv[i + 1] if i + 1 < len(sys.argv) else 'report.html'
        path = generate_html_report(rep, output_path=out)
        print(f"wrote {path}")
    else:
        plot_full_report(rep, save_path='report.png', show=True,
                         title=f"Replay: {rep['filepath'].split('/')[-1]}")
