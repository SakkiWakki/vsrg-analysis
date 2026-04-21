"""Quaver / Mania-Replay-Master style note visualizer.
For each note: draws the note, the player's press (offset from note), and judgment color.
Hit windows are shaded behind each note per the game's judge/OD.

Works for Etterna and osu!mania (any keycount). Axis units:
  Etterna: noterows (use --rows-per-ms to scale offsets; default reasonable).
  Osu!mania: milliseconds.
"""
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Slider

from analysis.games.etterna.replay import parse_replay as parse_etterna
from analysis.viz.plots import col_colors

plt.style.use('dark_background')


from analysis.games.etterna.judgment import (
    ETT_JUDGE_SCALES, windows_for as _ett_windows_for)

# Per-window color palette (viz only — the player uses JCLR from
# analysis.player.judgment for its hit marks).
_ETT_WCOLOR = {'marv': '#ffffff', 'perf': '#ffd54f', 'great': '#81c784',
               'good': '#4fc3f7', 'bad': '#ba68c8'}
MISS_CLR = '#e53935'


def etterna_windows(judge='J4', scale=None):
    """Hit windows (name, half-window seconds, color). Pass either a judge
    name ('J1'..'J8', 'J9'/'JUSTICE') or a numeric scale. Delegates the
    scale table + Bad-clamp to analysis.games.etterna.judgment so the
    viz layer and the player can't drift apart."""
    if scale is not None:
        # Numeric override: apply flat scale to J4 taps without the
        # Bad-clamp — matches the legacy behavior this argument enabled.
        base = [('marv', 0.0225), ('perf', 0.045), ('great', 0.090),
                ('good', 0.135), ('bad', 0.180)]
        return [(n, w * scale, _ETT_WCOLOR[n]) for (n, w) in base]
    return [(n, w, _ETT_WCOLOR[n]) for (n, w) in _ett_windows_for(judge)]


_OSU_LEGACY_NAMES = {'marv': '300g', 'perf': '300', 'great': '200',
                     'good': '100', 'bad': '50'}
_OSU_WCOLOR = {'300g': '#ffffff', '300': '#ffd54f', '200': '#81c784',
               '100': '#4fc3f7', '50': '#ba68c8'}


def osu_mania_windows(od=8):
    """Hit windows for osu!mania based on OD (ms). Delegates the formula
    to analysis.games.osu.judgment; keeps the legacy '300g'/'300'/…
    labels the plotting code already matches on."""
    from analysis.games.osu.judgment import windows_for
    out = []
    for name, sec in windows_for(od):
        legacy = _OSU_LEGACY_NAMES[name]
        out.append((legacy, sec, _OSU_WCOLOR[legacy]))
    return out


# osu! mod bit values
OSU_MOD_EASY = 1 << 1
OSU_MOD_HARDROCK = 1 << 4
OSU_MOD_DOUBLETIME = 1 << 6
OSU_MOD_HALFTIME = 1 << 8


def effective_osu_od(base_od, mods=0):
    """Apply HR (×1.4) / EZ (×0.5) mod to OD. DT/HT don't change OD values
    themselves in stable, though they stretch note spacing — we ignore that
    here since we display windows in absolute ms."""
    od = float(base_od)
    if mods & OSU_MOD_HARDROCK:
        od = min(10.0, od * 1.4)
    if mods & OSU_MOD_EASY:
        od = od * 0.5
    return od


def judge_for_offset(offset_s, windows, is_miss):
    if is_miss:
        return 'miss', MISS_CLR
    a = abs(offset_s)
    for name, w, clr in windows:
        if a <= w:
            return name, clr
    return 'miss', MISS_CLR


def render_chart(replay, window_units, start, ax,
                 windows=None, unit_label='noterow',
                 rows_per_ms=None, show_windows=True,
                 show_presses=True, title=None):
    """Render a window of the chart with notes, hit windows, and press markers.
    window_units: how many units (rows for Etterna, ms for osu) to show.
    start: starting position.
    windows: list of (name, half_window_seconds, color).
    rows_per_ms: if set (Etterna), convert offset seconds to row units for visual display.
                 None means offsets are already in the same units as noterows (osu=ms).
    """
    noterows = replay['noterows']
    columns = replay['columns']
    offsets = replay['offsets']
    misses = replay['misses']
    notetypes = replay['notetypes']
    holds_meta = replay.get('holds', [])  # Etterna: (row, col) drops; osu: (start, col, end)
    keycount = replay.get('keycount') or (int(columns.max()) + 1 if len(columns) else 4)
    palette = col_colors(keycount)

    end = start + window_units
    sel = (noterows >= start - 200) & (noterows < end + 200)
    r = noterows[sel]
    c = columns[sel]
    o = offsets[sel]
    mi = misses[sel]
    nt = notetypes[sel]

    lane_w = 1.0
    note_h = max(window_units * 0.008, 4 if rows_per_ms else 8)

    # Lane backgrounds
    for col in range(keycount):
        ax.axvspan(col * lane_w, (col + 1) * lane_w, color='#161616', zorder=0)

    # Hold bodies (osu: we have end times; Etterna: only drop flags, draw head-only)
    dropped_holds = set()
    hold_bodies = []  # (start, col, end)
    for h in holds_meta:
        if len(h) >= 3 and h[2] is not None:
            hold_bodies.append((h[0], h[1], h[2]))
        elif len(h) == 2:
            dropped_holds.add((int(h[0]), int(h[1])))
    for (hstart, hcol, hend) in hold_bodies:
        if hcol >= keycount:
            continue
        if hend < start or hstart >= end:
            continue
        y0 = max(hstart, start - 200)
        y1 = min(hend, end + 200)
        ax.add_patch(Rectangle(
            (hcol * lane_w + 0.18, y0),
            lane_w - 0.36, y1 - y0,
            facecolor=palette[hcol], alpha=0.35,
            edgecolor=palette[hcol], linewidth=0.6, zorder=3))

    # Hit window bands behind each note
    if show_windows and windows:
        for nr, cc, off, miss in zip(r, c, o, mi):
            if cc >= keycount:
                continue
            for name, w_s, clr in reversed(windows):
                if rows_per_ms is not None:
                    w_units = w_s * 1000 * rows_per_ms
                else:
                    w_units = w_s * 1000
                rect = Rectangle(
                    (cc * lane_w + 0.04, nr - w_units),
                    lane_w - 0.08, 2 * w_units,
                    color=clr, alpha=0.07, zorder=1)
                ax.add_patch(rect)

    # Lane separators
    for col in range(keycount + 1):
        ax.plot([col * lane_w, col * lane_w], [start, end],
                color='#2a2a2a', lw=0.5, zorder=2)

    # Notes (rectangles at their scheduled positions; hold heads thicker)
    note_patches, note_clrs, note_lws = [], [], []
    for nr, cc, note_t in zip(r, c, nt):
        if cc >= keycount:
            continue
        is_head = int(note_t) == 2
        nh = note_h * (1.6 if is_head else 1.0)
        note_patches.append(Rectangle(
            (cc * lane_w + 0.12, nr - nh / 2),
            lane_w - 0.24, nh))
        note_clrs.append(palette[cc])
        note_lws.append(1.2 if is_head else 0.5)
    if note_patches:
        pc = PatchCollection(note_patches, facecolors=note_clrs,
                             edgecolors='white', linewidths=note_lws, zorder=4)
        ax.add_collection(pc)

    # Dropped-hold markers (Etterna): X over the head
    for (dr, dc) in dropped_holds:
        if dc >= keycount or dr < start - 200 or dr > end + 200:
            continue
        ax.scatter([dc * lane_w + 0.5], [dr], marker='x',
                   s=60, c=MISS_CLR, linewidths=2, zorder=8)

    # Press markers (horizontal lines at note_pos + offset)
    if show_presses:
        for nr, cc, off, miss in zip(r, c, o, mi):
            if cc >= keycount:
                continue
            name, clr = judge_for_offset(off, windows or [], miss)
            cx_lo = cc * lane_w + 0.20
            cx_hi = (cc + 1) * lane_w - 0.20
            if miss:
                ax.plot([cx_lo, cx_hi], [nr, nr],
                        color=MISS_CLR, lw=1.6, alpha=0.85, zorder=6,
                        solid_capstyle='round')
                ax.plot([cc * lane_w + 0.5, (cc + 1) * lane_w - 0.2],
                        [nr - note_h * 2, nr + note_h * 2],
                        color=MISS_CLR, lw=0.8, alpha=0.5, zorder=6)
            else:
                if rows_per_ms is not None:
                    dy = off * 1000 * rows_per_ms
                else:
                    dy = off * 1000
                press_pos = nr + dy
                # Link line from note to press
                ax.plot([cc * lane_w + 0.5, cc * lane_w + 0.5],
                        [nr, press_pos], color=clr, lw=1.0, alpha=0.75, zorder=5)
                # Press marker
                ax.plot([cx_lo, cx_hi], [press_pos, press_pos],
                        color=clr, lw=2.0, zorder=7, solid_capstyle='round')

    ax.set_xlim(-0.05, keycount * lane_w + 0.05)
    ax.set_ylim(end, start)  # inverted so notes fall down
    ax.set_xticks([(i + 0.5) * lane_w for i in range(keycount)])
    ax.set_xticklabels([str(i) for i in range(keycount)])
    ax.set_ylabel(unit_label)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def _judgment_counts(replay, windows):
    offsets = replay['offsets']
    misses = replay['misses']
    counts = {name: 0 for name, _, _ in windows}
    counts['miss'] = 0
    for off, mi in zip(offsets, misses):
        name, _ = judge_for_offset(off, windows, mi)
        counts[name] = counts.get(name, 0) + 1
    return counts


def _legend_axes(ax, windows, counts=None):
    ax.axis('off')
    y = 0.95
    ax.text(0.02, y, 'Judgments', fontsize=11, color='w', weight='bold',
            transform=ax.transAxes)
    y -= 0.06
    for name, w_s, clr in windows:
        label = f'{name}  ±{w_s*1000:.1f}ms'
        if counts is not None:
            label += f'   n={counts.get(name, 0)}'
        ax.plot([0.04, 0.10], [y, y], color=clr, lw=4, transform=ax.transAxes)
        ax.text(0.14, y - 0.005, label, fontsize=9, color=clr,
                transform=ax.transAxes)
        y -= 0.055
    # miss
    ax.plot([0.04, 0.10], [y, y], color=MISS_CLR, lw=4, transform=ax.transAxes)
    miss_label = 'miss'
    if counts is not None:
        miss_label += f'   n={counts.get("miss", 0)}'
    ax.text(0.14, y - 0.005, miss_label, fontsize=9, color=MISS_CLR,
            transform=ax.transAxes)


def render_chart_full(replay, save_path='chart.png', rows_per_panel=None,
                      game='etterna', od=8, rows_per_ms=None, show=True,
                      panels_per_row=6, title=None):
    """Render the full chart as a grid of panels."""
    noterows = replay['noterows']
    if len(noterows) == 0:
        print("empty replay")
        return

    if game == 'osu':
        windows = osu_mania_windows(od=od)
        unit_label = 'time (ms)'
        rpm = None
    else:
        windows = etterna_windows('J4')
        unit_label = 'noterow'
        rpm = rows_per_ms if rows_per_ms is not None else 0.37

    if rows_per_panel is None:
        rows_per_panel = 8000 if game == 'osu' else 2400

    total = int(noterows.max()) + 1
    n_panels = (total + rows_per_panel - 1) // rows_per_panel
    keycount = replay.get('keycount') or int(replay['columns'].max()) + 1

    cols = min(panels_per_row, n_panels)
    rows = (n_panels + cols - 1) // cols + 1  # +1 for legend row

    fig = plt.figure(figsize=(cols * (keycount * 0.9 + 0.5), rows * 7))
    gs = fig.add_gridspec(rows, cols, hspace=0.28, wspace=0.25)

    counts = _judgment_counts(replay, windows)

    for i in range(n_panels):
        rr = i // cols
        cc = i % cols
        ax = fig.add_subplot(gs[rr, cc])
        st = i * rows_per_panel
        render_chart(replay, window_units=rows_per_panel, start=st, ax=ax,
                     windows=windows, unit_label=unit_label if cc == 0 else '',
                     rows_per_ms=rpm,
                     title=f'{st}–{st + rows_per_panel}')

    legend_ax = fig.add_subplot(gs[-1, :])
    _legend_axes(legend_ax, windows, counts)

    if title:
        fig.suptitle(title, fontsize=13, y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches='tight')
        print(f"saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def interactive(replay, game='etterna', od=8, window_units=None, rows_per_ms=None):
    """Interactive scrollable view."""
    if game == 'osu':
        windows = osu_mania_windows(od=od)
        unit_label = 'time (ms)'
        rpm = None
        win = window_units or 8000
    else:
        windows = etterna_windows('J4')
        unit_label = 'noterow'
        rpm = rows_per_ms if rows_per_ms is not None else 0.37
        win = window_units or 2400

    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(1, 4, width_ratios=[3, 0.05, 1, 0.05], wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    legend = fig.add_subplot(gs[0, 2])

    counts = _judgment_counts(replay, windows)
    _legend_axes(legend, windows, counts)

    total = int(replay['noterows'].max()) + 1 if len(replay['noterows']) else 1000
    render_chart(replay, window_units=win, start=0, ax=ax,
                 windows=windows, unit_label=unit_label, rows_per_ms=rpm)

    plt.subplots_adjust(bottom=0.10)
    ax_slider = plt.axes([0.12, 0.03, 0.65, 0.025])
    slider = Slider(ax_slider, 'pos', 0, max(1, total - win),
                    valinit=0, valstep=max(1, win // 40),
                    color='#ff8a65')

    def update(val):
        ax.clear()
        render_chart(replay, window_units=win, start=int(val), ax=ax,
                     windows=windows, unit_label=unit_label, rows_per_ms=rpm)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print("usage: note_visualizer.py <replay> [--osu chart.osu] [--od N] "
              "[--interactive] [--full] [-o file.png] [--rows-per-ms X]")
        sys.exit(1)
    path = args[0]

    save = None
    if '-o' in args:
        save = args[args.index('-o') + 1]
    od = 8
    if '--od' in args:
        od = float(args[args.index('--od') + 1])
    rpm = None
    if '--rows-per-ms' in args:
        rpm = float(args[args.index('--rows-per-ms') + 1])

    if '--osu' in args or path.endswith('.osr'):
        from analysis.games.osu.replay import parse_replay as parse_osu, find_osu_dirs
        osu_path = args[args.index('--osu') + 1] if '--osu' in args else None
        songs = find_osu_dirs().get('songs_dir')
        rep = parse_osu(path, osu_path=osu_path, songs_dir=songs)
        game = 'osu'
        title = f"{rep['chart_meta'].get('artist','?')} - {rep['chart_meta'].get('title','?')} [{rep['chart_meta'].get('version','')}]  {rep['keycount']}K OD{od}"
    else:
        rep = parse_etterna(path)
        game = 'etterna'
        title = f"Etterna replay: {path.split('/')[-1][:20]}... ({rep.get('keycount') or 'autoK'})"

    if '--interactive' in args:
        interactive(rep, game=game, od=od, rows_per_ms=rpm)
    elif '--full' in args:
        render_chart_full(rep, save_path=save or 'chart_full.png',
                          game=game, od=od, rows_per_ms=rpm, show=True,
                          title=title)
    else:
        win = 8000 if game == 'osu' else 2400
        if game == 'osu':
            windows = osu_mania_windows(od=od)
            unit_label = 'time (ms)'
            rpm_use = None
        else:
            windows = etterna_windows('J4')
            unit_label = 'noterow'
            rpm_use = rpm if rpm is not None else 0.37

        fig = plt.figure(figsize=(10, 11))
        gs = fig.add_gridspec(1, 3, width_ratios=[3, 0.08, 1])
        ax = fig.add_subplot(gs[0, 0])
        legend = fig.add_subplot(gs[0, 2])
        render_chart(rep, window_units=win, start=0, ax=ax,
                     windows=windows, unit_label=unit_label,
                     rows_per_ms=rpm_use, title=title)
        _legend_axes(legend, windows, _judgment_counts(rep, windows))
        if save:
            plt.savefig(save, dpi=120, bbox_inches='tight')
            print(f"saved: {save}")
        plt.show()
