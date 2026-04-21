"""Batch analysis across all scores in an Etterna profile."""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from analysis.games.etterna.replay import (parse_replay, parse_etterna_xml,
                             find_replay_for_score, find_etterna_dirs, clean_offsets)
from analysis.core.timing import hand_split, default_hands

plt.style.use('dark_background')


def _dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def analyze_all_scores(xml_path=None, replays_dir=None, filters=None, verbose=True):
    dirs = find_etterna_dirs()
    xml_path = xml_path or dirs['xml_path']
    replays_dir = replays_dir or dirs['replays_dir']
    if not xml_path or not replays_dir:
        raise FileNotFoundError("cannot locate Etterna.xml / ReplaysV2")

    scores = parse_etterna_xml(xml_path)
    if verbose:
        print(f"parsed {len(scores)} scores from {xml_path}")

    filters = filters or {}
    out = []
    for i, s in enumerate(scores):
        if filters.get('min_wife') and s.get('ssrnormpercent', 0) < filters['min_wife']:
            continue
        if filters.get('min_rate') and s.get('rate', 1.0) < filters['min_rate']:
            continue
        if filters.get('packs') and s.get('pack') not in filters['packs']:
            continue
        if filters.get('since'):
            dt = _dt(s.get('datetime'))
            if dt is None or dt < filters['since']:
                continue

        path = find_replay_for_score(s['scorekey'], replays_dir)
        if not path:
            continue
        try:
            rep = parse_replay(path)
        except Exception as e:
            if verbose:
                print(f"  skip {s['scorekey']}: {e}")
            continue
        clean = clean_offsets(rep)
        if len(clean['offsets']) < 50:
            continue
        keycount = int(clean['columns'].max()) + 1 if len(clean['columns']) else 4
        left, right = default_hands(keycount)
        hs = hand_split(clean['columns'], clean['offsets'], left, right)
        n_all = len(clean['offsets'])
        entry = {
            **s,
            'replay_path': path,
            'n_notes': n_all,
            'n_misses': int(rep['misses'].sum()),
            'mean_ms': float(np.mean(clean['offsets']) * 1000),
            'std_ms': float(np.std(clean['offsets']) * 1000),
            'abs_mean_ms': float(np.mean(np.abs(clean['offsets'])) * 1000),
            'left_mean_ms': hs['left']['mean'] * 1000,
            'left_std_ms': hs['left']['std'] * 1000,
            'right_mean_ms': hs['right']['mean'] * 1000,
            'right_std_ms': hs['right']['std'] * 1000,
            'hand_skew_ms': (hs['right']['mean'] - hs['left']['mean']) * 1000,
        }
        if filters.get('min_notes') and n_all < filters['min_notes']:
            continue
        out.append(entry)
        if verbose and (i + 1) % 250 == 0:
            print(f"  processed {i+1}/{len(scores)}, kept {len(out)}")
    if verbose:
        print(f"kept {len(out)} scores with replays")
    return out


def compare_hands_over_time(scores, save_path=None, show=True):
    dated = [(s, _dt(s.get('datetime'))) for s in scores]
    dated = [(s, d) for s, d in dated if d is not None]
    dated.sort(key=lambda x: x[1])
    if not dated:
        print("no dated scores")
        return

    dates = [d for _, d in dated]
    left = [s['left_mean_ms'] for s, _ in dated]
    right = [s['right_mean_ms'] for s, _ in dated]
    skew = [s['hand_skew_ms'] for s, _ in dated]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].scatter(dates, left, s=6, c='#4fc3f7', label='Left', alpha=0.5)
    axes[0].scatter(dates, right, s=6, c='#ff8a65', label='Right', alpha=0.5)
    axes[0].axhline(0, color='w', lw=0.5)
    axes[0].set_ylabel('mean offset (ms)')
    axes[0].legend()
    axes[0].set_title('Per-hand mean offset over time')
    axes[0].set_ylim(-40, 40)

    axes[1].scatter(dates, skew, s=6, c='#ba68c8', alpha=0.6)
    axes[1].axhline(0, color='w', lw=0.5)
    axes[1].set_ylabel('R − L skew (ms)')
    axes[1].set_xlabel('date')
    axes[1].set_title('Right − Left skew (positive = right later)')
    axes[1].set_ylim(-30, 30)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def compare_by_skillset(scores, save_path=None, show=True):
    groups = {}
    for s in scores:
        ssrs = s.get('ssrs', {})
        if not ssrs:
            continue
        dom = max((k for k in ssrs if k != 'Overall'), key=lambda k: ssrs[k])
        groups.setdefault(dom, []).append(s)
    if not groups:
        print("no skillset data")
        return

    names = sorted(groups.keys())
    l_mean = [np.mean([s['left_mean_ms'] for s in groups[n]]) for n in names]
    r_mean = [np.mean([s['right_mean_ms'] for s in groups[n]]) for n in names]
    l_std = [np.mean([s['left_std_ms'] for s in groups[n]]) for n in names]
    r_std = [np.mean([s['right_std_ms'] for s in groups[n]]) for n in names]
    ns = [len(groups[n]) for n in names]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    w = 0.38
    axes[0].bar(x - w / 2, l_mean, w, color='#4fc3f7', label='Left')
    axes[0].bar(x + w / 2, r_mean, w, color='#ff8a65', label='Right')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20)
    axes[0].set_ylabel('mean offset (ms)')
    axes[0].axhline(0, color='w', lw=0.5)
    axes[0].legend()
    axes[0].set_title('Hand mean by dominant skillset')
    for i, n in enumerate(ns):
        axes[0].text(i, max(l_mean[i], r_mean[i]) + 0.5, f'n={n}',
                     ha='center', fontsize=8, color='#888')

    axes[1].bar(x - w / 2, l_std, w, color='#4fc3f7', label='Left σ')
    axes[1].bar(x + w / 2, r_std, w, color='#ff8a65', label='Right σ')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20)
    axes[1].set_ylabel('std (ms)')
    axes[1].legend()
    axes[1].set_title('Hand variance by skillset')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def find_worst_coupling(scores, n=10):
    ranked = []
    for s in scores:
        if s['left_std_ms'] <= 0:
            continue
        ratio = s['right_std_ms'] / s['left_std_ms']
        ranked.append((ratio, s))
    ranked.sort(reverse=True)
    return [s for _, s in ranked[:n]]


def print_leaderboard(scores, n=15):
    if not scores:
        print("no scores")
        return
    print(f"\n{'song':<50s} {'pack':<30s} {'rate':>5} {'wife%':>7} {'L ms':>8} {'R ms':>8} {'skew':>7}")
    print('-' * 130)
    top = sorted(scores, key=lambda s: s.get('ssrnormpercent', 0), reverse=True)[:n]
    for s in top:
        song = (s.get('song') or '')[:48]
        pack = (s.get('pack') or '')[:28]
        wife = s.get('ssrnormpercent', 0) * 100
        print(f"{song:<50s} {pack:<30s} {s.get('rate',1):>5.2f} {wife:>7.2f} "
              f"{s['left_mean_ms']:>+8.2f} {s['right_mean_ms']:>+8.2f} {s['hand_skew_ms']:>+7.2f}")


if __name__ == '__main__':
    filters = {}
    if '--min-wife' in sys.argv:
        filters['min_wife'] = float(sys.argv[sys.argv.index('--min-wife') + 1])
    if '--min-notes' in sys.argv:
        filters['min_notes'] = int(sys.argv[sys.argv.index('--min-notes') + 1])
    scores = analyze_all_scores(filters=filters)

    print_leaderboard(scores)
    print("\ngenerating plots...")

    if '--no-plot' not in sys.argv:
        compare_hands_over_time(scores, save_path='batch_hands_over_time.png',
                                show=False)
        compare_by_skillset(scores, save_path='batch_by_skillset.png', show=False)
        print("saved: batch_hands_over_time.png, batch_by_skillset.png")

    print("\n=== WORST HAND RATIO (R std / L std) ===")
    for s in find_worst_coupling(scores, n=10):
        ratio = s['right_std_ms'] / max(s['left_std_ms'], 0.01)
        print(f"  {ratio:.2f}x  {s.get('song','?')[:50]:<50} "
              f"L={s['left_std_ms']:.1f}ms R={s['right_std_ms']:.1f}ms")
