"""Timing analysis for replays (any key count). All functions take raw arrays."""
import sys
import numpy as np
from analysis.etterna.replay import parse_replay, clean_offsets


def default_hands(keycount):
    """Split keycount columns into (left, right) tuples. Middle column (on odd K) goes to right."""
    half = keycount // 2
    left = tuple(range(half))
    right = tuple(range(half, keycount))
    return left, right


def _stats(offsets):
    if len(offsets) == 0:
        return {'n': 0, 'mean': 0, 'median': 0, 'std': 0, 'abs_mean': 0,
                'p25': 0, 'p75': 0, 'p05': 0, 'p95': 0}
    return {
        'n': int(len(offsets)),
        'mean': float(np.mean(offsets)),
        'median': float(np.median(offsets)),
        'std': float(np.std(offsets)),
        'abs_mean': float(np.mean(np.abs(offsets))),
        'p25': float(np.percentile(offsets, 25)),
        'p75': float(np.percentile(offsets, 75)),
        'p05': float(np.percentile(offsets, 5)),
        'p95': float(np.percentile(offsets, 95)),
    }


def hand_split(columns, offsets, left_cols=(0, 1), right_cols=(2, 3)):
    left = np.isin(columns, left_cols)
    right = np.isin(columns, right_cols)
    return {
        'left': _stats(offsets[left]),
        'right': _stats(offsets[right]),
        'left_cols': left_cols,
        'right_cols': right_cols,
    }


def per_column_stats(columns, offsets):
    out = {}
    for col in sorted(set(columns.tolist())):
        out[int(col)] = _stats(offsets[columns == col])
    return out


def timing_drift(noterows, offsets, columns, n_segments=4,
                 left_cols=(0, 1), right_cols=(2, 3)):
    if len(noterows) == 0:
        return {'segments': []}
    edges = np.linspace(noterows.min(), noterows.max() + 1, n_segments + 1)
    segments = []
    for i in range(n_segments):
        lo, hi = edges[i], edges[i + 1]
        mask = (noterows >= lo) & (noterows < hi)
        seg_off = offsets[mask]
        seg_col = columns[mask]
        left = np.isin(seg_col, left_cols)
        right = np.isin(seg_col, right_cols)
        segments.append({
            'index': i,
            'noterow_lo': float(lo),
            'noterow_hi': float(hi),
            'n': int(mask.sum()),
            'left': _stats(seg_off[left]),
            'right': _stats(seg_off[right]),
            'all': _stats(seg_off),
        })
    return {'segments': segments}


def coupling_analysis(noterows, offsets, columns,
                      left_cols=(0, 1), right_cols=(2, 3)):
    """For each column, compare timing when solo vs paired with same-hand neighbor
    at the same noterow (chord partner on same hand)."""
    out = {}
    unique_rows, inverse = np.unique(noterows, return_inverse=True)
    row_to_cols = {}
    for r_i, row in enumerate(unique_rows):
        row_to_cols[row] = set(columns[inverse == r_i].tolist())

    for col in sorted(set(columns.tolist())):
        same_hand = left_cols if col in left_cols else right_cols
        partners = [c for c in same_hand if c != col]
        solo_off, paired_off = [], []
        col_mask = columns == col
        col_rows = noterows[col_mask]
        col_offs = offsets[col_mask]
        for row, off in zip(col_rows, col_offs):
            rcols = row_to_cols.get(row, set())
            if any(p in rcols for p in partners):
                paired_off.append(off)
            else:
                solo_off.append(off)
        out[int(col)] = {
            'solo': _stats(np.array(solo_off)),
            'paired': _stats(np.array(paired_off)),
        }
    return out


def chord_vs_single(noterows, offsets, columns):
    unique_rows, counts = np.unique(noterows, return_counts=True)
    row_count = dict(zip(unique_rows.tolist(), counts.tolist()))
    sizes = np.array([row_count[int(r)] for r in noterows])
    single = sizes == 1
    jump = sizes == 2
    hand = sizes == 3
    quad = sizes == 4
    return {
        'single': _stats(offsets[single]),
        'jump': _stats(offsets[jump]),
        'hand': _stats(offsets[hand]),
        'quad': _stats(offsets[quad]),
    }


def rolling_stability(offsets, columns, window=200,
                      left_cols=(0, 1), right_cols=(2, 3)):
    if len(offsets) < window:
        window = max(10, len(offsets) // 4 or 1)
    n = len(offsets)
    idx = np.arange(n)
    stds_all, stds_l, stds_r, centers = [], [], [], []
    step = max(1, window // 4)
    left = np.isin(columns, left_cols)
    right = np.isin(columns, right_cols)
    for start in range(0, n - window + 1, step):
        end = start + window
        stds_all.append(float(np.std(offsets[start:end])))
        lo = offsets[start:end][left[start:end]]
        ro = offsets[start:end][right[start:end]]
        stds_l.append(float(np.std(lo)) if len(lo) > 1 else np.nan)
        stds_r.append(float(np.std(ro)) if len(ro) > 1 else np.nan)
        centers.append(int((start + end) // 2))
    return {
        'centers': centers,
        'std_all': stds_all,
        'std_left': stds_l,
        'std_right': stds_r,
        'window': window,
    }


def full_analysis(replay, left_cols=None, right_cols=None):
    clean = clean_offsets(replay)
    offs, cols, rows = clean['offsets'], clean['columns'], clean['noterows']
    keycount = replay.get('keycount') or (int(cols.max()) + 1 if len(cols) else 4)
    if left_cols is None or right_cols is None:
        left_cols, right_cols = default_hands(keycount)
    return {
        'keycount': keycount,
        'hand_split': hand_split(cols, offs, left_cols, right_cols),
        'per_column': per_column_stats(cols, offs),
        'drift': timing_drift(rows, offs, cols, left_cols=left_cols, right_cols=right_cols),
        'coupling': coupling_analysis(rows, offs, cols, left_cols=left_cols, right_cols=right_cols),
        'chord_vs_single': chord_vs_single(rows, offs, cols),
        'rolling': rolling_stability(offs, cols, left_cols=left_cols, right_cols=right_cols),
        'total_notes': int(len(replay['offsets'])),
        'misses': int(replay['misses'].sum()),
    }


def _fmt_stats(s, scale=1000):
    if s['n'] == 0:
        return "   (no data)"
    return (f"n={s['n']:>5}  mean={s['mean']*scale:+7.2f}ms  "
            f"median={s['median']*scale:+7.2f}ms  std={s['std']*scale:6.2f}ms  "
            f"|mean|={s['abs_mean']*scale:6.2f}ms")


def print_analysis(res):
    print("=== HAND SPLIT ===")
    print(f"  Left  (cols {res['hand_split']['left_cols']}): {_fmt_stats(res['hand_split']['left'])}")
    print(f"  Right (cols {res['hand_split']['right_cols']}): {_fmt_stats(res['hand_split']['right'])}")

    print("\n=== PER COLUMN ===")
    for col, s in res['per_column'].items():
        print(f"  col {col}: {_fmt_stats(s)}")

    print("\n=== DRIFT (chart segments) ===")
    for seg in res['drift']['segments']:
        print(f"  seg {seg['index']}: rows {int(seg['noterow_lo'])}..{int(seg['noterow_hi'])}")
        print(f"    L: {_fmt_stats(seg['left'])}")
        print(f"    R: {_fmt_stats(seg['right'])}")

    print("\n=== COUPLING (solo vs paired w/ same-hand partner) ===")
    for col, c in res['coupling'].items():
        print(f"  col {col}:")
        print(f"    solo  : {_fmt_stats(c['solo'])}")
        print(f"    paired: {_fmt_stats(c['paired'])}")

    print("\n=== CHORD vs SINGLE ===")
    for name in ('single', 'jump', 'hand', 'quad'):
        print(f"  {name:>6}: {_fmt_stats(res['chord_vs_single'][name])}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: etterna_timing.py <replay_file>")
        sys.exit(1)
    replay = parse_replay(sys.argv[1])
    res = full_analysis(replay)
    print(f"File: {replay['filepath']}")
    print(f"Notes: {res['total_notes']}, misses: {res['misses']}\n")
    print_analysis(res)
