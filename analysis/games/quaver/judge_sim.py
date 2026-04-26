"""Quaver judgement simulation.

Mirrors `analysis/games/osu/replay/judge.simulate_mania` but with
Quaver's windows (Marv/Perf/Great/Good/Okay/Miss) and Quaver's
release-multiplier semantics (long-note tails use windows * 1.5).

The output shape matches the osu sim's so the rest of the player
pipeline (`_build_arrays`, ghost-tap / miss-hold post-processing) can
consume Quaver replays without branching.
"""
from __future__ import annotations


_JUDGEMENTS = ['marv', 'perf', 'great', 'good', 'okay', 'miss']
_RANK = {j: i for i, j in enumerate(_JUDGEMENTS)}


def windows_ms(judge='Standard'):
    """Return windows as `[marv, perf, great, good, okay, miss]` in ms.
    Sourced from `analysis.games.quaver.judgment.windows_for` so the sim
    and the viz windows can never drift apart."""
    from analysis.games.quaver.judgment import windows_for
    return [w * 1000.0 for _, w in windows_for(judge)]


def _judgement_for(abs_diff, windows):
    for j, w in zip(_JUDGEMENTS, windows):
        if abs_diff <= w:
            return j
    return 'miss'


def _combine_head_tail(head_j, tail_j):
    """Quaver's post-judge clamp ; the worse of the two becomes the
    note's final judgement, matching `ScoreProcessorKeys.CalculateScore`
    where the tail can only equal-or-degrade the head."""
    return _JUDGEMENTS[max(_RANK[head_j], _RANK[tail_j])]


def _new_result(col, note):
    return {'col': col, 'time': note['time'], 'end_time': note.get('end_time'),
            'is_hold': note.get('end_time') is not None,
            'press_t': None, 'release_t': None,
            'head_off': None, 'tail_off': None,
            'judgement': None, 'broken': False, 'missed': False}


def _judge_tail(r, release_t, tail_windows):
    if not r['is_hold']:
        return
    diff = release_t - r['end_time']
    r['release_t'] = release_t
    r['tail_off'] = diff
    abs_diff = abs(diff)
    tail_miss_w = tail_windows[5]
    if abs_diff > tail_miss_w:
        r['broken'] = True
        r['judgement'] = 'miss'
        return
    tail_j = _judgement_for(abs_diff, tail_windows)
    head_j = r.get('_head_j', 'miss')
    if head_j == 'miss' or diff < -tail_miss_w:
        r['broken'] = True
        r['judgement'] = 'miss' if tail_j == 'miss' else 'okay'
    else:
        r['judgement'] = _combine_head_tail(head_j, tail_j)


def _advance_misses(per, next_unjudged, upto_t, late_expire_w):
    while next_unjudged < len(per):
        r = per[next_unjudged]
        if r['judgement'] is not None or r['missed']:
            next_unjudged += 1
            continue
        if upto_t - late_expire_w < r['time']:
            break
        r['missed'] = True
        r['judgement'] = 'miss'
        if r['is_hold']:
            r['broken'] = True
        next_unjudged += 1
    return next_unjudged


def _judge_press(r, t, windows, okay_w, early_w):
    diff = t - r['time']
    if diff < -early_w:
        return None
    r['press_t'] = t
    r['head_off'] = diff
    return _judgement_for(abs(diff), windows) if abs(diff) <= okay_w else 'miss'


def _simulate_column(col, notes, events, windows, tail_windows):
    per = [_new_result(col, n) for n in notes]
    early_w = windows[5]            # miss is the widest = early-press cutoff
    late_expire_w = windows[3]      # good defines auto-miss point
    okay_w = windows[4]
    held_idx = None
    next_unjudged = 0

    for t, is_press in events:
        next_unjudged = _advance_misses(per, next_unjudged, t, late_expire_w)

        if held_idx is not None:
            _judge_tail(per[held_idx], t, tail_windows)
            held_idx = None
        if not is_press:
            continue

        if next_unjudged >= len(per):
            continue
        r = per[next_unjudged]
        head_j = _judge_press(r, t, windows, okay_w, early_w)
        if head_j is None:
            continue
        if r['is_hold']:
            if head_j == 'miss':
                r['judgement'] = 'miss'
                r['missed'] = True
                r['broken'] = True
            else:
                r['_head_j'] = head_j
                held_idx = next_unjudged
        else:
            r['judgement'] = head_j
            if head_j == 'miss':
                r['missed'] = True
        next_unjudged += 1

    _advance_misses(per, next_unjudged, 10**18, late_expire_w)
    if held_idx is not None:
        per[held_idx]['broken'] = True
        per[held_idx]['judgement'] = 'miss'
    return per


def simulate_mania(notes_by_col, key_events_by_col, windows):
    """Run the Quaver judgement sim across every column.

    `notes_by_col[c]`: sorted list of `{'time', 'end_time'}`.
    `key_events_by_col[c]`: chronological `(t_ms, is_press)`.
    `windows`: `[marv, perf, great, good, okay, miss]` ms.

    Returns a flat list of per-note result dicts ; see `_new_result`."""
    from analysis.games.quaver.judgment import RELEASE_MULTIPLIER
    tail_windows = [w * RELEASE_MULTIPLIER for w in windows]
    results = []
    for c, notes in enumerate(notes_by_col):
        events = key_events_by_col[c] if c < len(key_events_by_col) else []
        results.extend(_simulate_column(c, notes, events, windows, tail_windows))
    return results
