"""osu!mania Classic judgment simulation.

Per column: walk the key-event stream, assign each press to the leftmost
unjudged head, auto-expire heads whose late boundary has passed, and for
LNs, judge the release against the tail window. Matches stable's
`ManiaHitWindows` (non-convert Classic branch).
"""

TAIL_RELEASE_LENIENCE = 1.5  # osu!lazer TailNote.RELEASE_WINDOW_LENIENCE

_JUDGEMENTS = ['MAX', '300', '200', '100', '50', 'miss']
_RANK = {j: i for i, j in enumerate(_JUDGEMENTS)}


def stable_hit_windows(od):
    """osu!stable / lazer-Classic hit windows, ms.
    Order: [MAX(300g), 300, 200, 100, 50, MISS]. MISS is the widest ;
    anything past it doesn't consume the note at all.
    Source: osu!lazer ManiaHitWindows.cs non-convert Classic branch."""
    inv = max(0.0, min(10.0, 10.0 - float(od)))
    return [16.0,
            34.0 + 3.0 * inv,
            67.0 + 3.0 * inv,
            97.0 + 3.0 * inv,
            121.0 + 3.0 * inv,
            158.0 + 3.0 * inv]


def _judgement_for(abs_diff, windows):
    for j, w in zip(_JUDGEMENTS, windows):
        if abs_diff <= w:
            return j
    return 'miss'


def _combine_head_tail(head_j, tail_j):
    """Stable combines head+tail by taking the worse of the two."""
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
    # Broken LN (head missed, or released too early) caps at 50.
    if head_j == 'miss' or diff < -tail_miss_w:
        r['broken'] = True
        r['judgement'] = 'miss' if tail_j == 'miss' else '50'
    else:
        r['judgement'] = _combine_head_tail(head_j, tail_j)


def _advance_misses(per, next_unjudged, upto_t, late_expire_w):
    """Auto-expire heads whose late boundary (`upto_t - late_expire_w`)
    has passed their note time. Returns the new `next_unjudged`."""
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


def _judge_press(r, t, windows, meh_w, early_w):
    """Decide the head judgement for note `r` given a press at time `t`.
    Returns the head judgement string, or `None` if the press is too
    early to consume the note."""
    diff = t - r['time']
    if diff < -early_w:
        return None
    r['press_t'] = t
    r['head_off'] = diff
    return _judgement_for(abs(diff), windows) if abs(diff) <= meh_w else 'miss'


def _simulate_column(col, notes, events, windows, tail_windows):
    per = [_new_result(col, n) for n in notes]
    early_w = windows[5]
    late_expire_w = windows[3]
    meh_w = windows[4]
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
        head_j = _judge_press(r, t, windows, meh_w, early_w)
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
    # Song ended mid-hold ; treat as tail miss.
    if held_idx is not None:
        per[held_idx]['broken'] = True
        per[held_idx]['judgement'] = 'miss'
    return per


def simulate_mania(notes_by_col, key_events_by_col, windows):
    """Simulate osu!mania Classic judgment per column.

    `notes_by_col[c]`: sorted list of `{'time', 'end_time'}` (end_time=None for rice).
    `key_events_by_col[c]`: chronological `(t_ms, is_press)` list.
    `windows`: `[MAX, 300, 200, 100, 50, MISS]` ms.

    Returns a flat list of result dicts ; see `_new_result` for shape.
    """
    tail_windows = [w * TAIL_RELEASE_LENIENCE for w in windows]
    results = []
    for c, notes in enumerate(notes_by_col):
        events = key_events_by_col[c] if c < len(key_events_by_col) else []
        results.extend(_simulate_column(c, notes, events, windows, tail_windows))
    return results
