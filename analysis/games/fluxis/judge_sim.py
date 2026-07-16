"""fluXis judgement simulation.

Ports the scoring rules from TeamFluXis/fluXis:

- Windows (`HitWindows.CreateTimings`): six judgements whose widths
  interpolate between three anchors by the chart's AccuracyDifficulty
  (`MapUtils.GetDifficulty`: 0 -> min, 5 -> mid, 10 -> max) and scale
  with the play rate. Releases and landmines use their own anchor sets.
- Input (`DrawableNote.OnPressed`): a press targets the earliest
  unjudged object in the lane regardless of distance (`Column.IsFirst`);
  if that object is outside the miss window the press is a no-op.
- Long notes: head judged like a tap; the tail judges against the
  release windows when the key comes up near EndTime. An early release
  breaks the hold (no regrab), but `ReleaseWindows.Lowest` is Alright:
  a broken, late, or never-happening release scores 'alright', never
  'miss'.
- Landmines (`DrawableLandmine`): a press -- or already holding the key
  -- inside the landmine Miss window detonates (scored Miss in-game);
  an avoided landmine scores Flawless. Detonations surface through the
  shared mine-hit contract; the avoided count is kept for accuracy
  bookkeeping.
- Tick notes (`DrawableTickNote`): Flawless when the key is already
  down as they cross; otherwise the first press inside the window
  judges them at that press's offset (fluXis's held-update loop /
  direct-hit paths); no press in the window is a Miss.

Sign convention: fluXis's `TimeDelta` is note-minus-now; everything
here uses this codebase's press-minus-note (late = positive) instead.
"""
from __future__ import annotations

from bisect import bisect_right

_JUDGEMENTS = ('flawless', 'perfect', 'great', 'alright', 'okay', 'miss')

# (name, window at difficulty 0 / 5 / 10), milliseconds.
_HIT_ANCHORS = (
    ('flawless', 22.0, 19.0, 13.0),
    ('perfect', 64.0, 49.0, 34.0),
    ('great', 97.0, 82.0, 67.0),
    ('alright', 127.0, 112.0, 97.0),
    ('okay', 151.0, 136.0, 121.0),
    ('miss', 188.0, 173.0, 158.0),
)
_RELEASE_ANCHORS = (
    ('flawless', 64.0, 49.0, 34.0),
    ('perfect', 97.0, 82.0, 67.0),
    ('great', 127.0, 112.0, 97.0),
    ('alright', 151.0, 136.0, 121.0),
)
_LANDMINE_ANCHORS = (
    # In-source comment: the Miss tier "matches Perfect from regular".
    ('miss', 64.0, 49.0, 34.0),
    ('flawless', 188.0, 173.0, 158.0),
)


def _difficulty_scale(difficulty, lo, mid, hi):
    if difficulty > 5:
        return mid + (hi - mid) * (difficulty - 5) / 5
    if difficulty < 5:
        return mid + (mid - lo) * (difficulty - 5) / 5
    return mid


def _windows(anchors, difficulty, rate):
    return [(name, _difficulty_scale(difficulty, lo, mid, hi) * rate)
            for name, lo, mid, hi in anchors]


def hit_windows_ms(difficulty, rate=1.0):
    return _windows(_HIT_ANCHORS, difficulty, rate)


def release_windows_ms(difficulty, rate=1.0):
    return _windows(_RELEASE_ANCHORS, difficulty, rate)


def landmine_windows_ms(difficulty, rate=1.0):
    return _windows(_LANDMINE_ANCHORS, difficulty, rate)


def _judgement_for(abs_off, windows):
    for name, w in windows:
        if abs_off <= w:
            return name
    return 'miss'


class _KeyState:
    """Per-lane key-down intervals, queryable by time. Built once from
    the press/release event list; `down_at(t)` and `down_within(a, b)`
    drive tick and landmine judging."""

    def __init__(self, events):
        self._starts = []
        self._ends = []
        open_t = None
        for t, is_press in events:
            if is_press and open_t is None:
                open_t = t
            elif not is_press and open_t is not None:
                self._starts.append(open_t)
                self._ends.append(t)
                open_t = None
        if open_t is not None:
            self._starts.append(open_t)
            self._ends.append(float('inf'))

    def down_at(self, t):
        i = bisect_right(self._starts, t) - 1
        return i >= 0 and t <= self._ends[i]

    def first_down_within(self, a, b):
        """Earliest instant in [a, b] with the key down, or None."""
        i = max(0, bisect_right(self._starts, a) - 1)
        while i < len(self._starts):
            start, end = self._starts[i], self._ends[i]
            if start > b:
                return None
            if end >= a:
                return max(start, a)
            i += 1
        return None


def _new_result(col, note):
    return {
        'time': note['time'], 'col': col,
        'head_off': None, 'judgement': 'miss',
        'is_hold': bool(note['is_hold']), 'end_time': note['end_time'],
        'tail_off': None, 'is_tick': False,
    }


def _worse(a, b):
    return a if _JUDGEMENTS.index(a) >= _JUDGEMENTS.index(b) else b


def _simulate_column(col, notes, events, windows, tail_windows):
    miss_w = windows[-1][1]
    tail_miss_w = tail_windows[-1][1]
    per = [_new_result(col, n) for n in notes]
    judged = [False] * len(notes)
    next_unjudged = 0
    held = None   # index of the LN currently held

    def advance_past(upto_t):
        nonlocal next_unjudged
        while next_unjudged < len(notes):
            if judged[next_unjudged]:
                next_unjudged += 1
                continue
            if notes[next_unjudged]['time'] + miss_w < upto_t:
                judged[next_unjudged] = True
                next_unjudged += 1
                continue
            break

    for t, is_press in events:
        advance_past(t)
        if is_press:
            if next_unjudged >= len(notes):
                continue
            note = notes[next_unjudged]
            off = t - note['time']
            if abs(off) > miss_w:
                continue   # IsFirst target out of window: press is a no-op
            r = per[next_unjudged]
            r['head_off'] = off
            r['judgement'] = _judgement_for(abs(off), windows)
            judged[next_unjudged] = True
            if r['is_hold']:
                held = next_unjudged
            next_unjudged += 1
        elif held is not None:
            r = per[held]
            rel_off = t - r['end_time']
            r['tail_off'] = rel_off
            if abs(rel_off) <= tail_miss_w:
                tail_j = _judgement_for(abs(rel_off), tail_windows)
            else:
                tail_j = 'alright'   # ReleaseWindows.Lowest
            r['judgement'] = _worse(r['judgement'], tail_j)
            held = None

    advance_past(float('inf'))
    if held is not None:
        # Never released: fluXis's non-user tail result is its lowest
        # release judgement, not a miss.
        r = per[held]
        r['judgement'] = _worse(r['judgement'], 'alright')
    return per


def _judge_ticks(col, ticks, events, key_state, results, windows):
    miss_w = windows[-1][1]
    presses = [t for t, is_press in events if is_press]
    for tick in ticks:
        r = _new_result(col, tick)
        r['is_tick'] = True
        t = tick['time']
        if key_state.down_at(t):
            r['head_off'] = 0.0
            r['judgement'] = 'flawless'
        else:
            i = bisect_right(presses, t - miss_w)
            if i < len(presses) and presses[i] <= t + miss_w:
                off = presses[i] - t
                r['head_off'] = off
                r['judgement'] = _judgement_for(abs(off), windows)
        results.append(r)


def simulate_landmines(mines_by_col, key_events_by_col, mine_miss_w):
    """Detonations per fluXis rules: a press or an already-held key
    inside the landmine Miss window (`mine_miss_w`) detonates. Returns
    `(hits, avoided_count)`; hits carry the shared mine-hit shape."""
    hits = []
    avoided = 0
    for c, mines in enumerate(mines_by_col):
        events = (key_events_by_col[c]
                  if c < len(key_events_by_col) else [])
        ks = _KeyState(events)
        for m in mines:
            end = m['end_time'] or m['time']
            t_hit = ks.first_down_within(m['time'] - mine_miss_w,
                                         end + mine_miss_w)
            if t_hit is None:
                avoided += 1
                continue
            hits.append({'col': c, 'mine_time': m['time'],
                         'end_time': end, 'press_time': t_hit})
    return hits, avoided


def simulate_mania(notes_by_col, ticks_by_col, key_events_by_col,
                   difficulty, rate=1.0):
    """Run the fluXis judgement sim across every column.

    `notes_by_col[c]` / `ticks_by_col[c]`: sorted `{'time', 'end_time',
    'is_hold'}` dicts (ms). `key_events_by_col[c]`: chronological
    `(t_ms, is_press)`. Returns a flat list of result dicts."""
    windows = hit_windows_ms(difficulty, rate)
    tail_windows = release_windows_ms(difficulty, rate)
    results = []
    for c, notes in enumerate(notes_by_col):
        events = key_events_by_col[c] if c < len(key_events_by_col) else []
        results.extend(_simulate_column(c, notes, events,
                                        windows, tail_windows))
        _judge_ticks(c, ticks_by_col[c], events, _KeyState(events),
                     results, windows)
    return results
