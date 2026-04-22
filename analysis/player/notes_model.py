"""Per-note arrays + ghost-tap / miss-hold bookkeeping.

Pulls the ~100 lines of Player.__init__ that used to build
`_noterows_list`, `_columns_list`, `_ln_tail_times`, `_ln_indices`,
`_ghost_times/_cols`, and the miss-hold arrays into one place.
The Player now holds a single `self.notes: NotesModel` and reads
attributes off it."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NotesModel:
    # Python-int copies of the noterow/column arrays. The draw loop hits
    # these thousands of times per frame and numpy scalar→int conversion
    # was surprisingly expensive on dense charts.
    noterows_list: list[int] = field(default_factory=list)
    columns_list: list[int] = field(default_factory=list)

    # Parallel to replay['noterows']/['columns']: tail time in seconds
    # for LN heads, NaN for taps. Replaces a (row, col) → float dict.
    ln_tail_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    ln_indices: list[int] = field(default_factory=list)

    # Ghost taps — (time_sec, column) for presses that missed every note.
    # osu only; Etterna .bin has no raw key stream.
    ghost_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    ghost_cols: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))

    # Miss holds — spans where the player held a key through a missed LN.
    # Press/release are stored unclipped so overholds stay visible.
    miss_hold_ln_heads_ms: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))
    miss_hold_press: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    miss_hold_release: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    miss_hold_cols: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))
    miss_hold_max_dur: float = 0.0

    # Miss → miss-hold index, filled once misses/offsets are known.
    miss_first_hold: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))
    miss_head_suppressed: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool))

    # SV-space caches — populated by the SV builder, kept here so every
    # "per-note stream" lives in one place.
    ghost_sv_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    miss_hold_press_sv: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    miss_hold_release_sv: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    miss_hold_max_sv_dur: float = 0.0

    # SM note streams. Mines, lifts, and fakes never
    # appear in the .bin replay — the adapter pulls them off the
    # matched .sm/.ssc and stashes time-converted arrays on the replay
    # dict. Empty for osu!mania.
    mine_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    mine_cols: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))
    lift_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    lift_cols: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))
    fake_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    fake_cols: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))
    # (head_row, col) for rolls — same key shape as hold_tails. The
    # renderer uses this to tint LN tails green for rolls.
    roll_head_keys: set = field(default_factory=set)


def build_notes_model(replay, times, hold_tails, game) -> NotesModel:
    """Populate a NotesModel from a parsed replay. Ghost taps / miss
    holds are osu-only; for Etterna we leave those arrays empty.
    Miss→hold linking happens later (needs misses/offsets/miss_pressed,
    which the Player computes after judging)."""
    m = NotesModel()
    m.noterows_list = [int(r) for r in replay['noterows']]
    m.columns_list = [int(c) for c in replay['columns']]

    m.ln_tail_times = np.full(len(times), np.nan, dtype=np.float64)
    for i, (row_val, col_val) in enumerate(zip(m.noterows_list,
                                                m.columns_list)):
        end_t = hold_tails.get((row_val, col_val))
        if end_t is not None:
            m.ln_tail_times[i] = end_t
            m.ln_indices.append(i)

    raw_ghosts = replay.get('ghost_taps') or []
    if raw_ghosts and game == 'osu':
        ghost_ts = np.array([t / 1000.0 for (t, _c) in raw_ghosts],
                            dtype=np.float64)
        ghost_cs = np.array([c for (_t, c) in raw_ghosts],
                            dtype=np.int32)
        order = np.argsort(ghost_ts, kind='stable')
        m.ghost_times = ghost_ts[order]
        m.ghost_cols = ghost_cs[order]

    raw_holds = replay.get('miss_holds') or []
    if raw_holds and game == 'osu':
        mh_heads = np.array([lh for (lh, _c, _pt, _rt) in raw_holds],
                            dtype=np.int64)
        mh_press = np.array([pt / 1000.0 for (_lh, _c, pt, _rt) in raw_holds],
                            dtype=np.float64)
        mh_rel = np.array([rt / 1000.0 for (_lh, _c, _pt, rt) in raw_holds],
                          dtype=np.float64)
        mh_cols = np.array([c for (_lh, c, _pt, _rt) in raw_holds],
                           dtype=np.int32)
        order = np.argsort(mh_press, kind='stable')
        m.miss_hold_ln_heads_ms = mh_heads[order]
        m.miss_hold_press = mh_press[order]
        m.miss_hold_release = mh_rel[order]
        m.miss_hold_cols = mh_cols[order]
        # Longest hold dur — conservative lookback when bisecting on
        # press_t so spans whose press is off-screen but release is on
        # still get picked up.
        m.miss_hold_max_dur = float(
            np.max(mh_rel - mh_press)) if mh_press.size else 0.0

    # Etterna chart-only streams. The adapter populates replay['mine_times']
    # etc. during prepare_replay_times when a chart match was found.
    for t_key, c_key, dst_t, dst_c in (
            ('mine_times', 'mine_cols', 'mine_times', 'mine_cols'),
            ('lift_times', 'lift_cols', 'lift_times', 'lift_cols'),
            ('fake_times', 'fake_cols', 'fake_times', 'fake_cols')):
        ts = replay.get(t_key)
        cs = replay.get(c_key)
        if ts is not None and cs is not None:
            setattr(m, dst_t, np.asarray(ts, dtype=np.float64))
            setattr(m, dst_c, np.asarray(cs, dtype=np.int32))
    roll_heads = replay.get('roll_heads')
    if roll_heads:
        m.roll_head_keys = set(roll_heads)
    return m


def link_miss_holds(m: NotesModel, offsets, misses, miss_pressed):
    """Attach each missed LN to its corresponding miss-hold span, if
    the player actually pressed the key within 2 ms of the recorded
    miss offset. When one press-span straddles multiple consecutive
    missed LNs, only the first LN gets the link and the others are
    flagged `miss_head_suppressed` so the renderer doesn't draw a
    redundant head indicator. Idempotent; call after judging."""
    m.miss_first_hold = np.full(len(offsets), -1, dtype=np.int32)
    m.miss_head_suppressed = np.zeros(len(offsets), dtype=bool)
    if not m.miss_hold_press.size:
        return

    by_head_col = {}
    for k, (head_ms, col) in enumerate(zip(m.miss_hold_ln_heads_ms,
                                            m.miss_hold_cols)):
        by_head_col.setdefault((int(head_ms), int(col)), []).append(k)

    linked_ln_keys = set()
    used_holds = set()
    tol_ms = 2
    for i, (head_ms, col) in enumerate(zip(m.noterows_list, m.columns_list)):
        if not (misses[i] and miss_pressed[i]):
            continue
        if math.isnan(m.ln_tail_times[i]):
            continue
        ln_key = (int(head_ms), int(col))
        press_ms = int(head_ms + round(float(offsets[i]) * 1000.0))
        matched = None
        for k in by_head_col.get(ln_key, []):
            if k in used_holds:
                continue
            hold_press_ms = int(round(float(m.miss_hold_press[k]) * 1000.0))
            if abs(hold_press_ms - press_ms) <= tol_ms:
                matched = k
                break
        if matched is None:
            continue
        used_holds.add(matched)
        if ln_key in linked_ln_keys:
            m.miss_head_suppressed[i] = True
            continue
        linked_ln_keys.add(ln_key)
        m.miss_first_hold[i] = matched
