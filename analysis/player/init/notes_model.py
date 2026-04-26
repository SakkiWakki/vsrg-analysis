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

    # Ghost taps ; (time_sec, column) for presses that missed every note.
    # osu only; Etterna .bin has no raw key stream.
    ghost_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    ghost_cols: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))

    # Miss holds ; spans where the player held a key through a missed LN.
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

    # SV-space caches ; populated by the SV builder, kept here so every
    # "per-note stream" lives in one place.
    ghost_sv_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    miss_hold_press_sv: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    miss_hold_release_sv: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    miss_hold_max_sv_dur: float = 0.0

    # SM note streams. Mines, lifts, and fakes never
    # appear in the .bin replay ; the adapter pulls them off the
    # matched .sm/.ssc and stashes time-converted arrays on the replay
    # dict. Empty for osu!mania.
    mine_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    mine_rows: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))
    mine_cols: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))
    mine_until: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    mine_sv: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    lift_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    lift_rows: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))
    lift_cols: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))
    lift_until: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    lift_sv: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    fake_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    fake_rows: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))
    fake_cols: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))
    fake_until: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    fake_sv: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    # (head_row, col) for rolls ; same key shape as hold_tails. The
    # renderer uses this to tint LN tails green for rolls.
    roll_head_keys: set = field(default_factory=set)



def build_notes_model(replay, times, hold_tails, adapter) -> NotesModel:
    """Populate a NotesModel from a parsed replay. The shared scaffolding
    (per-note row/col arrays + LN tail times) is built here; per-game
    extras (osu's ghost taps + miss-hold spans, Etterna's chart-only
    mines/lifts/fakes/rolls) are filled by the adapter's
    `populate_notes_model`. Miss-to-hold linking happens later via
    `link_miss_holds` once misses/offsets/miss_pressed are known."""
    m = NotesModel()
    m.noterows_list = [int(r) for r in replay['noterows']]
    m.columns_list = [int(c) for c in replay['columns']]
    _build_ln_times(m, times, hold_tails)
    adapter.populate_notes_model(replay, m)
    return m


def _build_ln_times(m, times, hold_tails):
    """Fill m.ln_tail_times and m.ln_indices from the adapter's hold_tails
    dict. Taps stay NaN; LN heads carry the tail time in seconds."""
    m.ln_tail_times = np.full(len(times), np.nan, dtype=np.float64)
    for i, (row, col) in enumerate(zip(m.noterows_list, m.columns_list)):
        end_t = hold_tails.get((row, col))
        if end_t is not None:
            m.ln_tail_times[i] = end_t
            m.ln_indices.append(i)



_HOLD_MATCH_TOL_MS = 2  # press-time tolerance for matching a miss to a hold span


def link_miss_holds(m: NotesModel, offsets, misses, miss_pressed):
    """Attach each missed LN to its corresponding miss-hold span, if
    the player actually pressed the key within _HOLD_MATCH_TOL_MS of the
    recorded miss offset. When one press-span straddles multiple consecutive
    missed LNs, only the first LN gets the link and the others are
    flagged `miss_head_suppressed` so the renderer doesn't draw a
    redundant head indicator. Idempotent; call after judging."""
    m.miss_first_hold = np.full(len(offsets), -1, dtype=np.int32)
    m.miss_head_suppressed = np.zeros(len(offsets), dtype=bool)
    if not m.miss_hold_press.size:
        return

    to_ms = lambda t: int(round(float(t) * 1000.0))

    by_head_col = {}
    for k, (head_ms, col) in enumerate(zip(m.miss_hold_ln_heads_ms,
                                            m.miss_hold_cols)):
        by_head_col.setdefault((int(head_ms), int(col)), []).append(k)

    linked_ln_keys = set()
    used_holds = set()
    for i, (head_ms, col) in enumerate(zip(m.noterows_list, m.columns_list)):
        is_candidate = (misses[i] and miss_pressed[i]
                        and not math.isnan(m.ln_tail_times[i]))
        ln_key = (int(head_ms), int(col))
        press_ms = int(head_ms) + to_ms(offsets[i])
        matched = next(
            (k for k in (by_head_col.get(ln_key, []) if is_candidate else ())
             if k not in used_holds
             and abs(to_ms(m.miss_hold_press[k]) - press_ms) <= _HOLD_MATCH_TOL_MS),
            None)

        if matched is not None:
            used_holds.add(matched)
            already_linked = ln_key in linked_ln_keys
            linked_ln_keys.add(ln_key)
            m.miss_head_suppressed[i] = already_linked
            m.miss_first_hold[i] = -1 if already_linked else matched
