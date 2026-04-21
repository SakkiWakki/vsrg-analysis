"""Per-note arrays + ghost-tap/ghost-hold bookkeeping.

Pulls the ~100 lines of Player.__init__ that used to build
`_noterows_list`, `_columns_list`, `_ln_tail_times`, `_ln_indices`,
`_ghost_times/_cols`, and the `_ghost_hold_*` arrays into one place.
The Player now holds a single `self.notes: NotesModel` and reads
attributes off it — no more dozen `self._ghost_hold_*` fields."""
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

    # Ghost holds — spans where the player held a key inside a missed LN.
    # Press/release are stored unclipped so overholds stay visible.
    ghost_hold_ln_heads_ms: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))
    ghost_hold_press: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    ghost_hold_release: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    ghost_hold_cols: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))
    ghost_hold_max_dur: float = 0.0

    # Miss → ghost-hold links, filled once misses/offsets are known.
    miss_first_ghost_hold: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))
    ghost_hold_extends_miss: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool))

    # SV-space caches — populated by the SV builder, kept here so every
    # "per-note stream" lives in one place.
    ghost_sv_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    ghost_hold_press_sv: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    ghost_hold_release_sv: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    ghost_hold_max_sv_dur: float = 0.0


def build_notes_model(replay, times, hold_tails, game) -> NotesModel:
    """Populate a NotesModel from a parsed replay. Ghost taps/holds are
    osu-only; for Etterna we leave the ghost arrays empty. Miss→ghost
    linking happens later (needs misses/offsets/miss_pressed, which the
    Player computes after judging)."""
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

    raw_holds = replay.get('ghost_holds') or []
    if raw_holds and game == 'osu':
        gh_heads = np.array([lh for (lh, _c, _pt, _rt) in raw_holds],
                            dtype=np.int64)
        gh_press = np.array([pt / 1000.0 for (_lh, _c, pt, _rt) in raw_holds],
                            dtype=np.float64)
        gh_rel = np.array([rt / 1000.0 for (_lh, _c, _pt, rt) in raw_holds],
                          dtype=np.float64)
        gh_cols = np.array([c for (_lh, c, _pt, _rt) in raw_holds],
                           dtype=np.int32)
        order = np.argsort(gh_press, kind='stable')
        m.ghost_hold_ln_heads_ms = gh_heads[order]
        m.ghost_hold_press = gh_press[order]
        m.ghost_hold_release = gh_rel[order]
        m.ghost_hold_cols = gh_cols[order]
        # Longest hold dur — conservative lookback when bisecting on
        # press_t so spans whose press is off-screen but release is on
        # still get picked up.
        m.ghost_hold_max_dur = float(
            np.max(gh_rel - gh_press)) if gh_press.size else 0.0
    return m


def link_miss_ghost_holds(m: NotesModel, offsets, misses, miss_pressed):
    """Attach each missed LN to its corresponding ghost-hold span, if
    the player did actually press the key within 2 ms of the recorded
    miss offset. Idempotent; call after judging."""
    m.miss_first_ghost_hold = np.full(len(offsets), -1, dtype=np.int32)
    m.ghost_hold_extends_miss = np.zeros(m.ghost_hold_press.size, dtype=bool)
    if not m.ghost_hold_press.size:
        return

    by_head_col = {}
    for k, (head_ms, col) in enumerate(zip(m.ghost_hold_ln_heads_ms,
                                            m.ghost_hold_cols)):
        by_head_col.setdefault((int(head_ms), int(col)), []).append(k)

    tol_ms = 2
    for i, (head_ms, col) in enumerate(zip(m.noterows_list, m.columns_list)):
        if not (misses[i] and miss_pressed[i]):
            continue
        if math.isnan(m.ln_tail_times[i]):
            continue
        press_ms = int(head_ms + round(float(offsets[i]) * 1000.0))
        for k in by_head_col.get((int(head_ms), int(col)), []):
            gh_press_ms = int(round(float(m.ghost_hold_press[k]) * 1000.0))
            if abs(gh_press_ms - press_ms) <= tol_ms:
                m.miss_first_ghost_hold[i] = k
                m.ghost_hold_extends_miss[k] = True
                break
