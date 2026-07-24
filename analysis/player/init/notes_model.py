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


# Chart-stream record kinds for the unified stream table below. A
# record's kind selects ONLY its sprite (and end-of-life semantics like
# `stream_until`); positioning, per-note mods, and visibility are the
# same pipeline taps use.
KIND_MINE = 0
KIND_LIFT = 1
KIND_FAKE = 2


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
    # Per-mine SV group ids (Quaver TimingGroups, parallel to
    # mine_times). Empty when every mine rides the default stream
    # (all other games).
    mine_groups: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=object))
    # Hold-mine spans (Quaver): end time per mine, NaN for point mines.
    mine_end_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    # Mine detonations (Quaver): index into the mine arrays + the press
    # time that set it off. Each detonation scored a Miss in-game.
    mine_hit_idx: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))
    mine_hit_press: np.ndarray = field(
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

    # Unified chart-stream table: one time-sorted struct-of-arrays
    # record set for every chart-only note above (mines + lifts +
    # fakes), with a KIND column. Built by `copy_chart_streams` from
    # the per-type boundary arrays; the render pipeline reads ONLY this
    # table (the per-type arrays stay as the adapter boundary and the
    # per-type SV caches). `stream_rows` holds beat rows where the game
    # supplies them, -1 otherwise (rows only feed the NotITG per-note
    # mod kernel, and NotITG streams always carry rows).
    stream_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    stream_cols: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))
    stream_rows: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))
    stream_kinds: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8))
    stream_until: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    stream_groups: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=object))
    # Span end per record (hold mines), NaN for point records.
    stream_end_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    # Cull-space projection of `stream_times`, gathered from the
    # per-type SV caches by the SV builder (see build_ghost_sv_caches).
    stream_sv: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64))
    # Sort permutation over the mines+lifts+fakes concatenation, so
    # per-type parallel data can be gathered into table order.
    stream_order: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))


_CHART_STREAM_FIELDS = (
    'mine_times', 'mine_rows', 'mine_cols', 'mine_until', 'mine_groups',
    'mine_end_times', 'mine_hit_idx', 'mine_hit_press',
    'lift_times', 'lift_rows', 'lift_cols', 'lift_until',
    'fake_times', 'fake_rows', 'fake_cols', 'fake_until',
)


def stream_groups_or_none(groups) -> np.ndarray | None:
    """Normalize a chart-stream group array for the SV projection: an
    empty (or absent) array means every entry uses the engine's default
    stream, which the projection expresses as `groups=None`. So does an
    all-None column -- the unified stream table keeps `stream_groups`
    parallel to its records, and games without TimingGroups fill it
    with None."""
    if groups is None or not len(groups):
        return None
    if all(g is None for g in groups):
        return None
    return groups


def copy_chart_streams(model: NotesModel, replay) -> None:
    """Copy chart-only stream arrays (mines/lifts/fakes + Quaver mine
    detonations) from the replay dict onto the model, preserving each
    field's declared dtype, then normalize them into the unified stream
    table. Adapters call this from `populate_notes_model`; absent keys
    keep the empty defaults."""
    for name in _CHART_STREAM_FIELDS:
        value = replay.get(name)
        if value is not None:
            dtype = getattr(model, name).dtype
            setattr(model, name, np.asarray(value, dtype=dtype))
    _build_stream_table(model)


def _family_column(arr, n, fill, dtype) -> np.ndarray:
    """A family's per-record column, or a `fill` column when the family
    left it empty (the boundary key was absent for this game)."""
    if arr is not None and len(arr) == n:
        return np.asarray(arr, dtype=dtype)
    return np.full(n, fill, dtype=dtype)


def _build_stream_table(m: NotesModel) -> None:
    """Merge the per-type stream families into the unified time-sorted
    table (see the NotesModel field block). The stable sort keeps
    equal-time records in family order (mines, lifts, fakes) and each
    family's own adapter order, so draw order is deterministic."""
    families = (
        (KIND_MINE, m.mine_times, m.mine_cols, m.mine_rows, m.mine_until,
         m.mine_groups, m.mine_end_times),
        (KIND_LIFT, m.lift_times, m.lift_cols, m.lift_rows, m.lift_until,
         None, None),
        (KIND_FAKE, m.fake_times, m.fake_cols, m.fake_rows, m.fake_until,
         None, None),
    )
    times, cols, rows, kinds, until, groups, ends = ([] for _ in range(7))
    for kind, f_times, f_cols, f_rows, f_until, f_groups, f_ends in families:
        n = len(f_times)
        if not n:
            continue
        times.append(np.asarray(f_times, dtype=np.float64))
        cols.append(np.asarray(f_cols, dtype=np.int32))
        kinds.append(np.full(n, kind, dtype=np.uint8))
        rows.append(_family_column(f_rows, n, -1, np.int64))
        until.append(_family_column(f_until, n, np.inf, np.float64))
        groups.append(_family_column(f_groups, n, None, object))
        ends.append(_family_column(f_ends, n, np.nan, np.float64))
    if not times:
        return

    order = np.argsort(np.concatenate(times), kind='stable')
    m.stream_order = order
    m.stream_times = np.concatenate(times)[order]
    m.stream_cols = np.concatenate(cols)[order]
    m.stream_rows = np.concatenate(rows)[order]
    m.stream_kinds = np.concatenate(kinds)[order]
    m.stream_until = np.concatenate(until)[order]
    m.stream_groups = np.concatenate(groups)[order]
    m.stream_end_times = np.concatenate(ends)[order]


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
