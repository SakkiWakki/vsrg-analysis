"""Top-level `parse_replay`: ties osr decoding, chart parsing, the
judgment sim, and ghost-tap / ghost-hold post-processing into a single
entry point that matches `etterna_replay.parse_replay`'s shape."""
import bisect

import numpy as np

from analysis.games.osu.replay.chart import parse_osu_file, find_osu_by_hash
from analysis.games.osu.replay.osr import (parse_osr_events,
                                            extract_key_events,
                                            OSU_MOD_RANDOM)
from analysis.games.osu.replay.judge import (stable_hit_windows,
                                              simulate_mania)
from analysis.games.osu.replay.random_mod import _random_column_permutation
from analysis.player.sv.replay_doc import SvReplayDoc, KIND_TIME_SPACE


def _group_notes_by_col(hitobjects, keycount, col_perm):
    """Return `(by_col_notes, holds_meta)`.
    `by_col_notes[c]`: list of `{'time', 'end_time'}` sorted by time.
    `holds_meta`: list of `(time, col, end_time)` for holds only."""
    by_col = [[] for _ in range(keycount)]
    holds_meta = []
    for h in hitobjects:
        c = h['column']
        if c >= keycount:
            continue
        if col_perm is not None:
            c = col_perm[c]
        end = int(h['end_time']) if h['is_hold'] else None
        by_col[c].append({'time': int(h['time']), 'end_time': end})
        if h['is_hold']:
            holds_meta.append((h['time'], c, h['end_time']))
    for col in by_col:
        col.sort(key=lambda n: n['time'])
    return by_col, holds_meta


def _collect_ln_intervals(sim, keycount):
    """Per-column sorted list of `(head, end)` for every LN. Intervals
    within a column don't overlap in a valid chart."""
    out = [[] for _ in range(keycount)]
    for r in sim:
        if r['is_hold'] and r['end_time'] is not None:
            out[r['col']].append((r['time'], r['end_time']))
    for col in out:
        col.sort()
    return out


def _inside_any_ln(intervals, t):
    """True if `t` falls inside any of `intervals` (sorted, non-overlapping)."""
    if not intervals:
        return False
    # Rightmost head <= t; check its end.
    idx = bisect.bisect_right(intervals, (t, float('inf'))) - 1
    if idx < 0:
        return False
    head, end = intervals[idx]
    return head <= t <= end


def _ghost_taps(sim, key_events_by_col, keycount):
    """Presses the sim never assigned to a note, excluding re-presses
    inside any LN interval (those are attempts at the LN, not ghosts)."""
    used = [set() for _ in range(keycount)]
    for r in sim:
        if r['press_t'] is not None:
            used[r['col']].add(int(r['press_t']))
    intervals = _collect_ln_intervals(sim, keycount)

    taps = []
    for c in range(keycount):
        for t, is_press in key_events_by_col[c]:
            t = int(t)
            if not is_press or t in used[c]:
                continue
            if _inside_any_ln(intervals[c], t):
                continue
            taps.append((t, c))
    return taps


def _iter_press_spans(events):
    """Yield `(press_t, release_t_or_None)` for each contiguous press
    span. `release_t` is None if the stream ends while still pressed."""
    press_start = None
    for t, is_press in events:
        if is_press:
            if press_start is None:
                press_start = int(t)
        else:
            if press_start is not None:
                yield press_start, int(t)
                press_start = None
    if press_start is not None:
        yield press_start, None


def _first_overlapping_missed_ln(missed, span_lo, span_hi):
    """Return `(ln_head, press_lo, press_hi)` for the first missed LN
    in `missed` that overlaps the press span, or None.

    `span_hi=None` means the stream ended while still held; we clip the
    reported release at the LN end so the renderer doesn't draw a bar
    off the end of the chart. An empty (lo==hi) span never overlaps."""
    if span_hi is None:
        for ln_head, ln_end in missed:
            if ln_end > span_lo:
                return ln_head, span_lo, ln_end
        return None
    if span_hi <= span_lo:
        return None
    for ln_head, ln_end in missed:
        if ln_end < span_lo or ln_head > span_hi:
            continue
        return ln_head, span_lo, span_hi
    return None


def _miss_holds(sim, key_events_by_col, keycount):
    """For each missed LN, emit `(ln_head, col, press_t, release_t)` for
    the first press span that overlaps it. Rendered as a red stroke
    showing how long the player actually held the key for a miss;
    deduped so a single span that straddles N consecutive missed LNs
    produces one entry, not N."""
    missed_by_col = [[] for _ in range(keycount)]
    for r in sim:
        if (r['is_hold'] and r['judgement'] == 'miss'
                and r['end_time'] is not None):
            missed_by_col[r['col']].append((r['time'], r['end_time']))
    for col in missed_by_col:
        col.sort()

    out = []
    for c in range(keycount):
        missed = missed_by_col[c]
        if not missed:
            continue
        for span_lo, span_hi in _iter_press_spans(key_events_by_col[c]):
            hit = _first_overlapping_missed_ln(missed, span_lo, span_hi)
            if hit is not None:
                ln_head, lo, hi = hit
                out.append((ln_head, c, lo, hi))
    return out


def _build_arrays(sim):
    """Flatten sim results into the parallel-array shape expected by the
    rest of the pipeline, plus `hold_releases` for the renderer.

    For misses, preserve the actual press offset when the player did
    press (even if out-of-window) so the renderer can draw a red
    hit-line at the press position; `miss_pressed` flags that case."""
    rows, offs, cols, nt, miss_flag, miss_pressed = [], [], [], [], [], []
    hold_releases = []
    for r in sim:
        rows.append(r['time'])
        cols.append(r['col'])
        nt.append(0)

        is_miss = (r['head_off'] is None or r['judgement'] == 'miss')
        if is_miss and r['head_off'] is None:
            offs.append(1.0)
            miss_pressed.append(False)
        else:
            offs.append(r['head_off'] / 1000.0)
            miss_pressed.append(is_miss)
        miss_flag.append(is_miss)

        if r['is_hold']:
            hold_releases.append(
                (r['time'], r['col'], r['end_time'], r['tail_off']))

    order = np.argsort(rows, kind='stable')
    return {
        'noterows': np.array(rows, dtype=np.int64)[order],
        'offsets': np.array(offs, dtype=np.float64)[order],
        'columns': np.array(cols, dtype=np.int32)[order],
        'notetypes': np.array(nt, dtype=np.int32)[order],
        'misses': np.array(miss_flag, dtype=bool)[order],
        'miss_pressed': np.array(miss_pressed, dtype=bool)[order],
        'hold_releases': hold_releases,
    }


def _resolve_osu_path(osr_path, osu_path, songs_dir, beatmap_hash):
    if osu_path is not None:
        return osu_path
    if songs_dir is None:
        raise ValueError("need osu_path or songs_dir")
    found = find_osu_by_hash(beatmap_hash, songs_dir)
    if found is None:
        raise FileNotFoundError(f"no .osu match for hash {beatmap_hash}")
    return found


def parse_replay(osr_path, osu_path=None, songs_dir=None, hit_window_ms=None):
    """Parse `.osr` + matching `.osu`. Runs the osu!mania Classic judge
    sim per column so each press/release is assigned to a specific note.

    `hit_window_ms` is retained for signature compatibility but ignored:
    windows come from the chart's OD via `stable_hit_windows()`."""
    del hit_window_ms
    keycount_from_osr, events, meta = parse_osr_events(osr_path)
    osu_path = _resolve_osu_path(osr_path, osu_path, songs_dir,
                                 meta['beatmap_hash'])
    chart = parse_osu_file(osu_path)
    keycount = chart.get('keycount') or max(keycount_from_osr, 4)

    # Re-run stable's Fisher-Yates with the replay's seed to reproduce
    # the Random-mod column permutation that was active during the play.
    col_perm = None
    if (meta.get('mods', 0) & OSU_MOD_RANDOM) and meta.get('rng_seed') is not None:
        col_perm = _random_column_permutation(keycount, meta['rng_seed'])

    by_col_notes, holds_meta = _group_notes_by_col(
        chart['hitobjects'], keycount, col_perm)
    key_events_by_col = extract_key_events(events, keycount)
    windows = stable_hit_windows(float(chart.get('od', 8.0)))
    sim = simulate_mania(by_col_notes, key_events_by_col, windows)

    arrays = _build_arrays(sim)
    sv_sections = chart.get('sv_sections', [])
    bpms = _osu_bpms_from_timing_points(chart.get('timing_points', []))
    sv_doc = SvReplayDoc(
        engine_kind=KIND_TIME_SPACE,
        engine_key='osu_time',
        sections=list(sv_sections),
        bpms=bpms,
    )
    return {
        **arrays,
        'sv': sv_doc,
        'holds': holds_meta,
        'ghost_taps': _ghost_taps(sim, key_events_by_col, keycount),
        'miss_holds': _miss_holds(sim, key_events_by_col, keycount),
        'keycount': keycount,
        'filepath': str(osr_path),
        'chart_path': str(osu_path),
        'meta': meta,
        'chart_meta': {k: chart[k] for k in
                       ('title', 'artist', 'creator', 'version', 'keycount')},
        'od': float(chart.get('od', 8.0)),
        'mods': int(meta.get('mods', 0)),
    }


def _perfect_sim(by_col_notes):
    """Autoplay sim records: every note hit dead-on. Mirrors the dict
    shape `simulate_mania` produces (see `judge._new_result`) so
    `_build_arrays` and the ghost/miss-hold post-passes treat a synth
    identically to a real play - zero offsets, no misses, LN tails
    released exactly on time."""
    sim = []
    for col, notes in enumerate(by_col_notes):
        for note in notes:
            end = note['end_time']
            is_hold = end is not None
            sim.append({
                'col': col, 'time': note['time'], 'end_time': end,
                'is_hold': is_hold,
                'press_t': note['time'], 'release_t': end,
                'head_off': 0.0, 'tail_off': 0.0 if is_hold else None,
                'judgement': 'MAX', 'broken': False, 'missed': False,
            })
    return sim


def autoplay_replay(osu_path):
    """Synthesize a perfect autoplay replay from a `.osu` chart alone -
    no `.osr` decode, no judge sim. Used by the library's unplayed-charts
    feature: the entry's `replay_path` is the chart file, and this fills
    the same dict `parse_replay` returns (minus the player-derived
    ghost taps / miss holds, which a flawless play has none of)."""
    chart = parse_osu_file(osu_path)
    keycount = chart.get('keycount') or 4
    by_col_notes, holds_meta = _group_notes_by_col(
        chart['hitobjects'], keycount, col_perm=None)
    sim = _perfect_sim(by_col_notes)

    arrays = _build_arrays(sim)
    sv_doc = SvReplayDoc(
        engine_kind=KIND_TIME_SPACE,
        engine_key='osu_time',
        sections=list(chart.get('sv_sections', [])),
        bpms=_osu_bpms_from_timing_points(chart.get('timing_points', [])),
    )
    return {
        **arrays,
        'sv': sv_doc,
        'holds': holds_meta,
        'ghost_taps': [],
        'miss_holds': [],
        'keycount': keycount,
        'filepath': str(osu_path),
        'chart_path': str(osu_path),
        'meta': {'mods': 0},
        'chart_meta': {k: chart[k] for k in
                       ('title', 'artist', 'creator', 'version', 'keycount')},
        'od': float(chart.get('od', 8.0)),
        'mods': 0,
    }


def _osu_bpms_from_timing_points(timing_points):
    """Project uninherited timing points to (beat, bpm) pairs in
    Etterna's beat-space convention. Beats accumulate from t=0 along the
    BPM segment in effect. Returns at least one segment so beat-space
    consumers always have a base BPM."""
    uninherited = sorted(
        ((t_ms, mpb) for t_ms, mpb in timing_points if mpb > 0),
        key=lambda x: x[0],
    )
    if not uninherited:
        return [(0.0, 120.0)]
    out = []
    cur_beat = 0.0
    cur_t_ms = 0.0
    cur_bpm = 60000.0 / uninherited[0][1]
    out.append((0.0, cur_bpm))
    for t_ms, mpb in uninherited[1:]:
        dt_ms = max(0.0, t_ms - cur_t_ms)
        cur_beat += (dt_ms / 1000.0) * (cur_bpm / 60.0)
        cur_bpm = 60000.0 / mpb
        cur_t_ms = t_ms
        out.append((cur_beat, cur_bpm))
    return out
