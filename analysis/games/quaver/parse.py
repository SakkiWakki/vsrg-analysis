"""Top-level Quaver `parse_replay`: ties `.qr` decoding, `.qua` parsing,
and the judgement sim into a single entry point matching the shape
`analysis/games/osu/replay/parse.parse_replay` produces.

The output dict carries everything `analysis/player/sv/render._build_registry`
needs to detect a Quaver chart (`_quaver_sv_sections`,
`_quaver_initial_velocity`) plus the column-major notes the player and
judgement layers consume.
"""
from __future__ import annotations

import numpy as np

from analysis.games.quaver.qr_replay import (parse_qr_events,
                                              extract_key_events)
from analysis.games.quaver.qua_chart import (parse_qua_file,
                                              find_qua_by_hash)
from analysis.games.quaver.judge_sim import (simulate_mania, simulate_mines,
                                              windows_ms)
from analysis.player.sv.replay_doc import SvReplayDoc, KIND_TIME_SPACE


def parse_replay(qr_path, qua_path=None, songs_dir=None, judge='Standard'):
    """Parse a `.qr` + matching `.qua` (resolved either explicitly or by
    MD5 lookup against `songs_dir`). Returns a dict shaped like the osu
    parser's output plus Quaver-specific SV fields."""
    keycount_from_qr, events, meta = parse_qr_events(qr_path)
    qua_path = _resolve_qua_path(qr_path, qua_path, songs_dir, meta['map_md5'])
    chart = parse_qua_file(qua_path)
    keycount = chart.get('keycount') or max(keycount_from_qr, 4)

    by_col_notes, holds_meta, mines_by_col = _group_notes_by_col(
        chart['hitobjects'], keycount)
    key_events_by_col = extract_key_events(events, keycount)
    windows = windows_ms(judge)
    sim = simulate_mania(by_col_notes, key_events_by_col, windows)
    mine_hits = simulate_mines(mines_by_col, key_events_by_col, windows[0])
    mine_arrays = _build_mine_arrays(mines_by_col, mine_hits)

    # Map each (time_ms, column) -> TimingGroup id so the renderer can
    # bind a per-note group array parallel to the simulator's output.
    # Charts without TimingGroups put every note in `$Default`.
    note_group_map = {(int(h['time']), int(h['column'])): h['group']
                      for h in chart['hitobjects']}
    arrays = _build_arrays(sim, note_group_map)
    note_groups = arrays.pop('_quaver_note_groups')
    bpms = _quaver_bpms_to_beat_space(chart['timing_points'])
    sv_doc = SvReplayDoc(
        engine_kind=KIND_TIME_SPACE,
        engine_key='quaver_time',
        sections=list(chart['sv_sections']),
        initial_velocity=float(chart['initial_velocity']),
        groups=chart['groups'],
        bpms=bpms,
        note_groups=note_groups,
        flags={'legacy_ln': bool(chart.get('legacy_ln_rendering', False))},
    )
    return {
        **arrays,
        **mine_arrays,
        'sv': sv_doc,
        'holds': holds_meta,
        'keycount': keycount,
        'filepath': str(qr_path),
        'chart_path': str(qua_path),
        'meta': meta,
        'chart_meta': {k: chart[k]
                       for k in ('title', 'artist', 'creator', 'version',
                                 'keycount')},
        '_quaver_audio_file': chart.get('audio', ''),
        'judge': judge,
        'mods': int(meta.get('mods', 0)),
    }


def _resolve_qua_path(qr_path, qua_path, songs_dir, map_md5):
    if qua_path is not None:
        return qua_path
    if songs_dir is None:
        raise ValueError('need qua_path or songs_dir')
    found = find_qua_by_hash(map_md5, songs_dir)
    if found is None:
        raise FileNotFoundError(f'no .qua match for hash {map_md5}')
    return found


def _group_notes_by_col(hitobjects, keycount):
    """Split hitobjects into per-column tap/hold notes, hold metadata,
    and per-column mines. Mines never enter the judgment note stream;
    they detonate via `simulate_mines` and render as a chart stream."""
    by_col = [[] for _ in range(keycount)]
    mines_by_col = [[] for _ in range(keycount)]
    holds_meta = []
    for h in hitobjects:
        c = h['column']
        if c >= keycount:
            continue
        record = {'time': int(h['time']), 'end_time': h['end_time']}
        if h.get('is_mine'):
            mines_by_col[c].append(record)
            continue
        by_col[c].append(record)
        if h['is_hold']:
            holds_meta.append((h['time'], c, h['end_time']))
    for col in by_col:
        col.sort(key=lambda n: n['time'])
    for col in mines_by_col:
        col.sort(key=lambda n: n['time'])
    return by_col, holds_meta, mines_by_col


def _build_mine_arrays(mines_by_col, mine_hits):
    """Chart-stream arrays for the renderer, same contract Etterna's
    adapter fills: `mine_times`/`mine_cols`/`mine_until` (time-sorted;
    no `mine_rows` -- Quaver is time-based, and empty rows route the SV
    projection through time-space). Plus the detonations:
    `mine_hit_idx` (index into the sorted mine arrays) and
    `mine_hit_press` (press time, seconds)."""
    flat = [(m['time'], c, m['end_time'] or 0)
            for c, mines in enumerate(mines_by_col) for m in mines]
    if not flat:
        return {}
    flat.sort()

    index_of = {(t, c): i for i, (t, c, _e) in enumerate(flat)}
    hit_pairs = [(index_of[(h['mine_time'], h['col'])],
                  h['press_time'] / 1000.0) for h in mine_hits]
    hit_pairs.sort()

    return {
        'mine_times': np.array([t / 1000.0 for t, _c, _e in flat],
                               dtype=np.float64),
        'mine_cols': np.array([c for _t, c, _e in flat], dtype=np.int32),
        'mine_until': np.full(len(flat), np.inf, dtype=np.float64),
        'mine_end_times': np.array(
            [e / 1000.0 if e else np.nan for _t, _c, e in flat],
            dtype=np.float64),
        'mine_hit_idx': np.array([i for i, _p in hit_pairs], dtype=np.int64),
        'mine_hit_press': np.array([p for _i, p in hit_pairs],
                                   dtype=np.float64),
    }


def _build_arrays(sim, note_group_map):
    """Mirror `analysis/games/osu/replay/parse._build_arrays`; misses
    that came with a press preserve `head_off` so the renderer draws
    the player's actual press position, not the note's spawn point.

    `note_group_map` keys each note (time_ms, column) to its TimingGroup
    id so we can return a `_quaver_note_groups` array parallel to the
    sorted noterows."""
    rows, offs, cols, nt, miss_flag, miss_pressed = [], [], [], [], [], []
    groups = []
    hold_releases = []
    for r in sim:
        rows.append(r['time'])
        cols.append(r['col'])
        nt.append(0)
        groups.append(note_group_map.get((int(r['time']), int(r['col'])),
                                          '$Default'))

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
        '_quaver_note_groups': np.array(groups, dtype=object)[order],
        'hold_releases': hold_releases,
    }


def _quaver_bpms_to_beat_space(timing_points):
    """Project Quaver's `(time_ms, bpm)` timing points to `(beat, bpm)`
    pairs, matching `analysis/games/osu/replay/parse._osu_bpms_from_timing_points`.
    Beats accumulate from `t=0` along the active BPM segment."""
    if not timing_points:
        return [(0.0, 120.0)]
    out = []
    cur_beat = 0.0
    cur_t_ms = 0.0
    cur_bpm = float(timing_points[0][1]) or 120.0
    out.append((0.0, cur_bpm))
    for t_ms, bpm in timing_points[1:]:
        dt_ms = max(0.0, t_ms - cur_t_ms)
        cur_beat += (dt_ms / 1000.0) * (cur_bpm / 60.0)
        cur_bpm = float(bpm) or cur_bpm
        cur_t_ms = t_ms
        out.append((cur_beat, cur_bpm))
    return out
