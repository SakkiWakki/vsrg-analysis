"""Top-level fluXis `parse_replay`: ties `.frp` decoding, `.fsc`/`.ffx`
parsing, and the judgement sim into a dict shaped like the other game
parsers' output.

SV routes through the shared time-density integral engine (the
`KIND_TIME_SPACE` integrator the Quaver charts use): fluXis scroll
velocities are the same `(time, multiplier)` density stream, negative
multipliers included. Lane switches become a `_fluxis_lane_mask`
timeline the adapter's `lane_mask()` hook exposes to the renderer; the
raw effect streams are stashed for the future modchart renderer.
"""
from __future__ import annotations

import numpy as np

from analysis.games.fluxis import fsc_chart
from analysis.games.fluxis.fsc_chart import parse_fsc
from analysis.games.fluxis.frp_replay import extract_key_events, parse_frp
from analysis.games.fluxis.judge_sim import (landmine_windows_ms,
                                              simulate_landmines,
                                              simulate_mania)
from analysis.games.fluxis.lane_switch import build_lane_mask_timeline
from analysis.games.quaver.parse import (_build_mine_arrays,
                                          _quaver_bpms_to_beat_space)
from analysis.player.sv.replay_doc import SvReplayDoc, KIND_TIME_SPACE


def parse_replay(frp_path, fsc_path, rate=1.0):
    chart = parse_fsc(fsc_path)
    keycount = chart['keycount']
    difficulty = chart['accuracy_difficulty']

    frames, player_id = parse_frp(frp_path)
    key_events = extract_key_events(frames, keycount)

    notes_by_col, ticks_by_col, mines_by_col = _split_hitobjects(
        chart['hitobjects'], keycount)
    sim = simulate_mania(notes_by_col, ticks_by_col, key_events,
                         difficulty, rate)
    arrays = _build_arrays(sim)

    mine_miss_w = landmine_windows_ms(difficulty, rate)[0][1]
    mine_hits, mines_avoided = simulate_landmines(
        mines_by_col, key_events, mine_miss_w)
    mine_arrays = _build_mine_arrays(mines_by_col, mine_hits)

    return {
        **arrays,
        **mine_arrays,
        'sv': _sv_doc(chart),
        'keycount': keycount,
        'rate': float(rate),
        'accuracy_difficulty': difficulty,
        'filepath': str(frp_path),
        'chart_path': str(fsc_path),
        'meta': {'player_id': player_id,
                 'mines_avoided': mines_avoided},
        **_chart_context(chart, keycount),
    }


def _sv_doc(chart):
    return SvReplayDoc(
        engine_kind=KIND_TIME_SPACE,
        engine_key='quaver_time',
        sections=[(t_ms / 1000.0, mult)
                  for t_ms, mult in chart['scroll_velocities']],
        initial_velocity=1.0,
        bpms=_quaver_bpms_to_beat_space(chart['timing_points']),
    )


def _chart_context(chart, keycount):
    """The chart-side half of a replay dict: display metadata plus the
    raw streams the renderer and the future modchart path read."""
    return {
        'chart_meta': {
            'title': chart['title'], 'artist': chart['artist'],
            'creator': chart['mapper'], 'version': chart['difficulty'],
            'keycount': keycount,
        },
        '_fluxis_audio_file': chart['audio'],
        '_fluxis_background_file': chart['background'],
        '_fluxis_storyboard_file': chart['storyboard'],
        '_fluxis_lane_mask': build_lane_mask_timeline(
            chart['lane_switches'], keycount, v2=chart['ls_v2']),
        '_fluxis_effect_streams': chart['effect_streams'],
        '_fluxis_colors': chart['colors'],
        '_fluxis_timing_points': chart['timing_points'],
        '_fluxis_hitobjects': chart['hitobjects'],
        '_fluxis_scroll_velocities': chart['scroll_velocities'],
        '_fluxis_end_time': _chart_end_ms(chart['hitobjects']),
    }


def autoplay_replay(fsc_path):
    """Synthesize a perfect autoplay replay from a `.fsc` chart alone -
    no `.frp` decode, no judge sim. Powers the library's unplayed-charts
    feature: the entry's `replay_path` is the chart file, and this fills
    the same dict `parse_replay` returns. Landmines are charted but
    never detonated (a flawless play hits none)."""
    chart = parse_fsc(fsc_path)
    keycount = chart['keycount']

    notes_by_col, ticks_by_col, mines_by_col = _split_hitobjects(
        chart['hitobjects'], keycount)
    arrays = _build_arrays(_perfect_sim(notes_by_col, ticks_by_col))
    return {
        **arrays,
        **_build_mine_arrays(mines_by_col, []),
        'sv': _sv_doc(chart),
        'keycount': keycount,
        'rate': 1.0,
        'accuracy_difficulty': chart['accuracy_difficulty'],
        'filepath': str(fsc_path),
        'chart_path': str(fsc_path),
        'meta': {'player_id': None,
                 'mines_avoided': sum(len(col) for col in mines_by_col)},
        **_chart_context(chart, keycount),
    }


def _perfect_sim(notes_by_col, ticks_by_col):
    """Autoplay sim records: every note hit dead-on and every hold held
    through its ticks. Mirrors the dict shape `simulate_mania` produces
    so `_build_arrays` treats a synth identically to a real play."""
    sim = []
    for col, notes in enumerate(notes_by_col):
        for note in notes:
            sim.append({
                'time': note['time'], 'col': col,
                'head_off': 0.0, 'judgement': 'flawless',
                'is_hold': bool(note['is_hold']),
                'end_time': note['end_time'],
                'tail_off': 0.0 if note['is_hold'] else None,
                'is_tick': False,
            })
    for col, ticks in enumerate(ticks_by_col):
        for tick in ticks:
            sim.append({
                'time': tick['time'], 'col': col,
                'head_off': 0.0, 'judgement': 'flawless',
                'is_hold': False, 'end_time': tick['end_time'],
                'tail_off': None, 'is_tick': True,
            })
    return sim


def _chart_end_ms(hitobjects):
    """Port MapInfo.EndTime: the last hit object's end time (its tail for
    holds), or 1000ms when the chart is empty."""
    if not hitobjects:
        return 1000.0
    return max(h['end_time'] if h['end_time'] is not None else h['time']
               for h in hitobjects)


def _split_hitobjects(hitobjects, keycount):
    notes = [[] for _ in range(keycount)]
    ticks = [[] for _ in range(keycount)]
    mines = [[] for _ in range(keycount)]
    for h in hitobjects:
        c = h['column']
        if c >= keycount:
            continue
        record = {'time': h['time'], 'end_time': h['end_time'],
                  'is_hold': h['is_hold']}
        match h['type']:
            case fsc_chart.HO_LANDMINE:
                mines[c].append(record)
            case fsc_chart.HO_TICK:
                ticks[c].append(record)
            case _:
                notes[c].append(record)
    for group in (notes, ticks, mines):
        for col in group:
            col.sort(key=lambda n: n['time'])
    return notes, ticks, mines


def _build_arrays(sim):
    """Mirror the Quaver `_build_arrays` contract: per-note rows (ms),
    offsets (s), misses, and hold-release metadata. Ticks ride in the
    main stream with `NT_TICK` so the renderer colors them distinctly."""
    from analysis.player.notetypes import NT_TAP, NT_TICK

    rows, offs, cols, nt = [], [], [], []
    miss_flag, miss_pressed = [], []
    hold_releases = []
    for r in sim:
        rows.append(int(round(r['time'])))
        cols.append(r['col'])
        nt.append(NT_TICK if r['is_tick'] else NT_TAP)

        is_miss = (r['head_off'] is None or r['judgement'] == 'miss')
        if r['head_off'] is None:
            offs.append(1.0)
            miss_pressed.append(False)
        else:
            offs.append(r['head_off'] / 1000.0)
            miss_pressed.append(is_miss)
        miss_flag.append(is_miss)

        if r['is_hold']:
            tail_off = (r['tail_off'] / 1000.0
                        if r['tail_off'] is not None else None)
            hold_releases.append(
                (int(round(r['time'])), r['col'],
                 r['end_time'], tail_off))

    order = np.argsort(rows, kind='stable')
    holds = [(t, c, e) for t, c, e, _off in hold_releases]
    return {
        'noterows': np.array(rows, dtype=np.int64)[order],
        'offsets': np.array(offs, dtype=np.float64)[order],
        'columns': np.array(cols, dtype=np.int32)[order],
        'notetypes': np.array(nt, dtype=np.int32)[order],
        'misses': np.array(miss_flag, dtype=bool)[order],
        'miss_pressed': np.array(miss_pressed, dtype=bool)[order],
        'hold_releases': hold_releases,
        'holds': holds,
    }
