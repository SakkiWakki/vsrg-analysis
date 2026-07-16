"""NotITG library scan: chart-only entries.

NotITG has no replay system, so the library lists the charts
themselves: one entry per difficulty of every simfile under
`Songs/<pack>/<song>/`, flagged `unplayed` (the library's unplayed
toggle filters on it) with score-ish fields zeroed. An entry's
`replay_path` is `<simfile path>::<chart index>`; the adapter's
`parse_replay` resolves that reference into a synthesized autoplay
replay, so the rest of the pipeline never knows the difference.
"""
from __future__ import annotations

from pathlib import Path

from analysis.games.etterna.sm_chart import (NT_HOLD_HEAD, NT_ROLL_HEAD,
                                             NT_TAP, is_beat_in_warp,
                                             parse_notes_block, parse_sm,
                                             stepstype_keycount)

# Replay-stream convention inherited from Etterna: roll heads are
# encoded as hold heads; the chart extras' `roll_heads` set recolors.
HEAD_TYPES = (NT_HOLD_HEAD, NT_ROLL_HEAD)


def judged_notes(chart) -> list:
    """The chart's judgeable (row, col, nt) notes: taps and hold/roll
    heads outside warp-eaten regions. Shared by the library scan
    (skip decoy charts with nothing to judge, e.g. UKSRT one-mine
    troll difficulties) and the autoplay synthesizer."""
    notes, _holds = parse_notes_block(chart['notedata'])
    warps = chart.get('warps') or []
    stops = chart.get('stops') or []
    delays = chart.get('delays') or []
    return sorted(
        (row, col, nt) for row, col, nt in notes
        if nt in (NT_TAP, *HEAD_TYPES)
        and not is_beat_in_warp(row / 48.0, warps, stops, delays))


def chart_ref(sm_path, chart_index: int) -> str:
    return f'{sm_path}::{chart_index}'


def split_chart_ref(ref) -> tuple:
    """(simfile Path, chart index) from a chart ref; a bare path means
    chart 0."""
    text = str(ref)
    base, sep, index = text.rpartition('::')
    if not sep or not index.isdigit():
        return Path(text), 0
    return Path(base), int(index)


def _song_entries(sm_path: Path, pack: str) -> list:
    data = parse_sm(sm_path)
    title = data.get('title') or sm_path.parent.name
    artist = data.get('artist') or '?'
    mtime = sm_path.stat().st_mtime

    entries = []
    for index, chart in enumerate(data['charts']):
        keycount = stepstype_keycount(chart.get('stepstype', ''))
        if not keycount or not chart.get('notedata', '').strip():
            continue
        if not judged_notes(chart):
            continue
        steps = ' '.join(part for part in (chart.get('difficulty', ''),
                                           chart.get('meter', ''))
                         if part)
        entries.append({
            'game': 'notitg',
            'unplayed': True,
            'replay_path': chart_ref(sm_path, index),
            'chart_path': str(sm_path),
            'song': f'{artist} - {title}',
            'pack': pack,
            'steps': steps,
            'keycount': keycount,
            'rate': 1.0,
            'modifiers': None,
            'wife': 0.0,
            'grade': '',
            'judgments': {},
            'datetime': '',
            'mtime': mtime,
            'ssrs': {},
            'maxcombo': 0,
        })
    return entries


def simfile_paths(songs_dir) -> list:
    return sorted(Path(songs_dir).glob('*/*/*.sm'))


def scan_songs(songs_dir, progress=None) -> list:
    entries = []
    paths = simfile_paths(songs_dir)
    for i, sm_path in enumerate(paths):
        if progress and i % 25 == 0:
            progress(f'notitg: scanning charts... {i}/{len(paths)}')
        try:
            entries.extend(_song_entries(sm_path, sm_path.parent.parent.name))
        except Exception:
            # Malformed community simfiles must never kill the scan.
            continue
    return entries
