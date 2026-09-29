"""NotITG library scan: the chart catalogue.

NotITG has no replay system, so every library entry is a chart: one
per difficulty of every simfile under `Songs/<pack>/<song>/`. The
shared unplayed pass (`analysis.core.unplayed`) turns these into
entries, and an entry's `replay_path` is a chart ref the adapter opens
as a perfect autoplay.
"""
from __future__ import annotations

from pathlib import Path

from analysis.games.etterna.chart_ref import chart_ref, judged_notes
from analysis.games.etterna.sm_chart import parse_sm, stepstype_keycount


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
            'replay_path': chart_ref(sm_path, index),
            'chart_path': str(sm_path),
            'song': f'{artist} - {title}',
            'pack': pack,
            'steps': steps,
            'keycount': keycount,
            'modifiers': None,
            'mtime': mtime,
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
