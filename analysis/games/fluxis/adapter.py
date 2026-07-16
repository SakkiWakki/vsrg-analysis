"""fluXis game adapter: library scan via the realm database dump.

The realm dump carries everything the library tab shows (song,
difficulty, keycount, accuracy, grade, judgment counts, date), so
building entries never parses a chart or replay. Playback parsing
(`.fsc`/`.frp` -> player) is not implemented yet; opening an entry
raises a clear error until those parsers land.
"""
from __future__ import annotations

import re
from pathlib import Path

from analysis.core.cache import Cache
from analysis.core.game import GameAdapter

_LIBRARY_CACHE = Cache('fluxis_library.pkl')

# fluXis rate mods appear in RealmScore.Mods as e.g. "1.2x" tokens.
_RATE_MOD_RE = re.compile(r'(\d+(?:\.\d+)?)x')

_JUDGEMENT_FIELDS = ('Flawless', 'Perfect', 'Great', 'Alright',
                     'Okay', 'Miss')


def _rate_from_mods(mods: str) -> float:
    m = _RATE_MOD_RE.search(mods or '')
    return float(m.group(1)) if m else 1.0


def _iso_datetime(raw: str) -> str:
    # Realm dates dump as ISO-8601 with offset; the library column uses
    # the same 'YYYY-MM-DD HH:MM:SS' shape as every other game.
    return (raw or '')[:19].replace('T', ' ')


def _score_entry(score, maps, dirs):
    m = maps.get(score.get('MapID'), {})
    set_id = m.get('MapSetID')
    file_name = m.get('FileName')

    chart_path = None
    if set_id and file_name and dirs['maps_dir'] is not None:
        candidate = dirs['maps_dir'] / str(set_id) / str(file_name)
        chart_path = str(candidate) if candidate.is_file() else None

    replay_path = dirs['replays_dir'] / f"{score['ID']}.frp"
    if not replay_path.is_file():
        return None

    artist = m.get('Metadata.Artist') or '?'
    title = m.get('Metadata.Title') or '?'
    judgments = {name.lower(): int(score.get(name) or 0)
                 for name in _JUDGEMENT_FIELDS}

    return {
        'game': 'fluxis',
        'replay_path': str(replay_path),
        'beatmap_hash': m.get('Hash', ''),
        'song': f'{artist} - {title}',
        'pack': m.get('Metadata.Mapper', ''),
        'steps': m.get('Difficulty', ''),
        'keycount': int(m.get('KeyCount') or 0) or None,
        'chart_path': chart_path,
        'rate': _rate_from_mods(score.get('Mods', '')),
        'modifiers': score.get('Mods') or None,
        'wife': float(score.get('Accuracy') or 0.0) / 100.0,
        'grade': score.get('Grade', ''),
        'judgments': judgments,
        'datetime': _iso_datetime(score.get('Date', '')),
        'mtime': replay_path.stat().st_mtime,
        'ssrs': {},
        'maxcombo': int(score.get('MaxCombo') or 0),
    }


class FluxisAdapter(GameAdapter):
    name = 'fluxis'

    def parse_replay(self, path, chart_path=None):
        raise NotImplementedError(
            'fluXis playback is not implemented yet; the library scan '
            'works, but .fsc/.frp parsing is still to come')

    # --- library scan --------------------------------------------------

    def scan_library(self, progress=None):
        from analysis.games.fluxis.paths import find_fluxis_dirs
        from analysis.games.fluxis.realm_reader import dump_realm

        dirs = find_fluxis_dirs()
        if dirs['realm_path'] is None:
            return []
        if progress:
            progress('fluxis: reading realm database…')
        dump = dump_realm(dirs['realm_path'], progress=progress)
        if dump is None:
            return []

        maps = {m['ID']: m for m in dump.get('RealmMap', [])}
        entries = []
        for score in dump.get('RealmScore', []):
            entry = _score_entry(score, maps, dirs)
            if entry is not None:
                entries.append(entry)
        return entries

    # --- library cache lifecycle ----------------------------------------

    def load_cached(self):
        return _LIBRARY_CACHE.load()

    def save_cached(self, entries):
        _LIBRARY_CACHE.save([e for e in entries if e.get('game') == 'fluxis'])

    def rebuild(self, progress=None):
        _LIBRARY_CACHE.clear()
        entries = self.scan_library(progress=progress)
        if entries:
            _LIBRARY_CACHE.save(entries)
        return entries

    def incremental_update(self, progress=None):
        """The realm dump is one cheap subprocess, but it still costs a
        dotnet startup; skip it when the replay folder matches the cache
        exactly (fluXis writes one .frp per score)."""
        cached = _LIBRARY_CACHE.load()
        if cached is None:
            return self.rebuild(progress=progress)

        from analysis.games.fluxis.paths import find_fluxis_dirs
        replays_dir = find_fluxis_dirs()['replays_dir']
        on_disk = (sorted(str(p) for p in replays_dir.glob('*.frp'))
                   if replays_dir is not None else [])
        known = sorted(e['replay_path'] for e in cached)
        if on_disk == known:
            return cached
        return self.rebuild(progress=progress)


ADAPTER = FluxisAdapter()
