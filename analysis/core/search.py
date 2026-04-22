"""Unified replay library search. Iterates over every registered
`GameAdapter` and merges their `scan_library()` results into one filterable
list. Adding a new game = register its adapter; no edits here."""
import os
import time
import pickle
from pathlib import Path
from datetime import datetime

from analysis import cache_dir
from analysis.core import game as game_mod


CACHE_PATH = cache_dir() / 'library.pkl'


def _ensure_cache_dir():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _hash_one(p):
    import hashlib
    try:
        with open(p, 'rb') as f:
            return str(p), hashlib.md5(f.read()).hexdigest()
    except OSError:
        return str(p), None


def enrich_osu_with_charts(entries, songs_dir=None, progress=None):
    """Resolve song titles for osu entries by hashing .osu files in parallel."""
    from analysis.games.osu.replay import find_osu_dirs, parse_osu_file
    from concurrent.futures import ThreadPoolExecutor
    if songs_dir is None:
        songs_dir = find_osu_dirs().get('songs_dir')
    if not songs_dir:
        return entries
    # Group by hash: multiple replays can share a beatmap_hash (same chart
    # played N times). The previous `{hash: entry}` dict dropped all but one,
    # leaving duplicates unenriched forever.
    hashes = {}
    for e in entries:
        if e['game'] == 'osu' and e.get('beatmap_hash'):
            hashes.setdefault(e['beatmap_hash'], []).append(e)
    if not hashes:
        return entries

    paths = list(Path(songs_dir).rglob('*.osu'))
    if progress:
        progress(f'hashing {len(paths)} .osu files…')
    max_workers = min(32, (os.cpu_count() or 4) * 4)
    matched_entries = 0
    matched_hashes = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, (path_str, h) in enumerate(ex.map(_hash_one, paths, chunksize=32)):
            if h is None:
                continue
            if h in hashes:
                try:
                    chart = parse_osu_file(path_str)
                    song = f"{chart.get('artist','?')} - {chart.get('title','?')}"
                    version = chart.get('version', '')
                    creator = chart.get('creator', '')
                    keycount = chart.get('keycount')
                    for e in hashes[h]:
                        e['song'] = song
                        e['steps'] = version
                        e['pack'] = creator or e.get('pack', '')
                        e['keycount'] = keycount
                        e['chart_path'] = path_str
                        matched_entries += 1
                    matched_hashes += 1
                except Exception:
                    pass
            if progress and i % 500 == 0:
                progress(f'hashed {i}/{len(paths)} — '
                         f'{matched_hashes}/{len(hashes)} charts matched')
    if progress:
        progress(f'enrichment done: {matched_hashes}/{len(hashes)} charts '
                 f'({matched_entries} replays)')
    return entries


def build_library(use_cache=True, max_age_s=24 * 3600, refresh=False,
                  enrich_osu=False, progress=None):
    """Build the unified library by asking each game adapter to scan its own
    replays. Set enrich_osu=True to resolve osu!mania song titles (slow:
    hashes every .osu file). `progress` is an optional callable(str)."""
    _ensure_cache_dir()
    if use_cache and CACHE_PATH.exists() and not refresh:
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < max_age_s:
            try:
                with open(CACHE_PATH, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
    entries = []
    for name, adapter in game_mod.all_games().items():
        if progress:
            progress(f'scanning {name}…')
        got = adapter.scan_library(progress=progress) or []
        entries.extend(got)
        if progress:
            progress(f'{name}: {len(got)} entries — total {len(entries)}')
    if enrich_osu:
        entries = enrich_osu_with_charts(entries, progress=progress)
    try:
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(entries, f)
    except OSError:
        pass
    return entries


def _parse_dt(s):
    if not s:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d'):
        try:
            return datetime.strptime(s[:19], fmt)
        except (ValueError, TypeError):
            continue
    return None


def search(entries, query=None, game=None, min_wife=None, min_rate=None,
           since=None, sort='recent', descending=True, limit=None):
    """Filter and sort entries.

    sort: 'recent' (mtime), 'date' (datetime field), 'wife', 'song', 'rate',
          'mean_offset', 'overall_ssr'.
    """
    out = list(entries)
    if game:
        out = [e for e in out if e['game'] == game]
    if min_wife is not None:
        out = [e for e in out if e.get('wife', 0) >= min_wife]
    if min_rate is not None:
        out = [e for e in out if e.get('rate', 1) >= min_rate]
    if since is not None:
        out = [e for e in out
               if (_parse_dt(e.get('datetime')) or
                   datetime.fromtimestamp(e['mtime'])) >= since]
    if query:
        q = query.lower()
        out = [e for e in out
               if q in (e.get('song') or '').lower()
               or q in (e.get('pack') or '').lower()
               or q in (e.get('steps') or '').lower()
               or q in (e.get('grade') or '').lower()]

    def key_recent(e):
        return e.get('mtime', 0)

    def key_date(e):
        d = _parse_dt(e.get('datetime'))
        return d.timestamp() if d else e.get('mtime', 0)

    def key_overall(e):
        return e.get('ssrs', {}).get('Overall', 0)

    keyfn = {
        'recent': key_recent,
        'date': key_date,
        'wife': lambda e: e.get('wife', 0),
        'song': lambda e: (e.get('song') or '').lower(),
        'pack': lambda e: (e.get('pack') or '').lower(),
        'rate': lambda e: e.get('rate', 1),
        'overall_ssr': key_overall,
        'game': lambda e: e.get('game', ''),
        'keys': lambda e: e.get('keycount') or 0,
        'grade': lambda e: (e.get('grade') or '').lower(),
        'maxcombo': lambda e: e.get('maxcombo', 0),
    }.get(sort, key_recent)

    out.sort(key=keyfn, reverse=descending)
    if limit:
        out = out[:limit]
    return out


def pretty_print(entries, limit=30):
    print(f"\n{'game':<7} {'K':>3} {'song':<46} {'pack':<22} {'rate':>5} {'wife%':>7} {'date':<19}")
    print('-' * 120)
    for e in entries[:limit]:
        song = (e.get('song') or '')[:44]
        pack = (e.get('pack') or '')[:20]
        kc = e.get('keycount') or '?'
        print(f"{e['game']:<7} {str(kc):>3} {song:<46} {pack:<22} {e.get('rate',1):>5.2f} "
              f"{e.get('wife',0)*100:>7.2f} {(e.get('datetime') or '')[:19]:<19}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('query', nargs='?', default='')
    p.add_argument('--game', choices=['etterna', 'osu'])
    p.add_argument('--sort', default='recent',
                   choices=['recent', 'date', 'wife', 'song', 'pack', 'rate',
                            'overall_ssr', 'keys', 'game', 'grade', 'maxcombo'])
    p.add_argument('--asc', action='store_true')
    p.add_argument('--min-wife', type=float)
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--refresh', action='store_true')
    p.add_argument('--no-enrich', action='store_true')
    a = p.parse_args()

    lib = build_library(refresh=a.refresh, enrich_osu=not a.no_enrich)
    from collections import Counter
    counts = Counter(e['game'] for e in lib)
    parts = ', '.join(f'{n} {g}' for g, n in counts.most_common())
    print(f"library: {len(lib)} entries ({parts})")
    res = search(lib, query=a.query or None, game=a.game,
                 min_wife=(a.min_wife / 100 if a.min_wife else None),
                 sort=a.sort, descending=not a.asc, limit=a.limit)
    pretty_print(res, limit=a.limit)
