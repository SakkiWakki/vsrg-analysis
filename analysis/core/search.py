"""Unified replay library search. Scans Etterna (ReplaysV2 + Etterna.xml)
and osu!mania (osu! Data/r + beatmap database) and returns a merged, filterable list.
"""
import os
import time
import pickle
from pathlib import Path
from datetime import datetime

from analysis.etterna.replay import (parse_etterna_xml, find_etterna_dirs,
                             find_replay_for_score)


CACHE_PATH = Path.home() / '.cache' / 'etterna-analysis' / 'library.pkl'


def _ensure_cache_dir():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _etterna_keycount_from_stepstype(st):
    mapping = {
        'dance-single': 4, 'dance-solo': 6, 'dance-double': 8,
        'pump-single': 5, 'pump-double': 10,
        'kb7-single': 7,
    }
    return mapping.get(st, 4)


def scan_etterna():
    """Return list of entry dicts for every Etterna score that has a replay file."""
    dirs = find_etterna_dirs()
    xml = dirs.get('xml_path')
    replays = dirs.get('replays_dir')
    out = []
    if not xml or not replays:
        return out
    scores = parse_etterna_xml(xml)
    rdir = Path(replays)
    for s in scores:
        rp = rdir / s['scorekey']
        if not rp.exists():
            continue
        out.append({
            'game': 'etterna',
            'replay_path': str(rp),
            'scorekey': s['scorekey'],
            'song': s.get('song', ''),
            'pack': s.get('pack', ''),
            'steps': s.get('steps', ''),
            'rate': s.get('rate', 1.0),
            'wife': s.get('ssrnormpercent', 0),
            'grade': s.get('grade', ''),
            'datetime': s.get('datetime', ''),
            'mtime': rp.stat().st_mtime,
            'ssrs': s.get('ssrs', {}),
            'maxcombo': s.get('maxcombo', 0),
            'chart_key': s.get('chartkey', ''),
            'keycount': _etterna_keycount_from_stepstype(
                s.get('stepstype', 'dance-single')),
            'judgescale': float(s.get('judgescale', 1.0)),
        })
    return out


def _parse_one_osr(p):
    import osrparse
    try:
        r = osrparse.Replay.from_path(str(p))
        mode = getattr(r, 'mode', None)
        mode_int = mode.value if hasattr(mode, 'value') else int(mode or 0)
        if mode_int != 3:
            return None
        total = (getattr(r, 'count_300', 0) + getattr(r, 'count_100', 0) +
                 getattr(r, 'count_50', 0) + getattr(r, 'count_miss', 0) +
                 getattr(r, 'count_geki', 0) + getattr(r, 'count_katu', 0))
        acc = 0.0
        if total:
            acc = (getattr(r, 'count_geki', 0) * 300 +
                   getattr(r, 'count_300', 0) * 300 +
                   getattr(r, 'count_katu', 0) * 200 +
                   getattr(r, 'count_100', 0) * 100 +
                   getattr(r, 'count_50', 0) * 50) / (total * 300) * 100
        return {
            'game': 'osu',
            'replay_path': str(p),
            'beatmap_hash': r.beatmap_hash,
            'song': f'[{r.beatmap_hash[:8]}]',
            'pack': r.username,
            'steps': '',
            'rate': 1.0,
            'wife': acc / 100.0,
            'grade': '',
            'datetime': str(r.timestamp),
            'mtime': p.stat().st_mtime,
            'ssrs': {},
            'maxcombo': r.max_combo,
        }
    except Exception:
        return None


def scan_osu(progress=None):
    """Scan osu!mania replays in parallel."""
    from analysis.osu.replay import find_osu_dirs
    from concurrent.futures import ThreadPoolExecutor
    dirs = find_osu_dirs()
    paths = []
    for rdir in dirs.get('replays_dirs') or []:
        paths.extend(Path(rdir).glob('*.osr'))
    if not paths:
        return []
    out = []
    max_workers = min(32, (os.cpu_count() or 4) * 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, res in enumerate(ex.map(_parse_one_osr, paths, chunksize=8)):
            if res is not None:
                out.append(res)
            if progress and i % 200 == 0:
                progress(f'osu: {i}/{len(paths)} replays…')
    return out


def _hash_one(p):
    import hashlib
    try:
        with open(p, 'rb') as f:
            return str(p), hashlib.md5(f.read()).hexdigest()
    except OSError:
        return str(p), None


def enrich_osu_with_charts(entries, songs_dir=None, progress=None):
    """Resolve song titles for osu entries by hashing .osu files in parallel."""
    from analysis.osu.replay import find_osu_dirs, parse_osu_file
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
    """Build the unified library.
    Set enrich_osu=True to resolve osu!mania song titles (slow: scans every .osu).
    progress: optional callable(stage_str) for UI updates.
    """
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
    if progress:
        progress('scanning Etterna…')
    entries.extend(scan_etterna())
    if progress:
        progress(f'Etterna: {len(entries)} scores — scanning osu!mania…')
    ent_osu = scan_osu(progress=progress)
    entries.extend(ent_osu)
    if progress:
        progress(f'osu!mania: {len(ent_osu)} replays — total {len(entries)}')
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
    print(f"library: {len(lib)} entries "
          f"({sum(1 for e in lib if e['game']=='etterna')} etterna, "
          f"{sum(1 for e in lib if e['game']=='osu')} osu!mania)")
    res = search(lib, query=a.query or None, game=a.game,
                 min_wife=(a.min_wife / 100 if a.min_wife else None),
                 sort=a.sort, descending=not a.asc, limit=a.limit)
    pretty_print(res, limit=a.limit)
