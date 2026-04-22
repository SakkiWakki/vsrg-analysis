"""Unified replay library search. Iterates over every registered
`GameAdapter` and merges its per-game cache into one filterable list.
Adapters own their own cache files (see `GameAdapter.load_cached` /
`incremental_update` / `rebuild`); this module is the glue that joins
them and respects the user's ``library.enabled_games`` config."""
from datetime import datetime

from analysis.config.store import get_config
from analysis.core import game as game_mod


def enabled_games() -> set[str]:
    """Games the user wants visible. Missing config = all registered
    games enabled. Unknown names in the config are ignored so removing
    a game doesn't hard-error the library."""
    stored = get_config().get('library.enabled_games')
    if stored is None:
        return set(game_mod.all_games().keys())
    registered = set(game_mod.all_games().keys())
    return {name for name in stored if name in registered}


def set_game_enabled(name: str, enabled: bool) -> None:
    current = enabled_games()
    if enabled:
        current.add(name)
    else:
        current.discard(name)
    # Persist as a sorted list for readability in config.json.
    get_config().set('library.enabled_games', sorted(current))


def build_library(refresh=False, progress=None):
    """Return the full library as a flat list of entry dicts.

    Each adapter owns its own cache. By default we ask each adapter for
    an incremental update (fast — picks up new replays, reuses cache).
    Pass `refresh=True` to force a full rebuild of every enabled game;
    callers that want to rebuild only one game should call
    `adapter.rebuild()` directly.
    """
    enabled = enabled_games()
    entries = []
    for name, adapter in game_mod.all_games().items():
        if name not in enabled:
            continue
        if progress:
            progress(f'{name}: checking for new replays…')
        if refresh:
            got = adapter.rebuild(progress=progress) or []
        else:
            got = adapter.incremental_update(progress=progress) or []
        entries.extend(got)
        if progress:
            progress(f'{name}: {len(got)} entries — total {len(entries)}')
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
    a = p.parse_args()

    lib = build_library(refresh=a.refresh)
    from collections import Counter
    counts = Counter(e['game'] for e in lib)
    parts = ', '.join(f'{n} {g}' for g, n in counts.most_common())
    print(f"library: {len(lib)} entries ({parts})")
    res = search(lib, query=a.query or None, game=a.game,
                 min_wife=(a.min_wife / 100 if a.min_wife else None),
                 sort=a.sort, descending=not a.asc, limit=a.limit)
    pretty_print(res, limit=a.limit)
