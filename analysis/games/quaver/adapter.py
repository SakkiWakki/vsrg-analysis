"""Quaver game adapter ; `.qua` charts, `.qr` replays, scroll modes."""
from __future__ import annotations

from pathlib import Path

from analysis.core.cache import Cache
from analysis.core.game import GameAdapter
from analysis.player import scroll


_LIBRARY_CACHE = Cache('quaver_library.pkl')
# Maps .qua path -> (mtime, size, md5, parsed_meta). Reused across scans
# so MD5-hashing the Songs folder is amortized after the first run.
_CHART_INDEX_CACHE = Cache('quaver_chart_index.pkl')


class QuaverAdapter(GameAdapter):
    name = 'quaver'

    def parse_replay(self, path, chart_path=None):
        from analysis.games.quaver.parse import parse_replay
        return parse_replay(path, qua_path=chart_path,
                            songs_dir=_quaver_songs_dir())

    def resolve_audio(self, replay, entry=None, progress=None):
        from analysis.games.quaver.qua_chart import parse_qua_file
        if not replay.get('chart_path'):
            return None
        chart = parse_qua_file(replay['chart_path'])
        audio = chart.get('audio')
        if not audio:
            return None
        cand = Path(replay['chart_path']).parent / audio
        return str(cand) if cand.exists() else None

    def resolve_chart_timing(self, replay, entry=None, progress=None):
        # Quaver replays carry absolute ms timings ; no offset needed.
        return None, 0.0

    def prepare_replay_times(self, replay, **_):
        import numpy as np
        times = replay['noterows'].astype(np.float64) / 1000.0
        hold_tails = {}
        for h in replay.get('holds', []):
            if len(h) == 3 and h[2] is not None:
                hold_tails[(h[0], h[1])] = h[2] / 1000.0
        return times, hold_tails, int(replay['keycount'])

    def judgement_windows(self, replay, judge=None, **_):
        from analysis.games.quaver.judgment import windows_for
        return windows_for(self._normalize_judge(replay, judge))

    def judge_label(self, replay, judge=None, **_):
        return self._normalize_judge(replay, judge)

    def player_kwargs(self, replay, judge=None, **_):
        # The shared init code stores `judge_kwarg_name()`'s value on
        # `_active_judge` ; for Quaver that's `judge` (string preset), so
        # we seed it from the chart instead of the player's `ett_judge`
        # default of 'J4' which doesn't apply here.
        return {'ett_judge': self._normalize_judge(replay, judge)}

    def nudge_judge(self, current, delta):
        from analysis.games.quaver.judgment import preset_names
        names = preset_names()
        if not names:
            return current
        cur = current if current in names else names[0]
        step = 1 if delta >= 0 else -1
        return names[(names.index(cur) + step) % len(names)]

    @staticmethod
    def _normalize_judge(replay, judge):
        from analysis.games.quaver.judgment import preset_names
        names = preset_names()
        candidate = judge if judge in names else (replay or {}).get('judge')
        return candidate if candidate in names else 'Standard'

    def default_scroll_mode(self):
        return 'quaver'

    def viz_windows(self, replay, judge=None, **_):
        from analysis.games.quaver.judgment import windows_for
        windows = [(name, w_s, _DEFAULT_COLORS.get(name, '#888'))
                   for name, w_s in windows_for(
                       judge or replay.get('judge', 'Standard'))]
        return windows, 'time (ms)', None

    # --- library scan -----------------------------------------------------
    def scan_library(self, progress=None):
        """Parse every .qr on disk into a placeholder entry. Enrichment
        fills song/artist/keycount from the matching .qua afterwards."""
        paths = _qr_paths()
        return _parse_qr_batch(paths, progress=progress)

    # --- library cache lifecycle -----------------------------------------
    def load_cached(self):
        cached = _LIBRARY_CACHE.load()
        if cached is None:
            return None
        # Migrate older cache entries whose `datetime` was stored in
        # Quaver's culture-formatted shape. New scans always write ISO,
        # but pre-existing caches from the first Quaver-scan implementation
        # leak `MM/dd/yyyy ...` into the library tab's date column.
        from datetime import datetime
        changed = False
        for e in cached:
            d = e.get('datetime')
            if not d or not isinstance(d, str):
                continue
            if '/' not in d:
                continue
            for fmt in ('%m/%d/%Y %H:%M:%S', '%m/%d/%Y %I:%M:%S %p'):
                try:
                    e['datetime'] = datetime.strptime(d[:19], fmt).strftime(
                        '%Y-%m-%d %H:%M:%S')
                    changed = True
                    break
                except ValueError:
                    continue
        if changed:
            _LIBRARY_CACHE.save(cached)
        return cached

    def save_cached(self, entries):
        _LIBRARY_CACHE.save([e for e in entries if e.get('game') == 'quaver'])

    def rebuild(self, progress=None):
        _LIBRARY_CACHE.clear()
        _CHART_INDEX_CACHE.clear()
        paths = _qr_paths()
        entries = _parse_qr_batch(paths, progress=progress)
        _enrich_entries(entries, progress=progress)
        # Persist only when we found something so the next click retries
        # cleanly when the install folder isn't configured yet.
        if entries:
            _LIBRARY_CACHE.save(entries)
        return entries

    def incremental_update(self, progress=None):
        cached = _LIBRARY_CACHE.load()
        if cached is None:
            return self.rebuild(progress=progress)

        known = {e['replay_path']: e for e in cached}
        all_paths = _qr_paths()
        new_paths = [p for p in all_paths if str(p) not in known]
        if not new_paths:
            return cached

        if progress:
            progress(f'quaver: {len(new_paths)} new replay(s)…')
        new_entries = _parse_qr_batch(new_paths, progress=progress)
        if new_entries:
            _enrich_entries(new_entries, progress=progress)

        merged = cached + new_entries
        _LIBRARY_CACHE.save(merged)
        return merged

    # --- cross-game mod display (PlayerDataSource) -----------------------
    def mods_short(self, replay) -> str:
        from analysis.games.quaver.qr_replay import rate_for_mods
        meta = (replay or {}).get('meta') or {}
        mods = int(meta.get('mods', (replay or {}).get('mods', 0)) or 0)
        rate = rate_for_mods(mods)
        return f'{rate:g}x' if rate != 1.0 else 'NM'

    def mods_rate_multiplier(self, replay) -> float:
        from analysis.games.quaver.qr_replay import rate_for_mods
        meta = (replay or {}).get('meta') or {}
        mods = int(meta.get('mods', (replay or {}).get('mods', 0)) or 0)
        return rate_for_mods(mods)

    def can_handle_path(self, path):
        return str(path).lower().endswith('.qr')

    def resolve_standalone(self, path, args=None):
        from analysis.games.quaver.parse import parse_replay
        from analysis.games.quaver.qua_chart import parse_qua_file
        args = args or []
        qua_path = args[args.index('--qua') + 1] if '--qua' in args else None
        rep = parse_replay(path, qua_path=qua_path,
                           songs_dir=_quaver_songs_dir())
        audio = args[args.index('--audio') + 1] if '--audio' in args else None
        if audio is None and rep.get('chart_path'):
            try:
                chart = parse_qua_file(rep['chart_path'])
            except Exception:
                chart = {}
            if chart.get('audio'):
                cand = Path(rep['chart_path']).parent / chart['audio']
                if cand.exists():
                    audio = str(cand)
        return rep, None, 0.0, audio, {}


# Match the colors used in `analysis/viz/note_visualizer` for the other
# games' viz_windows so the per-judge bars look consistent across tabs.
_DEFAULT_COLORS = {
    'marv': '#5cf', 'perf': '#5fc', 'great': '#cf5',
    'good': '#fc5', 'okay': '#f5c', 'miss': '#f55',
}


def _quaver_songs_dir():
    from analysis.games.quaver.paths import find_quaver_dirs
    return find_quaver_dirs().get('songs_dir')


# --- library scan helpers (module-level so ThreadPoolExecutor can pickle) -----


def _qr_paths():
    """Every `.qr` under the resolved Quaver install: user-curated
    `Replays/` exports plus auto-saves under `Data/r/`."""
    from analysis.games.quaver.paths import all_replay_dirs
    paths = []
    for rdir in all_replay_dirs():
        paths.extend(Path(rdir).rglob('*.qr'))
    return paths


def _parse_qr_batch(paths, progress=None):
    """Parse `.qr` headers in parallel ; the `parse_qr_events` call is
    cheap (binary header + a small LZMA blob) but throughput matters
    once `Data/r/` accumulates thousands of autosaves."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    if not paths:
        return []
    out = []
    max_workers = min(32, (os.cpu_count() or 4) * 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, res in enumerate(ex.map(_parse_one_qr, paths, chunksize=8)):
            if res is not None:
                out.append(res)
            if progress and i % 200 == 0:
                progress(f'quaver: parsed {i}/{len(paths)} replays…')
    return out


def _parse_one_qr(p):
    """Build a single library entry from a `.qr` header. The caller
    runs us in a worker pool, so this stays import-light and never
    raises ; failed reads return None and get filtered out."""
    from analysis.games.quaver.qr_replay import parse_qr_events
    try:
        keycount, _events, meta = parse_qr_events(str(p))
        return {
            'game': 'quaver',
            'replay_path': str(p),
            'beatmap_hash': meta['map_md5'],
            # Placeholder song label until enrichment finds the .qua;
            # the prefix matches osu's "[<hash>]" convention so the
            # generic `needs_enrichment` check (`song.startswith('[')`)
            # works the same way.
            'song': f"[{(meta['map_md5'] or '')[:8]}]",
            'pack': meta.get('player_name', ''),
            'steps': '',
            'keycount': keycount,
            'rate': meta.get('rate', 1.0),
            'mods': int(meta.get('mods', 0)),
            'wife': float(meta.get('accuracy', 0.0)) / 100.0,
            'grade': '',
            # Normalize the replay date to ISO so the library tab's date
            # column doesn't mix `MM/dd/yyyy` (Quaver's C# InvariantCulture)
            # with `YYYY-MM-DD` (osu/Etterna). `time_played` is ms-since-
            # epoch and is the ground truth ; `meta['date']` is the raw
            # culture-formatted string we fall back to if `time_played` is
            # zero/garbage.
            'datetime': _quaver_iso_datetime(meta),
            'mtime': p.stat().st_mtime,
            'ssrs': {},
            'maxcombo': int(meta.get('max_combo', 0)),
        }
    except Exception:
        return None


def _quaver_iso_datetime(meta):
    """`YYYY-MM-DD HH:MM:SS` from a `.qr`'s `time_played` (ms-since-epoch)
    or the raw culture-formatted `date` field. Empty on parse failure."""
    from datetime import datetime, timezone
    tp = int(meta.get('time_played', 0) or 0)
    if tp > 0:
        return datetime.fromtimestamp(tp / 1000.0,
                                       tz=timezone.utc).strftime(
            '%Y-%m-%d %H:%M:%S')
    raw = str(meta.get('date', '')).strip()
    if not raw:
        return ''
    for fmt in ('%m/%d/%Y %H:%M:%S', '%m/%d/%Y %I:%M:%S %p',
                '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(raw[:19], fmt).strftime(
                '%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue
    return raw[:19]


def _hash_chart(path):
    import hashlib
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None


def _hash_chart_from_path_str(path_str):
    """Module-level so ThreadPoolExecutor can pickle the callable."""
    return path_str, _hash_chart(path_str)


def _chart_meta(path):
    from analysis.games.quaver.qua_chart import parse_qua_file
    try:
        chart = parse_qua_file(path)
    except Exception:
        return None
    return {
        'song': f"{chart.get('artist', '?')} - {chart.get('title', '?')}",
        'steps': chart.get('version', ''),
        'creator': chart.get('creator', ''),
        'keycount': chart.get('keycount'),
    }


def _build_chart_index(songs_dir, progress=None):
    """`{path_str: (mtime, size, md5, meta)}` covering every `.qua` in
    `songs_dir`. Hashing only revisits files whose mtime/size changed
    since the last run, so subsequent calls are essentially free."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    cached = _CHART_INDEX_CACHE.load() or {}
    paths = list(Path(songs_dir).rglob('*.qua'))

    stale = []
    fresh = {}
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue
        key = str(p)
        prev = cached.get(key)
        if prev and prev[0] == st.st_mtime and prev[1] == st.st_size:
            fresh[key] = prev
        else:
            stale.append((key, st.st_mtime, st.st_size))

    if progress:
        if stale:
            progress(f'hashing {len(stale)} new/changed .qua files '
                     f'({len(fresh)} reused)…')
        else:
            progress(f'reusing {len(fresh)} cached chart hashes')

    if stale:
        max_workers = min(32, (os.cpu_count() or 4) * 4)
        path_strs = [s[0] for s in stale]
        stats = {s[0]: (s[1], s[2]) for s in stale}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, (path_str, md5) in enumerate(
                    ex.map(_hash_chart_from_path_str, path_strs, chunksize=32)):
                if md5 is None:
                    continue
                meta = _chart_meta(path_str)
                mtime, size = stats[path_str]
                fresh[path_str] = (mtime, size, md5, meta)
                if progress and i % 500 == 0:
                    progress(f'hashed {i}/{len(stale)} new .qua files')

    _CHART_INDEX_CACHE.save(fresh)
    return fresh


def _enrich_entries(entries, progress=None):
    """Fill song/steps/pack/keycount/chart_path on Quaver entries from
    the chart-hash index. Mirrors `analysis.games.osu.adapter._enrich_entries`."""
    targets = {}
    for e in entries:
        if e.get('game') == 'quaver' and e.get('beatmap_hash'):
            targets.setdefault(e['beatmap_hash'], []).append(e)
    if not targets:
        return
    songs_dir = _quaver_songs_dir()
    if not songs_dir:
        return

    index = _build_chart_index(songs_dir, progress=progress)
    hash_to_entry = {}
    for path_str, (_m, _s, md5, meta) in index.items():
        if md5 and md5 in targets and meta is not None:
            hash_to_entry[md5] = (path_str, meta)

    matched_hashes = 0
    matched_entries = 0
    for md5, group in targets.items():
        hit = hash_to_entry.get(md5)
        if not hit:
            continue
        path_str, meta = hit
        matched_hashes += 1
        for e in group:
            e['song'] = meta['song']
            e['steps'] = meta['steps']
            # The replay's `pack` field came from the player name ; once
            # we know the chart's creator, prefer that for parity with
            # how osu/etterna populate `pack` (mapper / pack name).
            e['pack'] = meta['creator'] or e.get('pack', '')
            e['keycount'] = meta['keycount']
            e['chart_path'] = path_str
            matched_entries += 1
    if progress:
        progress(f'quaver enrichment: {matched_hashes}/{len(targets)} charts '
                 f'({matched_entries} replays)')


# --- Quaver scroll mode -----------------------------------------------------
# Ported from Quaver's TimingGroupControllerKeys.ScrollSpeed + TrackRounding.
# `value` is the user-facing scroll speed shown in Quaver's options menu
# (5.0 to 100.0, default 15.0). Quaver stores this internally as an int
# 10x larger (50 to 1000, default 150) and divides by 10 in its formula;
# we skip that round-trip and work in the displayed scale directly.
_QUAVER_SKIN_SCALE = 1920.0 / 1366.0
_QUAVER_BASE_WINDOW_H = 768.0
_MS_PER_S = 1000.0


def _quaver_pxps_at_base_window(value):
    scroll_speed = value / 20.0 * _QUAVER_SKIN_SCALE
    return scroll_speed * _MS_PER_S


def _quaver_to_pxps(value, opts, p):
    window_scale = p.H / _QUAVER_BASE_WINDOW_H
    return _quaver_pxps_at_base_window(float(value)) * window_scale


def _quaver_from_pxps(pxps, opts, p):
    window_scale = p.H / _QUAVER_BASE_WINDOW_H
    return pxps / (_quaver_pxps_at_base_window(1.0) * window_scale)


scroll.register(scroll.ScrollMode(
    key='quaver',
    label='Quaver',
    game='quaver',
    to_pxps=_quaver_to_pxps,
    from_pxps=_quaver_from_pxps,
    default_value=15.0,
    value_bounds=(5.0, 100.0),
    nudge=scroll.integer_step_nudge,
    format_value=lambda v: (f'Q {int(v)}' if abs(v - round(v)) < 1e-4
                            else f'Q {v:.1f}'),
))


ADAPTER = QuaverAdapter()
