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
        chart_path = replay.get('chart_path')
        if not chart_path:
            return None
        audio = replay.get('_quaver_audio_file')
        if audio is None:
            from analysis.games.quaver.qua_chart import parse_qua_file
            chart = parse_qua_file(chart_path)
            audio = chart.get('audio')
        if not audio:
            return None
        cand = Path(chart_path).parent / audio
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

    def populate_notes_model(self, replay, model) -> None:
        # Mines + detonations from the .qua/.qr pair; the parser fills
        # the same chart-stream keys Etterna's adapter produces.
        from analysis.player.init.notes_model import copy_chart_streams
        copy_chart_streams(model, replay)

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

    def viz_panel_units(self, replay) -> int:
        return 8000

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
        merged = cached + new_entries
        _LIBRARY_CACHE.save(merged)
        return merged

    # --- cross-game mod display (PlayerDataSource) -----------------------
    def player_tab_kwargs(self, replay, entry, chart_ctx):
        return {
            'audio_chart_offset_s': _quaver_global_audio_offset_s(
                root=_quaver_root_for_replay(replay)),
            'audio_chart_offset_scales_with_rate': True,
        }

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


def _quaver_root_for_replay(replay) -> str | None:
    chart_path = (replay or {}).get('chart_path')
    if not chart_path:
        return None
    try:
        chart = Path(chart_path).resolve()
    except OSError:
        chart = Path(chart_path)
    for parent in chart.parents:
        if (parent / 'quaver.cfg').is_file():
            return str(parent)
        if parent.name == 'Songs':
            return str(parent.parent)
    return None


def _quaver_global_audio_offset_s(root: str | None = None) -> float:
    """GlobalAudioOffset from Quaver's own `quaver.cfg`, in seconds.

    Quaver's replay frame timestamps are captured in `CurrentAudioOffset`,
    which upstream computes as audio time plus this global offset multiplied
    by the active rate. The player renders in that chart-time domain, so the
    audio layer needs the inverse mapping when seeking.
    """
    import configparser
    import os
    import re
    from pathlib import Path

    env = os.environ.get('QUAVER_GLOBAL_AUDIO_OFFSET_MS')
    if env is not None:
        try:
            return float(env) / 1000.0
        except ValueError:
            pass

    if root is None:
        from analysis.games.quaver.paths import find_quaver_dirs
        root = find_quaver_dirs().get('root')
    if not root:
        return 0.0
    cfg_path = Path(root) / 'quaver.cfg'
    try:
        text = cfg_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return 0.0

    parser = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        parser.read_string(text)
        raw = parser.get('Config', 'GlobalAudioOffset', fallback=None)
    except configparser.Error:
        raw = None

    if raw is None:
        m = re.search(r'(?im)^\s*GlobalAudioOffset\s*=\s*([-+]?\d+)', text)
        raw = m.group(1) if m else None
    try:
        return float(raw) / 1000.0
    except (TypeError, ValueError):
        return 0.0


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
    from functools import partial
    if not paths:
        return []
    # Build the chart-hash lookup once; each worker stamps its replay
    # with song/steps/keycount/chart_path inline. Empty when no songs
    # dir is configured ; entries fall back to the `[hash[:8]]` placeholder.
    hash_to_chart = _build_chart_hash_lookup(progress=progress)
    worker = partial(_parse_one_qr, hash_to_chart=hash_to_chart)
    out = []
    max_workers = min(32, (os.cpu_count() or 4) * 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, res in enumerate(ex.map(worker, paths, chunksize=8)):
            if res is not None:
                out.append(res)
            if progress and i % 200 == 0:
                progress(f'quaver: parsed {i}/{len(paths)} replays…')
    return out


def _build_chart_hash_lookup(progress=None):
    """`md5 -> (chart_path_str, meta)` for every parseable .qua in the
    user's songs dir. Empty when no songs dir is configured."""
    songs_dir = _quaver_songs_dir()
    if not songs_dir:
        return {}
    index = _build_chart_index(songs_dir, progress=progress)
    return {md5: (path_str, meta)
            for path_str, (_m, _s, md5, meta) in index.items()
            if md5 and meta is not None}


def _parse_one_qr(p, hash_to_chart=None):
    """Build a single library entry from a `.qr` header. The caller
    runs us in a worker pool, so this stays import-light and never
    raises ; failed reads return None and get filtered out.
    `hash_to_chart` is `md5 -> (chart_path, meta)` ; when the replay's
    map_md5 hits, song/steps/keycount/chart_path get filled inline."""
    from analysis.games.quaver.qr_replay import parse_qr_events
    try:
        keycount, _events, meta = parse_qr_events(str(p))
    except Exception as exc:
        # Surfacing matters: a format bump (e.g. the 0.0.3 mine-hit
        # field) silently emptied the library of new replays once.
        print(f'quaver: failed to parse {p}: {exc}')
        return None
    try:
        md5 = meta['map_md5']
        hit = (hash_to_chart or {}).get(md5)
        if hit is not None:
            chart_path, chart_meta = hit
            song = chart_meta['song']
            steps = chart_meta['steps']
            # Prefer the chart's creator over the replay's player_name for
            # `pack`, matching how osu/etterna populate it (mapper / pack).
            pack = chart_meta['creator'] or meta.get('player_name', '')
            keycount_out = chart_meta['keycount'] or keycount
        else:
            chart_path = None
            # Placeholder song label when no chart match; the prefix
            # matches osu's "[<hash>]" convention so the generic
            # `needs_enrichment` check (`song.startswith('[')`) works.
            song = f"[{(md5 or '')[:8]}]"
            steps = ''
            pack = meta.get('player_name', '')
            keycount_out = keycount

        return {
            'game': 'quaver',
            'replay_path': str(p),
            'beatmap_hash': md5,
            'song': song,
            'pack': pack,
            'steps': steps,
            'keycount': keycount_out,
            'chart_path': chart_path,
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
