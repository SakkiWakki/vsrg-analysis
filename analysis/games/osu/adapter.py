"""osu!mania game adapter + scroll modes."""
from __future__ import annotations

from pathlib import Path

from analysis.core.cache import Cache
from analysis.core.game import GameAdapter
from analysis.player import scroll


_LIBRARY_CACHE = Cache('osu_library.pkl')
# Maps .osu path -> (mtime, size, md5, parsed_meta). Reused across both
# rebuild and incremental-update paths so hashing the Songs folder is a
# one-time cost, not a per-refresh cost.
_CHART_INDEX_CACHE = Cache('osu_chart_index.pkl')


class OsuAdapter(GameAdapter):
    name = 'osu'

    def parse_replay(self, path, chart_path=None):
        from analysis.games.osu.replay import parse_replay, find_osu_dirs
        songs = find_osu_dirs().get('songs_dir')
        return parse_replay(path, osu_path=chart_path, songs_dir=songs)

    def resolve_audio(self, replay, entry=None, progress=None):
        from analysis.games.osu.replay import parse_osu_file
        if not replay.get('chart_path'):
            return None
        chart = parse_osu_file(replay['chart_path'])
        audio = chart.get('audio')
        if not audio:
            return None
        cand = Path(replay['chart_path']).parent / audio
        return str(cand) if cand.exists() else None

    def resolve_chart_timing(self, replay, entry=None, progress=None):
        # osu! replays carry absolute ms timings; no sm-style offset needed.
        return None, 0.0

    def prepare_replay_times(self, replay, **_):
        import numpy as np
        times = replay['noterows'].astype(np.float64) / 1000.0
        hold_tails = {}
        for h in replay.get('holds', []):
            if len(h) == 3 and h[2] is not None:
                hold_tails[(h[0], h[1])] = h[2] / 1000.0
        return times, hold_tails, int(replay['keycount'])

    def build_sv_engine(self, replay):
        """osu!mania SV comes from the .osu timing points, already flattened
        to `[(time_sec, multiplier)]` by `_compute_sv_sections` during chart
        parse. Time-space integration matches how osu! positions notes
        (uniform scroll speed scaled by the SV multiplier at each moment)."""
        from analysis.player.sv.engine import TimeSpaceSVEngine
        sections = replay.get('sv_sections') or []
        if not sections:
            return None
        return TimeSpaceSVEngine(sections)

    def effective_od(self, replay, od=None):
        from analysis.viz.note_visualizer import effective_osu_od
        base = od if od is not None else float(replay.get('od', 8.0))
        mods = int(replay.get('mods', 0))
        return effective_osu_od(base, mods)

    def judgement_windows(self, replay, od=None, **_):
        from analysis.games.osu.judgment import windows_for
        return windows_for(self.effective_od(replay, od))

    def judge_kwarg_name(self):
        return 'od'

    def nudge_judge(self, current, delta):
        """osu!mania OD is continuous (float). The beatmap field caps
        at 10 but mods push effective OD higher (HR at OD10 ≈ 14, and
        charts can simulate stricter-than-stable windows too), so we
        allow 0..15 in the UI. Caller passes the physical delta
        (±0.1 from the sidebar buttons, or larger on keyboard)."""
        cur = float(current if current is not None else 8.0)
        return max(0.0, min(15.0, cur + float(delta)))

    def judge_label(self, replay, od=None, **_):
        return f'OD {self.effective_od(replay, od):g}'

    def default_scroll_mode(self):
        return 'osu'

    def player_kwargs(self, replay, od=None, **_):
        return {'od': self.effective_od(replay, od)}

    # ── Cross-game mod hooks (used by PlayerDataSource) ──

    def mods_short(self, replay) -> str:
        m = int((replay or {}).get('mods', 0) or 0)
        return _osu_mods_string(m)

    def mods_raw(self, replay) -> dict:
        m = int((replay or {}).get('mods', 0) or 0)
        return {'bitfield': m, 'rate': _osu_mods_rate(m)}

    def mods_rate_multiplier(self, replay) -> float:
        m = int((replay or {}).get('mods', 0) or 0)
        return _osu_mods_rate(m)

    def chart_stats_extra(self, replay):
        """Return (difficulty, rating, extra) for ChartStats.

        difficulty: effective OD (uncapped for HR/EZ).
        rating:     star rating if cached, else 0.
        extra:      full osu-native {ar, cs, hp, od, stars}.
        """
        if not isinstance(replay, dict):
            return 0.0, 0.0, {}
        base_od = float(replay.get('od', 0.0))
        mods = int(replay.get('mods', 0) or 0)
        eff_od = _apply_od_mods(base_od, mods)
        cm = replay.get('chart_meta') or {}
        cs = float(cm.get('keycount', 0) or 0)   # mania only
        ar = float(cm.get('ar', 0) or 0)
        hp = float(cm.get('hp', 0) or 0)
        stars = float(cm.get('stars', 0) or 0)
        return eff_od, stars, {
            'od': eff_od, 'cs': cs, 'ar': ar, 'hp': hp, 'stars': stars,
        }

    # --- library scan -----------------------------------------------------
    def scan_library(self, progress=None):
        """Parse every .osr on disk into a placeholder entry (no song
        title — enrichment fills those). Kept for parity with the old
        adapter contract; `rebuild` and `incremental_update` are the
        library pipeline's real entry points."""
        paths = _osr_paths()
        return _parse_osr_batch(paths, progress=progress)

    # --- library cache lifecycle -----------------------------------------
    def load_cached(self):
        return _LIBRARY_CACHE.load()

    def save_cached(self, entries):
        _LIBRARY_CACHE.save([e for e in entries if e.get('game') == 'osu'])

    def rebuild(self, progress=None):
        _LIBRARY_CACHE.clear()
        _CHART_INDEX_CACHE.clear()
        paths = _osr_paths()
        entries = _parse_osr_batch(paths, progress=progress)
        _enrich_entries(entries, progress=progress)
        # Same rationale as EtternaAdapter.rebuild: an empty result
        # typically means the replays dir isn't configured, not that
        # the user has zero replays. Leaving the cache absent means the
        # next click actually retries instead of reading [] forever.
        if entries:
            _LIBRARY_CACHE.save(entries)
        return entries

    def incremental_update(self, progress=None):
        cached = _LIBRARY_CACHE.load()
        if cached is None:
            return self.rebuild(progress=progress)

        known = {e['replay_path']: e for e in cached}
        all_paths = _osr_paths()
        new_paths = [p for p in all_paths if str(p) not in known]
        if not new_paths:
            return cached

        if progress:
            progress(f'osu: {len(new_paths)} new replay(s)…')
        new_entries = _parse_osr_batch(new_paths, progress=progress)
        if new_entries:
            # Warm chart index makes this near-instant on the common path
            # (user played a chart that was already in the Songs folder).
            _enrich_entries(new_entries, progress=progress)

        merged = cached + new_entries
        _LIBRARY_CACHE.save(merged)
        return merged

    # --- standalone-launch resolver --------------------------------------
    def can_handle_path(self, path):
        return str(path).lower().endswith('.osr')

    def resolve_standalone(self, path, args=None):
        from analysis.games.osu.replay import (parse_replay, find_osu_dirs,
                                               parse_osu_file)
        args = args or []
        osu_path = args[args.index('--osu') + 1] if '--osu' in args else None
        songs = find_osu_dirs().get('songs_dir')
        rep = parse_replay(path, osu_path=osu_path, songs_dir=songs)
        audio = args[args.index('--audio') + 1] if '--audio' in args else None
        if audio is None and rep.get('chart_path'):
            try:
                chart = parse_osu_file(rep['chart_path'])
            except Exception:
                chart = {}
            if chart.get('audio'):
                cand = Path(rep['chart_path']).parent / chart['audio']
                if cand.exists():
                    audio = str(cand)
        return rep, None, 0.0, audio, {}

    # --- PlayerTab kwargs -------------------------------------------------
    def player_tab_kwargs(self, replay, entry, chart_ctx):
        # osu carries OD + mods on the replay itself; nothing extra needed.
        return {}

    # --- note visualizer --------------------------------------------------
    def viz_windows(self, replay, judge=None, od=None):
        from analysis.viz.note_visualizer import osu_mania_windows
        return osu_mania_windows(od=od if od is not None else 8), 'time (ms)', None


def _osr_paths():
    from analysis.games.osu.replay import find_osu_dirs
    dirs = find_osu_dirs()
    paths = []
    for rdir in dirs.get('replays_dirs') or []:
        paths.extend(Path(rdir).glob('*.osr'))
    return paths


def _parse_osr_batch(paths, progress=None):
    import os
    from concurrent.futures import ThreadPoolExecutor
    if not paths:
        return []
    out = []
    max_workers = min(32, (os.cpu_count() or 4) * 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, res in enumerate(ex.map(_parse_one_osr, paths, chunksize=8)):
            if res is not None:
                out.append(res)
            if progress and i % 200 == 0:
                progress(f'osu: parsed {i}/{len(paths)} replays…')
    return out


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
    from analysis.games.osu.replay import parse_osu_file
    try:
        chart = parse_osu_file(path)
    except Exception:
        return None
    return {
        'song': f"{chart.get('artist', '?')} - {chart.get('title', '?')}",
        'steps': chart.get('version', ''),
        'creator': chart.get('creator', ''),
        'keycount': chart.get('keycount'),
    }


def _build_chart_index(songs_dir, progress=None):
    """Return a dict {path_str: (mtime, size, md5, meta)}. Reuses a
    persistent cache, rehashing only files whose mtime/size changed."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    cached = _CHART_INDEX_CACHE.load() or {}
    paths = list(Path(songs_dir).rglob('*.osu'))

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
            progress(f'hashing {len(stale)} new/changed .osu files '
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
                    progress(f'hashed {i}/{len(stale)} new .osu files')

    _CHART_INDEX_CACHE.save(fresh)
    return fresh


def _enrich_entries(entries, progress=None):
    """Fill song/steps/pack/keycount/chart_path in-place on osu entries,
    using a persistent chart-hash index keyed on (path, mtime, size)."""
    from analysis.games.osu.replay import find_osu_dirs
    targets = {}
    for e in entries:
        if e.get('game') == 'osu' and e.get('beatmap_hash'):
            targets.setdefault(e['beatmap_hash'], []).append(e)
    if not targets:
        return
    songs_dir = find_osu_dirs().get('songs_dir')
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
            e['pack'] = meta['creator'] or e.get('pack', '')
            e['keycount'] = meta['keycount']
            e['chart_path'] = path_str
            matched_entries += 1
    if progress:
        progress(f'osu enrichment: {matched_hashes}/{len(targets)} charts '
                 f'({matched_entries} replays)')


def _parse_one_osr(p):
    """Module-level helper so ThreadPoolExecutor can pickle the callable."""
    import osrparse
    from analysis.games.osu.replay import rate_for_mods
    try:
        r = osrparse.Replay.from_path(str(p))
        mode = getattr(r, 'mode', None)
        mode_int = mode.value if hasattr(mode, 'value') else int(mode or 0)
        if mode_int != 3:
            return None
        mods = int(r.mods.value) if hasattr(r.mods, 'value') else int(r.mods or 0)
        rate = rate_for_mods(mods)
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
            'rate': rate,
            'mods': mods,
            'wife': acc / 100.0,
            'grade': '',
            'datetime': str(r.timestamp),
            'mtime': p.stat().st_mtime,
            'ssrs': {},
            'maxcombo': r.max_combo,
        }
    except Exception:
        return None


# --- osu!mania scroll mode --------------------------------------------------
# Ported from osu-framework's SpeedMania.DistanceAt (px/ms = Speed * 21/600).
_OSU_PX_PER_MS_PER_SPEED = 21.0 / 600.0
_MS_PER_S = 1000.0


def _osu_pxps_at_reference_field(value):
    return value * _OSU_PX_PER_MS_PER_SPEED * _MS_PER_S


def _osu_to_pxps(value, opts, p):
    field_scale = p.H / p.REFERENCE_FIELD_H
    return _osu_pxps_at_reference_field(float(value)) * field_scale


def _osu_from_pxps(pxps, opts, p):
    field_scale = p.H / p.REFERENCE_FIELD_H
    return pxps / (_osu_pxps_at_reference_field(1.0) * field_scale)


scroll.register(scroll.ScrollMode(
    key='osu',
    label='osu!',
    game='osu',
    to_pxps=_osu_to_pxps,
    from_pxps=_osu_from_pxps,
    default_value=20.0,
    value_bounds=(1.0, 40.0),
    nudge=scroll.integer_step_nudge,
    format_value=lambda v: (f'osu {int(v)}' if abs(v - round(v)) < 1e-4
                            else f'osu {v:.2f}'),
))


ADAPTER = OsuAdapter()


# ── Mod bitfield helpers (shared by adapter methods) ──

# osu! stable mod bits. Order matters for display: NC follows DT
# because real stable activates DT alongside NC.
_OSU_MOD_BITS = (
    (1 << 0, 'NF'),  (1 << 1, 'EZ'),  (1 << 2, 'TD'),  (1 << 3, 'HD'),
    (1 << 4, 'HR'),  (1 << 5, 'SD'),  (1 << 6, 'DT'),  (1 << 7, 'RX'),
    (1 << 8, 'HT'),  (1 << 9, 'NC'),  (1 << 10, 'FL'), (1 << 11, 'AU'),
    (1 << 12, 'SO'), (1 << 13, 'AP'), (1 << 14, 'PF'), (1 << 15, '4K'),
    (1 << 16, '5K'), (1 << 17, '6K'), (1 << 18, '7K'), (1 << 19, '8K'),
    (1 << 20, 'FI'), (1 << 21, 'RD'), (1 << 22, 'CN'), (1 << 23, 'TP'),
    (1 << 24, '9K'), (1 << 25, 'KC'), (1 << 26, '1K'), (1 << 27, '3K'),
    (1 << 28, '2K'), (1 << 29, 'V2'), (1 << 30, 'MR'),
)


def _osu_mods_string(mods: int) -> str:
    if mods == 0:
        return 'NM'
    names = [name for bit, name in _OSU_MOD_BITS if mods & bit]
    return ''.join(names) if names else 'NM'


def _osu_mods_rate(mods: int) -> float:
    if mods & (1 << 6) or mods & (1 << 9):   # DT / NC
        return 1.5
    if mods & (1 << 8):                      # HT
        return 0.75
    return 1.0


def _apply_od_mods(od: float, mods: int) -> float:
    # HR multiplies OD by 1.4 (capped at 10 in stable, but effective
    # windows go higher); EZ halves it.
    if mods & (1 << 4):
        od *= 1.4
    if mods & (1 << 1):
        od *= 0.5
    return od
