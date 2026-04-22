"""osu!mania game adapter + scroll modes."""
from __future__ import annotations

from pathlib import Path

from analysis.core.game import GameAdapter
from analysis.player import scroll


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
        from analysis.player.timing import infer_keycount
        times = replay['noterows'].astype(np.float64) / 1000.0
        hold_tails = {}
        for h in replay.get('holds', []):
            if len(h) == 3 and h[2] is not None:
                hold_tails[(h[0], h[1])] = h[2] / 1000.0
        return times, hold_tails, infer_keycount(replay)

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

    # --- library scan -----------------------------------------------------
    def scan_library(self, progress=None):
        import os
        from concurrent.futures import ThreadPoolExecutor
        from analysis.games.osu.replay import find_osu_dirs
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
# From SpeedMania.cs:126 — DistanceAt: px/ms = Speed * 21 / 600 = Speed * 0.035
# in osu's 480-tall logical playfield. We scale by H / REFERENCE_FIELD_H in
# the Player so the visual fraction-of-screen-per-second matches osu stable.
_OSU_SPEED_PX_PER_MS = 0.035


def _osu_to_pxps(value, opts, p):
    field_scale = p.H / p.REFERENCE_FIELD_H
    return float(value) * _OSU_SPEED_PX_PER_MS * 1000.0 * field_scale


def _osu_from_pxps(pxps, opts, p):
    field_scale = p.H / p.REFERENCE_FIELD_H
    return pxps / (_OSU_SPEED_PX_PER_MS * 1000.0 * field_scale)


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
