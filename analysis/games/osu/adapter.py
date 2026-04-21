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
