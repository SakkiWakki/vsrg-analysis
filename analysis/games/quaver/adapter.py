"""Quaver game adapter ; `.qua` charts, `.qr` replays, scroll modes."""
from __future__ import annotations

from pathlib import Path

from analysis.core.game import GameAdapter
from analysis.player import scroll


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
        return windows_for(judge or replay.get('judge', 'Standard'))

    def judge_label(self, replay, judge=None, **_):
        return judge or replay.get('judge', 'Standard')

    def default_scroll_mode(self):
        return 'quaver'

    def viz_windows(self, replay, judge=None, **_):
        from analysis.games.quaver.judgment import windows_for
        windows = [(name, w_s, _DEFAULT_COLORS.get(name, '#888'))
                   for name, w_s in windows_for(
                       judge or replay.get('judge', 'Standard'))]
        return windows, 'time (ms)', None

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
    """Best-effort resolution of the local Quaver songs directory.
    Returns None if not configured ; the parser then needs an explicit
    `--qua` argument.

    Quaver stores maps under `<install>/Songs/<mapsetId>/`. We probe the
    obvious Linux/Wine + Windows locations ; users can override via the
    `QUAVER_SONGS_DIR` env var when their install is somewhere unusual."""
    import os
    explicit = os.environ.get('QUAVER_SONGS_DIR')
    if explicit and Path(explicit).is_dir():
        return explicit
    candidates = [
        Path.home() / 'Quaver' / 'Songs',
        Path.home() / '.steam' / 'steam' / 'steamapps' / 'common'
        / 'Quaver' / 'Songs',
        Path.home() / 'Games' / 'Quaver' / 'Songs',
    ]
    for c in candidates:
        if c.is_dir():
            return str(c)
    return None


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
