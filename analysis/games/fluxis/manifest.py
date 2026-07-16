"""fluXis game manifest -- the GUI's view of the game."""
from __future__ import annotations

from analysis.core.manifest import GameManifest, PathField

from analysis.games.fluxis.paths import (FLUXIS_DATA_KEY, autodetect_data_dir,
                                          find_fluxis_dirs, validate_data_dir)


_VIZ_COLORS = {
    'flawless': '#5cf', 'perfect': '#5fc', 'great': '#cf5',
    'alright': '#fc5', 'okay': '#f5c', 'miss': '#f55',
}


def _note_viz_config(replay, judge=None, od=None):
    from analysis.games.fluxis.judge_sim import hit_windows_ms
    difficulty = (float(od) if od is not None
                  else float(replay.get('accuracy_difficulty', 8.0))
                  if isinstance(replay, dict) else 8.0)
    rate = (float(replay.get('rate') or 1.0)
            if isinstance(replay, dict) else 1.0)
    windows = [(name, ms / 1000.0, _VIZ_COLORS[name])
               for name, ms in hit_windows_ms(difficulty, rate)]
    return {
        'windows': windows,
        'unit_label': f'time (ms)  ;  ACC {difficulty:g}',
        'rows_per_ms': None,
        'win': 8000,
    }


def _resolve_chart_context(replay, entry=None, progress=None):
    from analysis.core import game as game_mod
    audio = game_mod.get('fluxis').resolve_audio(replay)
    return None, 0.0, audio


MANIFEST = GameManifest(
    name='fluxis',
    path_fields=[
        PathField(
            key='data',
            label='fluXis data folder',
            hint=('Point at your fluXis data folder ; the one that '
                  'contains `fluxis.realm`, `maps/`, and `replays/`. '
                  'This is the game\'s storage directory, not the '
                  'install folder.'),
            placeholder={
                'win32': r'e.g. %APPDATA%\fluXis',
                'linux': 'e.g. ~/.local/share/fluXis',
                'darwin': 'e.g. ~/Library/Application Support/fluXis',
                'default': 'e.g. ~/.local/share/fluXis',
            },
            settings_key=FLUXIS_DATA_KEY,
            error_hint='folder missing fluxis.realm',
            autodetect=autodetect_data_dir,
            validate=validate_data_dir,
        ),
    ],
    find_dirs=find_fluxis_dirs,
    note_viz_config=_note_viz_config,
    resolve_chart_context=_resolve_chart_context,
)
