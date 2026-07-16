"""NotITG game manifest -- the GUI's view of the game."""
from __future__ import annotations

from analysis.core.manifest import GameManifest, PathField

from analysis.games.notitg.paths import (NOTITG_ROOT_KEY, autodetect_root,
                                         find_notitg_dirs, validate_root)

_VIZ_WINDOWS = (
    ('fantastic', 0.023, '#5df'),
    ('excellent', 0.0445, '#fd5'),
    ('great', 0.1035, '#7f7'),
    ('decent', 0.1365, '#c7f'),
    ('wayoff', 0.1815, '#e96'),
)


def _note_viz_config(replay, judge=None, od=None):
    return {
        'windows': list(_VIZ_WINDOWS),
        'unit_label': 'noterow  ;  ITG',
        'rows_per_ms': 0.37,
        'win': 2400,
    }


def _resolve_chart_context(replay, entry=None, progress=None):
    from analysis.core import game as game_mod
    return game_mod.get('notitg').resolve_all(
        replay, entry=entry, progress=progress)


MANIFEST = GameManifest(
    name='notitg',
    path_fields=[
        PathField(
            key='root',
            label='NotITG folder',
            hint=('Point at your NotITG folder ; the portable install '
                  'that contains `Songs/` and `Program/`. NotITG has no '
                  'replays, so its charts appear as unplayed library '
                  'entries that play back in autoplay.'),
            placeholder={
                'win32': r'e.g. C:\Games\NotITG',
                'linux': 'e.g. ~/Games/NotITG',
                'darwin': 'e.g. ~/Games/NotITG',
                'default': 'e.g. ~/Games/NotITG',
            },
            settings_key=NOTITG_ROOT_KEY,
            error_hint='folder missing Songs/',
            autodetect=autodetect_root,
            validate=validate_root,
        ),
    ],
    find_dirs=find_notitg_dirs,
    note_viz_config=_note_viz_config,
    resolve_chart_context=_resolve_chart_context,
)
