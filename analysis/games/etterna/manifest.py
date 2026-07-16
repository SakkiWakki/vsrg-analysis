"""Etterna game manifest -- the GUI's view of the game.

Replaces the old `EtternaGuiAdapter` ; the paths dialog reads
`MANIFEST.path_fields` and renders one row per field. Override storage
goes through `analysis.core.path_overrides` (the "shopkeeper") so
`find_etterna_dirs` doesn't need to import Qt to honor user overrides.
"""
from __future__ import annotations

from pathlib import Path

from analysis.core.manifest import GameManifest, PathField


_ETTERNA_ROOT_KEY = 'paths/etterna_root'


def _find_dirs():
    from analysis.games.etterna.replay import find_etterna_dirs
    return find_etterna_dirs()


def _autodetect_root():
    """The override field stores the install root (parent of `Save/`) so
    `Preferences.ini` AdditionalSongFolders resolution keeps working."""
    save = _find_dirs().get('save_dir')
    return str(Path(save).parent) if save else None


def _validate_root(path):
    if not path:
        return False
    p = Path(path)
    if not p.is_dir():
        return False
    save = p / 'Save'
    if save.is_dir() and ((save / 'LocalProfiles').is_dir()
                          or (save / 'Etterna.xml').is_file()):
        return True
    # Accept a bare Save/ dir too (back-compat for pre-root overrides).
    return (p / 'LocalProfiles').is_dir() or (p / 'Etterna.xml').is_file()


def _resolve_chart_context(replay, entry=None, progress=None):
    from analysis.core import game as game_mod
    return game_mod.get('etterna').resolve_all(
        replay, entry=entry, progress=progress)


def _note_viz_config(replay, judge=None, od=None):
    from analysis.viz.note_visualizer import etterna_windows
    j = judge or 'J4'
    return {
        'windows': etterna_windows(j),
        'unit_label': f'noterow  ;  {j}',
        'rows_per_ms': 0.37,
        'win': 2400,
    }


MANIFEST = GameManifest(
    name='etterna',
    path_fields=[
        PathField(
            key='root',
            label='Etterna install folder',
            hint=('Point at your Etterna install folder ; the one that '
                  'contains `Save/` and `Songs/`. Additional song folders '
                  'listed in `Preferences.ini` are picked up automatically.'),
            placeholder={
                'win32': r'e.g. C:\Games\Etterna',
                'linux': 'e.g. ~/.etterna or ~/etterna',
                'darwin': 'e.g. /Applications/Etterna',
                'default': 'e.g. ~/.etterna or ~/etterna',
            },
            settings_key=_ETTERNA_ROOT_KEY,
            error_hint='folder missing Save/ (or LocalProfiles/Etterna.xml)',
            autodetect=_autodetect_root,
            validate=_validate_root,
        ),
    ],
    find_dirs=_find_dirs,
    note_viz_config=_note_viz_config,
    resolve_chart_context=_resolve_chart_context,
)
