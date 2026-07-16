"""NotITG install discovery.

NotITG ships as a portable folder (no installer, no standard OS
location), so autodetection only probes a few conventional spots and
users normally point the path field at their install. The root is the
folder containing `Songs/` and `Program/`.
"""
from __future__ import annotations

from pathlib import Path

from analysis.core import path_overrides

NOTITG_ROOT_KEY = 'paths/notitg_root'

_CANDIDATES = ('~/NotITG', '~/Games/NotITG', '~/games/NotITG')


def validate_root(path) -> bool:
    if not path:
        return False
    return (Path(path) / 'Songs').is_dir()


def autodetect_root() -> str | None:
    for candidate in _CANDIDATES:
        expanded = Path(candidate).expanduser()
        if validate_root(expanded):
            return str(expanded)
    return None


def find_notitg_dirs() -> dict:
    root = path_overrides.get(NOTITG_ROOT_KEY) or autodetect_root()
    if not root or not validate_root(root):
        return {'root': None, 'songs_dir': None}
    return {'root': str(root), 'songs_dir': str(Path(root) / 'Songs')}
