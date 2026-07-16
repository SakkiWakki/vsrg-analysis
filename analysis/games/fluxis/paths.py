"""fluXis data-directory discovery.

fluXis (osu.Framework game) keeps everything under one storage folder:
`fluxis.realm` (the metadata/link database), `maps/<mapset-guid>/` with
`.fsc` charts + assets, and `replays/<score-guid>.frp`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from analysis.core import path_overrides

FLUXIS_DATA_KEY = 'paths/fluxis_data'


def _default_data_candidates():
    home = Path.home()
    match sys.platform:
        case 'win32':
            appdata = os.environ.get('APPDATA')
            return [Path(appdata) / 'fluXis'] if appdata else []
        case 'darwin':
            return [home / 'Library' / 'Application Support' / 'fluXis']
        case _:
            xdg = os.environ.get('XDG_DATA_HOME')
            base = Path(xdg) if xdg else home / '.local' / 'share'
            return [base / 'fluXis']


def validate_data_dir(path) -> bool:
    return (Path(path).expanduser() / 'fluxis.realm').is_file()


def autodetect_data_dir():
    for c in _default_data_candidates():
        if (c / 'fluxis.realm').is_file():
            return str(c)
    return None


def find_fluxis_dirs():
    """Returns dict with data_dir, realm_path, maps_dir, replays_dir;
    every value None when no install was found. User override from GUI
    settings wins over the env var, which wins over autodetection."""
    candidates = []

    override = path_overrides.get(FLUXIS_DATA_KEY)
    if override:
        candidates.append(Path(override).expanduser())
    env = os.environ.get('FLUXIS_DATA_DIR')
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(_default_data_candidates())

    for c in candidates:
        if (c / 'fluxis.realm').is_file():
            return {
                'data_dir': c,
                'realm_path': c / 'fluxis.realm',
                'maps_dir': c / 'maps',
                'replays_dir': c / 'replays',
            }
    return {'data_dir': None, 'realm_path': None,
            'maps_dir': None, 'replays_dir': None}
