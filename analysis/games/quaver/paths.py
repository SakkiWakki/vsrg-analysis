"""Quaver install-path discovery.

Quaver stores its Songs folder under the install root (`<root>/Songs`).
Replays live under `<root>/Data/Replays/`. We probe the common install
locations on Linux (Steam Proton, Wine, ~/Quaver) and Windows; an
explicit override saved by the GUI takes priority.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


# Folders we expect under a valid Quaver install root.
_REQUIRED_SUBDIRS = ('Songs',)


def _candidate_roots():
    """Return install-root candidates in priority order.

    Linux: covers `~/Quaver`, the standard Steam paths under `~/.steam`
    and `~/.local/share/Steam`, and the Wine prefix Quaver users sometimes
    keep at `~/Games/Quaver`. Steam libraries on alternative drives are
    discovered via `libraryfolders.vdf` (Steam writes one entry per
    library). Windows defaults to `%LOCALAPPDATA%\\Quaver` and Steam's
    install path under `Program Files`.
    """
    home = Path.home()
    out = [
        home / 'Quaver',
        home / '.steam' / 'steam' / 'steamapps' / 'common' / 'Quaver',
        home / '.local' / 'share' / 'Steam' / 'steamapps' / 'common' / 'Quaver',
        home / 'Games' / 'Quaver',
    ]
    out.extend(_steam_library_quaver_paths())
    if sys.platform == 'win32':
        local_appdata = os.environ.get('LOCALAPPDATA')
        if local_appdata:
            out.append(Path(local_appdata) / 'Quaver')
        program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
        out.append(Path(program_files) / 'Steam' / 'steamapps'
                   / 'common' / 'Quaver')
    return out


def _steam_library_quaver_paths():
    """Parse Steam's `libraryfolders.vdf` so secondary libraries (e.g. a
    drive at `/mnt/...`) are picked up automatically. Returns an empty
    list when the VDF can't be read or no libraries are listed."""
    home = Path.home()
    vdf_candidates = [
        home / '.steam' / 'steam' / 'steamapps' / 'libraryfolders.vdf',
        home / '.local' / 'share' / 'Steam' / 'steamapps' / 'libraryfolders.vdf',
    ]
    if sys.platform == 'win32':
        program_files = os.environ.get('ProgramFiles(x86)',
                                        r'C:\Program Files (x86)')
        vdf_candidates.append(
            Path(program_files) / 'Steam' / 'steamapps'
            / 'libraryfolders.vdf')

    out = []
    for vdf in vdf_candidates:
        if not vdf.is_file():
            continue
        try:
            text = vdf.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        # Steam VDF library entries look like `"path"  "/mnt/.../SteamLibrary"`.
        # Cheap regex-free split is fine here ; the format is line-oriented.
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith('"path"'):
                continue
            parts = stripped.split('"')
            if len(parts) < 5:
                continue
            lib_root = parts[3]
            out.append(Path(lib_root) / 'steamapps' / 'common' / 'Quaver')
        break
    return out


def find_quaver_dirs():
    """Return install-root + the well-known Quaver subfolders.

    Keys:
        root              install folder (`<root>/Songs`, `<root>/Data/r`, ...).
        songs_dir         `<root>/Songs/`.
        replays_dir       `<root>/Replays/` ; user-exported `.qr` files.
        auto_replays_dir  `<root>/Data/r/` ; every play is auto-saved here as
                          `<scoreId>.qr` (`LeaderboardScoreRightClickOptions.cs`
                          and `ConfigManager.cs::_dataDirectory`).
    Each value is None when the corresponding folder couldn't be located,
    so callers can probe `find_quaver_dirs().get('auto_replays_dir')`
    without crashing on a stub install.

    The install root is resolved via:
        1. GUI override (paths/quaver_root via QSettings).
        2. `QUAVER_ROOT` env var.
        3. `QUAVER_SONGS_DIR` env var (legacy ; covers older configs that
           pointed straight at Songs/ instead of the install root).
        4. Autodetect against `_candidate_roots()`.
    """
    root = _resolve_root()
    if root is None:
        # Legacy env var that pointed at Songs/ directly.
        legacy_songs = os.environ.get('QUAVER_SONGS_DIR')
        if legacy_songs and Path(legacy_songs).is_dir():
            return {'root': None, 'songs_dir': legacy_songs,
                    'replays_dir': None, 'auto_replays_dir': None}
        return {'root': None, 'songs_dir': None,
                'replays_dir': None, 'auto_replays_dir': None}

    rp = Path(root)
    songs = rp / 'Songs'
    replays = rp / 'Replays'
    auto_replays = rp / 'Data' / 'r'
    return {
        'root': str(rp),
        'songs_dir': str(songs) if songs.is_dir() else None,
        'replays_dir': str(replays) if replays.is_dir() else None,
        'auto_replays_dir': str(auto_replays) if auto_replays.is_dir() else None,
    }


def all_replay_dirs():
    """Return every directory we should walk for `.qr` files (exports +
    autosaves). Caller iterates and `.rglob('*.qr')` each. The returned
    list preserves priority ; `Data/r/` last because that's where the
    bulk of files live and the export folder is the user-curated source."""
    dirs = find_quaver_dirs()
    out = []
    for key in ('replays_dir', 'auto_replays_dir'):
        path = dirs.get(key)
        if path:
            out.append(path)
    return out


def validate_quaver_root(path):
    """True when `path` looks like a Quaver install root: it's a folder
    that contains the subfolders we depend on (currently just `Songs`)."""
    if not path:
        return False
    p = Path(path)
    if not p.is_dir():
        return False
    return all((p / sub).is_dir() for sub in _REQUIRED_SUBDIRS)


def _resolve_root():
    override = _qsettings_quaver_root()
    if override and Path(override).is_dir():
        return override
    env_root = os.environ.get('QUAVER_ROOT')
    if env_root and Path(env_root).is_dir():
        return env_root
    for c in _candidate_roots():
        if c.is_dir() and validate_quaver_root(c):
            return str(c)
    return None


def _qsettings_quaver_root():
    """Read the persisted GUI override without taking a hard dep on Qt
    in CLI/test paths -- if the GUI hasn't been imported (no QSettings
    available), we silently skip the override layer."""
    try:
        from analysis.gui.settings import get_quaver_root_override
    except Exception:
        return None
    try:
        return get_quaver_root_override()
    except Exception:
        return None
