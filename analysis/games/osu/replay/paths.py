"""Install-root + Songs + replays-dir discovery for osu!.

User override (from the path_overrides shopkeeper) wins. Otherwise we
walk a list of common install locations and pick the first that has at
least one `osu!.<user>.cfg` ; then read `BeatmapDirectory` from that cfg
to resolve the real Songs path."""
import os
import sys
from pathlib import Path

from analysis.core import path_overrides


def list_osu_profiles(root):
    """Return `osu!.<user>.cfg` filenames in `root`, sorted by mtime
    (newest first)."""
    try:
        entries = []
        for e in Path(root).iterdir():
            nl = e.name.lower()
            if nl.startswith('osu!.') and nl.endswith('.cfg'):
                try:
                    entries.append((e.stat().st_mtime, e.name))
                except OSError:
                    continue
    except OSError:
        return []
    entries.sort(reverse=True)
    return [n for _, n in entries]


def _pick_osu_cfg(root):
    """Honor the saved profile override if it still exists, else pick
    the most recently modified profile."""
    profiles = list_osu_profiles(root)
    if not profiles:
        return None
    want = path_overrides.get('paths/osu_profile')
    if want and want in profiles:
        return str(Path(root) / want)
    return str(Path(root) / profiles[0])


def _read_beatmap_directory(cfg_path):
    try:
        for line in Path(cfg_path).read_text(errors='ignore').splitlines():
            s = line.strip()
            if not s or s[0] in '#;' or '=' not in s:
                continue
            key, _, val = s.partition('=')
            if key.strip() == 'BeatmapDirectory':
                return val.strip() or None
    except OSError:
        return None
    return None


def _resolve_songs_from_root(root):
    root = Path(root)
    cfg = _pick_osu_cfg(root)
    if cfg:
        bmd = _read_beatmap_directory(cfg)
        if bmd:
            p = Path(bmd)
            if not p.is_absolute():
                p = root / bmd
            if p.is_dir():
                return str(p)
    default = root / 'Songs'
    return str(default) if default.is_dir() else None


def _osu_replays_for(root, songs_dir=None):
    """Replay directories, in priority order. `<root>/Data/r` is the
    primary; we fall back to traditional default locations so a stock
    Wine/native install still works without configuration."""
    out = []

    def _add(p):
        s = str(p)
        if p.exists() and s not in out:
            out.append(s)

    if root:
        _add(Path(root) / 'Data' / 'r')
    if songs_dir:
        _add(Path(songs_dir).parent / 'Data' / 'r')
    home = Path.home()
    _add(home / 'osu!' / 'Data' / 'r')
    if sys.platform == 'win32':
        local = os.environ.get('LOCALAPPDATA')
        if local:
            _add(Path(local) / 'osu!' / 'Data' / 'r')
    else:
        _add(home / '.local' / 'share' / 'osu-wine' / 'osu!' / 'Data' / 'r')
    return out


def _root_candidates():
    home = Path.home()
    if sys.platform == 'win32':
        # Native Windows osu! default: %LOCALAPPDATA%\osu!
        local = os.environ.get('LOCALAPPDATA')
        out = []
        if local:
            out.append(Path(local) / 'osu!')
        out += [home / 'osu!', home / 'Documents' / 'osu!']
        return out
    # Linux: bare install dir + wine prefixes + WSL interop.
    user = os.environ.get('USER', '')
    return [
        home / 'osu!',
        home / 'Games' / 'osu!',
        home / '.local' / 'share' / 'osu-wine' / 'osu!',
        home / '.local' / 'share' / 'osu!',
        home / 'Documents' / 'osu!',
        home / '.wine' / 'drive_c' / 'users' / user / 'AppData' / 'Local' / 'osu!',
        Path('/mnt/c/Users') / user / 'AppData' / 'Local' / 'osu!',
    ]


def _songs_only_candidates():
    home = Path.home()
    if sys.platform == 'win32':
        local = os.environ.get('LOCALAPPDATA')
        out = [home / 'osu!' / 'Songs']
        if local:
            out.append(Path(local) / 'osu!' / 'Songs')
        return out
    return [home / 'osu!' / 'Songs',
            home / '.local' / 'share' / 'osu-wine' / 'osu!' / 'Songs']


def find_osu_dirs():
    """Return `{'root', 'songs_dir', 'replays_dirs'}`. User override
    wins; `songs_dir` is resolved from the selected profile's
    `BeatmapDirectory`."""
    override = path_overrides.get('paths/osu_root')
    if override and Path(override).exists():
        songs = _resolve_songs_from_root(override)
        return {'root': str(override), 'songs_dir': songs,
                'replays_dirs': _osu_replays_for(override, songs)}

    for root in _root_candidates():
        if root.is_dir() and list_osu_profiles(root):
            songs = _resolve_songs_from_root(root)
            return {'root': str(root), 'songs_dir': songs,
                    'replays_dirs': _osu_replays_for(str(root), songs)}

    # Last resort: a stock Songs/ without a cfg ; preserves autodetect
    # for users who only have charts.
    for c in _songs_only_candidates():
        if c.is_dir():
            return {'root': str(c.parent), 'songs_dir': str(c),
                    'replays_dirs': _osu_replays_for(str(c.parent), str(c))}

    return {'root': None, 'songs_dir': None,
            'replays_dirs': _osu_replays_for(None)}
