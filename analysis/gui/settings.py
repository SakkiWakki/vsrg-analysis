"""QSettings wrapper for persistent app state (scroll speed, mode,
last-used library filters, window geometry, etc.).

Also owns the install-path overrides for Etterna and osu!. Core modules
call `find_etterna_dirs` / `find_osu_dirs`, which consult these overrides
before falling back to autodetection — so a user-configured path wins
even though the core modules don't depend on Qt."""
from pathlib import Path

from PySide6.QtCore import QSettings

_ORG = 'clanker'
_APP = 'vsrg-analysis'

_cached = None


def get_settings():
    global _cached
    if _cached is None:
        _cached = QSettings(_ORG, _APP)
    return _cached


# ---- install-path overrides ------------------------------------------------
# Stored as strings under paths/etterna_save and paths/osu_songs. Empty/None
# means "fall back to autodetect". We expose simple getters/setters so that
# core modules can consult them without importing Qt widgets.

ETTERNA_SAVE_KEY = 'paths/etterna_save'
OSU_SONGS_KEY = 'paths/osu_songs'
FIRST_RUN_KEY = 'paths/first_run_done'


def _str_or_none(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def get_etterna_save_override():
    return _str_or_none(get_settings().value(ETTERNA_SAVE_KEY))


def set_etterna_save_override(path):
    s = get_settings()
    p = _str_or_none(path)
    if p is None:
        s.remove(ETTERNA_SAVE_KEY)
    else:
        s.setValue(ETTERNA_SAVE_KEY, p)


def get_osu_songs_override():
    return _str_or_none(get_settings().value(OSU_SONGS_KEY))


def set_osu_songs_override(path):
    s = get_settings()
    p = _str_or_none(path)
    if p is None:
        s.remove(OSU_SONGS_KEY)
    else:
        s.setValue(OSU_SONGS_KEY, p)


def is_first_run_done():
    return bool(get_settings().value(FIRST_RUN_KEY, False, type=bool))


def mark_first_run_done():
    get_settings().setValue(FIRST_RUN_KEY, True)


def validate_etterna_save(path):
    """A valid Etterna save dir either has a LocalProfiles/ subtree or a
    direct Etterna.xml. We don't require ReplaysV2/ because some users only
    have XML-only imports from older installs."""
    if not path:
        return False
    p = Path(path)
    if not p.is_dir():
        return False
    return (p / 'LocalProfiles').is_dir() or (p / 'Etterna.xml').is_file()


def validate_osu_songs(path):
    """A valid osu! songs dir is any existing directory — mania beatmap folders
    don't have a consistent marker, but the dir itself must exist."""
    if not path:
        return False
    return Path(path).is_dir()
