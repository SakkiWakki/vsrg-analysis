"""QSettings wrapper for persistent app state (scroll speed, mode,
last-used library filters, window geometry, etc.).

Also owns the install-path overrides for Etterna and osu!. Core modules
call `find_etterna_dirs` / `find_osu_dirs`, which consult these overrides
before falling back to autodetection ; so a user-configured path wins
even though the core modules don't depend on Qt.

The overrides now store **install roots** (e.g. `~/etterna/`, not
`~/etterna/Save/`). Resolution into Save/Songs/Replays/XML happens in the
game-specific replay modules, which also read `Preferences.ini` and
`osu!.<user>.cfg` to honor user-configured subpaths (AdditionalSongFolders,
BeatmapDirectory, picked profile).

Player-settings validators live at the bottom: every persisted player
preference declares whether it's game-agnostic or game-dependent and
provides a validator. `load_player_settings(game)` returns a dict with
every value already coerced and game-checked, so caller sites can't
accidentally carry an incompatible value across a game switch."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
# Stored as strings under paths/etterna_root, paths/osu_root. Empty/None means
# "fall back to autodetect". Legacy keys paths/etterna_save and paths/osu_songs
# are migrated transparently on read: if the old value pointed at Save/ or
# Songs/, we fold it up to the install root.
#
# osu! profiles: a single install can have multiple per-user configs
# (osu!.<USER>.cfg). If the user has >1 we persist the chosen one under
# paths/osu_profile so later launches remember the selection.

ETTERNA_ROOT_KEY = 'paths/etterna_root'
OSU_ROOT_KEY = 'paths/osu_root'
OSU_PROFILE_KEY = 'paths/osu_profile'
QUAVER_ROOT_KEY = 'paths/quaver_root'
FIRST_RUN_KEY = 'paths/first_run_done'

def _str_or_none(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _fold_to_install_root(path, subdir_names):
    """If `path` ends in one of `subdir_names` (case-insensitive), return its
    parent ; otherwise return `path` unchanged. Used to migrate legacy
    Save/Songs paths to install roots."""
    p = Path(path)
    if p.name.lower() in {s.lower() for s in subdir_names}:
        return str(p.parent)
    return str(p)

def get_etterna_root_override():
    return _str_or_none(get_settings().value(ETTERNA_ROOT_KEY))


def set_etterna_root_override(path):
    s = get_settings()
    p = _str_or_none(path)
    if p is None:
        s.remove(ETTERNA_ROOT_KEY)
    else:
        s.setValue(ETTERNA_ROOT_KEY, p)


def get_osu_root_override():
    return _str_or_none(get_settings().value(OSU_ROOT_KEY))


def set_osu_root_override(path):
    s = get_settings()
    p = _str_or_none(path)
    if p is None:
        s.remove(OSU_ROOT_KEY)
    else:
        s.setValue(OSU_ROOT_KEY, p)


def get_osu_profile_override():
    """The selected osu!.<user>.cfg filename (not a path). None means 'pick
    automatically' ; resolver falls back to newest-mtime cfg."""
    return _str_or_none(get_settings().value(OSU_PROFILE_KEY))


def set_osu_profile_override(name):
    s = get_settings()
    p = _str_or_none(name)
    if p is None:
        s.remove(OSU_PROFILE_KEY)
    else:
        s.setValue(OSU_PROFILE_KEY, p)


def get_quaver_root_override():
    return _str_or_none(get_settings().value(QUAVER_ROOT_KEY))


def set_quaver_root_override(path):
    s = get_settings()
    p = _str_or_none(path)
    if p is None:
        s.remove(QUAVER_ROOT_KEY)
    else:
        s.setValue(QUAVER_ROOT_KEY, p)


# Back-compat shims ; migrate.py and any external callers still refer to these
# names. They now return/accept install roots, which is what callers want
# anyway (migrate.py just echoes the value into the config tree).
get_etterna_save_override = get_etterna_root_override
set_etterna_save_override = set_etterna_root_override
get_osu_songs_override = get_osu_root_override
set_osu_songs_override = set_osu_root_override


def is_first_run_done():
    return bool(get_settings().value(FIRST_RUN_KEY, False, type=bool))


def mark_first_run_done():
    get_settings().setValue(FIRST_RUN_KEY, True)


# Validators live on the per-game GuiAdapter; these are thin wrappers so
# pre-existing callers / tests keep working.
def validate_etterna_root(path):
    from analysis.core import gui_adapter as gui_mod
    return gui_mod.get('etterna').validate_root(path)


def validate_osu_root(path):
    from analysis.core import gui_adapter as gui_mod
    return gui_mod.get('osu').validate_root(path)


def validate_quaver_root(path):
    from analysis.core import gui_adapter as gui_mod
    return gui_mod.get('quaver').validate_root(path)


validate_etterna_save = validate_etterna_root
validate_osu_songs = validate_osu_root


# ---- player settings: typed, optionally game-scoped ------------------------
# Every persisted player preference lives here so that adding a new one is a
# single registration rather than a new branch in both player_tab and
# library_tab. A validator receives (raw_value, game) and returns the coerced
# value to use, or the declared default if the raw value is missing/invalid.


@dataclass(frozen=True)
class PlayerSetting:
    key: str                                      # QSettings key
    game_dependent: bool                          # True → validator uses game
    default: Any                                  # used when missing/invalid
    validate: Callable[[Any, str | None], Any]    # (raw, game) -> coerced


def _validate_skin(raw, _game):
    s = None if raw is None else str(raw)
    return s if s in ('bar', 'circle') else 'bar'


def _validate_bool(default):
    def _v(raw, _game):
        if raw is None:
            return default
        # QSettings already coerces when type=bool is passed, but this layer
        # runs before the Qt call so we normalize defensively.
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in ('true', '1', 'yes'):
            return True
        if s in ('false', '0', 'no', ''):
            return False
        return default
    return _v


def _validate_scroll_mode(raw, game):
    """Scroll mode must exist in the registry AND be compatible with `game`.
    A saved 'cmod' is fine under etterna and falls back to the osu default
    when the user opens an osu replay."""
    from analysis.player import scroll as scroll_registry
    scroll_registry.ensure_loaded()
    mode = None if raw is None else str(raw)
    # Back-compat: the old player wrote 'linear' for the ms mode.
    if mode == 'linear':
        mode = 'ms'
    if game and mode and scroll_registry.is_compatible(mode, game):
        return mode
    if game:
        return scroll_registry.default_for_game(game)
    return mode or 'ms'


PLAYER_SETTINGS: dict[str, PlayerSetting] = {
    s.key.split('/', 1)[1]: s for s in (
        PlayerSetting('player/skin', False, 'bar', _validate_skin),
        PlayerSetting('player/press_hide', False, False, _validate_bool(False)),
        PlayerSetting('player/pitch_correct', False, True, _validate_bool(True)),
        PlayerSetting('player/render_uncapped', False, True,
                  _validate_bool(True)),
        PlayerSetting('player/scroll_mode', True, None, _validate_scroll_mode),
    )
}


def load_player_settings(game: str | None = None) -> dict[str, Any]:
    """Return every registered player setting, validated. `game` is required
    for game-dependent entries; passing None leaves them un-coerced, which is
    only appropriate for UI code that doesn't yet know the game (library tab
    before it picks a replay)."""
    s = get_settings()
    out: dict[str, Any] = {}
    for name, desc in PLAYER_SETTINGS.items():
        if desc.game_dependent and game is None:
            out[name] = desc.default
            continue
        raw = s.value(desc.key, desc.default)
        out[name] = desc.validate(raw, game)
    return out


def save_player_setting(name: str, value: Any) -> None:
    """Write one player setting back. Central helper so call-sites aren't
    repeating the `player/...` key string."""
    desc = PLAYER_SETTINGS[name]
    get_settings().setValue(desc.key, value)
