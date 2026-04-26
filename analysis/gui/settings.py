"""QSettings wrapper for persistent app state (scroll speed, mode,
last-used library filters, window geometry, etc.).

Install-path overrides live in `analysis.core.path_overrides` (the
"shopkeeper") instead of here ; the Qt backend in
`analysis.gui.path_overrides_qt` is what writes them through this same
QSettings instance. That keeps `analysis.core` and the per-game `replay`
modules Qt-free.

Player-settings validators live below: every persisted player preference
declares whether it's game-agnostic or game-dependent and provides a
validator. `load_player_settings(game)` returns a dict with every value
already coerced and game-checked, so caller sites can't accidentally
carry an incompatible value across a game switch."""
from dataclasses import dataclass
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


FIRST_RUN_KEY = 'paths/first_run_done'


def is_first_run_done():
    return bool(get_settings().value(FIRST_RUN_KEY, False, type=bool))


def mark_first_run_done():
    get_settings().setValue(FIRST_RUN_KEY, True)


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
