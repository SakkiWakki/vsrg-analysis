"""Scroll-mode registry.

Each game contributes zero or more `ScrollMode` entries describing how a
game-native scalar (Etterna CMOD BPM, osu!mania Scroll Speed, Quaver SV...)
maps to effective px/sec. A small core ms-to-judgment mode is registered
here and is always present.

Modes are *globally* accessible: a player loaded from an Etterna replay
can switch to `osu` mode if the user has that game folder installed.

Adding a mode (see analysis/games/README.md for game setup):
    ScrollMode(
        key='cmod', label='CMOD', game='etterna',
        to_pxps=lambda value, opts, p: ...,
        from_pxps=lambda pxps, opts, p: ...,
        default_value=600.0,
        nudge=_multiplicative_nudge,
        options={'mini': 0.0, 'receptor_size': 1.0},
        on_enter=_cmod_on_enter,
        on_exit=_cmod_on_exit,
    )
    register(mode)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Any

from analysis.core import game as game_mod


# --- Mode descriptor ---------------------------------------------------------


@dataclass
class ScrollMode:
    key: str                       # unique id, e.g. 'cmod'
    label: str                     # UI label, e.g. 'CMOD'
    game: str | None               # 'etterna', 'osu', None (core/cross-game)
    to_pxps: Callable[[float, dict, Any], float]
    from_pxps: Callable[[float, dict, Any], float]
    default_value: float
    value_bounds: tuple[float, float] = (1.0, 5000.0)
    nudge: Callable[[float, float, dict], float] | None = None
    format_value: Callable[[float], str] | None = None
    # Options are per-mode knobs (e.g. Etterna mini, receptor size). Each
    # value in this dict is the default; the per-player state dict copies it.
    options: dict[str, Any] = field(default_factory=dict)
    # Optional lifecycle callbacks triggered by Player.set_scroll_mode:
    #   on_enter(player, mode_state): called after the mode becomes active
    #   on_exit(player,  mode_state): called before switching away
    on_enter: Callable[[Any, dict], None] | None = None
    on_exit: Callable[[Any, dict], None] | None = None


# --- Nudge helpers -----------------------------------------------------------


def multiplicative_nudge(value: float, factor: float, opts: dict) -> float:
    """Scale by `factor` (>1 faster, <1 slower). Used by ms-style modes where
    a log-uniform step feels right across the whole range."""
    return value * factor


def ms_nudge(value: float, factor: float, opts: dict) -> float:
    """ms-to-judgment: faster = smaller ms, so we invert the factor."""
    return value / factor


def integer_step_nudge(value: float, factor: float, opts: dict) -> float:
    """Snap toward integer units, then step by +/-1. A fractional value
    (from a cross-mode translation like C1115 -> osu 35.05) first
    floors/ceils to reach an integer boundary; thereafter stepping is by
    whole units, matching osu!mania's in-game F3/F4 binding."""
    d = 1 if factor >= 1 else -1
    snapped = math.ceil(value) if d > 0 else math.floor(value)
    return snapped + d * (value == snapped)


# --- Registry ----------------------------------------------------------------


_MODES: dict[str, ScrollMode] = {}
_ORDER: list[str] = []


def register(mode: ScrollMode) -> None:
    if mode.key in _MODES:
        return  # idempotent: re-import of adapters during hot-reload is fine
    _MODES[mode.key] = mode
    _ORDER.append(mode.key)


def get(key: str) -> ScrollMode | None:
    return _MODES.get(key)


def all_modes() -> list[ScrollMode]:
    """Return modes in registration order. Core 'ms' mode comes first, then
    per-game modes in the order games were discovered."""
    return [_MODES[k] for k in _ORDER]


def keys() -> list[str]:
    return list(_ORDER)


def is_compatible(mode_key: str, game: str) -> bool:
    """True if `mode_key` is safe to use under `game`. Core modes (game=None)
    are always compatible; per-game modes must match exactly. Unknown keys
    are incompatible so callers fall back to the game's default."""
    m = _MODES.get(mode_key)
    if m is None:
        return False
    return m.game is None or m.game == game


def default_for_game(game: str) -> str:
    """The preferred scroll mode for `game`, asked of its adapter. Returns
    the core 'ms' mode if the adapter or registry can't answer."""
    try:
        return game_mod.get(game).default_scroll_mode()
    except Exception:
        return 'ms'


# --- Core mode: ms-to-judgment (game-independent) ----------------------------


def _ms_to_pxps(value, opts, p):
    ms = max(1e-6, float(value))
    return (p.H * p.hit_line_y_frac) / (ms / 1000.0)


def _ms_from_pxps(pxps, opts, p):
    return (p.H * p.hit_line_y_frac) / max(1e-6, pxps) * 1000.0


register(ScrollMode(
    key='ms',
    label='ms',
    game=None,
    to_pxps=_ms_to_pxps,
    from_pxps=_ms_from_pxps,
    default_value=400.0,
    value_bounds=(50.0, 3000.0),
    nudge=ms_nudge,
    format_value=lambda v: f'{int(v)}ms',
))


# --- Auto-discovery trigger --------------------------------------------------


_discovered = False


def ensure_loaded() -> None:
    """Force a first-level scan of analysis/games/*/adapter.py. Each adapter
    module imports this module and calls register() at import time, so once
    we've imported them, the registry is populated. Idempotent."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    game_mod.discover_games()
