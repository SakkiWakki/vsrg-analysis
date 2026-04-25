"""Process-wide hook for injecting live game state into overlay components.

The overlay backend runs on its own thread (spun by the existing
``OverlayPublisher.run_draw_thread``). It needs access to the current
:class:`~analysis.overlay.api.OverlayGameState` at frame build time, but
game-state sources are game-specific (osu memory poller, etterna
adapter, ...). Rather than threading a state handle through every
registration and thread, the game adapter installs a *provider* once at
startup and the overlay backend reads through this hook.

Usage (from a game adapter's live-session module)::

    from analysis.components.provider import set_game_state_provider

    def _state_provider():
        return my_live_tracker.snapshot()   # -> OverlayGameState or None

    set_game_state_provider(_state_provider)

The provider is called on the overlay render thread, so it must be
thread-safe (or cheap enough that a short lock is fine). When no
provider is installed (sidebar-only runs, tests, headless CI),
overlay components silently render nothing ; better than crashing.
"""
from __future__ import annotations

import threading
from typing import Callable

from analysis.overlay.api import OverlayGameState


_lock = threading.Lock()
_provider: Callable[[], OverlayGameState | None] | None = None
_memory_provider: Callable[[], 'GameMemoryState | None'] | None = None


def set_game_state_provider(
        fn: Callable[[], OverlayGameState | None] | None) -> None:
    """Install (or clear with ``None``) the overlay game-state provider."""
    global _provider
    with _lock:
        _provider = fn


def set_game_memory_provider(fn: 'Callable[[], GameMemoryState | None] | None') -> None:
    """Install (or clear with ``None``) the native game-memory provider.
    Called by the osu live client when it starts polling."""
    global _memory_provider
    with _lock:
        _memory_provider = fn


def current_game_state() -> OverlayGameState | None:
    with _lock:
        fn = _provider
    if fn is None:
        return None
    try:
        return fn()
    except Exception as exc:
        print(f'[components.provider] game-state provider raised: {exc}')
        return None


def current_game_memory():
    with _lock:
        fn = _memory_provider
    if fn is None:
        return None
    try:
        return fn()
    except Exception as exc:
        print(f'[components.provider] game-memory provider raised: {exc}')
        return None


# Avoid a circular import: GameMemoryState lives in api.py which imports
# nothing from provider.py.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from analysis.components.api import GameMemoryState
    from typing import Callable
