"""Process-global registry of user-configured install-path overrides.

Acts as a "town shopkeeper": every component that needs to know where a
game lives -- core path resolvers, library scanners, the paths dialog --
calls `get(game, key)` here. Whoever owns the persistence layer
(currently the Qt GUI, via `analysis.gui.path_overrides_qt`) installs a
backend at startup; headless callers leave it unset and `get()` returns
None, falling back to autodetection.

Keeps `analysis.core` and `analysis.games.<game>.replay*` Qt-free so
nothing in those packages drags PySide6 into headless contexts.
"""
from __future__ import annotations

from typing import Protocol


class _Backend(Protocol):
    def get(self, settings_key: str) -> str | None: ...
    def set(self, settings_key: str, value: str | None) -> None: ...


_BACKEND: _Backend | None = None


def set_backend(backend: _Backend | None) -> None:
    """Install (or clear) the persistence backend. Called once at GUI
    bootstrap and once at GUI shutdown ; tests inject fakes here."""
    global _BACKEND
    _BACKEND = backend


def get(settings_key: str) -> str | None:
    """Return the user's saved override for `settings_key`, or None when
    no backend is installed or no value has been set. Empty strings are
    normalized to None so callers can do `override or autodetect()`."""
    if _BACKEND is None:
        return None
    raw = _BACKEND.get(settings_key)
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def set(settings_key: str, value: str | None) -> None:
    """Persist (or clear) the override for `settings_key`. No-op when no
    backend is installed -- headless callers shouldn't be writing paths."""
    if _BACKEND is None:
        return
    s = None if value is None else str(value).strip() or None
    _BACKEND.set(settings_key, s)
