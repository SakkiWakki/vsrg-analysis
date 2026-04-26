"""Qt-backed implementation of the path-overrides shopkeeper.

Installed once at GUI bootstrap (`analysis.gui.app`) so every component
that calls `analysis.core.path_overrides.get()` reads from QSettings
without dragging Qt into headless modules.
"""
from __future__ import annotations

from analysis.core import path_overrides


class QtSettingsBackend:
    """Reads/writes overrides through the shared `QSettings` instance.
    Empty values are mapped to a remove() so the file stays tidy."""

    def get(self, settings_key: str) -> str | None:
        from analysis.gui.settings import get_settings
        v = get_settings().value(settings_key)
        return None if v is None else str(v)

    def set(self, settings_key: str, value: str | None) -> None:
        from analysis.gui.settings import get_settings
        s = get_settings()
        if value is None:
            s.remove(settings_key)
        else:
            s.setValue(settings_key, value)


def install() -> None:
    """Wire the QtSettingsBackend in as the global override backend.
    Idempotent ; calling twice just replaces the backend with itself."""
    path_overrides.set_backend(QtSettingsBackend())
