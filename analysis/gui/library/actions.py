"""Registry for plugin-contributed actions on the library tab.

Plugins expose actions by adding a top-level ``register_library_actions``
to any bundle module::

    def register_library_actions(add):
        add('Live stats', open_live_stats)

``add(label, callback)`` takes a plain callable; clicking the button
calls it with no arguments. The registry is intentionally minimal — it's
a toolbar, not a full menu system. If a plugin needs richer UX
(submenus, icons, keyboard shortcuts), that lives in a future API.

The registry is process-wide: a single instance holds every plugin's
actions and fires a listener whenever a plugin adds or removes one.
``library_tab`` subscribes so that actions registered *after* the tab
builds (e.g. by a plugin enabled via the Plugins dialog) show up without
restart.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable


@dataclass
class LibraryAction:
    key: str
    label: str
    callback: Callable[[], None]
    module: str = ''


class LibraryActionRegistry:
    """Process-wide list of plugin-contributed toolbar actions.

    Thread-safe for the add/clear path; listeners are called
    synchronously inside ``add`` so UI updates happen on the calling
    thread (plugin registration happens on the Qt thread during
    discovery).
    """

    def __init__(self):
        self._actions: list[LibraryAction] = []
        self._listeners: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    def add(self, label: str, callback: Callable[[], None], *,
            key: str | None = None, module: str = '') -> LibraryAction:
        if not callable(callback):
            raise TypeError('callback must be callable')
        label = str(label).strip()
        if not label:
            raise ValueError('label is required')
        key = str(key or f'{module}:{label}')
        action = LibraryAction(
            key=key, label=label, callback=callback, module=str(module))
        with self._lock:
            # Replace any existing action with the same key — supports
            # re-registration after a plugin reload.
            self._actions = [a for a in self._actions if a.key != key]
            self._actions.append(action)
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn()
            except Exception as exc:
                print(f'library action listener failed: {exc}')
        return action

    def clear_module(self, module: str) -> int:
        """Drop every action owned by ``module``. Called by the plugin
        manager on rediscovery so a disabled bundle's buttons vanish.
        Returns the number of actions removed."""
        module = str(module)
        with self._lock:
            before = len(self._actions)
            self._actions = [a for a in self._actions if a.module != module]
            removed = before - len(self._actions)
            listeners = list(self._listeners) if removed else []
        for fn in listeners:
            try:
                fn()
            except Exception as exc:
                print(f'library action listener failed: {exc}')
        return removed

    def actions(self) -> list[LibraryAction]:
        with self._lock:
            return list(self._actions)

    def subscribe(self, fn: Callable[[], None]) -> Callable[[], None]:
        """Register ``fn`` to be called after any add/clear. Returns a
        handle that is itself the unsubscribe callable."""
        with self._lock:
            self._listeners.append(fn)

        def _unsub():
            with self._lock:
                try:
                    self._listeners.remove(fn)
                except ValueError:
                    pass
        return _unsub


_singleton: LibraryActionRegistry | None = None
_singleton_lock = threading.Lock()


def get_registry() -> LibraryActionRegistry:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = LibraryActionRegistry()
        return _singleton


def reset_for_tests() -> None:
    global _singleton
    with _singleton_lock:
        _singleton = None
