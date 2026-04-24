from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class LibraryAction:
    key: str
    label: str
    callback: Callable[[], None]
    module: str = ''


class LibraryActionRegistry:
    def __init__(self):
        self._actions: list[LibraryAction] = []
        self._listeners: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    def add(
        self,
        label: str,
        callback: Callable[[], None],
        *,
        key: str | None = None,
        module: str = '',
    ) -> LibraryAction:
        action = self._make_action(label, callback, key=key, module=module)

        with self._lock:
            self._actions = [
                existing for existing in self._actions
                if existing.key != action.key
            ]
            self._actions.append(action)
            listeners_snapshot = tuple(self._listeners)

        self._notify(listeners_snapshot)
        return action

    def clear_module(self, module: str) -> int:
        module = str(module)

        with self._lock:
            before = len(self._actions)
            self._actions = [
                action for action in self._actions
                if action.module != module
            ]
            removed = before - len(self._actions)
            listeners = tuple(self._listeners) if removed else ()

        self._notify(listeners)
        return removed

    def actions(self) -> list[LibraryAction]:
        with self._lock:
            return list(self._actions)

    def subscribe(self, fn: Callable[[], None]) -> Callable[[], None]:
        if not callable(fn):
            raise TypeError('listener must be callable')

        with self._lock:
            self._listeners.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._listeners.remove(fn)
                except ValueError:
                    pass

        return unsubscribe

    @staticmethod
    def _make_action(
        label: str,
        callback: Callable[[], None],
        *,
        key: str | None,
        module: str,
    ) -> LibraryAction:
        if not callable(callback):
            raise TypeError('callback must be callable')

        label = str(label).strip()
        if not label:
            raise ValueError('label is required')

        module = str(module)
        return LibraryAction(
            key=str(key or f'{module}:{label}'),
            label=label,
            callback=callback,
            module=module,
        )

    @staticmethod
    def _notify(listeners: tuple[Callable[[], None], ...]) -> None:
        for listener in listeners:
            try:
                listener()
            except Exception:
                _LOG.exception('library action listener failed')


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