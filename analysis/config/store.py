"""Unified, observable config store.

One nested dict holds every setting the app persists — paths, player
chrome state, per-plugin state. Readers access leaves by dotted path
(``'plugins.builtin:judgment.enabled'``) and subscribe to a prefix to
get fanout when any descendant changes. Writers mutate single leaves,
never whole subtrees, so a flip of one plugin's ``enabled`` flag
doesn't re-publish the rest of the tree.

Design:

  * **Single source of truth.** Legacy QSettings values and the two
    plugin-state JSONs (``player_plugins.json``,
    ``sidebar_sections.json``) are migrated on first load into one
    file: ``~/.config/vsrg-analysis/config.json``. Old files are
    deleted after migration so they can't drift.
  * **Dotted paths.** Plugin keys can contain colons
    (``'builtin:judgment'``), so dots are the path separator and the
    dict itself stores colon-bearing keys verbatim.
  * **Path-level subscriptions.** ``subscribe('plugins', fn)`` fires
    for any change under ``plugins.*``; ``subscribe('plugins.foo', fn)``
    only fires for that subtree. Root subscription is the empty path.
  * **Shared mutable state + fanout.** All windows hold a reference to
    the same store; a write in one window's dialog reaches the others
    through their subscriptions. No snapshotting, no COW at the object
    layer — the JSON file is the persisted snapshot.
  * **Debounced writes.** Bursts of ``set`` calls (e.g. the user
    toggling several checkboxes in a row) coalesce into one disk
    write. Tests can flush synchronously via ``flush()``.

Not covered by this module: cross-process sync. If a second OS process
writes the file, we won't pick it up — the app is single-process today.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from analysis.player.events import EventBus, Subscription


_SEP = '.'


def _split(path: str) -> tuple[str, ...]:
    """Split a dotted path into parts. Empty string → empty tuple
    (the root). Leading/trailing dots are ignored."""
    if not path:
        return ()
    return tuple(p for p in path.split(_SEP) if p)


def _deep_get(tree: dict, parts: tuple[str, ...], default=None):
    node: Any = tree
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def _deep_set(tree: dict, parts: tuple[str, ...], value) -> None:
    """Assign ``value`` at ``parts``, creating intermediate dicts as
    needed. Raises if an intermediate path is occupied by a non-dict."""
    if not parts:
        raise ValueError('cannot set the root of the config tree')
    node = tree
    for p in parts[:-1]:
        nxt = node.get(p)
        if nxt is None:
            nxt = {}
            node[p] = nxt
        elif not isinstance(nxt, dict):
            raise TypeError(
                f'config path {_SEP.join(parts)!r} passes through a '
                f'non-dict value at {p!r}')
        node = nxt
    node[parts[-1]] = value


def _deep_delete(tree: dict, parts: tuple[str, ...]) -> bool:
    if not parts:
        return False
    node = tree
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            return False
        node = node[p]
    if not isinstance(node, dict) or parts[-1] not in node:
        return False
    del node[parts[-1]]
    return True


def _is_under(path: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    """True if ``path`` is equal to or a descendant of ``prefix``."""
    if len(path) < len(prefix):
        return False
    return path[:len(prefix)] == prefix


class ConfigStore:
    """Observable nested config dict with path-level subscriptions."""

    CHANGE_KIND = 'changed'

    def __init__(self, file_path: Path, *, autosave: bool = True,
                 debounce_s: float = 0.2):
        self._path = Path(file_path)
        self._autosave = bool(autosave)
        self._debounce_s = float(debounce_s)
        self._tree: dict = {}
        self._bus = EventBus()
        self._lock = threading.RLock()
        self._save_timer: threading.Timer | None = None

    # ─── Load / save ──────────────────────────────────────────────────────

    def load(self) -> None:
        """Read the config file into memory. Missing file → empty tree
        (not an error; first run). Corrupt file → empty tree + warning,
        so one bad save doesn't lock the user out."""
        try:
            raw = self._path.read_text()
        except FileNotFoundError:
            self._tree = {}
            return
        except OSError as exc:
            print(f'config read failed: {exc}')
            self._tree = {}
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f'config parse failed ({self._path}): {exc}; '
                  'starting from empty tree')
            self._tree = {}
            return
        if not isinstance(data, dict):
            print(f'config root is not an object; ignoring')
            self._tree = {}
            return
        self._tree = data

    def flush(self) -> None:
        """Write pending changes to disk synchronously. Cancels any
        pending debounced save and does the write inline."""
        with self._lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None
            self._write_now()

    def _write_now(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Snapshot under lock so a concurrent set() can't mutate the
            # tree mid-serialize.
            with self._lock:
                snapshot = json.dumps(self._tree, indent=2, sort_keys=True)
            tmp = self._path.with_suffix(self._path.suffix + '.tmp')
            tmp.write_text(snapshot + '\n')
            tmp.replace(self._path)
        except OSError as exc:
            print(f'config write failed: {exc}')

    def _schedule_save(self) -> None:
        if not self._autosave:
            return
        if self._save_timer is not None:
            self._save_timer.cancel()
        self._save_timer = threading.Timer(self._debounce_s, self._write_now)
        self._save_timer.daemon = True
        self._save_timer.start()

    # ─── Read / write ─────────────────────────────────────────────────────

    def get(self, path: str, default: Any = None) -> Any:
        with self._lock:
            return _deep_get(self._tree, _split(path), default)

    def set(self, path: str, value: Any) -> bool:
        """Set a single leaf. No-op + returns False if the value is
        unchanged (by ``==``). Returns True if the tree mutated and
        subscribers were notified."""
        parts = _split(path)
        if not parts:
            raise ValueError('cannot set the root of the config tree')
        with self._lock:
            old = _deep_get(self._tree, parts, _MISSING)
            if old is not _MISSING and old == value:
                return False
            _deep_set(self._tree, parts, value)
        self._bus.emit(self.CHANGE_KIND, (parts, old, value))
        self._schedule_save()
        return True

    def delete(self, path: str) -> bool:
        """Remove a leaf. Returns True if something was removed. Emits
        a change with ``new=_MISSING`` sentinel so handlers can
        distinguish delete from 'set to None'."""
        parts = _split(path)
        if not parts:
            raise ValueError('cannot delete the root of the config tree')
        with self._lock:
            old = _deep_get(self._tree, parts, _MISSING)
            if old is _MISSING:
                return False
            _deep_delete(self._tree, parts)
        self._bus.emit(self.CHANGE_KIND, (parts, old, _MISSING))
        self._schedule_save()
        return True

    def snapshot(self) -> dict:
        """Return a deep copy of the current tree. Useful for debug
        dumps and tests — production code should prefer ``get`` on
        specific paths."""
        import copy
        with self._lock:
            return copy.deepcopy(self._tree)

    # ─── Subscriptions ────────────────────────────────────────────────────

    def subscribe(self, prefix: str,
                  fn: Callable[[tuple, Any, Any], None]) -> Subscription:
        """Register ``fn(path, old, new)`` for any change under
        ``prefix`` (or all changes if prefix is empty). The handler
        receives the full dotted path parts — a subscriber to
        ``'plugins'`` seeing a flip of ``plugins.foo.enabled`` gets
        ``path=('plugins','foo','enabled')``."""
        prefix_parts = _split(prefix)

        def _gate(path, old, new):
            if _is_under(path, prefix_parts):
                fn(path, old, new)

        _gate.__wrapped__ = fn  # aid debugging
        return self._bus.on(self.CHANGE_KIND, _gate)

    def unsubscribe(self, sub: Subscription) -> bool:
        return self._bus.off(sub)


class _Missing:
    """Sentinel for 'path did not exist'. Exposed so handlers can
    distinguish delete from write-of-None."""

    _singleton = None

    def __new__(cls):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __repr__(self):
        return '<MISSING>'

    def __bool__(self):
        return False


_MISSING = _Missing()
MISSING = _MISSING  # public alias


# ─── Singleton ────────────────────────────────────────────────────────────

_singleton: ConfigStore | None = None


def _default_path() -> Path:
    return Path.home() / '.config' / 'vsrg-analysis' / 'config.json'


def get_config() -> ConfigStore:
    """Return the process-wide config store. Creates it and runs the
    legacy-file migration on first call."""
    global _singleton
    if _singleton is None:
        _singleton = ConfigStore(_default_path())
        _singleton.load()
        from analysis.config.migrate import migrate_legacy
        migrate_legacy(_singleton)
    return _singleton


def reset_for_tests() -> None:
    """Drop the singleton so the next ``get_config`` call builds fresh.
    Test-only helper; production code must not call this."""
    global _singleton
    if _singleton is not None:
        try:
            _singleton.flush()
        except Exception:
            pass
    _singleton = None
