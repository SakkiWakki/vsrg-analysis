"""Narrow host API exposed to sandboxed plugins.

Sandboxed plugins cannot directly access ``player``, ``renderer``, or any
Qt/matplotlib objects. Instead they receive a ``PlayerState`` view through
``SidebarContext.player_state`` (v1) for read-only observation, and can
register UI via ``SidebarContext`` as before.

Persistence: plugins obtain a scoped config handle via
:func:`plugin_config`, which reads/writes/subscribes under
``plugins.<plugin_key>.settings`` in the shared config store. This is
the only way a sandboxed plugin gets durable per-user state -- the
sandbox blocks direct filesystem access.

Network: plugins may call :func:`http_get` to fetch a URL. Every URL
requires user consent; the host shows a dialog on first access and
persists the decision (always/never) per plugin+URL. Call only from a
plugin's own thread -- this call blocks until the user responds or the
request completes.

Timing: sandboxed plugins can call :func:`monotonic_seconds` for
frame-to-frame timing diagnostics without importing blocked stdlib time
modules directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import time



def monotonic_seconds() -> float:
    """Monotonic wall-clock seconds from the host process.

    Sandboxed plugins should use this helper for elapsed-time measurements
    instead of importing ``time`` directly.
    """
    return float(time.perf_counter())


# ── Network permission dialog hook ────────────────────────────────────────
#
# The Qt host sets this at startup. Signature:
#   _show_permission_dialog(plugin_key: str, url: str) -> str
# Returns one of: 'always', 'allow_once', 'deny_once', 'never'.
# Called on Qt's main thread (host marshals it there). The plugin thread
# waits on a threading.Event while this runs.
_show_permission_dialog: Callable[[str, str], str] | None = None


def set_permission_dialog(fn: Callable[[str, str], str]) -> None:
    """Called by the Qt host at startup to register the dialog callback."""
    global _show_permission_dialog
    _show_permission_dialog = fn


class NetworkAccessDenied(Exception):
    """Raised by :func:`http_get` when the user denied network access."""


def http_get(plugin_key: str, url: str, *, timeout: float = 5.0) -> bytes:
    """Fetch ``url`` and return the raw response body.

    Blocks the calling thread until the user grants or denies access (on
    first call for this plugin+URL combination), then performs the HTTP
    request. Must be called from a plugin's own thread, not the Qt main
    thread.

    Raises :class:`NetworkAccessDenied` if the user denies access.
    Raises :class:`urllib.error.URLError` on network errors.

    Examples:
        Permanent decisions (always/never) are stored and skip the dialog
        on future calls. One-time decisions (allow_once/deny_once) apply
        only to this call.
    """
    from analysis.plugins.permissions import Decision, stored, record

    decision = stored(plugin_key, url)

    if decision == Decision.NEVER:
        raise NetworkAccessDenied(f'{plugin_key} access to {url!r} is set to never')

    if decision != Decision.ALWAYS:
        result = _ask_permission(plugin_key, url)
        if result == 'always':
            record(plugin_key, url, Decision.ALWAYS)
        elif result == 'never':
            record(plugin_key, url, Decision.NEVER)
            raise NetworkAccessDenied(f'{plugin_key} denied access to {url!r}')
        elif result == 'deny_once':
            raise NetworkAccessDenied(f'{plugin_key} denied access to {url!r}')
        # 'allow_once' falls through

    import urllib.request
    req = urllib.request.Request(url, headers={'Accept': '*/*'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _ask_permission(plugin_key: str, url: str) -> str:
    """Block the calling thread until the Qt dialog resolves.

    Marshals the dialog to Qt's main thread via the registered callback,
    using a threading.Event so this thread waits without spinning."""
    if _show_permission_dialog is None:
        # No Qt host registered (tests, headless). Deny by default.
        return 'deny_once'

    import threading
    result_holder: list[str] = []
    done = threading.Event()

    def _on_main_thread():
        try:
            result_holder.append(_show_permission_dialog(plugin_key, url))
        except Exception:
            result_holder.append('deny_once')
        finally:
            done.set()

    # Qt host must call _on_main_thread() on the Qt main thread.
    # This is done via QMetaObject.invokeMethod or a queued signal
    # registered by set_permission_dialog's companion set_invoke_on_main.
    _invoke_on_main(_on_main_thread)
    done.wait()
    return result_holder[0] if result_holder else 'deny_once'


_invoke_on_main: Callable[[Callable], None] = lambda fn: fn()


def set_invoke_on_main(fn: Callable[[Callable], None]) -> None:
    """Called by the Qt host to register how to marshal a callable to the
    Qt main thread (e.g. via QMetaObject.invokeMethod + Qt.QueuedConnection).

    Examples:
        from PySide6.QtCore import QMetaObject, Qt
        set_invoke_on_main(
            lambda fn: QMetaObject.invokeMethod(app, fn, Qt.QueuedConnection))
    """
    global _invoke_on_main
    _invoke_on_main = fn


@dataclass(frozen=True)
class PlayerState:
    """Read-only snapshot of the current player state.

    Fields are deliberately minimal for v1 — add more as concrete plugin
    needs appear. All arrays are provided as tuples so plugins can't
    mutate the live game state.
    """
    t: float                # current playback time (s)
    play_rate: float
    paused: bool
    keycount: int
    note_count: int
    judge_counts: dict[str, int]  # {judgment_name: n}
    windows: tuple[tuple[str, float], ...]  # ((name, width_s), ...)

    @classmethod
    def from_player(cls, p, t_now: float):
        counts = {n: 0 for n, _ in p.windows}
        counts['miss'] = 0
        for j in p.note_judges:
            counts[j] = counts.get(j, 0) + 1
        return cls(
            t=float(t_now),
            play_rate=float(p.play_rate),
            paused=bool(p.paused),
            keycount=int(p.keycount),
            note_count=int(len(p.times)),
            judge_counts=dict(counts),
            windows=tuple((n, float(w)) for n, w in p.windows),
        )


def _escape_key(key: str) -> str:
    """Plugin keys can contain dots (rare but legal — bundle authors
    pick them). Dotted path parts are the store's separator, so rewrite
    any dots in a key to underscores. Matches ``_escape_key`` in
    :mod:`analysis.player.hud.sidebar_api` so all layers line up."""
    return key.replace('.', '_')


class PluginConfig:
    """Scoped config handle for one plugin.

    All reads, writes, and subscriptions are rooted at
    ``plugins.<escaped_key>.settings.<field...>`` in the shared config
    store. A plugin registered under ``'mybundle:hello'`` sees its own
    settings and cannot reach other plugins' subtrees or the top-level
    ``paths.*`` / ``player.*`` state.

    Writes fan out through the store's subscription graph, so another
    window's instance of the same plugin — or the plugin's own config
    UI — picks up the change on the next frame. No polling, no reload.

    This is a convenience boundary for sandboxed plugins; it isn't
    load-bearing security. Trusted plugins could bypass it by touching
    the store directly, but shouldn't.
    """

    def __init__(self, plugin_key: str, config=None):
        if not plugin_key:
            raise ValueError('plugin_key is required')
        from analysis.config import get_config
        self._store = config if config is not None else get_config()
        self._plugin_key = plugin_key
        self._root = f'plugins.{_escape_key(plugin_key)}.settings'
        self._root_parts = tuple(self._root.split('.'))

    def _path(self, field: str) -> str:
        if not field:
            return self._root
        return f'{self._root}.{field}'

    def get(self, field: str, default: Any = None) -> Any:
        """Read a setting. Missing fields return ``default``."""
        return self._store.get(self._path(field), default)

    def set(self, field: str, value: Any) -> bool:
        """Write a setting. Returns True if the value changed."""
        if not field:
            raise ValueError('field is required')
        return self._store.set(self._path(field), value)

    def delete(self, field: str) -> bool:
        if not field:
            raise ValueError('field is required')
        return self._store.delete(self._path(field))

    def snapshot(self) -> dict:
        """Shallow copy of this plugin's full settings dict. Useful for
        dumping to JSON or comparing two states. Deeper values are
        shared references — treat as read-only."""
        current = self._store.get(self._root, {}) or {}
        return dict(current) if isinstance(current, dict) else {}

    def subscribe(self, fn: Callable[[str, Any, Any], None]):
        """Listen for changes to this plugin's settings.

        ``fn(field, old, new)`` — ``field`` is the dotted path relative
        to this plugin's root (so ``set('volume', 0.5)`` fires with
        ``field='volume'``). Returns an opaque handle; pass to
        :meth:`unsubscribe` to stop listening.
        """
        root_parts = self._root_parts

        def _gate(path, old, new):
            if (len(path) < len(root_parts)
                    or path[:len(root_parts)] != root_parts):
                return
            field = '.'.join(path[len(root_parts):])
            fn(field, old, new)

        return self._store.subscribe('plugins', _gate)

    def unsubscribe(self, handle) -> bool:
        return self._store.unsubscribe(handle)


def plugin_config(plugin_key: str) -> PluginConfig:
    """Factory for :class:`PluginConfig`. Call from inside a plugin's
    ``register`` / ``register_sidebar`` (or lazily on first draw); the
    returned handle is safe to cache for the process lifetime."""
    return PluginConfig(plugin_key)
