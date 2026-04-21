"""Narrow host API exposed to sandboxed plugins.

Sandboxed plugins cannot directly access ``player``, ``renderer``, or any
Qt/matplotlib objects. Instead they receive a ``PlayerState`` view through
``SidebarContext.player_state`` (v1) for read-only observation, and can
register UI via ``SidebarContext`` as before.

Persistence: plugins obtain a scoped config handle via
:func:`plugin_config`, which reads/writes/subscribes under
``plugins.<plugin_key>.settings`` in the shared config store. This is
the only way a sandboxed plugin gets durable per-user state — the
sandbox blocks direct filesystem access.

Future work: expose read-only chart/replay snapshots, a scoped FS API
("load replay by id"), etc., without giving plugins arbitrary path access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


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
