"""Construction of the narrow ``LaunchableGame`` proxy plugins see on
``ctx.game``.

Lives outside the public ``components.api`` module so plugins can't
accidentally instantiate the host-side proxy class -- they only see the
:class:`~analysis.components.api.LaunchableGame` Protocol.
"""
from __future__ import annotations

from analysis.core.game import GameAdapter


class _AdapterProxy:
    """Backs ``ctx.game``. Forwards a fixed allow-list of methods to
    the underlying adapter; everything else is invisible.

    The proxy doesn't reach into adapter internals -- only methods
    declared on :class:`~analysis.components.api.LaunchableGame` are
    forwarded. Adapters that don't implement an optional method (e.g.
    Etterna has no ``launch``) propagate the ``NotImplementedError``
    to the caller, which is the documented contract.
    """

    __slots__ = ('_adapter',)

    def __init__(self, adapter: GameAdapter):
        self._adapter = adapter

    @property
    def name(self) -> str:
        return self._adapter.name

    def launch(self, *, with_overlay: bool = True):
        return self._adapter.launch(with_overlay=with_overlay)


def proxy_for(game_name: str | None):
    """Return a proxy for the named adapter, or ``None`` if no adapter
    is registered under that name (e.g. surface has no active game)."""
    if not game_name:
        return None
    from analysis.core.game import all_games
    adapter = all_games().get(game_name)
    if adapter is None:
        return None
    return _AdapterProxy(adapter)
