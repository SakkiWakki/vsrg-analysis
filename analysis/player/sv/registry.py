"""SV engine registry: per-chart catalog of available integration engines.

Built once per chart load, holds named lazy factories for each engine type
the chart can be played under. The chart's "native" engine is the one its
source game uses; other engines may be swappable depending on whether the
chart's timing data can be losslessly translated to that engine's measure
(see DESIGN.tex §4: time-space charts embed into beat-space trivially;
beat-space charts with warps cannot be exactly represented in time-space).

The registry is intentionally dumb about scroll modes (CMOD/MMOD/XMOD).
Those are a separate axis -- they live in `analysis.player.scroll.registry`
and control how px/sec is computed from the active SV engine's output.
The SV registry only controls which integrator computes the cumulative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


# Engine key constants used as registry keys and in UI.
KEY_ETTERNA_BEAT = 'etterna_beat'      # BeatSpaceSVEngine semantics
KEY_OSU_TIME = 'osu_time'              # TimeSpaceSVEngine semantics
KEY_IDENTITY = 'identity'              # No-op


# Display labels for each engine.
ENGINE_LABELS = {
    KEY_ETTERNA_BEAT: 'etterna (beat)',
    KEY_OSU_TIME: 'osu (time)',
    KEY_IDENTITY: 'identity',
}


# Primary game associated with each engine (used to coordinate scroll-mode
# selection when the engine is cycled). Identity has no associated game.
ENGINE_PRIMARY_GAME = {
    KEY_ETTERNA_BEAT: 'etterna',
    KEY_OSU_TIME: 'osu',
}


@dataclass
class _EngineSlot:
    """One entry in the registry."""
    key: str
    label: str
    factory: Callable[[], object]      # builds the engine on first access
    cached: Optional[object] = None    # populated lazily on first activate


class SVEngineRegistry:
    """Per-chart engine catalog.

    Constructed by SvRenderController at chart-load time. Each slot is a
    factory that lazily produces an engine; the active slot is the one
    currently driving the renderer. swap_engine() rebuilds the active
    cache through the controller -- the registry itself just remembers
    which key is active.
    """

    def __init__(self):
        self._slots: dict[str, _EngineSlot] = {}
        self._native: str | None = None
        self._active: str | None = None

    def register(self, key: str, label: str, factory: Callable[[], object],
                 *, native: bool = False, eager: bool = False) -> None:
        """Register an engine slot.

        `eager=True` instantiates immediately (used for the native engine,
        since it's always live at chart load). Otherwise the factory runs
        the first time the slot is activated.
        """
        slot = _EngineSlot(key=key, label=label, factory=factory)
        if eager:
            slot.cached = factory()
        self._slots[key] = slot
        if native:
            self._native = key
            if self._active is None:
                self._active = key

    def keys(self) -> list[str]:
        return list(self._slots.keys())

    def native_key(self) -> str | None:
        return self._native

    def active_key(self) -> str | None:
        return self._active

    def label(self, key: str) -> str:
        slot = self._slots.get(key)
        return slot.label if slot else key

    def get(self, key: str) -> object:
        """Return the engine for `key`, instantiating on first use."""
        slot = self._slots[key]
        if slot.cached is None:
            slot.cached = slot.factory()
        return slot.cached

    def active(self) -> object | None:
        if self._active is None:
            return None
        return self.get(self._active)

    def set_active(self, key: str) -> object:
        """Mark `key` active and return its engine. Caller is responsible
        for invalidating any caches that derived from the previous engine."""
        if key not in self._slots:
            raise KeyError(f'unknown engine key: {key!r}')
        self._active = key
        return self.get(key)

    def next_key(self) -> str | None:
        """Return the key that follows the active one in registration order
        (wrap-around). Does NOT mutate -- caller is responsible for the
        actual swap so cache invalidation can run alongside."""
        keys = self.keys()
        if not keys:
            return None
        cur = self._active or keys[0]
        try:
            idx = keys.index(cur)
        except ValueError:
            idx = -1
        return keys[(idx + 1) % len(keys)]
