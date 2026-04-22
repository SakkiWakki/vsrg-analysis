"""Core PAL types: ``OverlayPlatform`` protocol + neutral ``OverlayFrame``
command list.

The frame is the *portable* contract between the overlay backend and the
platform. It is a flat list of ``(kind, args)`` records — intentionally
dumb — so a future Windows or macOS platform can interpret it however
that OS's overlay API wants. The current Linux/Gamescope platform
serialises these records into the shared-memory widget format that
``osu_overlay.c`` already understands.

Capabilities are advertised separately so the overlay component backend
can refuse to render click-through buttons on platforms that don't yet
route input (the gamescope path today).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ── Command records ─────────────────────────────────────────────────

# Each record is ``(kind, payload)``. Keeping payloads as tuples (rather
# than full dataclasses) makes the whole frame cheap to build and trivial
# to serialise on the platform side.

CMD_RECT = 'rect'
CMD_TEXT = 'text'
CMD_GROUP_BEGIN = 'group_begin'
CMD_GROUP_END = 'group_end'


@dataclass
class OverlayFrame:
    """Neutral per-frame command list handed from the overlay component
    backend to the active :class:`OverlayPlatform`.

    All coordinates and sizes are *normalized* ``[0, 1]`` relative to the
    overlay target resolution. The overlay component backend converts
    from the component's local pixel coords on its way in, so platforms
    don't need to know the component size.
    """

    # Target resolution the frame was laid out against — used by the
    # platform to decide whether to remap on a resolution change.
    width: int = 1920
    height: int = 1080
    records: list[tuple[str, tuple]] = field(default_factory=list)

    def rect(self, id_str: str, x: float, y: float, w: float, h: float,
             *, color: int, anchor: int = 0) -> None:
        self.records.append((CMD_RECT, (id_str, x, y, w, h, color, anchor)))

    def text(self, id_str: str, s: str, x: float, y: float,
             *, px_scale: float = 2.0, color: int, anchor: int = 0) -> None:
        self.records.append(
            (CMD_TEXT, (id_str, s, x, y, px_scale, color, anchor)))

    def begin_group(self, name: str) -> None:
        self.records.append((CMD_GROUP_BEGIN, (name,)))

    def end_group(self) -> None:
        self.records.append((CMD_GROUP_END, ()))


# ── Handles + capabilities ─────────────────────────────────────────

@dataclass
class OverlayHandle:
    """Opaque-to-callers handle returned from :meth:`OverlayPlatform.setup`.

    Platforms may subclass to carry implementation state (shm fd, X
    window id, DX swap chain, …). Callers only pass it back to
    ``submit_frame`` and ``teardown``.
    """
    key: str
    width: int
    height: int
    # Platform-specific extras live here — typed as Any so the base
    # package doesn't leak Linux/X11 types.
    impl: Any = None


@dataclass(frozen=True)
class OverlayPlatformCapabilities:
    """What a platform can and can't do. The overlay component backend
    reads these to decide whether to emit interactive chrome, a drag
    handshake, etc.

    Growth policy: add flags as surfaces learn tricks; never rename.
    Components and backends branch on these, so renames ripple."""
    supports_input: bool = False         # click-through buttons
    supports_drag_edit: bool = False     # user-driven widget repositioning
    native_resolution: tuple = (1920, 1080)


# ── The platform contract ───────────────────────────────────────────

@runtime_checkable
class OverlayPlatform(Protocol):
    """One-overlay-per-game hosting contract.

    Lifecycle: ``setup(key) → handle``; zero-or-more ``submit_frame``
    calls; ``teardown(handle)``. ``is_available`` lets callers probe
    without running setup side-effects.
    """

    def is_available(self) -> bool: ...
    def capabilities(self) -> OverlayPlatformCapabilities: ...

    def setup(self, key: str, *, width: int, height: int) -> OverlayHandle: ...
    def submit_frame(self, handle: OverlayHandle,
                     frame: OverlayFrame) -> None: ...
    def teardown(self, handle: OverlayHandle) -> None: ...
