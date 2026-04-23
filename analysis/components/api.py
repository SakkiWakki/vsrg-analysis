"""Unified plugin-component API: one draw function, many surfaces.

A component is a plugin-supplied ``(manifest, draw)`` pair. The manifest
declares *where* the component is allowed to live (``sidebar``,
``overlay``, future surfaces) and what live data it needs. The draw
function is called once per frame on every surface that agreed to host
it — each surface's backend translates the component's calls into its
native primitives (QPainter for the sidebar, shared-memory widgets for
the gamescope overlay, and so on).

Guiding principle: the component never imports Qt, mmap, the player, or
the overlay publisher. It only touches ``ComponentContext``. This keeps
the port surface small and the sandbox story honest.

Shape:

    ComponentManifest
        key, name, supported_surfaces, data requirements, default layout

    ComponentContext   (protocol, see below)
        geometry (w, h, cursor y), primitives (text, rect, line, button,
        heading, ...), access to a ``ComponentGameState`` that each
        backend populates from the live source of truth (the ``Player``
        in-app, the ``OverlayGameState`` in-game).

    ComponentGameState (protocol)
        Methods a component may call to read live data. A backend
        implementation wraps its world (``Player`` / ``OverlayGameState``)
        and answers the subset it can. Unknown methods raise
        :class:`DataNotAvailable`.

The manifest's ``requires_data`` set is checked at registration time so
a mis-targeted component ("sidebar, overlay" but needs a data field
only overlay provides) is rejected on that surface rather than crashing
on first frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


# ── Surfaces ────────────────────────────────────────────────────────

# Surface identifiers are strings, open-set on purpose: a future
# "windows-directdraw overlay" backend registers its own name without
# editing this file. The two surfaces we ship today:
SURFACE_GUI = 'gui'        # in-app Qt surface (sidebar + any future GUI widgets)
SURFACE_OVERLAY = 'overlay'  # in-game overlay surface (gamescope, gl_layer, etc.)

# ── Regions ─────────────────────────────────────────────────────────

# REGION_FREE is universal: a component not docked in any surface panel,
# floating freely on the surface instead. Surface plugins define their
# own panel region names (e.g. REGION_PANEL in plugins/builtin/sidepanel).
REGION_FREE = 'free'


# ── Data access ─────────────────────────────────────────────────────

class DataNotAvailable(Exception):
    """Raised by a :class:`ComponentGameState` when a field the caller
    asked for isn't provided by the current surface. Components that
    want to degrade gracefully should catch this; manifests that list
    the field in ``requires_data`` never observe it (registration would
    have refused the surface)."""


@runtime_checkable
class ComponentGameState(Protocol):
    """Read-only view over whatever the surface uses as its source of
    truth. Methods grow over time as components need more. Each backend
    implements the subset that makes sense for its world; calling an
    unsupported method raises :class:`DataNotAvailable`.

    Methods are nullary: if a component wants derived values, derive them
    in the component. This keeps the contract flat and the two backends
    independent of each other's quirks.
    """

    def supports(self, field: str) -> bool:
        """Return True iff ``field`` is answerable on this source. Used
        by the component registry to gate surface mounting."""
        ...

    # ── Game identity / shape ──
    def game(self) -> str: ...
    def keycount(self) -> int: ...

    # ── Scoring state ──
    def combo(self) -> int: ...
    def accuracy(self) -> float: ...

    # ── Judgments ──
    # Windows: ordered list of (name, half-width-seconds) for the
    # component to display. Overlay-side game state may not know the
    # window widths at all (osu! only gives counts), in which case the
    # source raises DataNotAvailable and the component falls back to
    # "counts only" rendering.
    def judgment_windows(self) -> list[tuple[str, float]]: ...
    def judgment_counts(self) -> dict[str, int]: ...
    def judgment_colors(self) -> dict[str, tuple]: ...
    def judge_label(self) -> str: ...

    # ── Raw game memory (osu! only for now) ──
    # Returns None when the native reader is unavailable or the player
    # is not in an active gameplay session. Components should degrade
    # gracefully rather than requiring this field via requires_data.
    def game_memory(self) -> 'GameMemoryState | None': ...


# ── Game memory snapshot ───────────────────────────────────────────

@dataclass(frozen=True)
class GameMemoryState:
    """Read-only snapshot of live game memory, delivered via
    ComponentGameState.game_memory(). Fields reflect what the native
    reader can extract cross-platform; platform-specific internals stay
    in the trusted host layer.

    All fields are as-read at poll time. None values indicate the field
    was not populated (e.g. hit_errors_ms when not in gameplay)."""
    in_gameplay: bool
    combo: int
    max_combo: int
    accuracy: float                    # 0.0 - 1.0
    hit_300: int
    hit_100: int
    hit_50: int
    hit_miss: int
    hit_geki: int
    hit_katu: int
    hit_errors_ms: tuple[int, ...]     # raw per-hit offset errors in ms
    map_md5: str
    map_title: str


# ── Drawing primitives ─────────────────────────────────────────────

# All coordinates passed to :class:`ComponentContext` are *component-local
# pixels* — the component thinks of itself as painting into its own
# ``(0, 0, ctx.w, ctx.h)`` box. Each backend translates to its native
# coord system: sidebar adds the column offset; overlay normalises to
# [0, 1] of the framebuffer.


@runtime_checkable
class ComponentContext(Protocol):
    """Per-frame context handed to a component's draw callable.

    The plugin calls geometry helpers + primitives; the backend decides
    what painting actually means. ``measure_only=True`` lets the sidebar
    pre-measure a component (for pinned-bottom layout) without touching
    the painter — overlay backends ignore the flag.
    """

    surface: str                  # one of SURFACE_*
    region: str                   # current region within the surface (surface-defined)
    w: int                        # component-local width, px
    h: int                        # component-local height, px (0 == grow)
    y: int                        # paint cursor, advances as rows emit
    measure_only: bool
    data: ComponentGameState

    # ── Geometry helpers ──
    def split_row(self, n: int = 2, gap: int = 4) -> list[tuple[int, int]]:
        """``n`` equal-width (x, w) slots across the component's content
        column, with ``gap`` pixels between them."""
        ...

    # ── Raw primitives (coords are local px, origin top-left) ──
    def text(self, s: str, x: int, baseline: int,
             color: tuple = None) -> None: ...
    def rect(self, rect: tuple, color: tuple = None,
             outline: tuple = None, outline_w: int = 1) -> None: ...
    def line(self, start: tuple, end: tuple, color: tuple,
             width: int = 1) -> None: ...

    # ── Cursor-advancing rows ──
    def spacer(self, h: int = None) -> None: ...
    def draw_heading(self, text: str, color: tuple = None) -> None: ...
    def draw_text(self, text: str, color: tuple = None, indent: int = 0,
                  height: int = None) -> None: ...
    def draw_hint(self, text: str, color: tuple = None) -> None: ...

    # ── Interactive primitives. ──
    def draw_button(self, label: str, action: str, payload=None, *,
                    enabled: bool = True, height: int = None,
                    center: bool = False) -> tuple: ...
    def button_at(self, rect: tuple, label: str, action: str, payload=None,
                  *, enabled: bool = True, center: bool = False) -> None: ...
    def checkbox(self, x: int, y: int, checked: bool) -> tuple: ...

    # ── Capability probe (for components that want to branch) ──
    @property
    def supports_input(self) -> bool:
        """True when a backend routes user clicks back to ``action``
        names. False for display-only surfaces like the current
        gamescope overlay."""
        ...


# ── Manifest ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ComponentManifest:
    """Declarative spec for a component. One per plugin; lists every
    surface the component is allowed on and what data it needs. The
    registry refuses to mount a component on a surface whose data source
    doesn't cover ``requires_data`` — prevents first-frame crashes from
    mis-targeted components.

    Surface-specific layout hints live in ``plugin_fields`` under the
    surface's name. Each surface backend reads its own entry and ignores
    the rest; absent entries fall back to the backend's defaults.
    """

    key: str
    name: str
    supported_surfaces: frozenset[str]
    requires_data: frozenset[str] = field(default_factory=frozenset)
    optional_data: frozenset[str] = field(default_factory=frozenset)
    plugin_fields: dict[str, Any] = field(default_factory=dict)
    module: str = ''

    def __post_init__(self):
        object.__setattr__(self, 'supported_surfaces',
                           frozenset(self.supported_surfaces))
        object.__setattr__(self, 'requires_data',
                           frozenset(self.requires_data))
        object.__setattr__(self, 'optional_data',
                           frozenset(self.optional_data))


# The draw callable's type alias. ``None`` return — the backend owns
# cursor flushing and hitbox commit.
DrawFn = Callable[[ComponentContext], None]


@dataclass(frozen=True)
class Component:
    """Registered pair of (manifest, draw). Produced by registration
    helpers; consumed by each surface backend."""
    manifest: ComponentManifest
    draw: DrawFn
