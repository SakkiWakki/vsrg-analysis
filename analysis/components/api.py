"""Unified plugin-component API: one draw function, many surfaces.

A component is a plugin-supplied ``(manifest, draw)`` pair. The manifest
declares *where* the component is allowed to live (``sidebar``,
``overlay``, future surfaces) and what live data it needs. The draw
function is called once per frame on every surface that agreed to host
it — each surface's backend translates the component's calls into its
native primitives (QPainter for the sidebar, shared-memory widgets for
the gamescope overlay, and so on).

Guiding principle: the component never imports Qt, mmap, the player, or
the overlay publisher. It only touches ``Context``. This keeps
the port surface small and the sandbox story honest.

Shape:

    Manifest
        key, name, supported_surfaces, data requirements, default layout

    Context   (protocol, see below)
        geometry (w, h, cursor y), primitives (text, rect, line, button,
        heading, ...), access to a ``GameState`` that each
        backend populates from the live source of truth (the ``Player``
        in-app, the ``OverlayGameState`` in-game).

    GameState (protocol)
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
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


# ── Surfaces ────────────────────────────────────────────────────────

# Surface identifiers are strings, open-set on purpose: a future
# "windows-directdraw overlay" backend registers its own name without
# editing this file. The two surfaces we ship today:
SURFACE_GUI = 'gui'        # in-app Qt surface (sidebar + any future GUI widgets)
SURFACE_OVERLAY = 'overlay'  # in-game overlay surface (gamescope, gl_layer, etc.)
SURFACE_VIZ = 'viz'        # visualization host surface (tab, window, etc.)

# ── Regions ─────────────────────────────────────────────────────────

# REGION_FREE is universal: a component not docked in any surface panel,
# floating freely on the surface instead. Surface plugins define their
# own panel region names (e.g. REGION_PANEL in plugins/builtin/sidepanel).
REGION_FREE = 'free'


# -- Layers ---------------------------------------------------------

LAYER_GROUP = 'group'
LAYER_LEAF = 'leaf'
LAYER_BEFORE = 'before'
LAYER_AFTER = 'after'
LAYER_INSIDE = 'inside'


@dataclass(frozen=True)
class LayerPlacement:
    relation: str
    target: str

    def __post_init__(self):
        object.__setattr__(self, 'relation', str(self.relation))
        object.__setattr__(self, 'target', str(self.target))


@dataclass(frozen=True)
class LayerDeclaration:
    key: str
    name: str
    placement: LayerPlacement
    kind: str = LAYER_LEAF
    default_visible: bool = True
    can_hide: bool = True
    listed: bool = True
    accepts_children: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        object.__setattr__(self, 'key', str(self.key))
        object.__setattr__(self, 'name', str(self.name))
        object.__setattr__(self, 'kind', str(self.kind))
        object.__setattr__(self, 'accepts_children',
                           frozenset(self.accepts_children))


@dataclass(frozen=True)
class LayerState:
    key: str
    name: str
    kind: str
    owner: str
    parent: str | None
    depth: int
    local_visible: bool
    visible: bool
    can_hide: bool
    listed: bool
    children: tuple['LayerState', ...] = field(default_factory=tuple)


# ── Data access ─────────────────────────────────────────────────────

class DataNotAvailable(Exception):
    """Raised by a :class:`GameState` when a field the caller
    asked for isn't provided by the current surface. Components that
    want to degrade gracefully should catch this; manifests that list
    the field in ``requires_data`` never observe it (registration would
    have refused the surface)."""


@runtime_checkable
class GameState(Protocol):
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

    # ── Chart snapshots ──
    # All three return a dataclass populated with whatever the current
    # source knows; unknown string fields are '', unknown numeric fields
    # are 0. Components needing partial info should read specific fields.
    def chart_metadata(self) -> 'ChartMetadata': ...
    def chart_stats(self) -> 'ChartStats': ...
    def chart_paths(self) -> 'ChartPaths': ...

    # ── Play identity / state ──
    def player_name(self) -> str: ...
    def score(self) -> int: ...
    def max_combo(self) -> int: ...
    def current_grade(self) -> str: ...
    def mods_short(self) -> str:
        """Game-agnostic short mod string (e.g. 'HDDT' for osu,
        'MX1.2' for etterna rate 1.2, '' for none). Suitable for
        display; machine-readable variants are game-specific."""
        ...
    def mods_raw(self) -> dict:
        """Game-defined raw mod representation. Keys are per-game:
        osu uses {'bitfield': int, 'rate': float}; etterna uses
        {'rate': float, 'flags': list[str]}. Consumers that read this
        MUST branch on ``game()`` to interpret."""
        ...
    def play_rate_effective(self) -> float:
        """Effective time scale incl. rate mods (DT=1.5, HT=0.75 for
        osu; XML rate for etterna). 1.0 = nominal."""
        ...
    def hit_errors_ms(self) -> tuple[int, ...]:
        """Per-hit signed offsets in ms (misses excluded). Replay surface
        returns the full array; live-memory surface returns what the
        native reader has in its buffer (usually last N)."""
        ...
    def unstable_rate(self) -> float: ...

    # ── Playback state (GUI surface only) ──
    # All raise DataNotAvailable on the overlay surface.
    def t_now(self) -> float: ...
    def play_rate(self) -> float: ...
    def paused(self) -> bool: ...
    def note_count(self) -> int: ...
    def sv_enabled(self) -> bool: ...
    def sv_suspended(self) -> bool: ...
    def sv_sections(self) -> list: ...
    def skin(self) -> str: ...
    def press_hide(self) -> bool: ...
    def scroll_mode(self) -> str: ...
    def scroll_value(self) -> float: ...
    def effective_scroll_ms(self) -> float: ...
    def layer_visible(self, layer: str) -> bool: ...
    def layer_tree(self) -> tuple[LayerState, ...]: ...


# ── Game memory snapshot ───────────────────────────────────────────

@dataclass(frozen=True)
class GameMemoryState:
    """Read-only snapshot of live game memory, delivered via
    GameState.game_memory(). Cross-game fields only; judgment counts
    and any game-specific extras are stashed in ``judgment_counts`` and
    ``extra`` so this struct stays neutral.

    All fields are as-read at poll time. Empty tuple / {} indicate the
    field was not populated (e.g. hit_errors_ms when not in gameplay)."""
    in_gameplay: bool
    combo: int
    max_combo: int
    accuracy: float                    # 0.0 - 1.0
    judgment_counts: dict              # {judge_name: count}, game-defined keys
    hit_errors_ms: tuple[int, ...]     # raw per-hit offset errors in ms
    map_md5: str
    map_title: str
    # Game-specific extras (osu: 'grade', etterna: 'rate_str', etc.).
    # Consumers must branch on the surrounding GameState.game() key.
    extra: dict = field(default_factory=dict)


# ── Chart metadata snapshot ─────────────────────────────────────────

@dataclass(frozen=True)
class ChartMetadata:
    """Read-only snapshot of chart-identifying metadata, delivered via
    GameState.chart_metadata(). All string fields default to '' and
    integer IDs to 0 when unknown. Fields here are stable across the
    play and may be computed once at chart load time."""
    artist: str = ''
    artist_unicode: str = ''
    title: str = ''
    title_unicode: str = ''
    creator: str = ''       # mapper name (.osu "Creator")
    version: str = ''       # difficulty name (.osu "Version")
    md5: str = ''           # beatmap file hash
    beatmap_id: int = 0
    beatmap_set_id: int = 0
    source: str = ''
    tags: str = ''


# ── Chart stats snapshot ────────────────────────────────────────────

@dataclass(frozen=True)
class ChartStats:
    """Read-only snapshot of chart difficulty stats, delivered via
    GameState.chart_stats(). Cross-game fields only; anything specific
    to a single game (osu AR/CS/HP, etterna MSD breakdown, etc.) goes
    into ``extra`` keyed by a game-defined string.

    Mode name is the parent-game label ('osu' / 'etterna' / ...); rely
    on GameState.keycount() for the column count when relevant.
    """
    mode_name: str = ''
    difficulty: float = 0.0   # neutral difficulty scalar (OD for osu,
                              # MSD for etterna, 0 if none)
    rating: float = 0.0       # computed rating (stars for osu, 0 if n/a)
    bpm_common: float = 0.0
    bpm_min: float = 0.0
    bpm_max: float = 0.0
    length_ms: int = 0        # chart length from first to last note
    first_object_ms: int = 0
    last_object_ms: int = 0
    total_objects: int = 0
    hold_count: int = 0
    max_combo: int = 0        # theoretical chart max combo
    # Game-specific extras (osu: {'ar', 'cs', 'hp', 'stars'};
    # etterna: {'msd_stream', 'msd_jumpstream', ...}).
    extra: dict = field(default_factory=dict)


# ── Chart paths snapshot ────────────────────────────────────────────

@dataclass(frozen=True)
class ChartPaths:
    """Filesystem locations for chart assets. Relative paths are
    interpreted against the library root; empty strings mean unknown.
    Delivered via GameState.chart_paths(); may be called from any
    surface that has a loaded chart."""
    chart_folder: str = ''         # chart's containing directory name
    audio_filename: str = ''       # audio file within chart folder
    background_filename: str = ''
    skin_folder: str = ''          # active skin folder name (osu),
                                   # noteskin name (etterna), etc.
    library_root: str = ''         # game's chart library root
                                   # (osu! Songs dir, etterna Songs dir)


# ── HUD flags snapshot ────────────────────────────────────────────

@dataclass(frozen=True)
class HudFlags:
    """Read-only snapshot of the HUD overlay state for this frame.
    Delivered via ctx.hud_flags. Fields relevant to components that
    need to react to edit mode or panel open/close state."""
    edit_mode: bool
    layers_panel_open: bool
    plugin_panel_open: bool
    open_flyout: str | None


# ── Scoped config access ───────────────────────────────────────────

@runtime_checkable
class Config(Protocol):
    """Scoped, per-component config handle. Reads and writes are
    restricted to this component's own namespace in the shared config
    store. A component cannot access another component's settings."""

    def get(self, field: str, default: Any = None) -> Any: ...
    def set(self, field: str, value: Any) -> bool: ...
    def delete(self, field: str) -> bool: ...
    def subscribe(self, fn) -> Any: ...
    def unsubscribe(self, handle) -> bool: ...


# ── Replay state ──────────────────────────────────────────────────

@runtime_checkable
class ReplayState(Protocol):
    """Read-only view over the post-analysis replay data for the current
    session. Available on surfaces that host a live Player (GUI sidebar).
    Raises DataNotAvailable on surfaces without replay context (overlay).

    `_clean` variants have misses filtered out -- use these for timing
    analysis. Raw variants include misses so the renderer can draw miss
    markers.
    """

    def offsets(self) -> 'np.ndarray': ...
    def offsets_clean(self) -> 'np.ndarray': ...
    def columns(self) -> 'np.ndarray': ...
    def columns_clean(self) -> 'np.ndarray': ...
    def noterows(self) -> 'np.ndarray': ...
    def noterows_clean(self) -> 'np.ndarray': ...
    def misses(self) -> 'np.ndarray': ...
    def notetypes(self) -> 'np.ndarray': ...
    def keycount(self) -> int: ...
    def game(self) -> str: ...


# ── Data analysis utilities ────────────────────────────────────────

@runtime_checkable
class DataAnalysis(Protocol):
    """Pure data-analysis utilities over replay arrays. All methods take
    explicit array arguments and have no side effects -- they're stateless
    helpers exposed through the component API so plugins don't need to
    import game-specific modules directly.

    Examples:
        left, right = ctx.analysis.default_hands(ctx.replay.keycount())
        stats = ctx.analysis.per_column_stats(
            ctx.replay.columns_clean(), ctx.replay.offsets_clean())
    """

    def default_hands(self, keycount: int,
                      ) -> 'tuple[tuple[int,...], tuple[int,...]]': ...
    """Split keycount into (left_cols, right_cols). Middle column on odd
    key counts goes to the right hand."""

    def hand_split(self, columns: 'np.ndarray', offsets: 'np.ndarray',
                   left_cols: tuple, right_cols: tuple) -> dict: ...
    """Per-hand timing statistics dict with keys 'left', 'right',
    'left_cols', 'right_cols'. Each hand dict has n, mean, std, etc."""

    def per_column_stats(self, columns: 'np.ndarray',
                         offsets: 'np.ndarray') -> dict: ...
    """Timing statistics keyed by column index."""

    def timing_drift(self, noterows: 'np.ndarray', offsets: 'np.ndarray',
                     columns: 'np.ndarray', *,
                     n_segments: int = 4,
                     left_cols: tuple = (0, 1),
                     right_cols: tuple = (2, 3)) -> dict: ...
    """Chart segmented into n_segments; per-segment timing stats."""

    def rolling_stability(self, offsets: 'np.ndarray',
                          columns: 'np.ndarray', *,
                          window: int = 200,
                          left_cols: tuple = (0, 1),
                          right_cols: tuple = (2, 3)) -> dict: ...
    """Rolling window std -- tracks consistency across a session."""

    def coupling_analysis(self, noterows: 'np.ndarray',
                          offsets: 'np.ndarray',
                          columns: 'np.ndarray', *,
                          left_cols: tuple = (0, 1),
                          right_cols: tuple = (2, 3)) -> dict: ...
    """Solo vs paired timing per column (chord partner effect)."""

    def chord_vs_single(self, noterows: 'np.ndarray',
                        offsets: 'np.ndarray',
                        columns: 'np.ndarray') -> dict: ...
    """Timing split by chord size (single, jump, hand, quad)."""


# ── Drawing primitives ─────────────────────────────────────────────

# All coordinates passed to :class:`Context` are *component-local
# pixels* — the component thinks of itself as painting into its own
# ``(0, 0, ctx.w, ctx.h)`` box. Each backend translates to its native
# coord system: sidebar adds the column offset; overlay normalises to
# [0, 1] of the framebuffer.


@runtime_checkable
class Context(Protocol):
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
    data: GameState
    replay: ReplayState
    analysis: DataAnalysis
    hud_flags: HudFlags
    config: Config

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
    def image(self, rect: tuple, frame) -> None:
        """Blit a :class:`~analysis.components.pal.web.WebTextureFrame`
        (or a raw ``QPixmap`` on GUI surfaces that accept it) into the
        component's local-coord rect.

        Frame-kind dispatch:
          - ``qpixmap``       -- direct blit on GUI backends.
          - ``gl_texture_id`` / ``qsg_texture`` -- future backends that
            composite zero-copy. A backend that doesn't recognise the
            kind is expected to attempt a best-effort downgrade (e.g.
            readback to QPixmap via the WebTexture's latest_frame) or
            raise :class:`DataNotAvailable` if that isn't possible.
        """
        ...

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
class Manifest:
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
    layers: tuple[LayerDeclaration, ...] = field(default_factory=tuple)
    plugin_fields: dict[str, Any] = field(default_factory=dict)
    module: str = ''

    def __post_init__(self):
        object.__setattr__(self, 'supported_surfaces',
                           frozenset(self.supported_surfaces))
        object.__setattr__(self, 'requires_data',
                           frozenset(self.requires_data))
        object.__setattr__(self, 'optional_data',
                           frozenset(self.optional_data))
        object.__setattr__(self, 'layers', tuple(self.layers))


# The draw callable's type alias. ``None`` return — the backend owns
# cursor flushing and hitbox commit.
DrawFn = Callable[[Context], None]


@dataclass(frozen=True)
class Component:
    """Registered pair of (manifest, draw). Produced by registration
    helpers; consumed by each surface backend."""
    manifest: Manifest
    draw: DrawFn
