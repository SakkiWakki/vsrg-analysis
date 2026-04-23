"""Overlay surface backend for the unified component API.

Components paint in local pixels; this backend normalises their coords
into the overlay target resolution and emits PAL
:class:`~analysis.components.pal.base.OverlayFrame` records. The PAL
then hands the frame to whatever overlay implementation is active
(today: gamescope + shm).

Live data on the overlay side is an :class:`OverlayGameState` snapshot
published by the game adapter. :class:`OverlayGameStateDataSource`
wraps it and answers the fields it can; windows / per-judge colors are
typically unavailable live (osu!'s memory only exposes counts), in
which case :class:`DataNotAvailable` fires and components either
degrade or were filtered out at registration time by ``requires_data``.

Interactive primitives (buttons, checkboxes) still render their chrome
so the overlay and sidebar look the same, but no hitboxes are
registered — the gamescope path routes no click events back. Components
that care about this branch on ``ctx.supports_input``.
"""
from __future__ import annotations

from dataclasses import dataclass

from analysis.components.api import (
    ComponentGameState,
    DataNotAvailable,
    REGION_FREE,
    SURFACE_OVERLAY,
)
from analysis.components.pal.base import OverlayFrame
from analysis.overlay.api import (
    ANCHOR_TL,
    BLACK_DIM,
    WHITE,
    rgba,
)
from analysis.player.render import theme


# ── Layout hints ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class OverlayFields:
    """Overlay-specific layout hints for a component manifest.

    The overlay has no panel region — all components float freely.
    ``default_xy`` and ``default_size`` are normalized to [0, 1] of the
    framebuffer dimensions."""
    hz: float = 30.0
    default_xy: tuple = (0.02, 0.04)
    default_size: tuple = (0.18, 0.18)
    region: str = REGION_FREE


# ── Data source ─────────────────────────────────────────────────────


class OverlayGameStateDataSource:
    """Implements :class:`ComponentGameState` against an
    :class:`~analysis.overlay.api.OverlayGameState` snapshot.

    Each field the overlay snapshot actually carries is answered from
    the snapshot; live fields the snapshot lacks (windows, judge
    colors) raise :class:`DataNotAvailable`. Components with optional
    dependencies on those fields should catch the exception and render
    a reduced layout.
    """

    _FIELDS = frozenset({
        'game', 'keycount', 'combo', 'accuracy',
        'judgment_counts',
    })

    def __init__(self, state):
        self._s = state

    def supports(self, field: str) -> bool:
        return str(field) in self._FIELDS

    # Game identity
    def game(self) -> str:
        return str(self._s.game)

    def keycount(self) -> int:
        return int(self._s.keycount)

    # Scoring
    def combo(self) -> int:
        return int(self._s.combo)

    def accuracy(self) -> float:
        return float(self._s.accuracy)

    # Judgments
    def judgment_windows(self) -> list[tuple[str, float]]:
        raise DataNotAvailable(
            'overlay game state does not expose judgment window widths')

    def judgment_counts(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self._s.judgments}

    def judgment_colors(self) -> dict[str, tuple]:
        raise DataNotAvailable('overlay game state has no judge-color table')

    def judge_label(self) -> str:
        raise DataNotAvailable('overlay game state has no judge label')

    def game_memory(self):
        # The overlay backend delivers the live native snapshot produced
        # by OsuLiveClient and published via the game adapter's state hook.
        # Returns None when no snapshot is available (osu! not running,
        # native extension missing, or not in active gameplay).
        from analysis.components.provider import current_game_memory
        return current_game_memory()


# ── Color + font conversions ────────────────────────────────────────


def _color_to_rgba(color) -> int:
    """``ComponentContext`` colors match the theme convention: 3-tuple
    ``(r, g, b)`` or 4-tuple with alpha. The overlay wants a packed
    RGBA uint32. Accept either.
    """
    if color is None:
        return WHITE
    if isinstance(color, int):
        return int(color) & 0xffffffff
    if len(color) == 3:
        r, g, b = color
        a = 255
    elif len(color) == 4:
        r, g, b, a = color
    else:
        return WHITE
    return rgba(int(r), int(g), int(b), int(a))


# Glyph-scale chosen so a line of sidebar-style text renders at roughly
# the same visual size on a 1440p overlay as it does in the embedded
# player's 14-px font. Plugins that want bigger callouts override via
# the primitive's ``px_scale`` after porting (not supported yet; keep
# uniform for now).
_OVERLAY_PX_SCALE = 1.8


# ── Replay / analysis stubs for the overlay surface ─────────────────
# The overlay has no post-analysis replay session -- it reads live game
# state from memory. Both protocols raise DataNotAvailable so components
# can degrade gracefully rather than crash.

class _OverlayNoReplay:
    def _na(self, *_, **__):
        raise DataNotAvailable(
            'replay data is not available on the overlay surface')
    offsets = offsets_clean = columns = columns_clean = _na
    noterows = noterows_clean = misses = notetypes = _na
    keycount = game = _na


class _OverlayNoAnalysis:
    def _na(self, *_, **__):
        raise DataNotAvailable(
            'analysis utilities require replay data; '
            'not available on the overlay surface')
    default_hands = hand_split = per_column_stats = _na
    timing_drift = rolling_stability = coupling_analysis = chord_vs_single = _na


# ── Context ─────────────────────────────────────────────────────────


class OverlayComponentContext:
    """Component context that serializes primitives to an
    :class:`OverlayFrame`.

    Coord mapping: component-local px → absolute overlay px (add
    ``_origin``) → normalised (divide by framebuffer w/h). Widget ids
    are derived from the component key + a monotonically increasing
    counter so they stay stable across frames (required for drag
    persistence by the C renderer).
    """

    surface = SURFACE_OVERLAY

    def __init__(self, frame: OverlayFrame, *,
                 component_key: str, origin_px: tuple, size_px: tuple,
                 fb_w: int, fb_h: int,
                 data_source: ComponentGameState,
                 supports_input: bool = False):
        self._frame = frame
        self._key = str(component_key)
        self._origin = (int(origin_px[0]), int(origin_px[1]))
        self.w = int(size_px[0])
        self.h = int(size_px[1])
        self._fb_w = max(1, int(fb_w))
        self._fb_h = max(1, int(fb_h))
        self.measure_only = False  # overlay has nothing to measure
        self.data = data_source
        self.replay = _OverlayNoReplay()
        self.analysis = _OverlayNoAnalysis()
        self.y = 0
        self._id_counter = 0
        self._input = bool(supports_input)

    @property
    def supports_input(self) -> bool:
        return self._input

    # ── Coord helpers ──
    def _norm_xy(self, x_px: int, y_px: int) -> tuple[float, float]:
        abs_x = self._origin[0] + int(x_px)
        abs_y = self._origin[1] + int(y_px)
        return (abs_x / self._fb_w, abs_y / self._fb_h)

    def _norm_wh(self, w_px: int, h_px: int) -> tuple[float, float]:
        return (int(w_px) / self._fb_w, int(h_px) / self._fb_h)

    def _next_id(self, kind: str) -> str:
        self._id_counter += 1
        return f'{self._key}:{kind}{self._id_counter}'

    # ── Geometry ──
    def split_row(self, n: int = 2, gap: int = 4) -> list[tuple[int, int]]:
        if n <= 0:
            return []
        total_gap = gap * (n - 1)
        slot_w = (self.w - total_gap) // n
        slots = []
        x = 0
        for i in range(n):
            slot = slot_w if i < n - 1 else self.w - x
            slots.append((x, slot))
            x += slot_w + gap
        return slots

    # ── Raw primitives ──
    def text(self, s, x, baseline, color=None) -> None:
        nx, ny = self._norm_xy(x, baseline - 12)  # approximate baseline→top
        self._frame.text(
            self._next_id('t'), str(s), nx, ny,
            px_scale=_OVERLAY_PX_SCALE,
            color=_color_to_rgba(color if color is not None else theme.BTN_FG),
            anchor=ANCHOR_TL)

    def rect(self, rect, color=None, outline=None, outline_w=1) -> None:
        rx, ry, rw, rh = rect
        nx, ny = self._norm_xy(rx, ry)
        nw, nh = self._norm_wh(rw, rh)
        if color is not None:
            self._frame.rect(self._next_id('r'), nx, ny, nw, nh,
                             color=_color_to_rgba(color), anchor=ANCHOR_TL)
        if outline is not None:
            # Fake outlines with four 1-px rects. The overlay renderer
            # has no stroke primitive, so this is the honest fallback.
            t = max(1, int(outline_w))
            oc = _color_to_rgba(outline)
            tw_norm = t / self._fb_w
            th_norm = t / self._fb_h
            self._frame.rect(self._next_id('ol'), nx, ny, nw, th_norm,
                             color=oc, anchor=ANCHOR_TL)
            self._frame.rect(self._next_id('ol'),
                             nx, ny + nh - th_norm, nw, th_norm,
                             color=oc, anchor=ANCHOR_TL)
            self._frame.rect(self._next_id('ol'),
                             nx, ny, tw_norm, nh,
                             color=oc, anchor=ANCHOR_TL)
            self._frame.rect(self._next_id('ol'),
                             nx + nw - tw_norm, ny, tw_norm, nh,
                             color=oc, anchor=ANCHOR_TL)

    def line(self, start, end, color, width=1) -> None:
        # Overlay renderer doesn't draw lines. Best-effort: render as a
        # 1-pixel-thick rect along the axis-aligned span (diagonals
        # silently skip). This keeps checkbox ticks etc. from looking
        # invisible; future work: teach the C renderer KIND_LINE.
        sx, sy = start
        ex, ey = end
        if sx == ex:  # vertical
            y0, y1 = sorted((int(sy), int(ey)))
            self.rect((int(sx), y0, max(1, int(width)), y1 - y0), color)
        elif sy == ey:  # horizontal
            x0, x1 = sorted((int(sx), int(ex)))
            self.rect((x0, int(sy), x1 - x0, max(1, int(width))), color)
        # Diagonal: unsupported on this surface; drop.

    # ── Cursor-advancing rows ──
    def spacer(self, h=None) -> None:
        self.y += int(h) if h is not None else int(theme.SECTION_SPACER)

    def draw_heading(self, text, color=None) -> None:
        col = color if color is not None else theme.COLOR_HEADING
        self.text(text, 0, self.y + 18, color=col)
        self.y += int(theme.HEADING_H)

    def draw_text(self, text, color=None, indent=0, height=None) -> None:
        col = color if color is not None else theme.BTN_FG
        self.text(text, int(indent), self.y + theme.TEXT_BASELINE_ROW,
                  color=col)
        self.y += int(height) if height is not None else int(theme.ROW_TEXT_H)

    def draw_hint(self, text, color=None) -> None:
        col = color if color is not None else theme.COLOR_HINT
        self.draw_text(text, color=col, height=theme.ROW_HINT_H)

    # ── Interactive (chrome-only on overlay today) ──
    def draw_button(self, label, action, payload=None, *, enabled=True,
                    height=None, center=False):
        h = int(height) if height is not None else int(theme.ROW_BUTTON_H)
        rect_local = (0, self.y, self.w, h)
        self.button_at(rect_local, label, action, payload,
                       enabled=enabled, center=center)
        self.y += h
        return rect_local

    def button_at(self, rect, label, action, payload=None, *,
                  enabled=True, center=False) -> None:
        rx, ry, rw, rh = rect
        fill = theme.BTN_FILL if enabled else theme.BTN_FILL_DISABLED
        self.rect((int(rx), int(ry), int(rw), int(rh)), fill,
                  outline=theme.BTN_BORDER)
        fg = theme.BTN_FG if enabled else theme.BTN_FG_DISABLED
        if center:
            tx = int(rx) + max(0, (int(rw) - len(str(label)) * 6) // 2)
        else:
            tx = int(rx) + theme.TEXT_INDENT
        self.text(label, tx, int(ry) + theme.TEXT_BASELINE_BUTTON, color=fg)
        # No hitbox: overlay has no click routing today. supports_input
        # is False, so components that need interactivity will have
        # gated themselves off this surface.

    def checkbox(self, x, y, checked) -> tuple:
        size = int(theme.CHECKBOX_SIZE)
        box = (int(x), int(y), size, size)
        self.rect(box, theme.COLOR_CHECKBOX_FILL,
                  outline=theme.COLOR_CHECKBOX_BORDER)
        if checked:
            # Two line segments form the tick — ``line`` supports
            # axis-aligned only; the diagonal parts drop cleanly.
            self.line((int(x) + 2, int(y) + 5),
                      (int(x) + 4, int(y) + 8),
                      theme.COLOR_CHECKBOX_MARK, 2)
            self.line((int(x) + 4, int(y) + 8),
                      (int(x) + 9, int(y) + 2),
                      theme.COLOR_CHECKBOX_MARK, 2)
        return box


# ── Driver ─────────────────────────────────────────────────────────


def draw_component_into_frame(component, frame: OverlayFrame, *,
                              game_state, origin_px: tuple,
                              size_px: tuple,
                              supports_input: bool = False) -> None:
    """Run a unified-API component inside an overlay frame build pass.

    ``frame`` is the PAL-neutral command list being assembled; the
    component's output is grouped under its key so the C renderer's
    drag-per-group logic naturally treats the whole component as one
    draggable unit.
    """
    data_source = OverlayGameStateDataSource(game_state)
    frame.begin_group(component.manifest.key)
    try:
        cctx = OverlayComponentContext(
            frame,
            component_key=component.manifest.key,
            origin_px=origin_px, size_px=size_px,
            fb_w=frame.width, fb_h=frame.height,
            data_source=data_source,
            supports_input=supports_input)
        component.draw(cctx)
    finally:
        frame.end_group()


__all__ = [
    'OverlayGameStateDataSource',
    'OverlayComponentContext',
    'draw_component_into_frame',
]
