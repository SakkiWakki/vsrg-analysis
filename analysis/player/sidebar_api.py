"""Public API for replay-player sidebar sections.

Sidebar sections are modular panels rendered in the right-hand HUD of the
embedded player. A section is a callable ``draw(sctx)`` that paints rows
into the column handed out by the renderer.

Plugins expose sections via a top-level ``register_sidebar(add)`` function
in the same modules discovered by ``PluginManager``::

    def register_sidebar(add):
        add('My section', my_draw_fn, priority=500)

The renderer iterates sections in ascending priority order. Built-in
sections use priorities in 100..900; user sections default to 1000 and can
override to interleave. Set ``pin_bottom=True`` to stick a section to the
bottom of the sidebar.

``SidebarContext`` exposes the full drawing vocabulary sections need
(``text``, ``rect``, ``line``, ``button``, ``checkbox``, ``split_row``,
cursor helpers). Plugin authors should use these rather than importing Qt
helpers directly — this way the sidebar stays consistent with the theme
and stays backend-agnostic for future renderers.

Design tokens (colors, row heights, paddings) live in
``analysis.player.theme``.
"""
from __future__ import annotations

from dataclasses import dataclass

from analysis.player import theme


# Monospace character width (approx) used to center short labels.
_CHAR_PX = 6


@dataclass
class SidebarSection:
    key: str
    name: str
    draw: object
    priority: int = 1000
    module: str = ''
    pin_bottom: bool = False


class SidebarContext:
    """Per-frame context passed to each section's draw callable.

    Holds the paint cursor and drawing primitives. Sections advance ``y``
    as they render; the renderer lays out top-pinned sections from the top
    and bottom-pinned sections from a measured offset so they hug ``p.H``.

    When ``measure_only`` is True, primitives skip painting and hitbox
    registration but still advance ``y``. This lets the renderer compute
    each bottom section's height without a dry-run painter hack.
    """

    def __init__(self, render_ctx, painter, renderer, sidebar_x, sidebar_w,
                 y, *, measure_only=False):
        self.render_ctx = render_ctx
        self.painter = painter
        self.renderer = renderer
        self.sidebar_x = int(sidebar_x)
        self.sidebar_w = int(sidebar_w)
        self.y = int(y)
        self.measure_only = bool(measure_only)

    # ── Geometry ─────────────────────────────────────────────────────────
    @property
    def player(self):
        return self.render_ctx.player

    @property
    def col_x(self):
        return self.sidebar_x + theme.SIDEBAR_INSET

    @property
    def col_w(self):
        return self.sidebar_w - 2 * theme.SIDEBAR_INSET

    def split_row(self, n=2, gap=4):
        """Return ``n`` equal-width (x, w) slots across the column, with
        ``gap`` pixels between them. Handy for side-by-side buttons."""
        if n <= 0:
            return []
        total_gap = gap * (n - 1)
        slot_w = (self.col_w - total_gap) // n
        slots = []
        x = self.col_x
        for i in range(n):
            w = slot_w if i < n - 1 else self.col_w - (x - self.col_x)
            slots.append((x, w))
            x += slot_w + gap
        return slots

    # ── Raw primitives (no-op when measuring) ────────────────────────────
    def text(self, text, x, baseline, color=theme.BTN_FG):
        if self.measure_only:
            return
        from analysis.player.qt_renderer import _text
        _text(self.painter, text, color, x, baseline)

    def rect(self, rect, color, outline=None, outline_w=1):
        if self.measure_only:
            return
        from analysis.player.qt_renderer import _rect, _rect_outline
        if color is not None:
            _rect(self.painter, color, rect)
        if outline is not None:
            _rect_outline(self.painter, outline, rect, outline_w)

    def line(self, start, end, color, width=1):
        if self.measure_only:
            return
        from analysis.player.qt_renderer import _line
        _line(self.painter, color, start, end, width)

    def add_hitbox(self, rect, action, payload=None):
        if self.measure_only:
            return
        self.player._hud_hitboxes.append((tuple(rect), action, payload))

    # ── Cursor-advancing rows ────────────────────────────────────────────
    def spacer(self, h=theme.SECTION_SPACER):
        self.y += int(h)

    def draw_heading(self, text, color=theme.COLOR_HEADING):
        if not self.measure_only:
            self.painter.setFont(self.renderer.big_font)
            self.text(text, self.col_x, self.y + 18, color)
            self.painter.setFont(self.renderer.font)
        self.y += theme.HEADING_H

    def draw_text(self, text, color=theme.BTN_FG, indent=0,
                  height=theme.ROW_TEXT_H):
        self.text(text, self.col_x + indent,
                  self.y + theme.TEXT_BASELINE_ROW, color)
        self.y += height

    def draw_hint(self, text, color=theme.COLOR_HINT):
        self.draw_text(text, color=color, height=theme.ROW_HINT_H)

    def draw_button(self, label, action, payload=None, *, enabled=True,
                    height=theme.ROW_BUTTON_H, center=False):
        """Full-width button at the current cursor, advances ``y``."""
        rect = (self.col_x, self.y, self.col_w, height)
        self.button_at(rect, label, action, payload,
                       enabled=enabled, center=center)
        self.y += height
        return rect

    def button_at(self, rect, label, action, payload=None, *,
                  enabled=True, center=False):
        """Button at an explicit rect — does NOT advance the cursor.

        Use when you need multiple buttons on one row (call with rects from
        ``split_row``) or an inline control inside a larger layout.
        """
        fill = theme.BTN_FILL if enabled else theme.BTN_FILL_DISABLED
        fg = theme.BTN_FG if enabled else theme.BTN_FG_DISABLED
        self.rect(rect, fill, outline=theme.BTN_BORDER)
        rx, ry, rw, _rh = rect
        if center:
            tx = rx + max(0, (rw - len(str(label)) * _CHAR_PX) // 2)
        else:
            tx = rx + theme.TEXT_INDENT
        self.text(label, tx, ry + theme.TEXT_BASELINE_BUTTON, fg)
        if enabled:
            self.add_hitbox(rect, action, payload)

    def checkbox(self, x, y, checked, *,
                 size=theme.CHECKBOX_SIZE,
                 fill=theme.COLOR_CHECKBOX_FILL,
                 border=theme.COLOR_CHECKBOX_BORDER,
                 mark=theme.COLOR_CHECKBOX_MARK):
        """Draw a tick-box at (x, y). Does not advance the cursor."""
        box = (x, y, size, size)
        self.rect(box, fill, outline=border)
        if checked:
            self.line((box[0] + 2, box[1] + 5),
                      (box[0] + 4, box[1] + 8), mark, 2)
            self.line((box[0] + 4, box[1] + 8),
                      (box[0] + 9, box[1] + 2), mark, 2)
        return box


class SidebarSectionRegistry:
    """Holds registered sidebar sections and yields them in draw order."""

    def __init__(self):
        self._sections: list[SidebarSection] = []

    def add(self, name, draw, *, priority=1000, key=None, module='',
            pin_bottom=False):
        key = str(key or f'{module}:{name}')
        self._sections.append(SidebarSection(
            key=key,
            name=str(name),
            draw=draw,
            priority=int(priority),
            module=str(module),
            pin_bottom=bool(pin_bottom),
        ))
        self._sections.sort(key=lambda s: (s.priority, s.name))

    def top_sections(self):
        return [s for s in self._sections if not s.pin_bottom]

    def bottom_sections(self):
        return [s for s in self._sections if s.pin_bottom]

    def all_sections(self):
        return list(self._sections)
