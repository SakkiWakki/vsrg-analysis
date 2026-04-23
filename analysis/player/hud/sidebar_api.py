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
``analysis.player.render.theme``.
"""
from __future__ import annotations

from dataclasses import dataclass

from analysis.player.render import theme


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
    enabled: bool = True
    # Optional flyout: when set, the sidebar renders ``draw`` as a
    # collapsed one-line header; the full ``draw_expanded`` content is
    # painted into a flyout panel to the left of the sidebar when
    # ``player.hud.open_flyout`` equals this section's key. One flyout
    # may be open at a time across the whole sidebar.
    draw_expanded: object | None = None
    # Layout regions. A section opts in to drag-and-drop with
    # ``draggable=True``; in edit mode the user can move it between the
    # sidepanel and the free (floating) region. ``default_region`` is
    # the initial region when no saved layout exists; ``'sidepanel'`` is
    # handled by the sidebar itself, ``'free'`` is the hardcoded
    # floating region. Unknown names fall back to ``'free'``.
    draggable: bool = False
    default_region: str = 'sidepanel'
    # Initial free-region placement in normalized screen coords and
    # explicit size (px). Used only when the component is first placed
    # in 'free' and no saved layout exists.
    default_free_xy: tuple = (0.5, 0.5)
    default_size: tuple = (210, 120)


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
                 y, *, measure_only=False, hitbox_clip=None):
        self.render_ctx = render_ctx
        self.painter = painter
        self.renderer = renderer
        self.sidebar_x = int(sidebar_x)
        self.sidebar_w = int(sidebar_w)
        self.y = int(y)
        self.measure_only = bool(measure_only)
        # (y_min, y_max) in screen coords: hitboxes fully outside this
        # band are dropped so scrolled-off rows don't catch clicks. None
        # disables the clip (default, for pinned-bottom sections etc.).
        self.hitbox_clip = hitbox_clip

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
        from analysis.player.render.qt_renderer import _text
        _text(self.painter, text, color, x, baseline)

    def rect(self, rect, color, outline=None, outline_w=1):
        if self.measure_only:
            return
        from analysis.player.render.qt_renderer import _rect, _rect_outline
        if color is not None:
            _rect(self.painter, color, rect)
        if outline is not None:
            _rect_outline(self.painter, outline, rect, outline_w)

    def line(self, start, end, color, width=1):
        if self.measure_only:
            return
        from analysis.player.render.qt_renderer import _line
        _line(self.painter, color, start, end, width)

    def add_hitbox(self, rect, action, payload=None):
        if self.measure_only:
            return
        if self.hitbox_clip is not None:
            rx, ry, rw, rh = rect
            ymin, ymax = self.hitbox_clip
            if ry + rh <= ymin or ry >= ymax:
                return
        self.player.hud.add_hitbox(rect, action, payload)

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


def _escape_key(key: str) -> str:
    """Plugin keys may contain dots; the config store uses dots as a path
    separator. Rewrite dots to underscores for the on-disk path. Matches
    :func:`analysis.config.migrate._escape` so legacy files line up."""
    return key.replace('.', '_')


class SidebarSectionRegistry:
    """Holds registered sidebar sections and yields them in draw order.

    Enabled/disabled state is owned by the process-wide
    :class:`analysis.config.ConfigStore` (read from
    ``plugins.<key>.sidebar_disabled``). Every registry instance
    subscribes to ``plugins`` so that a toggle in one window's Plugins
    dialog reaches the sidebar of every other window on the next
    frame."""

    def __init__(self, config=None):
        from analysis.config import get_config
        self._config = config if config is not None else get_config()
        self._sections: list[SidebarSection] = []
        self._config_sub = self._config.subscribe(
            'plugins', self._on_config_change)

    def add(self, name, draw, *, priority=1000, key=None, module='',
            pin_bottom=False, draw_expanded=None, draggable=False,
            default_region='sidepanel', default_free_xy=(0.5, 0.5),
            default_size=(210, 120)):
        key = str(key or f'{module}:{name}')
        self._sections.append(SidebarSection(
            key=key,
            name=str(name),
            draw=draw,
            priority=int(priority),
            module=str(module),
            pin_bottom=bool(pin_bottom),
            enabled=not self._is_disabled(key),
            draw_expanded=draw_expanded,
            draggable=bool(draggable),
            default_region=str(default_region),
            default_free_xy=(float(default_free_xy[0]),
                             float(default_free_xy[1])),
            default_size=(int(default_size[0]), int(default_size[1])),
        ))
        self._sections.sort(key=lambda s: (s.priority, s.name))

    def flyout_section(self, key):
        """Return the section registered with this flyout key, or None."""
        key = str(key)
        for s in self._sections:
            if s.enabled and s.draw_expanded is not None and s.key == key:
                return s
        return None

    def top_sections(self):
        return [s for s in self._sections
                if s.enabled and not s.pin_bottom
                and self._effective_region(s) == 'sidepanel']

    def bottom_sections(self):
        return [s for s in self._sections
                if s.enabled and s.pin_bottom
                and self._effective_region(s) == 'sidepanel']

    def free_sections(self):
        """Draggable sections that currently live in the free region."""
        out = [s for s in self._sections
               if s.enabled and s.draggable
               and self._effective_region(s) == 'free']
        out.sort(key=lambda s: (self.section_order(s), s.name))
        return out

    def all_sections(self):
        return list(self._sections)

    def find_section(self, key: str):
        """Look up a section by key, or None if no section with that key
        is registered. Callers don't have to reimplement linear search
        over `all_sections`."""
        for s in self._sections:
            if s.key == key:
                return s
        return None

    # ── Per-section layout (region, order, free-region rect) ────────

    def _layout_path(self, key: str, field: str) -> str:
        return f'player.sidebar_layout.{_escape_key(key)}.{field}'

    def _effective_region(self, section: 'SidebarSection') -> str:
        """'sidepanel' or 'free'. Non-draggable sections are always
        locked to the sidepanel; draggable sections read their saved
        region, or fall back to the plugin-declared default."""
        if not section.draggable:
            return 'sidepanel'
        saved = self._config.get(self._layout_path(section.key, 'region'),
                                 None)
        if saved in ('sidepanel', 'free'):
            return saved
        declared = section.default_region
        return 'free' if declared == 'free' else 'sidepanel'

    def section_region(self, key: str) -> str:
        section = self.find_section(key)
        return self._effective_region(section) if section else 'sidepanel'

    def section_order(self, section: 'SidebarSection') -> float:
        """User-facing drop order. Falls back to the plugin's declared
        priority so components without a saved order keep their
        original rank."""
        saved = self._config.get(self._layout_path(section.key, 'order'),
                                 None)
        if saved is not None:
            try:
                return float(saved)
            except (TypeError, ValueError):
                pass
        return float(section.priority)

    def section_free_rect(self, section: 'SidebarSection',
                          screen_w: int, screen_h: int) -> tuple:
        """Return (x, y, w, h) for a free-region section in screen px.
        Reads the saved rect if any, else derives from the plugin's
        declared default (normalized xy × screen, explicit size)."""
        base = self._config.get(
            self._layout_path(section.key, 'rect'), None)
        if isinstance(base, dict):
            try:
                return (int(base['x']), int(base['y']),
                        int(base['w']), int(base['h']))
            except (KeyError, TypeError, ValueError):
                pass
        w, h = section.default_size
        fx, fy = section.default_free_xy
        x = int(max(0, min(screen_w - w, fx * screen_w)))
        y = int(max(0, min(screen_h - h, fy * screen_h)))
        return (x, y, int(w), int(h))

    def set_section_region(self, key: str, region: str) -> None:
        if region not in ('sidepanel', 'free'):
            region = 'free'
        self._config.set(self._layout_path(key, 'region'), region)

    def set_section_order(self, key: str, order: float) -> None:
        self._config.set(self._layout_path(key, 'order'), float(order))

    def set_section_free_rect(self, key: str, x: int, y: int,
                              w: int, h: int) -> None:
        self._config.set(self._layout_path(key, 'rect'),
                         {'x': int(x), 'y': int(y),
                          'w': int(w), 'h': int(h)})

    def set_enabled(self, key, enabled):
        """Flip a section on/off. Writes through the config store so
        every other window's registry sees the change via its
        subscription. Returns True if the key exists."""
        key = str(key)
        if not any(s.key == key for s in self._sections):
            return False
        self._config.set(
            f'plugins.{_escape_key(key)}.sidebar_disabled',
            not bool(enabled))
        return True

    def toggle_enabled(self, key):
        for s in self._sections:
            if s.key == key:
                return self.set_enabled(key, not s.enabled)
        return False

    def close(self):
        """Unsubscribe from the config store. Call when the owning
        player tab is being disposed to avoid stale handlers firing
        against a dead registry."""
        if self._config_sub is not None:
            self._config.unsubscribe(self._config_sub)
            self._config_sub = None

    def _is_disabled(self, key: str) -> bool:
        return bool(self._config.get(
            f'plugins.{_escape_key(key)}.sidebar_disabled', False))

    def _on_config_change(self, path, old, new):
        """Config changed somewhere under ``plugins``. Refresh the
        enabled flag on any section whose key matches."""
        # path shape: ('plugins', <escaped_key>, <field>, ...)
        if len(path) < 3 or path[-1] != 'sidebar_disabled':
            return
        escaped = path[1]
        for s in self._sections:
            if _escape_key(s.key) == escaped:
                s.enabled = not bool(new) if new is not None else True
