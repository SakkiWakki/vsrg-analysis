"""Render a ``Component`` tree into a ``SidebarContext``.

The renderer is the glue between the declarative component tree and
the imperative ``SidebarContext`` primitives that paint + register
hitboxes. Plugins produce a tree; the host calls ``render`` to turn it
into pixels.

Layout model:

  * ``Column`` stacks children top-to-bottom.
  * ``Row`` splits the available width into equal slots and renders
    each child inside its slot, advancing ``y`` by the tallest child.
  * Everything else advances ``y`` by its natural height.
"""
from __future__ import annotations

from analysis.player.render import theme
from analysis.ui.components import (Box, Button, Checkbox, Column, Heading,
                                     Row, Spacer, Text)


def render(sctx, component) -> None:
    """Render ``component`` starting at ``sctx.y`` across the full
    sidebar column width."""
    _render_at(sctx, component, sctx.col_x, sctx.col_w)


def _render_at(sctx, component, x, width) -> None:
    """Render ``component`` at (``x``, ``sctx.y``) within ``width`` px.
    Advances ``sctx.y`` past the component."""
    if component is None:
        return
    if isinstance(component, Text):
        color = theme.BTN_FG if component.color is None else component.color
        sctx.text(component.text, x + component.indent,
                  sctx.y + theme.TEXT_BASELINE_ROW, color)
        sctx.y += theme.ROW_TEXT_H
        return
    if isinstance(component, Heading):
        # Heading uses the renderer's big_font; reach through sctx to
        # match the existing draw_heading primitive rather than
        # re-deriving font sizing here.
        if not sctx.measure_only:
            sctx.painter.setFont(sctx.renderer.big_font)
            sctx.text(component.text, x, sctx.y + 18, theme.COLOR_HEADING)
            sctx.painter.setFont(sctx.renderer.font)
        sctx.y += theme.HEADING_H
        return
    if isinstance(component, Spacer):
        sctx.y += (theme.SECTION_SPACER if component.height is None
                   else int(component.height))
        return
    if isinstance(component, Button):
        rect = (x, sctx.y, width, theme.ROW_BUTTON_H)
        sctx.button_at(rect, component.label, component.action,
                       component.payload)
        sctx.y += theme.ROW_BUTTON_H
        return
    if isinstance(component, Checkbox):
        h = 18
        rect = (x, sctx.y, width, h)
        sctx.add_hitbox(rect, component.action, component.payload)
        sctx.checkbox(x + 6, sctx.y + 3, checked=component.checked)
        sctx.text(component.label, x + 22,
                  sctx.y + theme.TEXT_BASELINE_ROW, theme.BTN_FG)
        sctx.y += h
        return
    if isinstance(component, Box):
        w = min(width, component.width)
        sctx.rect((x, sctx.y, w, component.height),
                  component.fill, outline=component.outline)
        sctx.y += component.height
        return
    if isinstance(component, Column):
        for child in component.children:
            _render_at(sctx, child, x, width)
        return
    if isinstance(component, Row):
        _render_row(sctx, component, x, width)
        return
    raise TypeError(f'unknown component type: {type(component).__name__}')


def _render_row(sctx, row, x, width):
    children = row.children
    if not children:
        return
    n = len(children)
    gap = row.gap
    total_gap = gap * (n - 1)
    slot_w = max(1, (width - total_gap) // n)
    start_y = sctx.y
    max_y = start_y
    for i, child in enumerate(children):
        # Last slot absorbs rounding so the row fills ``width`` exactly.
        w = slot_w if i < n - 1 else width - (slot_w + gap) * (n - 1)
        slot_x = x + i * (slot_w + gap)
        sctx.y = start_y
        _render_at(sctx, child, slot_x, w)
        if sctx.y > max_y:
            max_y = sctx.y
    sctx.y = max_y
