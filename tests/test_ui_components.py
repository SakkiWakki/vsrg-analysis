"""Tests for the declarative component system + sidebar renderer.

The components are plain dataclasses — trivial to test in isolation.
The renderer is exercised against a fake ``SidebarContext`` that
records each primitive call, so we can assert tree-of-components →
expected-sequence-of-primitives without dragging in Qt.
"""
from __future__ import annotations

import pytest

from analysis.player import theme
from analysis.ui.components import (Box, Button, Checkbox, Column, Component,
                                     Heading, Row, Spacer, Text,
                                     collect_actions, iter_tree, section)
from analysis.ui.render_sidebar import render


# ─── Component values ──────────────────────────────────────────────────────

def test_components_are_frozen():
    """Tree nodes must be immutable so two renderers walking the same
    tree can't see each other's mutations."""
    t = Text('hi')
    with pytest.raises(Exception):
        t.text = 'mutated'  # frozen dataclass raises FrozenInstanceError


def test_column_normalizes_children_to_tuple():
    """Accept lists for ergonomics (``Column([a, b])``) but store as
    tuple so the instance is hashable."""
    c = Column([Text('a'), Text('b')])
    assert isinstance(c.children, tuple)
    assert len(c.children) == 2
    hash(c)  # must not raise


def test_row_normalizes_children_to_tuple():
    r = Row([Button('ok', 'do_it')])
    assert isinstance(r.children, tuple)
    hash(r)


def test_section_helper():
    s = section('Title', Text('body'))
    assert isinstance(s, Column)
    assert isinstance(s.children[0], Heading)
    assert s.children[0].text == 'Title'
    assert s.children[1] == Text('body')


def test_iter_tree_depth_first():
    tree = Column((
        Heading('a'),
        Row((Button('b', 'x'), Button('c', 'y'))),
        Text('d'),
    ))
    nodes = list(iter_tree(tree))
    # Column, Heading, Row, Button, Button, Text
    assert [type(n).__name__ for n in nodes] == [
        'Column', 'Heading', 'Row', 'Button', 'Button', 'Text']


def test_collect_actions_reports_interactive_leaves():
    tree = Column((
        Heading('title'),
        Button('a', 'action_a'),
        Row((Button('b', 'action_b'),
             Checkbox('c', True, 'action_c'))),
        Text('not interactive'),
    ))
    assert collect_actions(tree) == ['action_a', 'action_b', 'action_c']


# ─── Renderer ──────────────────────────────────────────────────────────────

class _FakeCtx:
    """Records every SidebarContext primitive call and advances ``y``
    when the renderer asks it to. Mirrors just enough of the real
    SidebarContext to drive the renderer."""

    def __init__(self, col_x=10, col_w=200, y=0):
        self.col_x = col_x
        self.col_w = col_w
        self.y = y
        self.measure_only = False
        self.calls: list = []
        # Stub painter/renderer attrs so Heading's font swap is a no-op.
        self.painter = _StubPainter(self.calls)
        self.renderer = _StubRenderer()

    def text(self, text, x, baseline, color):
        self.calls.append(('text', text, x, baseline, color))

    def rect(self, rect, fill, outline=None, outline_w=1):
        self.calls.append(('rect', rect, fill, outline))

    def add_hitbox(self, rect, action, payload=None):
        self.calls.append(('hitbox', tuple(rect), action, payload))

    def button_at(self, rect, label, action, payload=None, *,
                  enabled=True, center=False):
        self.calls.append(('button', tuple(rect), label, action, payload))
        # Mirror the real implementation: register a hitbox too.
        self.calls.append(('hitbox', tuple(rect), action, payload))

    def checkbox(self, x, y, checked, **kwargs):
        self.calls.append(('checkbox', x, y, bool(checked)))


class _StubPainter:
    def __init__(self, calls):
        self._calls = calls

    def setFont(self, font):
        self._calls.append(('setFont', font))


class _StubRenderer:
    big_font = 'BIG'
    font = 'NORMAL'


def test_render_text():
    ctx = _FakeCtx()
    render(ctx, Text('hello'))
    assert ('text', 'hello', ctx.col_x + 0,
            0 + theme.TEXT_BASELINE_ROW, theme.BTN_FG) in ctx.calls
    assert ctx.y == theme.ROW_TEXT_H


def test_render_text_with_indent_and_color():
    ctx = _FakeCtx()
    render(ctx, Text('x', color=(255, 0, 0), indent=8))
    text_calls = [c for c in ctx.calls if c[0] == 'text']
    assert text_calls == [('text', 'x', ctx.col_x + 8,
                           theme.TEXT_BASELINE_ROW, (255, 0, 0))]


def test_render_heading_swaps_font():
    ctx = _FakeCtx()
    render(ctx, Heading('title'))
    # Font set to big_font, text drawn, font restored.
    assert ('setFont', 'BIG') in ctx.calls
    assert ('setFont', 'NORMAL') in ctx.calls
    assert any(c[0] == 'text' and c[1] == 'title' for c in ctx.calls)
    assert ctx.y == theme.HEADING_H


def test_render_spacer_default_height():
    ctx = _FakeCtx()
    render(ctx, Spacer())
    assert ctx.y == theme.SECTION_SPACER
    # No primitives drawn — spacer is pure layout.
    assert all(c[0] == 'setFont' or c[0] not in ('text', 'rect', 'hitbox',
                                                  'button', 'checkbox')
               for c in ctx.calls)


def test_render_spacer_custom_height():
    ctx = _FakeCtx()
    render(ctx, Spacer(height=42))
    assert ctx.y == 42


def test_render_button_registers_hitbox():
    ctx = _FakeCtx()
    render(ctx, Button('click me', 'do_thing', payload='payload_x'))
    button_calls = [c for c in ctx.calls if c[0] == 'button']
    hitbox_calls = [c for c in ctx.calls if c[0] == 'hitbox']
    assert len(button_calls) == 1
    assert button_calls[0][2] == 'click me'
    assert button_calls[0][3] == 'do_thing'
    assert button_calls[0][4] == 'payload_x'
    assert hitbox_calls[0][2] == 'do_thing'
    assert ctx.y == theme.ROW_BUTTON_H


def test_render_checkbox_has_hitbox_tick_and_label():
    ctx = _FakeCtx()
    render(ctx, Checkbox('enable', True, 'toggle_it', payload='k'))
    kinds = [c[0] for c in ctx.calls]
    # Must register a hitbox, draw a tick, and draw the label text.
    assert 'hitbox' in kinds
    assert 'checkbox' in kinds
    assert 'text' in kinds
    hitbox = next(c for c in ctx.calls if c[0] == 'hitbox')
    assert hitbox[2:] == ('toggle_it', 'k')
    tick = next(c for c in ctx.calls if c[0] == 'checkbox')
    assert tick[3] is True


def test_render_column_stacks_children():
    ctx = _FakeCtx()
    render(ctx, Column((Text('a'), Text('b'), Text('c'))))
    assert ctx.y == theme.ROW_TEXT_H * 3


def test_render_row_splits_width_and_advances_by_max_child():
    ctx = _FakeCtx(col_x=0, col_w=200)
    render(ctx, Row((Button('l', 'la'), Button('r', 'ra')), gap=4))
    buttons = [c for c in ctx.calls if c[0] == 'button']
    assert len(buttons) == 2
    # Two equal slots of 98 with a 4px gap between.
    left_rect = buttons[0][1]
    right_rect = buttons[1][1]
    assert left_rect[0] == 0  # left slot starts at x=0
    assert left_rect[2] == 98  # slot width
    assert right_rect[0] == 102  # left + slot + gap
    assert right_rect[2] == 98
    # Row height is the taller child — both buttons are the same here.
    assert ctx.y == theme.ROW_BUTTON_H


def test_render_row_last_slot_absorbs_rounding():
    """Column width 201 / 2 slots / 4px gap → 98 + 99. Last slot takes
    the extra px so the row fills the column exactly."""
    ctx = _FakeCtx(col_x=0, col_w=201)
    render(ctx, Row((Button('l', 'la'), Button('r', 'ra')), gap=4))
    buttons = [c for c in ctx.calls if c[0] == 'button']
    assert buttons[0][1][2] == 98   # left slot
    assert buttons[1][1][2] == 99   # right slot, absorbs the +1
    # Row spans exactly 0..201: last slot x + w = 201.
    assert buttons[1][1][0] + buttons[1][1][2] == 201


def test_render_row_empty_is_noop():
    ctx = _FakeCtx()
    render(ctx, Row(()))
    assert ctx.y == 0
    assert ctx.calls == []


def test_render_box_draws_rect_and_advances():
    ctx = _FakeCtx(col_x=5, col_w=50)
    render(ctx, Box(width=40, height=20, fill=(1, 2, 3), outline=(4, 5, 6)))
    rect_calls = [c for c in ctx.calls if c[0] == 'rect']
    assert rect_calls == [('rect', (5, 0, 40, 20), (1, 2, 3), (4, 5, 6))]
    assert ctx.y == 20


def test_render_box_clamps_to_available_width():
    ctx = _FakeCtx(col_x=0, col_w=30)
    render(ctx, Box(width=100, height=10, fill=(0, 0, 0)))
    rect = next(c for c in ctx.calls if c[0] == 'rect')[1]
    assert rect[2] == 30  # clamped to col_w


def test_render_none_is_noop():
    ctx = _FakeCtx()
    render(ctx, None)
    assert ctx.y == 0
    assert ctx.calls == []


def test_render_unknown_type_raises():
    ctx = _FakeCtx()
    with pytest.raises(TypeError):
        render(ctx, object())


def test_nested_column_flat_stacks():
    ctx = _FakeCtx()
    render(ctx, Column((
        Column((Text('a'), Text('b'))),
        Text('c'),
    )))
    assert ctx.y == theme.ROW_TEXT_H * 3


def test_section_helper_renders_like_heading_plus_children():
    ctx = _FakeCtx()
    render(ctx, section('Group', Text('item')))
    assert ctx.y == theme.HEADING_H + theme.ROW_TEXT_H


def test_same_tree_renders_deterministically():
    """Immediate-mode means rendering the same tree twice produces the
    same primitive sequence — no hidden state."""
    tree = Column((Heading('t'), Text('body'), Button('go', 'do')))
    a = _FakeCtx()
    b = _FakeCtx()
    render(a, tree)
    render(b, tree)
    assert a.calls == b.calls
    assert a.y == b.y
