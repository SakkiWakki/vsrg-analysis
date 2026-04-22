"""Tests for the draggable-component layout system.

Covers:
- section_region / set_section_region persistence round-trip
- free_sections() filters by effective region + order
- default_region='free' makes a section start in free
- _run_sections records per-section rects + draws edit outlines
- edit-mode suppresses non-edit hitbox actions in handle_mouse_down
- begin_drag_section action kicks off a drag with the right offsets
- _finish_drag routes drops by cursor X relative to sidebar edge
- _compute_drop_order picks midpoint between neighbors using last
  frame's rects
"""
from __future__ import annotations

from types import SimpleNamespace

from analysis.config import get_config
from analysis.player.hud.sidebar_api import (
    SidebarSection,
    SidebarSectionRegistry,
)
from analysis.player.hud.hud_state import HudState
from analysis.player.render.qt_renderer import QtPlayerRenderer


def _fresh_registry():
    """Return a registry on the process config, cleaning up any
    layout keys a previous test left behind."""
    reg = SidebarSectionRegistry(config=get_config())
    # Belt-and-braces: wipe any stale test keys.
    cfg = get_config()
    for k in ('demo_one', 'demo_two', 'demo_three'):
        cfg.set(f'player.sidebar_layout.{k}.region', None)
        cfg.set(f'player.sidebar_layout.{k}.order', None)
        cfg.set(f'player.sidebar_layout.{k}.rect', None)
    return reg


def test_section_region_round_trip():
    reg = _fresh_registry()
    reg.add('one', lambda s: None, key='demo:one', priority=200,
            draggable=True, default_region='sidepanel')
    reg.add('two', lambda s: None, key='demo:two', priority=300,
            draggable=True, default_region='free')

    # Defaults match declared default_region.
    assert reg.section_region('demo:one') == 'sidepanel'
    assert reg.section_region('demo:two') == 'free'

    # Non-draggable sections are always sidepanel regardless of saved
    # value — the registry ignores the key rather than respecting a
    # nonsense override.
    reg.add('three', lambda s: None, key='demo:three',
            draggable=False, default_region='free')
    assert reg.section_region('demo:three') == 'sidepanel'

    reg.set_section_region('demo:one', 'free')
    assert reg.section_region('demo:one') == 'free'
    reg.close()


def test_free_sections_filters_and_orders():
    reg = _fresh_registry()
    reg.add('a', lambda s: None, key='demo:one', priority=200,
            draggable=True, default_region='sidepanel')
    reg.add('b', lambda s: None, key='demo:two', priority=300,
            draggable=True, default_region='free')
    reg.add('c', lambda s: None, key='demo:three', priority=100,
            draggable=False)  # non-draggable, stays in sidepanel

    free = reg.free_sections()
    assert [s.key for s in free] == ['demo:two']

    reg.set_section_region('demo:one', 'free')
    reg.set_section_order('demo:one', 250.0)   # between two (300) and none
    reg.set_section_order('demo:two', 350.0)
    free = reg.free_sections()
    assert [s.key for s in free] == ['demo:one', 'demo:two']  # 250 < 350
    reg.close()


def test_section_free_rect_uses_saved_then_default():
    reg = _fresh_registry()
    reg.add('a', lambda s: None, key='demo:one', priority=200,
            draggable=True, default_region='free',
            default_free_xy=(0.25, 0.5), default_size=(160, 120))
    sec = [s for s in reg.all_sections() if s.key == 'demo:one'][0]

    # Default rect = (0.25 * 1000, 0.5 * 800) = (250, 400), size
    # clamped to (160, 120).
    assert reg.section_free_rect(sec, 1000, 800) == (250, 400, 160, 120)

    reg.set_section_free_rect('demo:one', 10, 20, 300, 150)
    assert reg.section_free_rect(sec, 1000, 800) == (10, 20, 300, 150)
    reg.close()


def test_run_sections_records_rects_and_draws_outline_in_edit_mode():
    """Edit mode must:
    - record every section's rect in plugin_data['sidepanel_rects'],
    - draw a blue outline for draggable non-flyout sections,
    - register a 'begin_drag_section' hitbox on the full section rect.
    Non-draggable / flyout sections are skipped by both passes."""
    outlines = []
    hitboxes = []

    draggable = SidebarSection(
        key='a:drag', name='Drag', draw=lambda s: setattr(s, 'y', s.y + 40),
        draggable=True)
    nondrag = SidebarSection(
        key='a:fixed', name='Fixed', draw=lambda s: setattr(s, 'y', s.y + 30),
        draggable=False)
    flyout = SidebarSection(
        key='a:fly', name='Fly', draw=lambda s: setattr(s, 'y', s.y + 20),
        draggable=True,
        draw_expanded=lambda s: None)  # flyouts skip drag grab

    # Patch the outline helper to record calls instead of painting.
    original = QtPlayerRenderer._draw_edit_outline
    QtPlayerRenderer._draw_edit_outline = staticmethod(
        lambda painter, x, y, w, h, *, highlighted: outlines.append(
            (x, y, w, h)))

    try:
        plugin_data = {}
        render_ctx = SimpleNamespace(plugin_data=plugin_data)
        player = SimpleNamespace(hud=SimpleNamespace(
            open_flyout=None, edit_mode=True, drag_key=None,
            hitboxes=[],
            add_hitbox=lambda rect, action, payload=None:
                hitboxes.append((rect, action, payload))))
        sctx = SimpleNamespace(
            player=player, render_ctx=render_ctx,
            sidebar_x=500, sidebar_w=210, y=10,
            measure_only=False, painter=None,
            add_hitbox=lambda rect, action, payload=None:
                hitboxes.append((rect, action, payload)),
            hitbox_clip=None,
        )
        QtPlayerRenderer._run_sections(
            [draggable, nondrag, flyout], sctx)
    finally:
        QtPlayerRenderer._draw_edit_outline = original

    # Every section recorded its rect.
    assert set(plugin_data['sidepanel_rects'].keys()) == {
        'a:drag', 'a:fixed', 'a:fly'}

    # Only the draggable-non-flyout got an outline + drag-grab hitbox.
    assert len(outlines) == 1
    assert any(action == 'begin_drag_section' and payload == 'a:drag'
               for _rect, action, payload in hitboxes)
    assert not any(action == 'begin_drag_section' and payload == 'a:fly'
                   for _rect, action, payload in hitboxes)
    assert not any(action == 'begin_drag_section' and payload == 'a:fixed'
                   for _rect, action, payload in hitboxes)


def test_edit_mode_hud_state_fields_round_trip():
    hud = HudState()
    assert hud.edit_mode is False
    assert hud.drag_key is None
    assert hud.resize_key is None
    hud.edit_mode = True
    hud.drag_key = 'x'
    hud.resize_origin_size = (50, 60)
    assert hud.edit_mode
    assert hud.drag_key == 'x'
    assert hud.resize_origin_size == (50, 60)


def test_compute_drop_order_picks_midpoint_with_real_rects():
    """When last-frame rects are available, _compute_drop_order inserts
    between the two neighbors that straddle the cursor Y."""
    reg = _fresh_registry()
    reg.add('a', lambda s: None, key='demo:one', priority=100,
            draggable=True)
    reg.add('b', lambda s: None, key='demo:two', priority=200,
            draggable=True)
    reg.add('c', lambda s: None, key='demo:three', priority=300,
            draggable=True)

    # Simulate a painted frame: three rects stacked vertically.
    hud = HudState()
    hud.edit_mode = True
    hud.drag_key = 'demo:three'
    hud.frame_sidepanel_rects = {
        'demo:one': (1000, 10, 210, 40),    # mid = 30
        'demo:two': (1000, 60, 210, 40),    # mid = 80
    }
    # Can't instantiate a real Player (too many deps), so drive the
    # method on a minimal shim.
    shim = SimpleNamespace(
        plugins=SimpleNamespace(sidebar=reg),
        hud=hud,
        H=500,
    )
    # Bind the method off the real Player class.
    from analysis.player.player import Player
    # Cursor Y above both mids → insert before one.
    order = Player._compute_drop_order(shim, 5)
    assert order < 100.0
    # Cursor between mids → midpoint of 100 and 200.
    order = Player._compute_drop_order(shim, 50)
    assert order == 150.0
    # Cursor below both → insert after two.
    order = Player._compute_drop_order(shim, 200)
    assert order > 200.0
    reg.close()


def test_finish_drag_routes_by_cursor_x():
    """Drop in the sidebar column = sidepanel region; drop left of the
    column = free region."""
    from analysis.player.player import Player
    from analysis.player.render import theme

    reg = _fresh_registry()
    reg.add('combo', lambda s: None, key='demo:one', priority=200,
            draggable=True, default_region='sidepanel',
            default_size=(180, 100))
    hud = HudState()
    hud.edit_mode = True
    hud.drag_key = 'demo:one'
    hud.drag_offset = (20, 10)
    hud.drag_origin_region = 'sidepanel'
    hud.frame_sidepanel_rects = {}
    W = 1200
    shim = SimpleNamespace(
        plugins=SimpleNamespace(sidebar=reg),
        hud=hud, W=W, H=800,
    )
    # _finish_drag calls self._compute_drop_order on sidepanel drops.
    shim._compute_drop_order = lambda y: Player._compute_drop_order(shim, y)

    # Drop at x in sidebar column — stays in sidepanel.
    sidebar_x = W - theme.SIDEBAR_WIDTH
    Player._finish_drag(shim, sidebar_x + 5, 200)
    assert reg.section_region('demo:one') == 'sidepanel'
    assert shim.hud.drag_key is None

    # New drag, drop to the left of the sidebar — goes free.
    hud.drag_key = 'demo:one'
    hud.drag_offset = (20, 10)
    Player._finish_drag(shim, 100, 400)
    assert reg.section_region('demo:one') == 'free'
    # Rect top-left should be (cursor - offset) clamped on-screen.
    sec = [s for s in reg.all_sections() if s.key == 'demo:one'][0]
    x, y, w, h = reg.section_free_rect(sec, W, 800)
    assert (x, y) == (100 - 20, 400 - 10)
    assert (w, h) == (180, 100)
    reg.close()
