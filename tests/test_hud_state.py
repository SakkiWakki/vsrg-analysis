"""Tests for the HudState container."""
from __future__ import annotations

from analysis.player.hud_state import HudState


def test_defaults():
    h = HudState()
    assert h.sidebar_scroll == 0
    assert h.sidebar_scroll_max == 0
    assert h.plugin_panel_open is False
    assert h.hitboxes == []


def test_add_hitbox_tuple_normalization():
    """Rects may arrive as lists or tuples; hitbox entries should always
    be tuples so downstream unpacking is consistent."""
    h = HudState()
    h.add_hitbox([1, 2, 3, 4], 'toggle_plugin', 'my_key')
    assert h.hitboxes == [((1, 2, 3, 4), 'toggle_plugin', 'my_key')]


def test_add_hitbox_default_payload():
    h = HudState()
    h.add_hitbox((0, 0, 10, 10), 'toggle_plugin_panel')
    rect, action, payload = h.hitboxes[0]
    assert action == 'toggle_plugin_panel'
    assert payload is None


def test_clear_hitboxes():
    h = HudState()
    h.add_hitbox((0, 0, 10, 10), 'a')
    h.add_hitbox((0, 0, 10, 10), 'b')
    assert len(h.hitboxes) == 2
    h.clear_hitboxes()
    assert h.hitboxes == []


def test_hitboxes_per_instance():
    """Independent instances must not share the default list — a classic
    mutable-default pitfall guard."""
    a = HudState()
    b = HudState()
    a.add_hitbox((0, 0, 1, 1), 'x')
    assert b.hitboxes == []


def test_state_fields_are_mutable_scalars():
    h = HudState()
    h.sidebar_scroll = 50
    h.sidebar_scroll_max = 200
    h.plugin_panel_open = True
    assert (h.sidebar_scroll, h.sidebar_scroll_max, h.plugin_panel_open) \
        == (50, 200, True)
