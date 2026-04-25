"""Tests for the InputRouter and Region protocol.

The concrete ``SidebarRegion`` / ``LanesRegion`` pull in Qt + theme +
the full ``Player``; exercised here via a minimal fake player so the
tests stay fast and don't need a real replay. Qt is imported lazily in
the wheel handler for ``LanesRegion``, so the shift-modifier branch is
tested separately with a live QApplication.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from analysis.gui.region import (InputRouter, LanesRegion, Region,
                                  SidebarRegion)
from analysis.player.hud.hud_state import HudState


class _FakeRegion:
    def __init__(self, rect, wheel_ret=True, mouse_ret=True):
        self.rect = rect
        self.wheel_ret = wheel_ret
        self.mouse_ret = mouse_ret
        self.wheel_calls = []
        self.mouse_calls = []

    def contains(self, x, y):
        rx, ry, rw, rh = self.rect
        return rx <= x < rx + rw and ry <= y < ry + rh

    def on_wheel(self, x, y, dy, mods):
        self.wheel_calls.append((x, y, dy, mods))
        return self.wheel_ret

    def on_mouse_down(self, x, y, btn, mods):
        self.mouse_calls.append((x, y, btn, mods))
        return self.mouse_ret


# ─── Router ────────────────────────────────────────────────────────────────

def test_router_empty_returns_false():
    r = InputRouter()
    assert r.dispatch_wheel(10, 10, 120, 0) is False
    assert r.dispatch_mouse_down(10, 10, 1, 0) is False


def test_router_picks_containing_region():
    r = InputRouter()
    left = _FakeRegion((0, 0, 100, 100))
    right = _FakeRegion((100, 0, 100, 100))
    r.add(left)
    r.add(right)

    r.dispatch_wheel(50, 50, 120, 0)
    assert left.wheel_calls == [(50, 50, 120, 0)]
    assert right.wheel_calls == []

    r.dispatch_wheel(150, 50, -120, 0)
    assert left.wheel_calls == [(50, 50, 120, 0)]
    assert right.wheel_calls == [(150, 50, -120, 0)]


def test_router_first_match_wins_for_overlap():
    """Registration order determines priority ; overlays register first."""
    r = InputRouter()
    overlay = _FakeRegion((0, 0, 200, 200))
    base = _FakeRegion((0, 0, 200, 200))
    r.add(overlay)
    r.add(base)

    r.dispatch_mouse_down(10, 10, 1, 0)
    assert overlay.mouse_calls == [(10, 10, 1, 0)]
    assert base.mouse_calls == []


def test_router_returns_false_outside_any_region():
    r = InputRouter()
    r.add(_FakeRegion((0, 0, 10, 10)))
    assert r.dispatch_wheel(100, 100, 120, 0) is False


def test_router_region_without_hook_returns_false():
    """A region with ``contains`` but no ``on_wheel`` should not crash ;
    the router treats the missing hook as 'didn't handle'."""
    r = InputRouter()
    no_wheel = SimpleNamespace(contains=lambda x, y: True)
    r.add(no_wheel)
    assert r.dispatch_wheel(5, 5, 120, 0) is False


def test_router_handler_exception_returns_false():
    r = InputRouter()

    class Boom:
        def contains(self, x, y): return True

        def on_wheel(self, *a): raise RuntimeError('boom')

    r.add(Boom())
    assert r.dispatch_wheel(1, 1, 120, 0) is False


def test_router_remove():
    r = InputRouter()
    a = _FakeRegion((0, 0, 100, 100))
    r.add(a)
    assert r.remove(a) is True
    assert r.remove(a) is False
    assert r.dispatch_wheel(10, 10, 120, 0) is False


def test_router_region_at():
    r = InputRouter()
    a = _FakeRegion((0, 0, 100, 100))
    b = _FakeRegion((100, 0, 100, 100))
    r.add(a); r.add(b)
    assert r.region_at(10, 10) is a
    assert r.region_at(150, 10) is b
    assert r.region_at(500, 10) is None


def test_region_protocol_is_structural():
    """Any object with ``contains`` satisfies the protocol ; no base
    class required. Guards against accidental inheritance coupling."""
    obj = SimpleNamespace(contains=lambda x, y: True)
    assert isinstance(obj, Region)


# ─── SidebarRegion + LanesRegion ───────────────────────────────────────────

def _fake_player(w=1200, h=700):
    p = SimpleNamespace()
    p.W = w
    p.H = h
    p.hud = HudState()
    p.hud.sidebar_scroll_max = 500
    p.hud.sidebar_scroll = 100
    p._mouse_returns = True
    p.handle_mouse_down = lambda x, y: p._mouse_returns
    return p


def test_sidebar_region_contains_matches_theme_width():
    from analysis.player.render import theme
    p = _fake_player(w=1600, h=800)
    r = SidebarRegion(p)
    assert r.contains(1600 - theme.SIDEBAR_WIDTH + 5, 10)
    assert r.contains(1599, 799)
    assert not r.contains(1600 - theme.SIDEBAR_WIDTH - 1, 10)


def test_sidebar_region_wheel_scrolls_and_clamps():
    p = _fake_player()
    r = SidebarRegion(p)
    # Wheel up (positive dy) → scroll toward top.
    assert r.on_wheel(0, 0, 360, 0) is True
    assert p.hud.sidebar_scroll == 0  # 100 - 120 clamped at 0
    # Wheel down (negative dy) → scroll toward bottom.
    p.hud.sidebar_scroll = 100
    r.on_wheel(0, 0, -360, 0)
    assert p.hud.sidebar_scroll == 220


def test_sidebar_region_wheel_clamps_at_max():
    p = _fake_player()
    p.hud.sidebar_scroll = 490
    p.hud.sidebar_scroll_max = 500
    r = SidebarRegion(p)
    r.on_wheel(0, 0, -3600, 0)
    assert p.hud.sidebar_scroll == 500


def test_sidebar_region_mouse_delegates_to_player():
    p = _fake_player()
    r = SidebarRegion(p)
    p._mouse_returns = True
    assert r.on_mouse_down(10, 10, 1, 0) is True
    p._mouse_returns = False
    assert r.on_mouse_down(10, 10, 1, 0) is False


def test_lanes_region_contains_stops_at_sidebar():
    from analysis.player.render import theme
    p = _fake_player(w=1600, h=800)
    r = LanesRegion(p, seek_fn=lambda s: None)
    sidebar_start = 1600 - theme.SIDEBAR_WIDTH
    assert r.contains(0, 0)
    assert r.contains(sidebar_start - 1, 10)
    assert not r.contains(sidebar_start, 10)
    assert not r.contains(sidebar_start + 10, 10)


def test_lanes_region_wheel_seeks(_qapp):
    from PySide6.QtCore import Qt
    p = _fake_player()
    seeks = []
    r = LanesRegion(p, seek_fn=seeks.append)
    r.on_wheel(0, 0, 120, Qt.KeyboardModifier.NoModifier)
    assert seeks == [pytest.approx(0.5)]
    # Shift accelerates by 10x.
    r.on_wheel(0, 0, 120, Qt.ShiftModifier)
    assert seeks[-1] == pytest.approx(5.0)
