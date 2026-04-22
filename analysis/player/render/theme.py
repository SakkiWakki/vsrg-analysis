"""Global UI design tokens for the replay player.

Acts like a tiny Tailwind config: every color, padding, and row height used
by the sidebar / HUD / plugin UI resolves through this module, so layout
is consistent and tunable from one place. Callers should import token
names normally (``from analysis.player.render import theme`` then ``theme.BTN_FG``)
— attribute access transparently falls through to the *active* theme,
letting bundle-provided themes override tokens at runtime.

The defaults below are also the "Built-in" theme. A plugin bundle exposes
its own theme by shipping a ``theme/__init__.py`` (or ``theme.py``) that
defines the tokens it wants to override; missing tokens fall through to
these defaults.

Only one theme is active at a time (see ``set_active`` /
``active_theme``). Layered overrides are a possible future extension.
"""
from __future__ import annotations

import sys
from types import ModuleType


# ─── Sidebar geometry ──────────────────────────────────────────────────────
SIDEBAR_WIDTH = 210
SIDEBAR_INSET = 8
SIDEBAR_TOP = 14
SIDEBAR_BOTTOM_MARGIN = 12
SIDEBAR_BG = (20, 20, 22)

# ─── Flyout (expanded sidebar-section panel) ───────────────────────────────
FLYOUT_WIDTH = 220
FLYOUT_GAP = 6              # gap between flyout and sidebar edge
FLYOUT_BG = (26, 26, 30)
FLYOUT_BORDER = (68, 68, 76)
FLYOUT_INSET = 8

# ─── Sidebar divider (page-break style, used above pinned sections) ────────
DIVIDER_COLOR = (90, 90, 96)
DIVIDER_WIDTH_FRAC = 0.45   # fraction of the sidebar column
DIVIDER_MARGIN_Y = 6        # vertical breathing room above+below
DIVIDER_THICKNESS = 3       # stroke width of the page-break divider

# ─── Row heights ───────────────────────────────────────────────────────────
ROW_BUTTON_H = 20
ROW_TEXT_H = 18
ROW_HINT_H = 16
ROW_TALL_H = 24
HEADING_H = 26
SECTION_SPACER = 12

# ─── Button palette ────────────────────────────────────────────────────────
BTN_FILL = (32, 32, 36)
BTN_FILL_DISABLED = (24, 24, 26)
BTN_BORDER = (68, 68, 76)
BTN_FG = (220, 220, 220)
BTN_FG_DISABLED = (110, 110, 116)

# ─── Text ──────────────────────────────────────────────────────────────────
TEXT_INDENT = 8
TEXT_BASELINE_BUTTON = 14
TEXT_BASELINE_ROW = 13

# ─── Semantic colors ───────────────────────────────────────────────────────
COLOR_HEADING = (255, 171, 145)
COLOR_HINT = (120, 120, 130)
CHECKBOX_SIZE = 10
COLOR_CHECKBOX_FILL = (16, 16, 18)
COLOR_CHECKBOX_BORDER = (110, 110, 120)
COLOR_CHECKBOX_MARK = (160, 230, 160)
COLOR_PLUGIN_ENABLED = (210, 210, 215)
COLOR_PLUGIN_DISABLED = (110, 110, 116)


_BUILTIN_THEME_NAME = 'builtin'
_active_theme: ModuleType | None = None
_active_theme_name: str = _BUILTIN_THEME_NAME

# Tokens exported at module level; the proxy below looks these up in the
# active theme first, then falls back to the defaults here.
_DEFAULT_TOKENS = {k: v for k, v in dict(globals()).items()
                   if k.isupper()}


def set_active(theme_module: ModuleType | None, name: str | None = None):
    """Activate a theme. Pass ``None`` to revert to the built-in defaults."""
    global _active_theme, _active_theme_name
    _active_theme = theme_module
    _active_theme_name = str(name or _BUILTIN_THEME_NAME)


def active_theme():
    return _active_theme


def active_theme_name():
    return _active_theme_name


class _ThemeModule(ModuleType):
    """Module subclass that resolves token lookups against the active
    theme, falling back to this module's own defaults."""

    def __getattr__(self, name):
        if name.startswith('_') or name.islower():
            raise AttributeError(name)
        if _active_theme is not None:
            try:
                return getattr(_active_theme, name)
            except AttributeError:
                pass
        if name in _DEFAULT_TOKENS:
            return _DEFAULT_TOKENS[name]
        raise AttributeError(name)


sys.modules[__name__].__class__ = _ThemeModule
