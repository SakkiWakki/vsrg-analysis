"""Declarative component tree for sidebar-style UIs.

Components are plain immutable values describing *what* to draw ; not
how. A renderer (see ``render_sidebar.py``) walks the tree and calls
the imperative primitives on ``SidebarContext`` to actually paint +
register hitboxes.

Why this shape:

  * Plugins return a ``Component`` from a ``build(ctx)`` function
    instead of calling imperative ``sctx.draw_*`` methods. This keeps
    the plugin code side-effect-free and trivially testable (you can
    snapshot the tree).
  * Immediate-mode: the tree is rebuilt every frame. No reconciliation,
    no component state, no lifecycle ; the surrounding replay state is
    the single source of truth. Svelte minus the reactivity.
  * Renderer-agnostic: ``Text``, ``Button``, etc. don't know about Qt
    or the painted HUD. A future Qt-widget renderer or a headless text
    dump can traverse the same tree.

The existing imperative ``SidebarContext`` API (``draw_button``,
``draw_text``, etc.) is **not** deprecated ; built-in sections may
still use it. Components are the recommended surface for new plugins.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union


# ─── Component types ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Text:
    """A line of text. ``color=None`` means "renderer chooses" (theme
    default). Use ``Text('')`` as a one-line blank."""
    text: str
    color: Optional[tuple] = None
    indent: int = 0


@dataclass(frozen=True)
class Heading:
    """A section heading (styled differently from ``Text``)."""
    text: str


@dataclass(frozen=True)
class Button:
    """An interactive row. ``action`` is the hitbox-action string the
    player will receive on click; ``payload`` is passed along. Use the
    existing action vocabulary (``toggle_plugin``, ``scroll_nudge``,
    etc.) or register a new one on ``Player.handle_mouse_down``."""
    label: str
    action: str
    payload: object = None
    # Optional per-row style overrides. None means "use theme defaults".
    bg: Optional[tuple] = None
    fg: Optional[tuple] = None


@dataclass(frozen=True)
class Checkbox:
    """A labelled checkbox row. Renders as ``[x] label`` / ``[ ] label``
    and registers a hitbox spanning the whole row."""
    label: str
    checked: bool
    action: str
    payload: object = None


@dataclass(frozen=True)
class Spacer:
    """Vertical gap. ``height=None`` uses the theme's default spacer."""
    height: Optional[int] = None


@dataclass(frozen=True)
class Box:
    """An explicit rectangle. Rare ; most layout comes from Column/Row.
    ``fill=None`` leaves the background; ``outline`` draws a border."""
    width: int
    height: int
    fill: Optional[tuple] = None
    outline: Optional[tuple] = None


@dataclass(frozen=True)
class Column:
    """Stack children top-to-bottom. This is the default layout for a
    sidebar section."""
    children: tuple = field(default_factory=tuple)

    def __post_init__(self):
        # Normalize list → tuple so the instance stays frozen-hashable.
        object.__setattr__(self, 'children', tuple(self.children))


@dataclass(frozen=True)
class Row:
    """Lay children out left-to-right across the available width.
    Children are given equal width by the sidebar renderer; non-layout
    children (``Text``/``Button``) render in their assigned slot."""
    children: tuple = field(default_factory=tuple)
    gap: int = 4

    def __post_init__(self):
        object.__setattr__(self, 'children', tuple(self.children))


# Union of everything the renderer needs to handle. Plugin code should
# import the individual names; this exists for internal type hints.
Component = Union[Text, Heading, Button, Checkbox, Spacer, Box, Column, Row]


# ─── Helpers ───────────────────────────────────────────────────────────────

def section(title: str, *children) -> Column:
    """Convenience: ``Column`` with a ``Heading`` prepended. Matches the
    common pattern of "titled group of controls" so plugins don't need
    to type ``Heading`` manually every time."""
    return Column((Heading(title), *children))


def iter_tree(component: Component):
    """Depth-first walk over a component and its children. Handy for
    tests and for renderers that want to know "does this tree contain
    any interactive components?"."""
    yield component
    children = getattr(component, 'children', ())
    for child in children:
        yield from iter_tree(child)


def collect_actions(component: Component) -> list[str]:
    """Return every ``Button``/``Checkbox`` action string in the tree.
    Used by tests; also useful for future tooling that wants to verify
    a plugin only emits known actions."""
    return [c.action for c in iter_tree(component)
            if isinstance(c, (Button, Checkbox))]
