"""Declarative UI primitives for plugins.

This is the blessed drawing surface for **plugins** — especially
sandboxed ones, which can't touch Qt directly. Plugins return a
``Component`` tree; the host renders it via
:mod:`analysis.ui.render_sidebar`.

Built-in sidebar sections deliberately *don't* use this — they draw
imperatively via ``SidebarContext`` primitives (``draw_button``,
``draw_text``, …). The component layer exists to give untrusted code a
safe declarative path, not to replace host-owned rendering. If you're
editing a built-in panel, stay with the imperative API.
"""
from analysis.ui.components import (Box, Button, Checkbox, Column, Component,
                                     Heading, Row, Spacer, Text)

__all__ = ['Box', 'Button', 'Checkbox', 'Column', 'Component', 'Heading',
           'Row', 'Spacer', 'Text']
