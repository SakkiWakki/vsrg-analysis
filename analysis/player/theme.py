"""Global UI design tokens for the replay player.

Acts like a tiny Tailwind config: every color, padding, and row height used
by the sidebar / HUD / plugin UI comes from here so layout is consistent
and tunable from one file. Plugin authors can import these names to match
the built-in look, or override them locally.

Grouped by domain:
  * SIDEBAR_*  — the right-hand HUD column
  * ROW_*      — shared row heights
  * BTN_*      — bordered-button palette
  * TEXT_*     — text baselines / indents
  * COLOR_*    — semantic palette entries used across sections
"""
from __future__ import annotations

# ─── Sidebar geometry ──────────────────────────────────────────────────────
SIDEBAR_WIDTH = 210
SIDEBAR_INSET = 8          # horizontal padding from the sidebar background
SIDEBAR_TOP = 14           # y where the first section starts
SIDEBAR_BOTTOM_MARGIN = 12  # gap between the last bottom-pinned row and p.H
SIDEBAR_BG = (20, 20, 22)

# ─── Row heights ───────────────────────────────────────────────────────────
ROW_BUTTON_H = 20
ROW_TEXT_H = 18
ROW_HINT_H = 16
ROW_TALL_H = 24   # scroll/rate/game rows that want more breathing room
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
TEXT_BASELINE_BUTTON = 14   # y-offset inside a ROW_BUTTON_H rect
TEXT_BASELINE_ROW = 13      # y-offset for unboxed text rows

# ─── Semantic colors ───────────────────────────────────────────────────────
COLOR_HEADING = (255, 171, 145)
COLOR_HINT = (120, 120, 130)
CHECKBOX_SIZE = 10
COLOR_CHECKBOX_FILL = (16, 16, 18)
COLOR_CHECKBOX_BORDER = (110, 110, 120)
COLOR_CHECKBOX_MARK = (160, 230, 160)
COLOR_PLUGIN_ENABLED = (210, 210, 215)
COLOR_PLUGIN_DISABLED = (110, 110, 116)
