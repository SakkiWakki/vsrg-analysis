"""GUI surface backend for the unified component API.

Translates a plugin's calls against :class:`~analysis.components.api.Context`
into the existing ``SidebarContext`` primitives (QPainter + ``hud.add_hitbox``).
The component thinks in *local pixels* (``0..w`` x ``0..h``) from its own
top-left; this backend shifts those coords into the sidebar column and
advances the section's paint cursor to match.

Data: the GUI surface source of truth is the live ``Player`` instance.
:class:`PlayerDataSource` wraps it and answers the subset of
``GameState`` fields the player actually has. Methods not
answerable (e.g. live ``accuracy`` -- not meaningful mid-replay) raise
:class:`DataNotAvailable`.
"""
from __future__ import annotations

import numpy as np

from analysis.components.api import (
    GameState,
    HudFlags,
    DataNotAvailable,
    SURFACE_GUI,
)
from analysis.player.render import theme
from analysis.plugins.host_api import PluginConfig


# ── Data source ─────────────────────────────────────────────────────

class PlayerDataSource:
    """Implements :class:`GameState` against a live
    :class:`~analysis.player.player.Player`.

    Kept read-only on purpose. Components that want to *change* state
    (nudge a judge, toggle a setting) dispatch an action via the
    context's button helper; actions route through the player's
    ``handle_mouse_down`` hitbox table as before.
    """

    # The set of fields this source knows how to answer. Declared as a
    # class attribute so the registry's ``requires_data`` check can
    # inspect the set before we even instantiate.
    _FIELDS = frozenset({
        'game', 'keycount', 'combo', 'judgment_windows',
        'judgment_counts', 'judgment_colors', 'judge_label',
    })

    def __init__(self, player):
        self._p = player

    # ── Plumbing ──
    def supports(self, field: str) -> bool:
        return str(field) in self._FIELDS

    # ── Game identity ──
    def game(self) -> str:
        return str(self._p.game)

    def keycount(self) -> int:
        return int(getattr(self._p, 'keycount', 0))

    # ── Scoring ──
    def combo(self) -> int:
        # Replay-side: the player tracks a running "max combo" snapshot
        # by playhead; live ``combo`` (the current streak) is game-
        # specific and not always exposed. Fall back to 0 rather than
        # raising so the simple "print combo" case still works.
        return int(getattr(self._p, 'combo', 0))

    def accuracy(self) -> float:
        raise DataNotAvailable(
            'accuracy is not tracked mid-replay; use judgment_counts')

    # ── Judgments ──
    def judgment_windows(self) -> list[tuple[str, float]]:
        w = getattr(self._p, 'windows', None)
        if w is None:
            raise DataNotAvailable('judgment_windows')
        return [(str(n), float(ms)) for n, ms in w]

    def judgment_counts(self) -> dict[str, int]:
        windows = getattr(self._p, 'windows', None)
        note_judges = getattr(self._p, 'note_judges', None)
        if windows is None or note_judges is None:
            raise DataNotAvailable('judgment_counts')
        counts = {n: 0 for n, _ in windows}
        counts['miss'] = 0
        for j in note_judges:
            counts[j] = counts.get(j, 0) + 1
        return counts

    def judgment_colors(self) -> dict[str, tuple]:
        c = getattr(self._p, 'judge_colors', None)
        if c is None:
            raise DataNotAvailable('judgment_colors')
        return dict(c)

    def judge_label(self) -> str:
        lbl = getattr(self._p, 'judge_label', None)
        if lbl is None:
            raise DataNotAvailable('judge_label')
        return str(lbl)

    def game_memory(self):
        # GUI backend reads from the replay player, not from live memory.
        # Components that need live game state should target SURFACE_OVERLAY.
        return None

    # ── Playback state ──
    def t_now(self) -> float:
        t = getattr(self._p, '_render_t_now', None)
        if t is None:
            raise DataNotAvailable('t_now')
        return float(t)

    def play_rate(self) -> float:
        return float(getattr(self._p, 'play_rate', 1.0))

    def paused(self) -> bool:
        return bool(getattr(self._p, 'paused', True))

    def sv_enabled(self) -> bool:
        return bool(getattr(self._p, 'sv_enabled', False))

    def sv_suspended(self) -> bool:
        return bool(getattr(self._p, 'sv_suspended', lambda: False)())

    def skin(self) -> str:
        return str(getattr(self._p, 'skin', 'bar'))

    def press_hide(self) -> bool:
        return bool(getattr(self._p, 'press_hide', False))

    def scroll_mode(self) -> str:
        return str(getattr(self._p, 'scroll_mode', 'ms'))

    def scroll_value(self) -> float:
        try:
            return float(self._p._current_mode_value())
        except Exception:
            raise DataNotAvailable('scroll_value')

    def effective_scroll_ms(self) -> float:
        v = getattr(self._p, 'effective_scroll_ms', None)
        if v is None:
            raise DataNotAvailable('effective_scroll_ms')
        return float(v)

    def note_count(self) -> int:
        times = getattr(self._p, 'times', None)
        return int(len(times)) if times is not None else 0

    def sv_sections(self) -> list:
        return list(getattr(self._p, 'sv_sections', []))

    def layer_visible(self, layer: str) -> bool:
        from analysis.config import get_config
        return bool(get_config().get(f'player.layer_visibility.{layer}', True))


# Sanity check at import time: a component declaring this field really
# can assume the source will answer.
assert isinstance(PlayerDataSource.__mro__[0]._FIELDS, frozenset)


# ── Replay state ─────────────────────────────────────────────────────

class PlayerReplayState:
    """Implements ReplayState against a live Player instance.
    Raises DataNotAvailable when the player has no replay loaded."""

    def __init__(self, player):
        self._p = player

    def _require(self, attr):
        val = getattr(self._p, attr, None)
        if val is None:
            raise DataNotAvailable(f'replay.{attr}')
        return val

    def _clean_mask(self):
        misses = getattr(self._p, 'misses', None)
        if misses is None:
            raise DataNotAvailable('replay.misses')
        return ~misses

    def offsets(self) -> np.ndarray:
        return self._require('offsets')

    def offsets_clean(self) -> np.ndarray:
        return self._require('offsets')[self._clean_mask()]

    def columns(self) -> np.ndarray:
        return self._require('columns')

    def columns_clean(self) -> np.ndarray:
        return self._require('columns')[self._clean_mask()]

    def noterows(self) -> np.ndarray:
        return self._require('noterows') if hasattr(self._p, 'noterows') \
            else self._require('times')

    def noterows_clean(self) -> np.ndarray:
        return self.noterows()[self._clean_mask()]

    def misses(self) -> np.ndarray:
        return self._require('misses')

    def notetypes(self) -> np.ndarray:
        return self._require('notetypes')

    def keycount(self) -> int:
        return int(self._require('keycount'))

    def game(self) -> str:
        return str(getattr(self._p, 'game', 'unknown'))


# ── Data analysis utilities ───────────────────────────────────────────

class PlayerDataAnalysis:
    """Implements DataAnalysis using analysis/core/timing.py.
    Stateless -- all methods are pure functions over the provided arrays.
    The instance is cheap to construct; all heavy work is in the methods."""

    @staticmethod
    def default_hands(keycount):
        from analysis.core.timing import default_hands
        return default_hands(keycount)

    @staticmethod
    def hand_split(columns, offsets, left_cols, right_cols):
        from analysis.core.timing import hand_split
        return hand_split(columns, offsets, left_cols, right_cols)

    @staticmethod
    def per_column_stats(columns, offsets):
        from analysis.core.timing import per_column_stats
        return per_column_stats(columns, offsets)

    @staticmethod
    def timing_drift(noterows, offsets, columns, *,
                     n_segments=4, left_cols=(0, 1), right_cols=(2, 3)):
        from analysis.core.timing import timing_drift
        return timing_drift(noterows, offsets, columns,
                            n_segments=n_segments,
                            left_cols=left_cols, right_cols=right_cols)

    @staticmethod
    def rolling_stability(offsets, columns, *,
                          window=200, left_cols=(0, 1), right_cols=(2, 3)):
        from analysis.core.timing import rolling_stability
        return rolling_stability(offsets, columns,
                                 window=window,
                                 left_cols=left_cols, right_cols=right_cols)

    @staticmethod
    def coupling_analysis(noterows, offsets, columns, *,
                          left_cols=(0, 1), right_cols=(2, 3)):
        from analysis.core.timing import coupling_analysis
        return coupling_analysis(noterows, offsets, columns,
                                 left_cols=left_cols, right_cols=right_cols)

    @staticmethod
    def chord_vs_single(noterows, offsets, columns):
        from analysis.core.timing import chord_vs_single
        return chord_vs_single(noterows, offsets, columns)


_SHARED_ANALYSIS = PlayerDataAnalysis()


# ── Null config (used when no manifest key is provided) ──────────────

class _NullConfig:
    """Stub config used when a context has no manifest key. All reads
    return the default; writes are silently dropped."""
    def get(self, *args, **kwargs): return kwargs.get('default', args[1] if len(args) > 1 else None)
    def set(self, *_): return False
    def delete(self, *_): return False
    def subscribe(self, *_): return None
    def unsubscribe(self, *_): return False


# ── HUD flags ────────────────────────────────────────────────────────

def _hud_flags_from_player(player) -> HudFlags:
    hud = getattr(player, 'hud', None)
    if hud is None:
        return HudFlags(edit_mode=False, layers_panel_open=False,
                        plugin_panel_open=False, open_flyout=None)
    return HudFlags(
        edit_mode=bool(getattr(hud, 'edit_mode', False)),
        layers_panel_open=bool(getattr(hud, 'layers_panel_open', False)),
        plugin_panel_open=bool(getattr(hud, 'plugin_panel_open', False)),
        open_flyout=getattr(hud, 'open_flyout', None),
    )


# ── Context ─────────────────────────────────────────────────────────

_CHAR_PX = 6  # matches SidebarContext._CHAR_PX for label centering


class SidebarContext:
    """:class:`Context` implementation that defers to a
    wrapped :class:`~analysis.player.hud.sidebar_api.SidebarContext`.

    The key translation is coordinates: the component thinks in its own
    local pixel box (``0..w`` × ``0..h``); the backend shifts every
    primitive by ``(self._x0, self._y0)`` so it lands in the right
    column. The cursor ``self.y`` (local) is kept in sync with the
    underlying ``sctx.y`` (absolute) so cursor-advancing helpers work
    identically to sidebar-native code.
    """

    surface = SURFACE_GUI

    def __init__(self, sctx, *, x0: int, y0: int, w: int, h: int,
                 data_source: GameState, manifest_key: str = ''):
        self._sctx = sctx
        self._x0 = int(x0)
        self._y0 = int(y0)
        self.w = int(w)
        self.h = int(h)
        self.measure_only = bool(sctx.measure_only)
        self.data = data_source
        self.replay = PlayerReplayState(sctx.player)
        self.analysis = _SHARED_ANALYSIS
        self.hud_flags = _hud_flags_from_player(sctx.player)
        self.config = PluginConfig(manifest_key) if manifest_key else _NullConfig()
        # Local cursor starts at 0. We mirror sctx.y so advancing either
        # keeps them in lockstep.
        self.y = 0

    @property
    def supports_input(self) -> bool:
        return True

    # ── Absolute-coord helpers (private) ──
    def _ax(self, x: int) -> int:
        return self._x0 + int(x)

    def _ay(self, y: int) -> int:
        return self._y0 + int(y)

    def _sync_from_sctx(self) -> None:
        """Pull absolute cursor → local after an sctx-side advance."""
        self.y = int(self._sctx.y) - self._y0

    def _sync_to_sctx(self) -> None:
        """Push local cursor → absolute before calling an sctx row
        helper so it paints at the right Y."""
        self._sctx.y = self._y0 + int(self.y)

    # ── Geometry ──
    def split_row(self, n: int = 2, gap: int = 4) -> list[tuple[int, int]]:
        if n <= 0:
            return []
        total_gap = gap * (n - 1)
        slot_w = (self.w - total_gap) // n
        slots = []
        x = 0
        for i in range(n):
            slot = slot_w if i < n - 1 else self.w - x
            slots.append((x, slot))
            x += slot_w + gap
        return slots

    # ── Raw primitives ──
    def text(self, s, x, baseline, color=None) -> None:
        self._sctx.text(s, self._ax(x), self._ay(baseline),
                        color=color if color is not None else theme.BTN_FG)

    def rect(self, rect, color=None, outline=None, outline_w=1) -> None:
        rx, ry, rw, rh = rect
        self._sctx.rect((self._ax(rx), self._ay(ry), int(rw), int(rh)),
                        color, outline=outline, outline_w=outline_w)

    def line(self, start, end, color, width=1) -> None:
        sx, sy = start
        ex, ey = end
        self._sctx.line((self._ax(sx), self._ay(sy)),
                        (self._ax(ex), self._ay(ey)), color, width)

    # ── Cursor-advancing rows ──
    def spacer(self, h=None) -> None:
        step = int(h) if h is not None else int(theme.SECTION_SPACER)
        self.y += step
        self._sync_to_sctx()

    def draw_heading(self, text, color=None) -> None:
        self._sync_to_sctx()
        if color is None:
            color = theme.COLOR_HEADING
        if not self._sctx.measure_only:
            self._sctx.painter.setFont(self._sctx.renderer.big_font)
            self._sctx.text(text, self._ax(0), self._ay(self.y) + 18, color)
            self._sctx.painter.setFont(self._sctx.renderer.font)
        self.y += int(theme.HEADING_H)
        self._sync_to_sctx()

    def draw_text(self, text, color=None, indent=0, height=None) -> None:
        if color is None:
            color = theme.BTN_FG
        if height is None:
            height = theme.ROW_TEXT_H
        self.text(text, indent, self.y + theme.TEXT_BASELINE_ROW, color=color)
        self.y += int(height)
        self._sync_to_sctx()

    def draw_hint(self, text, color=None) -> None:
        self.draw_text(text,
                       color=theme.COLOR_HINT if color is None else color,
                       height=theme.ROW_HINT_H)

    # ── Interactive ──
    def draw_button(self, label, action, payload=None, *, enabled=True,
                    height=None, center=False):
        h = int(height) if height is not None else int(theme.ROW_BUTTON_H)
        rect_local = (0, self.y, self.w, h)
        self.button_at(rect_local, label, action, payload,
                       enabled=enabled, center=center)
        self.y += h
        self._sync_to_sctx()
        return rect_local

    def button_at(self, rect, label, action, payload=None, *,
                  enabled=True, center=False) -> None:
        rx, ry, rw, rh = rect
        abs_rect = (self._ax(rx), self._ay(ry), int(rw), int(rh))
        fill = theme.BTN_FILL if enabled else theme.BTN_FILL_DISABLED
        fg = theme.BTN_FG if enabled else theme.BTN_FG_DISABLED
        self._sctx.rect(abs_rect, fill, outline=theme.BTN_BORDER)
        if center:
            tx = abs_rect[0] + max(
                0, (int(rw) - len(str(label)) * _CHAR_PX) // 2)
        else:
            tx = abs_rect[0] + theme.TEXT_INDENT
        self._sctx.text(label, tx,
                        abs_rect[1] + theme.TEXT_BASELINE_BUTTON, fg)
        if enabled:
            self._sctx.add_hitbox(abs_rect, action, payload)

    def checkbox(self, x, y, checked) -> tuple:
        size = int(theme.CHECKBOX_SIZE)
        box_local = (int(x), int(y), size, size)
        self.rect(box_local, theme.COLOR_CHECKBOX_FILL,
                  outline=theme.COLOR_CHECKBOX_BORDER)
        if checked:
            self.line((int(x) + 2, int(y) + 5),
                      (int(x) + 4, int(y) + 8),
                      theme.COLOR_CHECKBOX_MARK, 2)
            self.line((int(x) + 4, int(y) + 8),
                      (int(x) + 9, int(y) + 2),
                      theme.COLOR_CHECKBOX_MARK, 2)
        return box_local


# ── Driver ─────────────────────────────────────────────────────────

def draw_component_in_sidebar(component, sctx, *, player) -> None:
    """Run a unified-API component inside a sidebar draw pass.

    Shapes the component context (local coords, data source, bounds)
    around ``sctx`` and hands it to ``component.draw``. After the call
    we copy the component's local cursor advance back onto ``sctx.y``
    so the sidebar's outer layout picks up the consumed vertical space.
    """
    data_source = PlayerDataSource(player)
    # The component occupies the full sidebar column from the current
    # cursor downward. Height 0 means "grow as the draw function
    # advances the cursor."
    x0 = sctx.col_x
    y0 = sctx.y
    w = sctx.col_w
    cctx = SidebarContext(
        sctx, x0=x0, y0=y0, w=w, h=0, data_source=data_source)
    component.draw(cctx)
    # Ensure the outer sidebar cursor reflects whatever the component
    # consumed. ``_sync_to_sctx`` already runs on every row helper,
    # but manual ``y +=`` updates inside the component would not
    # propagate otherwise.
    cctx._sync_to_sctx()
