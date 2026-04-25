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
    :class:`~analysis.player.player_api.Player`.

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
        't_now', 'play_rate', 'paused', 'note_count',
        'sv_enabled', 'sv_suspended', 'sv_sections',
        'skin', 'press_hide', 'scroll_mode', 'scroll_value',
        'effective_scroll_ms', 'layer_visible', 'layer_tree',
        'audio_status',
        # Chart + play snapshots (all surfaces with a loaded replay)
        'chart_metadata', 'chart_stats', 'chart_paths',
        'player_name', 'score', 'max_combo', 'current_grade',
        'mods_short', 'mods_raw', 'play_rate_effective',
        'hit_errors_ms', 'unstable_rate',
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

    # ── Chart snapshots ──

    def chart_metadata(self):
        from analysis.components.api import ChartMetadata
        rep = getattr(self._p, 'replay', None)
        if not isinstance(rep, dict):
            return ChartMetadata()
        cm = rep.get('chart_meta') or {}
        meta = rep.get('meta') or {}
        return ChartMetadata(
            artist=str(cm.get('artist') or ''),
            artist_unicode=str(cm.get('artist_unicode') or cm.get('artist') or ''),
            title=str(cm.get('title') or ''),
            title_unicode=str(cm.get('title_unicode') or cm.get('title') or ''),
            creator=str(cm.get('creator') or ''),
            version=str(cm.get('version') or ''),
            md5=str(rep.get('beatmap_hash') or meta.get('beatmap_hash') or ''),
            beatmap_id=int(cm.get('beatmap_id') or 0),
            beatmap_set_id=int(cm.get('beatmap_set_id') or 0),
            source=str(cm.get('source') or ''),
            tags=str(cm.get('tags') or ''),
        )

    def chart_stats(self):
        from analysis.components.api import ChartStats
        rep = getattr(self._p, 'replay', None)
        times = getattr(self._p, 'times', None)
        first_ms = int(times[0] * 1000) if times is not None and len(times) else 0
        last_ms = int(times[-1] * 1000) if times is not None and len(times) else 0
        holds = rep.get('holds') if isinstance(rep, dict) else None
        hold_count = int(len(holds)) if holds is not None else 0
        total_objects = int(len(times)) if times is not None else 0

        # Let the adapter fill in its own difficulty scalar and extras.
        adapter = self._adapter()
        if adapter is not None and hasattr(adapter, 'chart_stats_extra'):
            try:
                difficulty, rating, extra = adapter.chart_stats_extra(rep)
            except Exception:
                difficulty, rating, extra = 0.0, 0.0, {}
        else:
            # Fallback: osu stores 'od' at top level of the replay dict.
            difficulty = (float(rep.get('od', 0.0))
                          if isinstance(rep, dict) else 0.0)
            rating = 0.0
            extra = {}

        bpm = self._bpm_common()
        return ChartStats(
            mode_name=str(getattr(self._p, 'game', '') or ''),
            difficulty=float(difficulty),
            rating=float(rating),
            bpm_common=bpm, bpm_min=bpm, bpm_max=bpm,
            length_ms=max(0, last_ms - first_ms),
            first_object_ms=first_ms,
            last_object_ms=last_ms,
            total_objects=total_objects,
            hold_count=hold_count,
            max_combo=total_objects + hold_count,
            extra=dict(extra),
        )

    def chart_paths(self):
        from analysis.components.api import ChartPaths
        import os
        rep = getattr(self._p, 'replay', None)
        if not isinstance(rep, dict):
            return ChartPaths()
        chart_path = rep.get('chart_path') or ''
        chart_folder = (os.path.basename(os.path.dirname(chart_path))
                        if chart_path else '')
        library_root = (os.path.dirname(os.path.dirname(chart_path))
                        if chart_path else '')
        cm = rep.get('chart_meta') or {}
        return ChartPaths(
            chart_folder=str(chart_folder),
            audio_filename=str(cm.get('audio') or ''),
            background_filename=str(cm.get('background') or ''),
            skin_folder=str(getattr(self._p, 'skin', '') or ''),
            library_root=str(library_root),
        )

    def _bpm_common(self) -> float:
        ref = getattr(self._p, '_xmod_reference_bpm', None)
        if ref is not None:
            return float(ref)
        rep = getattr(self._p, 'replay', None)
        bpms = rep.get('bpms') if isinstance(rep, dict) else None
        if bpms:
            try:
                return float(bpms[0][1])
            except (IndexError, TypeError, ValueError):
                pass
        return 0.0

    # ── Play identity / state ──

    def player_name(self) -> str:
        rep = getattr(self._p, 'replay', None)
        if not isinstance(rep, dict):
            return ''
        return str(rep.get('player') or (rep.get('meta') or {}).get('player') or '')

    def score(self) -> int:
        rep = getattr(self._p, 'replay', None)
        if not isinstance(rep, dict):
            return 0
        return int(rep.get('score') or (rep.get('meta') or {}).get('score') or 0)

    def max_combo(self) -> int:
        # Running max combo from note_judges up to the current frame.
        note_judges = getattr(self._p, 'note_judges', None)
        if not note_judges:
            return 0
        cur = best = 0
        for j in note_judges:
            if j == 'miss':
                cur = 0
            else:
                cur += 1
                if cur > best:
                    best = cur
        return best

    def current_grade(self) -> str:
        try:
            counts = self.judgment_counts()
            windows = self.judgment_windows()
        except DataNotAvailable:
            return ''
        total = sum(counts.values()) if counts else 0
        if total == 0:
            return ''
        names = [n for n, _ in windows]
        n = len(names)
        # osu!mania: weight best=n, worst=1, miss=0
        weighted = sum((n - i) * counts.get(nm, 0) for i, nm in enumerate(names))
        max_weighted = total * n
        acc = weighted / max_weighted if max_weighted else 0.0
        misses = counts.get('miss', 0)
        if acc >= 1.0:
            return 'X'
        if misses == 0 and acc >= 0.95:
            return 'S'
        if acc >= 0.95:
            return 'A'
        if acc >= 0.90:
            return 'B'
        if acc >= 0.80:
            return 'C'
        if acc > 0.0:
            return 'D'
        return 'F'

    def mods_short(self) -> str:
        adapter = self._adapter()
        if adapter is not None and hasattr(adapter, 'mods_short'):
            try:
                return str(adapter.mods_short(self._p.replay) or '')
            except Exception:
                pass
        return ''

    def mods_raw(self) -> dict:
        adapter = self._adapter()
        if adapter is not None and hasattr(adapter, 'mods_raw'):
            try:
                return dict(adapter.mods_raw(self._p.replay) or {})
            except Exception:
                pass
        return {}

    def play_rate_effective(self) -> float:
        base = float(getattr(self._p, 'play_rate', 1.0))
        adapter = self._adapter()
        if adapter is not None and hasattr(adapter, 'mods_rate_multiplier'):
            try:
                base *= float(adapter.mods_rate_multiplier(self._p.replay))
            except Exception:
                pass
        return base

    def _adapter(self):
        """Return the game adapter, or None if not resolvable. Cached."""
        adapter = getattr(self._p, '_adapter', None)
        if adapter is not None:
            return adapter
        try:
            from analysis.core import game as game_mod
            return game_mod.get(getattr(self._p, 'game', ''))
        except Exception:
            return None

    def hit_errors_ms(self) -> tuple[int, ...]:
        offsets = getattr(self._p, 'offsets', None)
        misses = getattr(self._p, 'misses', None)
        if offsets is None:
            return ()
        if misses is not None:
            offsets = offsets[~misses]
        return tuple(int(round(float(v) * 1000)) for v in offsets)

    def unstable_rate(self) -> float:
        errs = self.hit_errors_ms()
        n = len(errs)
        if n < 2:
            return 0.0
        mean = sum(errs) / n
        var = sum((x - mean) ** 2 for x in errs) / n
        return 10.0 * (var ** 0.5)

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

    def audio_status(self) -> tuple[int, str]:
        # Stubs / non-AudioEngine surfaces (e.g. tests with a bare player)
        # don't carry the snapshot getter; default to "no events seen".
        snap = getattr(self._p, 'audio_status_snapshot', None)
        if snap is None:
            return 0, ''
        try:
            count, last = snap()
            return int(count), str(last)
        except Exception:
            return 0, ''

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
        registry = getattr(getattr(self._p, 'plugins', None), 'layers', None)
        if registry is not None:
            return registry.layer_visible(layer)
        from analysis.config import get_config
        return bool(get_config().get(f'player.layer_visibility.{layer}', True))

    def layer_tree(self):
        registry = getattr(getattr(self._p, 'plugins', None), 'layers', None)
        if registry is None:
            return ()
        return registry.layer_tree()


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

    def __init__(self):
        from analysis.core.timing import (
            chord_vs_single,
            coupling_analysis,
            default_hands,
            hand_split,
            per_column_stats,
            rolling_stability,
            timing_drift,
        )
        self._chord_vs_single = chord_vs_single
        self._coupling_analysis = coupling_analysis
        self._default_hands = default_hands
        self._hand_split = hand_split
        self._per_column_stats = per_column_stats
        self._rolling_stability = rolling_stability
        self._timing_drift = timing_drift

    def default_hands(self, keycount):
        return self._default_hands(keycount)

    def hand_split(self, columns, offsets, left_cols, right_cols):
        return self._hand_split(columns, offsets, left_cols, right_cols)

    def per_column_stats(self, columns, offsets):
        return self._per_column_stats(columns, offsets)

    def timing_drift(self, noterows, offsets, columns, *,
                     n_segments=4, left_cols=(0, 1), right_cols=(2, 3)):
        return self._timing_drift(
            noterows,
            offsets,
            columns,
            n_segments=n_segments,
            left_cols=left_cols,
            right_cols=right_cols,
        )

    def rolling_stability(self, offsets, columns, *,
                          window=200, left_cols=(0, 1), right_cols=(2, 3)):
        return self._rolling_stability(
            offsets,
            columns,
            window=window,
            left_cols=left_cols,
            right_cols=right_cols,
        )

    def coupling_analysis(self, noterows, offsets, columns, *,
                          left_cols=(0, 1), right_cols=(2, 3)):
        return self._coupling_analysis(
            noterows,
            offsets,
            columns,
            left_cols=left_cols,
            right_cols=right_cols,
        )

    def chord_vs_single(self, noterows, offsets, columns):
        return self._chord_vs_single(noterows, offsets, columns)


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


def _pixmap_from(frame_or_pixmap):
    """Resolve a ctx.image() argument to a ``QPixmap``.

    Accepts either a bare ``QPixmap`` or a ``WebTextureFrame``. Other
    frame kinds return None so the caller skips. A future GL-backed
    frame may stash a downgrade copy under ``meta['qpixmap_fallback']``
    for hosts that can't sample GL; we honor that if present.
    """
    from PySide6.QtGui import QPixmap
    if isinstance(frame_or_pixmap, QPixmap):
        return frame_or_pixmap
    kind = getattr(frame_or_pixmap, 'kind', None)
    handle = getattr(frame_or_pixmap, 'handle', None)
    if kind == 'qpixmap' and isinstance(handle, QPixmap):
        return handle
    meta = getattr(frame_or_pixmap, 'meta', None)
    if isinstance(meta, dict):
        fallback = meta.get('qpixmap_fallback')
        if isinstance(fallback, QPixmap):
            return fallback
    return None


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

    def image(self, rect, frame) -> None:
        """Blit a WebTextureFrame (or raw QPixmap) into ``rect``.

        Measure passes skip the actual draw. Frame kinds this backend
        can handle directly: ``qpixmap``. Other kinds are downgraded
        via their ``handle`` if it happens to be a QPixmap; otherwise
        we silently skip so a missing frame never aborts the paint pass.
        """
        if self._sctx.measure_only:
            return

        pix = _pixmap_from(frame)
        if pix is None or pix.isNull():
            return

        from PySide6.QtCore import QRect
        rx, ry, rw, rh = rect
        dest = QRect(self._ax(rx), self._ay(ry), int(rw), int(rh))
        self._sctx.painter.drawPixmap(dest, pix)

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
