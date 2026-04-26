"""Native Qt renderer for the embedded replay player."""
from __future__ import annotations

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPen


from analysis.player.render import culling, theme
from analysis.player.hud.sidebar_api import SidebarContext
from analysis.player.plugin.plugin_api import Stage
from analysis.player.render.render_context import RenderContext
# Re-export primitives so sidebar plugins / tests can keep importing them
# from this module. Implementations live in primitives.py; they're only
# re-exposed here to avoid churn in plugin code.
from analysis.player.render.primitives import (  # noqa: F401
    _qcolor, _qpen, _qbrush, _NO_PEN, _NO_BRUSH,
    _line, _rect, _rect_outline, _ellipse, _ellipse_outline, _text,
    _fmt_num, _rect_tuple,
)
from analysis.player.render.layers import field as _field_layer
from analysis.player.render.layers import notes as _notes_layer
# Re-export so tests / external code can keep importing _NoteView from
# this module; the implementation lives in layers/notes.py.
from analysis.player.render.layers.notes import _NoteView  # noqa: F401


def _precompute_candidate_ys(ctx) -> None:
    """Batch head + LN-tail + press-y for every candidate into parallel
    arrays aligned to `ctx.candidates`. `_build_note_view` and the
    press-mark drawer read from these by position, so the per-note loop
    never calls into the SV engine.

    On a dense Etterna chart with 60 visible notes and ~40% LN density
    that replaces ~3 * 60 = 180 Python->SV-engine bisects per frame
    with three numpy+searchsorted passes."""
    p = ctx.player
    cand = ctx.candidates
    if not cand:
        ctx.candidate_head_y = np.empty(0, dtype=np.float64)
        ctx.candidate_tail_y = np.empty(0, dtype=np.float64)
        ctx.candidate_press_y = np.empty(0, dtype=np.float64)
        return
    idx = np.asarray(cand, dtype=np.int64)
    head_times = p.times[idx]
    # Per-note groups: parallel to `p.times`, populated by Quaver
    # replays only. Pass through to the SV engine so each note's
    # `TimingGroup` selects the right SV stream. Other games leave the
    # attr unset and the engine ignores it.
    note_groups = getattr(p, '_note_sv_groups', None)
    cand_groups = note_groups[idx] if note_groups is not None else None
    ctx.candidate_head_y = p.batch_time_to_y(head_times, ctx.frame,
                                              groups=cand_groups)

    # LN tails only ; for non-LN candidates the cached tail array holds
    # NaN, which the batched path happily propagates; `_build_note_view`
    # only reads tail_y when is_ln is True so those entries are ignored.
    tail_times = p.notes.ln_tail_times[idx]
    ctx.candidate_tail_y = p.batch_time_to_y(tail_times, ctx.frame,
                                              groups=cand_groups)

    # Press-time y for the press-mark drawer. press_t = note_t + offset;
    # offsets is parallel to p.times so we can index it the same way.
    # Missed-and-not-pressed notes still go through this batch -- their
    # press_y is unused by the drawer (the early-out at the top of
    # `_draw_press_mark` skips them) but keeping the batch dense is much
    # faster than building a sub-array.
    press_times = head_times + p.offsets[idx]
    ctx.candidate_press_y = p.batch_time_to_y(press_times, ctx.frame,
                                               groups=cand_groups)


class QtPlayerRenderer:
    def __init__(self, plugin_manager):
        self.plugins = plugin_manager
        self.font = QFont('monospace', 10)
        self.big_font = QFont('monospace', 14)
        self.big_font.setBold(True)

    def build_context(self, player, painter, t_now):
        player._render_t_now = float(t_now)
        x0, lane_w = player._lane_geom()
        ctx = RenderContext(
            player=player,
            colors=player.judge_colors,
            t_now=float(t_now),
            x0=x0,
            lane_w=lane_w,
            judge_y=int(player.H * player.hit_line_y_frac),
            painter=painter,
            _scroll_speed=float(player.scroll_speed),
        )
        # Bind the per-replay sprite cache and invalidate any pixmaps
        # that were rasterized at an old geometry. Draw sites pull
        # rasterized pixmaps via `ctx.sprite_cache.get(name, ctx, ...)`.
        ctx.sprite_cache = player._sprite_cache
        ctx.sprite_cache.check_geometry(ctx.lane_w, ctx.note_h)
        culling.prepare_time_window(ctx)
        ctx.candidates = culling.select_note_candidates(ctx)
        # Precompute Y positions for every candidate's head + LN tail in one
        # numpy pass. Saves N*(2..4) scalar `ctx.time_to_y` calls per frame ;
        # on dense Etterna charts that's ~600 Python→SV-engine bisects per
        # frame collapsed to two numpy operations.
        _precompute_candidate_ys(ctx)
        # Build every per-candidate `_NoteView` once, up front. The
        # split `taps` / `lns` layers read from the same list so we
        # don't pay two separate prepasses.
        _notes_layer.prepare(ctx)
        player.debug_log_sv_frame(ctx)
        return ctx

    @property
    def _layers(self):
        registry = getattr(self.plugins, 'layers', None)
        if registry is None:
            # Fallback used only when no plugin manager is attached
            # (unit tests). Real replays register their NoteTypes via
            # `LayerRegistry.register_note_types`, so note-type leaves
            # don't need entries here.
            return (
                ('background',    _field_layer.draw_background, None),
                ('lanes',         _field_layer.draw_lanes,      Stage.AFTER_LANES),
                ('judgment',      _field_layer.draw_judgment,   Stage.AFTER_JUDGMENT),
                ('free_sections', self._draw_free_sections, None),
                ('hud',           self._draw_hud,          Stage.HUD),
            )
        return registry.render_plan(self._layer_draw_fns())

    def draw(self, player, painter, t_now):
        ctx = self.build_context(player, painter, t_now)
        # Hitboxes are frame-scoped ; clear them up front so the free
        # region (painted *before* the sidebar) can register its buttons
        # + drag handles without the sidebar pass wiping them later.
        hud = getattr(player, 'hud', None) if player is not None else None
        if hud is not None:
            hud.clear_hitboxes()
        self.plugins.draw(Stage.PRE_FRAME, ctx)
        visibility = self._layer_visibility(ctx)
        for name, fn, stage in self._layers:
            if visibility.get(name, True):
                if fn is not None:
                    fn(ctx, painter)
            if stage is not None:
                self.plugins.draw(stage, ctx)
        # Drag affordances: ghost + blue insertion line. Drawn last so
        # they sit above both the HUD and the free-region panels.
        if hud is not None and hud.edit_mode and hud.drag_key is not None:
            self._draw_drag_overlay(ctx, painter)
        self.plugins.draw(Stage.POST_FRAME, ctx)
        # Record per-frame metrics if profiling is enabled. Cheap when
        # disabled (single attribute check + early return).
        try:
            from analysis.gui import paint_profiler
            paint_profiler.record_frame(ctx)
        except ImportError:
            pass

    def _layer_draw_fns(self):
        # Only layers whose `draw` is a string lookup (plugin manifest
        # layers) need entries here. Note-type layers carry their draw
        # callable directly on the Layer and skip this mapping.
        return {
            'background': _field_layer.draw_background,
            'lanes': _field_layer.draw_lanes,
            'judgment': _field_layer.draw_judgment,
            'free_sections': self._draw_free_sections,
            'hud': self._draw_hud,
        }

    def _layer_visibility(self, ctx):
        tree = ctx.plugin_data.get('layer_visibility_tree')
        if tree is not None:
            out = {}
            self._flatten_layer_tree(tree, out)
            return out
        return ctx.plugin_data.get('layer_visibility') or {}

    @staticmethod
    def _flatten_layer_tree(states, out):
        for state in states:
            out[state.key] = state.visible
            QtPlayerRenderer._flatten_layer_tree(state.children, out)


    def _draw_hud(self, ctx, painter):
        p = ctx.player
        sidebar_x = p.W - theme.SIDEBAR_WIDTH
        # If the open flyout's section has been unregistered/disabled (e.g.
        # plugin reload), close it so a stale key doesn't paint an empty
        # panel.
        if p.hud.open_flyout is not None:
            if self.plugins.sidebar.flyout_section(p.hud.open_flyout) is None:
                p.hud.open_flyout = None
        _rect(painter, theme.SIDEBAR_BG,
              (sidebar_x, 0, theme.SIDEBAR_WIDTH, p.H))
        painter.setFont(self.font)

        top = self.plugins.sidebar.top_sections()
        bottom = self.plugins.sidebar.bottom_sections()

        # Measure bottom first ; top viewport ends where bottom starts, so
        # we need bottom's total height to figure out the top viewport's
        # max y and thus the clamp for the top scroll offset.
        bottom_h = 0
        if bottom:
            measure_bot = SidebarContext(ctx, painter, self, sidebar_x,
                                         theme.SIDEBAR_WIDTH, 0,
                                         measure_only=True)
            self._run_sections(bottom, measure_bot)
            bottom_h = measure_bot.y

        bottom_start_y = (p.H - bottom_h - theme.SIDEBAR_BOTTOM_MARGIN
                          if bottom_h else p.H)
        top_viewport_bottom = max(theme.SIDEBAR_TOP, bottom_start_y)
        top_viewport_h = max(0, top_viewport_bottom - theme.SIDEBAR_TOP)

        # Measure top sections to know whether they need scrolling.
        measure_top = SidebarContext(ctx, painter, self, sidebar_x,
                                     theme.SIDEBAR_WIDTH, theme.SIDEBAR_TOP,
                                     measure_only=True)
        self._run_sections(top, measure_top)
        top_content_h = max(0, measure_top.y - theme.SIDEBAR_TOP)

        # Clamp the player-owned scroll offset to the legal range so the
        # Qt wheel handler can write to it freely without knowing the
        # layout. Max scroll is "content_h - viewport_h" (zero when the
        # content already fits).
        overflow = max(0, top_content_h - top_viewport_h)
        p.hud.sidebar_scroll_max = overflow
        if p.hud.sidebar_scroll < 0:
            p.hud.sidebar_scroll = 0
        elif p.hud.sidebar_scroll > overflow:
            p.hud.sidebar_scroll = overflow
        scroll = p.hud.sidebar_scroll

        # Clip the top region so scrolled-off content can't draw over the
        # pinned-bottom area or bleed above the top margin. Hitboxes
        # registered outside the clip still get recorded, but the wheel
        # handler ignores clicks outside the sidebar anyway.
        painter.save()
        painter.setClipRect(QRectF(sidebar_x, theme.SIDEBAR_TOP,
                                   theme.SIDEBAR_WIDTH, top_viewport_h))
        top_ctx = SidebarContext(
            ctx, painter, self, sidebar_x,
            theme.SIDEBAR_WIDTH, theme.SIDEBAR_TOP - scroll,
            hitbox_clip=(theme.SIDEBAR_TOP, top_viewport_bottom))
        self._run_sections(top, top_ctx)
        painter.restore()

        if bottom:
            divider_w = int(theme.SIDEBAR_WIDTH * theme.DIVIDER_WIDTH_FRAC * 2)
            divider_x = sidebar_x + (theme.SIDEBAR_WIDTH - divider_w) // 2
            divider_y = bottom_start_y - theme.DIVIDER_MARGIN_Y - 8
            pen = QPen(_qcolor(theme.DIVIDER_COLOR),
                       int(theme.DIVIDER_THICKNESS))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(divider_x, divider_y),
                             QPointF(divider_x + divider_w, divider_y))

            real = SidebarContext(ctx, painter, self, sidebar_x,
                                  theme.SIDEBAR_WIDTH, bottom_start_y)
            self._run_sections(bottom, real)

        # Scroll indicator: thin vertical bar on the sidebar's left edge
        # when content overflows. Skipped when everything fits so it
        # doesn't add visual noise.
        if overflow > 0 and top_viewport_h > 0:
            thumb_h = max(20, int(top_viewport_h * top_viewport_h
                                  / max(1, top_content_h)))
            max_thumb_y = top_viewport_h - thumb_h
            thumb_y = (theme.SIDEBAR_TOP
                       + int(max_thumb_y * scroll / max(1, overflow)))
            _rect(painter, (120, 120, 120),
                  (sidebar_x + 1, thumb_y, 3, thumb_h))

        if p.hud.open_flyout is not None:
            self._draw_flyout(ctx, painter, sidebar_x)

        ctx.plugin_data['hud_y'] = top_ctx.y
        ctx.plugin_data['sidebar_x'] = sidebar_x
        # Snapshot per-frame rects for drag-drop drop-order math; the
        # player reads these in handle_mouse_up to decide where a
        # dropped section should land.
        p.hud.frame_sidepanel_rects = dict(
            ctx.plugin_data.get('sidepanel_rects', {}))
        p.hud.frame_free_rects = dict(
            ctx.plugin_data.get('free_rects', {}))

    def _draw_free_sections(self, ctx, painter):
        """Render every sidebar section whose effective region is 'free'.

        Each section gets its own panel at the saved ``(x, y, w, h)``,
        with an edit-mode outline + resize handle when the user is
        editing layout. The section's existing ``draw(sctx)`` callable
        is reused ; sections work identically in both regions because
        they paint into whatever column the sidebar context hands out.
        """
        # No-op for narrowly-mocked contexts (e.g. layer-gating tests
        # that construct a SimpleNamespace ctx with no player). The
        # real paint path always has player + sidebar attached.
        p = getattr(ctx, 'player', None)
        sidebar_reg = getattr(self.plugins, 'sidebar', None)
        if p is None or sidebar_reg is None:
            return
        free = sidebar_reg.free_sections()
        if not free:
            # Still record an empty "free rects" map so drag hit-testing
            # can tell the region has no occupants.
            ctx.plugin_data['free_rects'] = {}
            return

        rects: dict = {}
        for section in free:
            # Skip the dragged component here ; it's drawn as a floating
            # ghost at the cursor instead (see `_draw_drag_ghost`), so
            # painting it in place would double-render.
            if p.hud.edit_mode and p.hud.drag_key == section.key:
                continue
            rect = self.plugins.sidebar.section_free_rect(
                section, p.W, p.H)
            x, y, w, h = rect
            # Keep the panel on-screen even if the window shrunk since
            # the layout was saved.
            x = max(0, min(p.W - w, x))
            y = max(0, min(p.H - h, y))
            rects[section.key] = (x, y, w, h)

            _rect(painter, theme.FREE_BG, (x, y, w, h))
            _rect_outline(painter, theme.FREE_BORDER, (x, y, w, h), 1)

            sctx = SidebarContext(
                ctx, painter, self, x, w,
                y + theme.FREE_INSET)
            try:
                section.draw(sctx)
            except Exception as exc:
                src = f' ({section.module})' if section.module else ''
                print(f'free section failed: {section.name}{src}: {exc}')

            if p.hud.edit_mode:
                self._draw_edit_outline(painter, x, y, w, h,
                                        highlighted=False)
                self._draw_resize_handle(painter, x + w, y + h)
                # Whole-component drag handle hitbox (edit mode only).
                # Using a dedicated action so the sidepanel's per-button
                # handlers don't compete with the drag grab.
                p.hud.add_hitbox((x, y, w, h),
                                 'begin_drag_section', section.key)
                # Resize handle hitbox: bottom-right corner.
                hs = theme.RESIZE_HANDLE_SIZE
                p.hud.add_hitbox(
                    (x + w - hs, y + h - hs, hs, hs),
                    'begin_resize_section', section.key)

        ctx.plugin_data['free_rects'] = rects

    @staticmethod
    def _draw_edit_outline(painter, x, y, w, h, *, highlighted):
        color = theme.COLOR_EDIT_ACCENT
        pen = QPen(_qcolor(color), 2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
        painter.drawRect(QRectF(float(x), float(y), float(w), float(h)))

    @staticmethod
    def _draw_resize_handle(painter, x_br, y_br):
        size = theme.RESIZE_HANDLE_SIZE
        rect = (x_br - size, y_br - size, size, size)
        _rect(painter, theme.COLOR_EDIT_ACCENT, rect)

    def _draw_drag_overlay(self, ctx, painter):
        """Paint the drag ghost at the cursor + the insertion indicator
        in the sidepanel. Called last so it sits above everything."""
        p = ctx.player
        key = p.hud.drag_key
        section = None
        for s in self.plugins.sidebar.all_sections():
            if s.key == key:
                section = s
                break
        if section is None:
            return

        sidebar_x = p.W - theme.SIDEBAR_WIDTH
        cur_x, cur_y = p.hud.drag_pointer
        over_sidepanel = cur_x >= sidebar_x

        # Insertion line: blue bar between sidepanel neighbors under
        # the cursor. Only relevant when the drop target is the
        # sidepanel.
        if over_sidepanel:
            self._draw_insertion_line(p, painter, sidebar_x, cur_y)

        # Ghost: outlined rect at cursor, same size the section would
        # occupy in the free region (so the user sees the drop footprint).
        dx, dy = p.hud.drag_offset
        _rx, _ry, w, h = self.plugins.sidebar.section_free_rect(
            section, p.W, p.H)
        gx = cur_x - dx
        gy = cur_y - dy
        # Semi-transparent fill so the cursor + underlying content
        # show through.
        painter.setBrush(QBrush(QColor(
            theme.FREE_BG[0], theme.FREE_BG[1], theme.FREE_BG[2], 180)))
        painter.setPen(QPen(_qcolor(theme.COLOR_EDIT_ACCENT), 2))
        painter.drawRect(QRectF(float(gx), float(gy), float(w), float(h)))

    def _draw_insertion_line(self, player, painter, sidebar_x, cur_y):
        """Blue 2px bar at the nearest-neighbor boundary inside the
        sidepanel for the current cursor Y. Uses the last frame's
        sidepanel rects so the indicator lines up with what the user
        sees."""
        rects = player.hud.frame_sidepanel_rects or {}
        # Only non-pinned sections participate in reordering; pinned-
        # bottom ones stay fixed. Pinned status is on the section
        # object, not on the rect, so filter by registry.
        key_set = {
            s.key for s in self.plugins.sidebar.all_sections()
            if s.enabled and not s.pin_bottom
            and self.plugins.sidebar.section_region(s.key) == 'sidepanel'
            and s.key != player.hud.drag_key
        }
        band = [(k, rects[k]) for k in rects if k in key_set]
        # Sort by Y.
        band.sort(key=lambda kv: kv[1][1])
        if not band:
            return
        # Pick the insertion y: above the first rect whose mid-Y
        # exceeds the cursor; else below the last rect.
        insert_y = None
        for _, (_rx, ry, _rw, rh) in band:
            mid = ry + rh / 2
            if cur_y < mid:
                insert_y = ry
                break
        if insert_y is None:
            _, (_rx, ry, _rw, rh) = band[-1]
            insert_y = ry + rh

        x0 = sidebar_x + theme.SIDEBAR_INSET
        x1 = sidebar_x + theme.SIDEBAR_WIDTH - theme.SIDEBAR_INSET
        pen = QPen(_qcolor(theme.COLOR_EDIT_ACCENT), 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(float(x0), float(insert_y)),
                         QPointF(float(x1), float(insert_y)))

    def _draw_flyout(self, ctx, painter, sidebar_x):
        p = ctx.player
        section = self.plugins.sidebar.flyout_section(p.hud.open_flyout)
        if section is None:
            return
        anchor = ctx.plugin_data.get('flyout_anchors', {}).get(
            p.hud.open_flyout)

        flyout_w = theme.FLYOUT_WIDTH
        flyout_x = sidebar_x - flyout_w - theme.FLYOUT_GAP
        if flyout_x < 0:
            flyout_x = 0
            flyout_w = max(40, sidebar_x - theme.FLYOUT_GAP)

        # Measure the expanded content's height against the real width so
        # the panel fits snugly instead of spanning the full window.
        measure = SidebarContext(
            ctx, painter, self, flyout_x, flyout_w,
            theme.FLYOUT_INSET, measure_only=True,
        )
        try:
            section.draw_expanded(measure)
        except Exception as exc:
            src = f' ({section.module})' if section.module else ''
            print(f'flyout measure failed: {section.name}{src}: {exc}')
            return
        content_h = max(0, measure.y - theme.FLYOUT_INSET)
        panel_h = content_h + 2 * theme.FLYOUT_INSET

        # Align the panel's top with the header button; clamp so it
        # doesn't run off the top or past the bottom margin.
        anchor_top = anchor[1] if anchor else theme.SIDEBAR_TOP
        panel_y = min(
            max(theme.SIDEBAR_TOP, anchor_top),
            max(theme.SIDEBAR_TOP, p.H - theme.SIDEBAR_BOTTOM_MARGIN - panel_h),
        )
        panel_rect = (flyout_x, panel_y, flyout_w, panel_h)
        _rect(painter, theme.FLYOUT_BG, panel_rect)
        _rect_outline(painter, theme.FLYOUT_BORDER, panel_rect, 1)

        flyout_ctx = SidebarContext(
            ctx, painter, self, flyout_x, flyout_w,
            panel_y + theme.FLYOUT_INSET,
        )
        try:
            section.draw_expanded(flyout_ctx)
        except Exception as exc:
            src = f' ({section.module})' if section.module else ''
            print(f'flyout draw failed: {section.name}{src}: {exc}')

    @staticmethod
    def _run_sections(sections, sctx):
        """Paint each section's in-place ``draw``. Flyout sections always
        paint their collapsed header here (regardless of open state) ;
        the expanded panel is drawn separately by ``_draw_flyout`` and
        anchored next to the header, so the header must stay visible as
        the anchor point and re-click target. When a flyout is open,
        record the header's rect in ``plugin_data['flyout_anchors']``
        so ``_draw_flyout`` can position the panel against it.

        Also records every section's sidepanel rect in
        ``plugin_data['sidepanel_rects']`` keyed by section.key. Drag
        hit-testing + the reorder insertion-line use these rects; in
        edit mode, draggable sections get an outline and a
        ``begin_drag_section`` hitbox spanning their full rect."""
        p = sctx.player
        open_flyout = p.hud.open_flyout
        anchors = sctx.render_ctx.plugin_data.setdefault(
            'flyout_anchors', {})
        rects = sctx.render_ctx.plugin_data.setdefault(
            'sidepanel_rects', {})
        for section in sections:
            try:
                y_before = sctx.y
                section.draw(sctx)
                rect = (sctx.sidebar_x, y_before,
                        sctx.sidebar_w, sctx.y - y_before)
                rects[section.key] = rect
                if (section.draw_expanded is not None
                        and section.key == open_flyout):
                    anchors[section.key] = rect
                # Edit-mode affordances: outline + full-rect drag grab
                # for sections the plugin marked draggable. Flyout
                # sections skip drag-grab entirely ; they're complex
                # controls that'd be confusing to move around.
                if (p.hud.edit_mode and section.draggable
                        and section.draw_expanded is None
                        and not sctx.measure_only
                        and p.hud.drag_key != section.key):
                    QtPlayerRenderer._draw_edit_outline(
                        sctx.painter, rect[0], rect[1], rect[2], rect[3],
                        highlighted=False)
                    sctx.add_hitbox(rect, 'begin_drag_section',
                                    section.key)
            except Exception as exc:
                src = f' ({section.module})' if section.module else ''
