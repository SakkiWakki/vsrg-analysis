"""Native Qt renderer for the embedded replay player."""
from __future__ import annotations

import os
import time

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QPainter, QPen,
                           QTransform)

# The HUD (sidebar + free-region panels) re-renders at most this often;
# between renders the cached pixmap blits in one call. Text readouts at
# 60 Hz stay fluid while dense-chart frames stop paying the sidebar's
# measure+draw passes.
_HUD_REDRAW_HZ = 60
_HUD_LAYERS = frozenset(('free_sections', 'hud'))

# A field instance is `(transform, opacity)` or, for a per-copy capture
# scope, `(transform, opacity, scope)`. A missing scope is 'field'
# (transparent field-only capture) - the zero-cost path other games use.
# 'full' additionally carries the backdrop (background + below-draws).
# The screen scopes model SM's ActorFrameTexture: the AFT node captures
# the chart area as drawn so far in the SAME frame (backdrop + field
# blits, before any post-node sampler draws); 'screen' samplers show
# that fresh capture, 'screen_prev' samplers drew before the node so
# they show the previous frame's retained capture.
_DEFAULT_FIELD_SCOPE = 'field'
_SCREEN_SCOPE = 'screen'
_SCREEN_PREV_SCOPE = 'screen_prev'
# An AFT-rig curtain quad: a flat color fill at its tree position among
# the instance blits (covers earlier blits, capped by later ones).
_FILL_SCOPE = 'fill'

# Field captures render with this much extra margin per side (fraction
# of the window): a proxied/transformed field instance maps the capture
# boundary into view, and any note a mod pushed past the window would
# get sliced on that edge. The engine has no such intermediate boundary
# for proxies (it re-renders through the transform), so the margin
# hides the divergence for everything but extreme excursions. AFT
# ('screen') captures deliberately stay window-sized - the engine's AFT
# texture IS the screen, and its hard edge is chart-visible.
_FIELD_OVERSCAN_FRAC = 0.25
# The second, independently-modded playfield capture (dual-player NotITG).
# A 'field2' copy blits the second field capture instead of the
# primary; produced only when the frame carries a second_field spec.
_FIELD2_SCOPE = 'field2'

# Debug kill switch: force the pooled-QPixmap capture backend even on a
# GL host, for A/B comparison against the FBO composite path.
_FORCE_RASTER_CAPTURE = os.environ.get('VSRG_CAPTURE_BACKEND') == 'raster'
# A chart-time advance larger than this (or any backward step) between
# rendered frames reads as a seek, not smooth playback: the retained
# previous-frame screen composite is no longer this frame's visual
# predecessor and is dropped. At 205 BPM gat the frame delta is a few ms;
# this leaves a wide margin for slow render frames while still catching
# every user seek (seek10 = +/-10s).
_SEEK_GAP_S = 0.5


def _field_entry(entry):
    """(transform, opacity, scope) for a field instance, defaulting the
    scope when the producer supplied only (transform, opacity)."""
    if len(entry) >= 3:
        return entry[0], entry[1], entry[2]
    return entry[0], entry[1], _DEFAULT_FIELD_SCOPE


def _field_scope(entry) -> str:
    return entry[2] if len(entry) >= 3 else _DEFAULT_FIELD_SCOPE


def _field_extra(entry):
    """The scope's optional payload (4th element): a fill's rgb, or an
    aft sampler's (source name, capture-live?) freeze key."""
    return entry[3] if len(entry) >= 4 else None


from analysis.player.render import culling, gl_capture, theme
from analysis.player.render.capture import RasterCaptureBackend
from analysis.player.render.frame_stats import FrameStats
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
        self._hud_pixmap = None
        self._hud_rendered_at = 0.0
        self._hud_snapshot = None
        # The GL backend's cached HUD target: the 'hud' capture slot's
        # handle (None on the raster path, which uses `_hud_pixmap`).
        self._hud_src = None
        self._hud_slot_open = False
        self._hud_painter = None
        # Offscreen capture slots + blit/snapshot ops. Raster (pooled
        # QPixmaps, capture.py) by default; frames painting on a GL
        # context route to the FBO-backed backend (gl_capture.py). Both
        # persist so their pooled targets survive across frames.
        self._raster_capture = RasterCaptureBackend()
        self._gl_capture = None
        self._capture = self._raster_capture
        # Drawable handles for this frame's closed capture slots, reset
        # by each begin. `_field2_src` is the second, independently-
        # modded field capture for dual-player NotITG charts
        # (EffectFrame.second_field); None on every frame without a
        # second_field spec - the zero-cost path for every other game.
        self._field_src = None
        self._field2_src = None
        self._backdrop_src = None
        self._backdrop_painter = None
        # Previous frame's AFT capture (the chart area as of the node's
        # draw position), retained for this frame's 'screen_prev' copies.
        # None until a frame with screen copies captures one, and dropped
        # on a seek discontinuity. `_prev_screen_t` is the chart time it
        # was captured at, used to detect that discontinuity.
        self._prev_screen = None
        self._prev_screen_t = None
        # Whether the 'screen' slot is open this frame (chart painting
        # redirected into the offscreen composite).
        self._screen_open = False
        # This frame's node-point capture, taken lazily during the
        # instance blits and promoted to `_prev_screen` at composite end.
        self._field_overscan = (0, 0)
        self._screen_capture = None
        # The host painter bracketed by an open 'post' capture slot
        # (the unified GL shader stage); None outside that window.
        self._post_host = None
        # Preserve-texture freezes: the last capture each AFT source
        # blitted while its node was visible, held across hidden frames.
        self._aft_frozen: dict = {}
        self._frame_stats = FrameStats()
        # Set by GL hosts (PlayerCanvas); frames whose effects carry
        # shader passes route chart painting through it. None = raster
        # host, shaders skipped.
        self.shader_pipeline = None

    def build_context(self, player, painter, t_now):
        player._render_t_now = float(t_now)
        x0, lane_w = player._lane_geom()
        ctx = RenderContext(
            player=player,
            colors=player.judge_colors,
            t_now=float(t_now),
            x0=x0,
            lane_w=lane_w,
            judge_y=int(player.judge_y_px()),
            painter=painter,
            _scroll_speed=player.sv_render.effective_scroll_speed(t_now),
        )
        # Bind the per-replay sprite cache and invalidate any pixmaps
        # that were rasterized at an old geometry. Draw sites pull
        # rasterized pixmaps via `ctx.sprite_cache.get(name, ctx, ...)`.
        ctx.sprite_cache = player._sprite_cache
        ctx.sprite_cache.check_geometry(ctx.lane_w, ctx.note_h)
        # Composite effects up front: the lane-switch effect writes
        # `ctx.lane_xs`/`lane_ws`, which the notes prepass reads via
        # `ctx.lane_x`, so it must run before `_notes_layer.prepare`.
        ctx.effect_frame = self._composite_effects(player, ctx)
        culling.prepare_time_window(ctx)
        ctx.candidates = culling.select_note_candidates(ctx)
        # Precompute Y positions for every candidate's head + LN tail in one
        # numpy pass. Saves N*(2..4) scalar `ctx.time_to_y` calls per frame ;
        # on dense Etterna charts that's ~600 Python→SV-engine bisects per
        # frame collapsed to two numpy operations.
        _precompute_candidate_ys(ctx)
        # Per-note mods (NotITG ArrowEffects): mutate the candidate y
        # arrays + stash dx/alpha before the views are built.
        note_mods = getattr(player, '_note_mods', None)
        if note_mods is not None:
            note_mods.apply(ctx)
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
        self._select_capture_backend(painter)
        ctx = self.build_context(player, painter, t_now)
        # Per-rendered-frame cadence sample; published on the player so
        # the frame-analyzer component (whose own draw is throttled to
        # the HUD cache cadence) reads real frame timing via
        # `data.frame_stats()`.
        self._frame_stats.tick()
        if player is not None:
            player._render_frame_stats = self._frame_stats
        effect_frame = getattr(ctx, 'effect_frame', None)
        hud = getattr(player, 'hud', None) if player is not None else None
        # Narrowly-mocked contexts (layer-gating tests) have no player
        # or painter; for those, HUD layers draw directly like any other
        # layer instead of through the cache.
        cache_enabled = (painter is not None
                         and getattr(ctx, 'player', None) is not None)
        hud_due = not cache_enabled or self._hud_redraw_due(ctx, hud)
        # Hitboxes all belong to the HUD region (sidebar, free panels,
        # drag/resize affordances); the chart layers register none. They
        # are cleared only when the HUD actually re-renders so cached
        # frames keep last render's clickable regions.
        if hud is not None and hud_due:
            hud.clear_hitboxes()
        self.plugins.draw(Stage.PRE_FRAME, ctx)
        visibility = self._layer_visibility(ctx)
        try:
            self._draw_chart(ctx, painter, effect_frame, visibility,
                             cache_enabled)
        except Exception:
            # A mid-frame exception would otherwise leave capture
            # painters and native-painting brackets open, and every
            # later frame then paints against a corrupted target/state
            # (HUD content flashing into the chart area). Unwind
            # everything so only the throwing frame is lost, then let
            # the exception surface to name the real culprit.
            self._abort_frame_captures()
            raise
        # The HUD renders at the frame tail, after every capture
        # bracket has closed: on the GL host it targets its own capture
        # slot, which needs the host painter free to bracket.
        if cache_enabled and hud_due:
            self._render_hud(ctx, painter, visibility)
        if cache_enabled:
            self._blit_hud(painter)
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


    def _draw_chart(self, ctx, painter, effect_frame, visibility,
                    cache_enabled):
        """The capture-active middle of `draw`: chart layers, field
        instance composite, screen composite, and the shader stage.
        Split out so `draw` can unwind the capture brackets when any
        layer or plugin throws mid-frame. HUD layers are skipped here
        (rendered at the frame tail by `_render_hud`)."""
        # Shader passes post-process the chart as a whole: capture
        # background + field layers into the GL pipeline's FBO and let
        # it blit the shaded result back before the HUD (which never
        # gets post-processed) goes on top. When no pipeline is
        # attached or capture can't start, chart painting stays on the
        # host painter and shaders are simply skipped.
        chart_painter = self._begin_shader_capture(effect_frame, ctx, painter)
        capturing = chart_painter is not None
        if not capturing:
            chart_painter = painter
        # Screen-scope field copies blit the AFT node's chart-area
        # capture. When any are live, composite the whole chart region
        # into an offscreen pixmap this frame so the capture can be
        # snapshotted mid-blit at the node's draw position ('screen' =
        # this frame's snapshot, 'screen_prev' = last frame's retained
        # one), blit it to the real target, and retain the snapshot.
        # Zero-cost otherwise: `chart_painter` stays the real target and
        # _prev_screen machinery is untouched.
        self._sync_prev_screen(ctx)
        underlying_chart_painter = chart_painter
        screen_target = self._begin_screen_composite(
            effect_frame, ctx, chart_painter)
        if screen_target is not None:
            chart_painter = screen_target
        # Effects transform + bracket only the playfield layers. The
        # 'background' layer is the whole-canvas clear and must stay in
        # screen space (a transformed clear leaves stale pixels outside
        # the moved field -- visible as frame-trail smearing), and the
        # HUD never moves with the playfield. Below-draws (storyboards)
        # paint between the clear and the transformed field.
        #
        # Field copies (EffectFrame.fields) come in two scopes. A 'field'
        # copy replicates only the transparent field layers, so the shared
        # background shows through every copy (fluXis extra playfields;
        # NotITG ActorProxy copies). A 'full' copy replicates the whole
        # chart region incl. background + below-draws (NotITG AFT copies
        # whose ShowAFTBG grabs the background). To serve both from one
        # frame, the field layers ALWAYS capture into a transparent
        # the transparent field slot, and when any 'full' copy is present the
        # background clear + below-draws capture into a separate
        # a separate
        # backdrop slot (also blitted to screen once as the real
        # backdrop). A 'full' copy then blits backdrop+field under its
        # transform; a 'field' copy blits only the field pixmap.
        full_capture = self._full_field_capture(effect_frame, ctx)
        chart_wrapped = False
        scene_wrapped = False
        below_drawn = False
        field_painter = None
        for name, fn, stage in self._layers:
            is_hud = name in _HUD_LAYERS and cache_enabled
            in_field = not is_hud and name != 'background'
            captured = in_field or (full_capture and name == 'background')
            if not captured and (chart_wrapped or field_painter is not None):
                if chart_wrapped:
                    self._end_effect_transform(field_painter or chart_painter)
                    chart_wrapped = False
                if field_painter is not None:
                    self._end_field_capture()
                    self._capture_second_field(effect_frame, ctx,
                                               chart_painter, visibility)
                    self._blit_field_instances(effect_frame, ctx,
                                               chart_painter)
                    field_painter = None
            if full_capture and name == 'background':
                # A 'full' copy exists: redirect the background clear +
                # below-draws into the backdrop pixmap (blitted to screen
                # as the base backdrop by _blit_field_instances). A plain
                # screen clear runs on the host first so the area outside
                # the chart region (behind the HUD) is cleared; the
                # background layer itself (this iteration's `fn`) then
                # draws into the backdrop painter via `target` below.
                _field_layer.draw_background(ctx, chart_painter)
                self._begin_backdrop_capture(effect_frame, ctx, chart_painter)
            if in_field and not below_drawn:
                # Scene (camera) bracket opens before the below-draws
                # so background storyboards ride the camera; the
                # canvas clear stays outside it in screen space. Under a
                # 'full' capture the below-draws land in the backdrop
                # pixmap alongside the background clear.
                below_target = self._backdrop_painter or chart_painter
                scene_wrapped = self._begin_scene_transform(
                    effect_frame, below_target, ctx)
                self._draw_effect_below(effect_frame, ctx, below_target)
                below_drawn = True
                self._end_backdrop_capture()
                # Field instances: the field layer group (its own
                # transform bracket included) renders once into a
                # transparent offscreen buffer, then blits per instance.
                field_painter = self._begin_field_capture(
                    effect_frame, ctx, chart_painter)
                chart_wrapped = self._begin_effect_transform(
                    effect_frame, field_painter or chart_painter, ctx)
            if is_hud:
                # HUD layers render into a cached offscreen target at
                # most _HUD_REDRAW_HZ, at the frame TAIL (`_render_hud`)
                # once every capture bracket has closed - on the GL host
                # the target is a capture slot that must bracket the
                # host painter. Skipped entirely here.
                continue
            if captured:
                # Captured layers draw into whichever offscreen buffer is
                # open: the field pixmap for the field layers, or (for the
                # background layer under a 'full' copy) the backdrop pixmap.
                target = (field_painter or self._backdrop_painter
                          or chart_painter)
            else:
                target = chart_painter
            if visibility.get(name, True):
                if fn is not None:
                    self._draw_layer(fn, ctx, target, name, is_hud)
            if stage is not None:
                prev_painter = getattr(ctx, 'painter', None)
                ctx.painter = target
                self.plugins.draw(stage, ctx)
                ctx.painter = prev_painter
        if chart_wrapped:
            self._end_effect_transform(field_painter or chart_painter)
        if field_painter is not None:
            self._end_field_capture()
            self._capture_second_field(effect_frame, ctx, chart_painter,
                                       visibility)
            self._blit_field_instances(effect_frame, ctx, chart_painter)
        self._draw_effect_above(effect_frame, ctx, chart_painter)
        if scene_wrapped:
            self._end_effect_transform(chart_painter)
        self._draw_effect_top(effect_frame, ctx, chart_painter)
        # Close the screen composite: blit it to the real chart target and
        # retain it for next frame's 'screen' copies. `chart_painter` is
        # restored to the underlying target so the shader/HUD path below is
        # unaffected.
        if screen_target is not None:
            self._end_screen_composite(underlying_chart_painter, ctx)
            chart_painter = underlying_chart_painter
        if capturing:
            self._end_shader_capture(effect_frame, ctx)

    def _composite_effects(self, player, ctx):
        effects = getattr(player, '_render_effects', None) if player else None
        if not effects:
            return None
        from analysis.player.render.effects import composite
        frame = composite(effects, ctx)
        return None if frame.is_identity else frame

    def _begin_shader_capture(self, effect_frame, ctx, painter):
        """Start routing chart painting into the shader stage when this
        frame carries shader passes. Returns the capture painter, or
        None to paint direct (no passes, raster host, GL failure).

        With the GL capture backend active the capture is just another
        slot ('post') - the unified chain field FBOs -> instance
        composite -> shader passes -> screen; the pipeline's own
        capture pair serves the forced-raster fallback."""
        self._post_host = None
        if (effect_frame is None or not effect_frame.shaders
                or self.shader_pipeline is None or painter is None
                or getattr(ctx, 'player', None) is None):
            return None
        p = ctx.player
        if isinstance(self._capture, gl_capture.GLCaptureBackend):
            self._post_host = painter
            return self._capture.open('post', painter, p.W, p.H)
        return self.shader_pipeline.begin_capture(painter, p.W, p.H)

    def _end_shader_capture(self, effect_frame, ctx) -> None:
        """Close the shader stage opened by `_begin_shader_capture`:
        run the frame's passes over the capture, last pass into the
        real target."""
        if self._post_host is None:
            self.shader_pipeline.end_capture(effect_frame.shaders,
                                             ctx.t_now)
            return
        host = self._post_host
        self._post_host = None
        handle = self._capture.close('post')
        if isinstance(handle, gl_capture._GLHandle):
            self.shader_pipeline.run_over(
                host, handle.fbo, effect_frame.shaders, ctx.t_now,
                handle.fbo.width(), handle.fbo.height())
        elif handle is not None:
            # Mid-frame GL breakage handed the slot to the raster
            # fallback: present the capture unshaded.
            host.drawPixmap(0, 0, handle)

    @staticmethod
    def _begin_effect_transform(frame, painter, ctx) -> bool:
        """Push the composited transform + opacity for the chart-layer
        group, clipped to the chart region so no transform can paint
        into the sidebar. Returns whether a save() was made (so the
        caller knows to restore)."""
        if frame is None or (frame.transform is None and frame.opacity >= 1.0):
            return False
        painter.save()
        painter.setClipRect(QRectF(*ctx.chart_rect))
        if frame.transform is not None:
            painter.setTransform(frame.transform, True)
        if frame.opacity < 1.0:
            painter.setOpacity(frame.opacity)
        return True

    @staticmethod
    def _end_effect_transform(painter) -> None:
        painter.restore()

    @staticmethod
    def _full_field_capture(frame, ctx) -> bool:
        """True when any field copy this frame is 'full' scope (carries
        the background), so the backdrop (background clear + below-draws)
        must be captured for those copies to blit. The identity original
        and 'field' copies never need it."""
        if frame is None or not frame.fields:
            return False
        return any(_field_scope(entry) == 'full' for entry in frame.fields)

    @staticmethod
    def _has_screen_copy(frame) -> bool:
        """True when any field copy this frame is a screen scope, so the
        whole chart region must composite offscreen and the node-point
        capture be taken for this frame's 'screen' copies and next
        frame's 'screen_prev' ones."""
        if frame is None or not frame.fields:
            return False
        return any(_field_scope(entry) in (_SCREEN_SCOPE, _SCREEN_PREV_SCOPE)
                   for entry in frame.fields)

    def _abort_frame_captures(self) -> None:
        """Unwind every capture opened this frame after a mid-frame
        exception: end slot painters, restore render targets, close
        native brackets, and drop this frame's handles. Retained
        cross-frame captures re-prime afterwards, as after a seek."""
        self._capture.abort()
        if self._post_host is None and self.shader_pipeline is not None:
            abort = getattr(self.shader_pipeline, 'abort_capture', None)
            if abort is not None:
                abort()
        self._post_host = None
        self._backdrop_painter = None
        self._screen_open = False
        self._field_src = None
        self._field2_src = None
        self._backdrop_src = None
        self._hud_slot_open = False
        self._hud_painter = None
        self._capture.release(self._screen_capture)
        self._screen_capture = None

    def _select_capture_backend(self, painter) -> None:
        """Route this frame's capture slots to the FBO backend when the
        host painter renders on a GL context (the canvas widget),
        raster otherwise (tests, offscreen platform, GL breakage,
        VSRG_CAPTURE_BACKEND=raster). A switch drops every retained
        capture - the handles belong to the other backend - and the
        retention re-primes over the next frame, like after a seek."""
        use_gl = (not _FORCE_RASTER_CAPTURE and gl_capture.usable(painter)
                  and (self._gl_capture is None
                       or not self._gl_capture.broken))
        if use_gl and self._gl_capture is None:
            self._gl_capture = gl_capture.GLCaptureBackend()
        backend = self._gl_capture if use_gl else self._raster_capture
        if backend is not self._capture:
            self._drop_retained_captures()
            self._capture = backend

    def _drop_retained_captures(self) -> None:
        """Release every capture handle the renderer retains across
        frames, returning their storage to the current backend."""
        self._capture.release(self._prev_screen)
        self._prev_screen = None
        self._prev_screen_t = None
        self._capture.release(self._screen_capture)
        self._screen_capture = None
        for handle in self._aft_frozen.values():
            self._capture.release(handle)
        self._aft_frozen.clear()
        self._field_src = None
        self._field2_src = None
        self._backdrop_src = None
        # The cached HUD target belongs to the old backend too; the
        # first-render trigger in _hud_redraw_due re-renders it.
        self._hud_src = None
        self._hud_pixmap = None

    def _sync_prev_screen(self, ctx) -> None:
        """Drop the retained previous-frame AFT capture on a seek
        discontinuity. Playback advances chart time smoothly forward by
        one small frame delta; a seek jumps it (backward, or forward past
        several frames). A retained capture from before the jump is not
        the visual predecessor of this frame, so it must not feed the
        'screen_prev' feedback - those copies skip one frame, then the
        capture re-primes."""
        if getattr(ctx, 'player', None) is None:
            return
        t = float(ctx.t_now)
        prev_t = self._prev_screen_t
        if prev_t is not None and not (0.0 <= t - prev_t <= _SEEK_GAP_S):
            self._capture.release(self._prev_screen)
            self._prev_screen = None
            self._prev_screen_t = None
            # Retained freezes predate the jump too; they re-prime from
            # the next frame their source node draws.
            for handle in self._aft_frozen.values():
                self._capture.release(handle)
            self._aft_frozen.clear()

    def _begin_screen_composite(self, frame, ctx, painter):
        """Redirect the whole chart region into an offscreen pixmap when
        this frame carries screen copies; returns the composite painter
        or None (no screen copies, direct painting). The composite is
        what the AFT node's capture snapshots mid-blit; compositing
        offscreen keeps that snapshot cheap and consistent."""
        self._screen_open = False
        self._capture.release(self._screen_capture)
        self._screen_capture = None
        if (not self._has_screen_copy(frame) or painter is None
                or getattr(ctx, 'player', None) is None):
            return None
        p = ctx.player
        self._screen_open = True
        return self._capture.open('screen', painter, p.W, p.H)

    def _end_screen_composite(self, painter, ctx) -> None:
        """Finish the screen composite: end its painter, blit it to the
        real chart target, and retain this frame's node-point capture
        (with its chart time) as `_prev_screen` for next frame's
        'screen_prev' copies. The FINAL composite is never retained -
        it includes the screen blits themselves, and feeding it back
        makes an identity opaque sampler a fixed point that freezes the
        chart area."""
        if not self._screen_open:
            return
        self._capture.close('screen')
        self._screen_open = False
        self._capture.present(painter, 'screen')
        if self._screen_capture is not None:
            self._capture.release(self._prev_screen)
            self._prev_screen = self._screen_capture
            self._prev_screen_t = float(ctx.t_now)
            self._screen_capture = None

    def _begin_backdrop_capture(self, frame, ctx, painter) -> None:
        """Open the backdrop slot capturing the background clear +
        below-draws, used as the source for 'full' field copies (their
        capture includes the background). No-op unless this frame has a
        'full' copy. The captured backdrop also becomes the base backdrop
        blitted to screen, so it is never double-drawn."""
        self._backdrop_src = None
        self._backdrop_painter = None
        if (not self._full_field_capture(frame, ctx) or painter is None
                or getattr(ctx, 'player', None) is None):
            return
        p = ctx.player
        self._backdrop_painter = self._capture.open(
            'backdrop', painter, p.W, p.H)

    def _end_backdrop_capture(self) -> None:
        if self._backdrop_painter is not None:
            self._backdrop_src = self._capture.close('backdrop')
            self._backdrop_painter = None

    def _begin_field_capture(self, frame, ctx, painter):
        """Redirect the field layer group into the transparent field
        slot when this frame carries field instances; returns the
        capture painter or None (no fields, direct painting)."""
        self._field_src = None
        if (frame is None or not frame.fields or painter is None
                or getattr(ctx, 'player', None) is None):
            return None
        p = ctx.player
        mx, my = self._field_overscan = (int(p.W * _FIELD_OVERSCAN_FRAC),
                                         int(p.H * _FIELD_OVERSCAN_FRAC))
        capture_painter = self._capture.open('field', painter,
                                             p.W + 2 * mx, p.H + 2 * my)
        if capture_painter is not None:
            capture_painter.translate(mx, my)
        return capture_painter

    def _end_field_capture(self) -> None:
        self._field_src = self._capture.close('field')

    def _capture_second_field(self, frame, ctx, painter, visibility) -> None:
        """Render the field layers a second time with a dual-player chart's
        alternate (player-1) note-mod consumer, into the field2 slot, so
        'field2'-scope copies blit an independently-modded playfield.

        Engine parity (item 43/ENGINE_ORACLE 2b): an ActorProxy of a
        DIFFERENTLY-modded Player must re-render that side's note pipeline,
        not blit player 0's pixels. The two captures share the chart and
        candidate set; only the sampled (mod, player) channels differ. The
        player's `_note_mods` is swapped for the second consumer, the
        candidate pipeline + note views are rebuilt against it, the field
        layers draw into a fresh transparent pixmap (with the same effect
        transform bracket the primary capture uses so field transforms
        replicate), then player-0 state is restored for the rest of the
        frame. Zero cost when no second_field spec is present."""
        self._field2_src = None
        spec = getattr(frame, 'second_field', None)
        if (spec is None or painter is None
                or getattr(ctx, 'player', None) is None):
            return
        player = ctx.player
        primary = getattr(player, '_note_mods', None)
        mx, my = self._field_overscan
        fp = self._capture.open('field2', painter,
                                player.W + 2 * mx, player.H + 2 * my)
        if fp is not None:
            fp.translate(mx, my)
        try:
            player._note_mods = spec.note_mods
            self._rebuild_note_mods(ctx)
            wrapped = self._begin_effect_transform(frame, fp, ctx)
            self._draw_field_layers(ctx, fp, visibility)
            if wrapped:
                self._end_effect_transform(fp)
        finally:
            self._field2_src = self._capture.close('field2')
            player._note_mods = primary
            self._rebuild_note_mods(ctx)

    @staticmethod
    def _rebuild_note_mods(ctx) -> None:
        """Recompute the candidate y arrays + per-note mod stashes + note
        views for the ctx's current `player._note_mods`. Idempotent given a
        fixed candidate set, so it both applies the second consumer and,
        on restore, rebuilds player-0 state exactly."""
        _precompute_candidate_ys(ctx)
        note_mods = getattr(ctx.player, '_note_mods', None)
        if note_mods is not None:
            note_mods.apply(ctx)
        _notes_layer.prepare(ctx)

    def _draw_field_layers(self, ctx, painter, visibility) -> None:
        """Draw only the captured field layers (everything but the
        background clear and the HUD) into `painter`, matching what the
        main loop routes into the field pixmap. Used for the second-field
        capture; the primary capture is produced inline by the main loop."""
        for name, fn, _stage in self._layers:
            if name == 'background' or name in _HUD_LAYERS or fn is None:
                continue
            if visibility.get(name, True):
                self._draw_layer(fn, ctx, painter, name, is_hud=False)

    def _blit_field_instances(self, frame, ctx, painter) -> None:
        """Blit the base backdrop then one blit per field instance, each
        clipped to the chart region so no copy lands on the sidebar.

        A 'full' copy blits the backdrop capture (background + below-draws)
        then the field capture under its transform, so the whole screen is
        replicated. A 'field' copy blits only the field capture, so the
        real backdrop shows through. The screen scopes model the AFT
        node's capture, snapshotted from the in-progress composite at
        the first 'screen' blit (or after all blits when only
        'screen_prev' copies are live) - the node's draw position,
        holding backdrop + field blits + any pre-node sampler blits,
        never the screen blits made after it. A 'screen' copy blits THIS
        frame's capture (identity is a no-op re-draw, a transform is a
        screen-copy toss); a 'screen_prev' copy blits last frame's
        retained capture - its own blit lands in this frame's, the
        one-frame feedback that accumulates trails. When none is
        retained yet (first frame after start or a seek), 'screen_prev'
        skips a frame and re-primes. Each copy is additionally clipped to
        the mapped design box in its own source space, so it shows only the
        hard-cropped 640x480 screen (offscreen content never bleeds in).

        The backdrop pixmap, when present, was captured in place of the
        direct background/below-draws, so it is blitted here as the base
        backdrop exactly once."""
        from analysis.games.notitg.field_instances import design_box
        box = (design_box(ctx.chart_rect)
               if (self._full_field_capture(frame, ctx)
                   or self._has_screen_copy(frame)) else None)
        with self._capture.blits(painter, QRectF(*ctx.chart_rect)) as batch:
            if self._backdrop_src is not None:
                batch.blit(self._backdrop_src)
            for entry in frame.fields:
                self._blit_field_instance(batch, entry, box)
            if self._screen_open and self._screen_capture is None:
                # No 'screen' sampler drew this frame, but the node
                # still captures - its draw position follows the
                # instance blits - so 'screen_prev' copies have next
                # frame's source.
                self._take_screen_capture()
        self._backdrop_src = None

    def _take_screen_capture(self) -> None:
        """This frame's node-point AFT capture: the in-progress screen
        composite as of the node's draw position, reusing the retention
        slot freed when a seek dropped the previous capture."""
        self._screen_capture = self._capture.snapshot('screen')

    def _blit_field_instance(self, batch, entry, box) -> None:
        """One instance blit into the open batch (or the fill scope's
        curtain quad), honouring the scope's capture source. Skips
        instances whose source doesn't exist this frame ('screen_prev'
        before any capture is retained, 'field2' without a second
        capture)."""
        transform, opacity, scope = _field_entry(entry)
        extra = _field_extra(entry)
        if opacity < 1.0 / 255.0:
            return
        if scope == _FILL_SCOPE:
            # An AFT-rig curtain quad at its tree position: covers
            # every blit made before it, capped by the ones after.
            # The AFT node sits BEFORE the curtains in the rig's
            # tree (gat: nodes at 5718/5738, quads after), so the
            # node-point capture must snapshot the composite before
            # the curtain lands - otherwise a 'screen' sampler
            # above the quad blits the blackout back at itself.
            if self._screen_open and self._screen_capture is None:
                self._take_screen_capture()
            batch.fill(extra or (1.0, 1.0, 1.0), opacity)
            return
        if scope == _SCREEN_SCOPE and not self._screen_open:
            return
        if scope == _SCREEN_PREV_SCOPE and self._prev_screen is None:
            return
        if scope == _FIELD2_SCOPE and self._field2_src is None:
            return
        if scope == _SCREEN_SCOPE and self._screen_capture is None:
            self._take_screen_capture()
        match scope:
            case 'screen':
                source = self._aft_source(extra, self._screen_capture)
            case 'screen_prev':
                source = self._aft_source(extra, self._prev_screen)
            case 'field2':
                # The second player's independently-modded field capture.
                source = self._field2_src
            case _:
                if scope == 'full' and self._backdrop_src is not None:
                    batch.blit(self._backdrop_src, transform=transform,
                               src_box=box, opacity=opacity)
                source = self._field_src
        if scope != _SCREEN_SCOPE and scope != _SCREEN_PREV_SCOPE:
            transform, box = self._overscan_blit(transform, box)
        batch.blit(source, transform=transform, src_box=box,
                   opacity=opacity)

    def _overscan_blit(self, transform, box):
        """Blit args for an overscanned field source: the window origin
        sits at (+mx, +my) inside the capture, so the draw shifts back
        and a source-space clip box shifts forward."""
        mx, my = self._field_overscan
        if not mx and not my:
            return transform, box
        offset = QTransform.fromTranslate(-mx, -my)
        composed = offset * transform if transform is not None else offset
        shifted = box.translated(mx, my) if box is not None else None
        return composed, shifted

    def _aft_source(self, extra, live_capture):
        """The capture handle an AFT sampler blits, honouring
        preserve-texture freezes. `extra` is the sampler's (source name,
        capture-live?) pair: while the source node draws (live), the
        fresh capture is blitted and retained; while the node is hidden
        its texture stops updating, so the retained capture is blitted
        instead (gat's DelayFrame still-frames and the frozen ending
        toss). No pair, or no retained capture yet, falls back to the
        live capture. Freeze retention shares the capture handle with
        the screen retention chain, so it holds its own backend
        reference (retain/release)."""
        if extra is None:
            return live_capture
        name, live = extra
        if live:
            frozen = self._aft_frozen.get(name)
            if frozen is not live_capture:
                self._capture.release(frozen)
                self._aft_frozen[name] = self._capture.retain(live_capture)
            return live_capture
        return self._aft_frozen.get(name, live_capture)

    @staticmethod
    def _begin_scene_transform(frame, painter, ctx) -> bool:
        """Push the scene-wide camera transform around below-draws,
        chart layers, and effect draws under SCENE_TOP_Z, clipped to
        the chart region. Returns whether a save() was made."""
        if (frame is None or frame.scene_transform is None
                or painter is None):
            return False
        painter.save()
        painter.setClipRect(QRectF(*ctx.chart_rect))
        painter.setTransform(frame.scene_transform, True)
        return True

    @staticmethod
    def _draw_layer(fn, ctx, painter, name, is_hud) -> None:
        """Draw one layer, applying its layerfade alpha when a field layer
        (never the HUD) has one below 1."""
        opacities = getattr(ctx, 'layer_opacities', None)
        alpha = None if is_hud or not opacities else opacities.get(name)
        if alpha is None or alpha >= 1.0:
            fn(ctx, painter)
            return
        painter.save()
        painter.setOpacity(alpha)
        fn(ctx, painter)
        painter.restore()

    @staticmethod
    def _draw_effect_draws(draws, ctx, painter) -> None:
        """Run overlay draws clipped to the chart region; effects never
        paint into the sidebar."""
        if not draws:
            return
        painter.save()
        painter.setClipRect(QRectF(*ctx.chart_rect))
        for _z, fn in draws:
            fn(ctx, painter)
        painter.restore()

    def _draw_effect_below(self, frame, ctx, painter) -> None:
        if frame is not None:
            self._draw_effect_draws(frame.below, ctx, painter)

    def _draw_effect_above(self, frame, ctx, painter) -> None:
        if frame is not None:
            self._draw_effect_draws(frame.above, ctx, painter)

    def _draw_effect_top(self, frame, ctx, painter) -> None:
        """Draws above SCENE_TOP_Z: screen-space overlays (pulse,
        foreground flash) that never ride the camera."""
        if frame is not None:
            self._draw_effect_draws(frame.top, ctx, painter)

    def _hud_redraw_due(self, ctx, hud) -> bool:
        """The HUD pixmap re-renders when its content can actually have
        changed faster than the steady cadence: interaction state
        (edit mode, drag, flyout, wheel scroll), geometry, or layer
        visibility. Otherwise at most every 1/_HUD_REDRAW_HZ."""
        if self._hud_pixmap is None and self._hud_src is None:
            return True
        p = ctx.player
        interacting = hud is not None and (
            hud.edit_mode or hud.drag_key is not None
            or hud.open_flyout is not None)
        visibility = self._layer_visibility(ctx)
        snapshot = (
            p.W, p.H,
            hud.sidebar_scroll if hud is not None else 0,
            tuple(sorted(
                (k, v) for k, v in visibility.items() if k in _HUD_LAYERS)),
        )
        now = time.monotonic()
        stale = now - self._hud_rendered_at >= 1.0 / _HUD_REDRAW_HZ
        if interacting or stale or snapshot != self._hud_snapshot:
            self._hud_rendered_at = now
            self._hud_snapshot = snapshot
            return True
        return False

    def _render_hud(self, ctx, painter, visibility) -> None:
        """Render the HUD layers (+ their plugin stages) into the
        cached HUD target: a capture slot on the GL backend, a pooled
        transparent pixmap otherwise.

        On the GL host the HUD must NOT be a QPixmap: blitting a
        window-sized pixmap routes through Qt's shared pixmap-texture
        cache, and re-uploading ~12MB at the HUD redraw cadence
        thrashes that cache past its eviction limit - entries cross and
        drawPixmap starts serving the WRONG texture (the HUD appearing
        compressed inside sprite rects while its own region goes
        empty). A capture slot keeps the HUD on the GPU with no cache
        involvement. Runs at the frame tail so the slot's native
        bracket has the host painter free."""
        hud_painter = self._begin_hud_target(ctx, painter)
        try:
            for name, fn, stage in self._layers:
                if name not in _HUD_LAYERS:
                    continue
                if visibility.get(name, True) and fn is not None:
                    self._draw_layer(fn, ctx, hud_painter, name,
                                     is_hud=True)
                if stage is not None:
                    prev_painter = getattr(ctx, 'painter', None)
                    ctx.painter = hud_painter
                    self.plugins.draw(stage, ctx)
                    ctx.painter = prev_painter
        finally:
            self._end_hud_target()

    def _begin_hud_target(self, ctx, painter):
        p = ctx.player
        if isinstance(self._capture, gl_capture.GLCaptureBackend):
            self._hud_pixmap = None
            hud_painter = self._capture.open('hud', painter, p.W, p.H)
            self._hud_slot_open = True
        else:
            from PySide6.QtGui import QPixmap
            self._hud_src = None
            self._hud_slot_open = False
            dpr = float(painter.device().devicePixelRatioF())
            size = (int(p.W * dpr), int(p.H * dpr))
            pm = self._hud_pixmap
            if pm is None or (pm.width(), pm.height()) != size:
                pm = QPixmap(*size)
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.GlobalColor.transparent)
            self._hud_pixmap = pm
            hud_painter = QPainter(pm)
        hud_painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        hud_painter.setFont(self.font)
        self._hud_painter = hud_painter
        return hud_painter

    def _end_hud_target(self) -> None:
        if self._hud_slot_open:
            self._hud_slot_open = False
            self._hud_src = self._capture.close('hud')
        elif self._hud_painter is not None:
            self._hud_painter.end()
        self._hud_painter = None

    def _blit_hud(self, painter) -> None:
        """Composite the cached HUD target over the frame: the slot
        handle as a textured quad on the GL backend, the pooled pixmap
        otherwise."""
        if self._hud_src is not None:
            handle = self._hud_src
            self._capture.blit(painter, handle,
                               QRectF(0, 0, handle.w, handle.h))
        elif self._hud_pixmap is not None:
            painter.drawPixmap(0, 0, self._hud_pixmap)

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
        # The page-break divider floats `divider_gap` above the pinned
        # sections; the scrollable viewport ends the same gap above the
        # divider, so scrolled content keeps symmetric breathing room
        # instead of running through the separator.
        divider_gap = theme.DIVIDER_MARGIN_Y + 8
        divider_y = bottom_start_y - divider_gap
        top_region_end = divider_y - divider_gap if bottom_h else p.H
        top_viewport_bottom = max(theme.SIDEBAR_TOP, top_region_end)
        top_viewport_h = max(0, top_viewport_bottom - theme.SIDEBAR_TOP)

        # Clamp the player-owned scroll offset to the legal range so the
        # Qt wheel handler can write to it freely without knowing the
        # layout. The top region is deliberately NOT pre-measured: its
        # content height is whatever the previous real draw painted
        # (recorded below), so scrollability follows actual content and
        # never depends on plugins implementing measure hooks honestly.
        # A size change (e.g. expanding the layer tree) corrects the
        # clamp on the next HUD render.
        overflow = max(0, p.hud.sidebar_content_h - top_viewport_h)
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

        # Observe what the sections actually painted; this feeds the
        # next render's scroll clamp and this render's thumb.
        top_content_h = max(0, top_ctx.y - (theme.SIDEBAR_TOP - scroll))
        p.hud.sidebar_content_h = top_content_h
        overflow = max(0, top_content_h - top_viewport_h)
        p.hud.sidebar_scroll_max = overflow

        if bottom:
            divider_w = int(theme.SIDEBAR_WIDTH * theme.DIVIDER_WIDTH_FRAC * 2)
            divider_x = sidebar_x + (theme.SIDEBAR_WIDTH - divider_w) // 2
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
