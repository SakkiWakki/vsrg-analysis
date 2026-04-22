"""Native Qt renderer for the embedded replay player."""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter, QPen)


from analysis.player.render import culling, theme
from analysis.player.hud.sidebar_api import SidebarContext
from analysis.player.plugin_api import Stage
from analysis.player.render.render_context import RenderContext


# Roll bodies/tails use a fixed green per Etterna convention; keep it
# local to the renderer since nothing else branches on roll colors.
_ROLL_BODY = (90, 210, 90)
_ROLL_TAIL = (60, 160, 60)
_MISS_TAP_BODY = (77, 77, 77)
_MISS_LN_BODY = (38, 38, 38)
_RELEASE_GUIDE = (220, 220, 220)
_MISS_X_OUTLINE = (255, 60, 60, 110)
_BG_BASE = (14, 14, 16)
_LANE_BG = (22, 22, 24)
_LANE_LINE = (40, 40, 44)


def _head_rect(lx, y, lane_w, note_h):
    return (lx + 4, y - note_h // 2, lane_w - 8, note_h)


def _circle_head_radius(lane_w):
    return max(6, int((lane_w - 4) * 0.46))


def _dim(color, factor=2):
    return tuple(v // factor for v in color)


@dataclass
class _NoteView:
    """Per-note values used across the draw helpers. Built once per
    `_draw_notes` iteration so the sub-draws don't recompute them."""
    i: int
    col: int
    y: int           # y of the note head in screen pixels
    y_end: int       # y of the LN tail (== y when not an LN)
    lx: int          # left edge of the lane
    off: float       # hit offset in seconds (signed)
    press_t: float   # head + off, seconds
    release_t: float | None  # LN end + release offset, seconds
    rel_off: float | None
    end_t: float | None
    is_ln: bool
    is_roll: bool
    miss: bool
    ln_state: str    # 'upcoming' | 'tap' | 'held' | 'released' | 'missed' | 'missed_note'
    note_color: tuple
    jcolor: tuple    # judgement color for this note


def _qcolor(color):
    if isinstance(color, QColor):
        return color
    if len(color) == 4:
        return QColor(int(color[0]), int(color[1]), int(color[2]),
                      int(color[3]))
    return QColor(int(color[0]), int(color[1]), int(color[2]))


class _QtSurface:
    def __init__(self, painter):
        self.painter = painter

    def blit(self, surface, pos):
        if hasattr(surface, 'image'):
            self.painter.drawImage(QPointF(float(pos[0]), float(pos[1])),
                                   surface.image)


class _QtOffscreenSurface:
    def __init__(self, size):
        self.w, self.h = int(size[0]), int(size[1])
        self.image = QImage(max(1, self.w), max(1, self.h),
                            QImage.Format_ARGB32_Premultiplied)
        self.image.fill(QColor(0, 0, 0, 0))

    def fill(self, color):
        self.image.fill(_qcolor(color))

    def get_rect(self):
        return (0, 0, self.w, self.h)

    def blit(self, surface, pos):
        painter = QPainter(self.image)
        try:
            painter.drawImage(QPointF(float(pos[0]), float(pos[1])),
                              surface.image)
        finally:
            painter.end()


class _QtDraw:
    @staticmethod
    def _paint(surface):
        if hasattr(surface, 'painter'):
            return surface.painter, False
        painter = QPainter(surface.image)
        return painter, True

    def rect(self, surface, color, rect, width=0):
        painter, owned = self._paint(surface)
        try:
            x, y, w, h = _rect_tuple(rect)
            painter.setPen(QPen(_qcolor(color), int(width)) if width else
                           QPen(QColor(0, 0, 0, 0)))
            painter.setBrush(QBrush(QColor(0, 0, 0, 0)) if width else
                             QBrush(_qcolor(color)))
            painter.drawRect(QRectF(float(x), float(y), float(w), float(h)))
        finally:
            if owned:
                painter.end()

    def line(self, surface, color, start, end, width=1):
        painter, owned = self._paint(surface)
        try:
            painter.setPen(QPen(_qcolor(color), int(width)))
            painter.drawLine(QPointF(float(start[0]), float(start[1])),
                             QPointF(float(end[0]), float(end[1])))
        finally:
            if owned:
                painter.end()

    def circle(self, surface, color, center, radius, width=0):
        painter, owned = self._paint(surface)
        try:
            painter.setPen(QPen(_qcolor(color), int(width)) if width else
                           QPen(QColor(0, 0, 0, 0)))
            painter.setBrush(QBrush(QColor(0, 0, 0, 0)) if width else
                             QBrush(_qcolor(color)))
            r = float(radius)
            painter.drawEllipse(QPointF(float(center[0]), float(center[1])),
                                r, r)
        finally:
            if owned:
                painter.end()


class QtPygameCompat:
    SRCALPHA = 1
    draw = _QtDraw()

    @staticmethod
    def Rect(x, y=None, w=None, h=None):
        if y is None:
            return tuple(x)
        return (x, y, w, h)

    @staticmethod
    def Surface(size, _flags=0):
        return _QtOffscreenSurface(size)


def _rect_tuple(rect):
    if hasattr(rect, 'getRect'):
        return rect.getRect()
    return tuple(rect)


class QtPlayerRenderer:
    def __init__(self, plugin_manager):
        self.plugins = plugin_manager
        self.font = QFont('monospace', 10)
        self.big_font = QFont('monospace', 14)
        self.big_font.setBold(True)
        self.compat = QtPygameCompat()
        self._defaults = self._default_drawers()
        # Cache adapter-resolved drawer maps, keyed by adapter id so
        # switching replays (and thus adapters) picks up the right set
        # without rebuilding on every frame.
        self._drawer_cache: dict = {}

    # -----------------------------------------------------------------
    # Drawer registry. Each key here is a per-note-type draw op that a
    # game adapter may override via `GameAdapter.note_drawers()`.
    #
    # Signatures (every drawer takes `painter` first):
    #   tap_head(painter, skin, lx, y, lane_w, note_h, color)
    #   ln_body(painter, skin, lx, y_top, y_bot, lane_w, color)
    #   ln_tail(painter, skin, lx, y, lane_w, note_h, color)
    #   ln_release_guide(painter, lx, lane_w, y_tail, y_release)
    #   mine(painter, lx, y, lane_w)
    #   lift(painter, skin, lx, y, lane_w, note_h, color)
    #   fake(painter, skin, lx, y, lane_w, note_h, color)
    #   ghost_tap(painter, lx, y, lane_w, note_h)
    #   press_mark(painter, lx, lane_w, y_head, y_press, color)
    #   miss_x(painter, lx, y, lane_w, note_h, jcolor)
    #   miss_hold_stroke(painter, lx, lane_w, y_top, y_bot,
    #                    y_press, y_release, color)
    # -----------------------------------------------------------------
    def _default_drawers(self) -> dict:
        return {
            'tap_head': self._draw_note_head,
            'ln_body': self._draw_ln_body,
            'ln_tail': self._draw_ln_tail,
            'ln_release_guide': self._default_ln_release_guide,
            'mine': self._draw_mine,
            'lift': self._draw_lift,
            'fake': self._draw_fake,
            'ghost_tap': self._draw_ghost_tap,
            'press_mark': self._default_press_mark,
            'miss_x': self._default_miss_x,
            'miss_hold_stroke': self._default_miss_hold_stroke,
        }

    def _resolve_drawers(self, player) -> dict:
        adapter = getattr(player, '_adapter', None)
        cache_key = id(adapter)
        cached = self._drawer_cache.get(cache_key)
        if cached is not None:
            return cached
        drawers = dict(self._defaults)
        if adapter is not None:
            drawers.update(adapter.note_drawers() or {})
        self._drawer_cache[cache_key] = drawers
        return drawers

    def build_context(self, player, painter, t_now):
        x0, lane_w = player._lane_geom()
        ctx = RenderContext(
            player=player,
            screen=_QtSurface(painter),
            pygame=self.compat,
            colors=player.judge_colors,
            t_now=float(t_now),
            x0=x0,
            lane_w=lane_w,
            judge_y=int(player.H * player.hit_line_y_frac),
            painter=painter,
        )
        ctx.drawers = self._resolve_drawers(player)
        culling.prepare_time_window(ctx)
        ctx.candidates = culling.select_note_candidates(ctx)
        return ctx

    # Ordered draw layers. Each entry is (name, draw_fn, after_stage).
    # Layer names drive the `layer_visibility` dict in `ctx.plugin_data`
    # — missing/True = draw, False = skip the built-in (plugins still
    # fire, so a plugin can replace the layer). `after_stage` is the
    # plugin Stage fired after that layer (None for no hook).
    @property
    def _layers(self):
        return (
            ('background',    self._draw_background,   None),
            ('lanes',         self._draw_lanes,        Stage.AFTER_LANES),
            ('judgment',      self._draw_judgment,     Stage.AFTER_JUDGMENT),
            ('notes',         self._draw_notes,        None),
            ('chart_extras',  self._draw_chart_extras, Stage.AFTER_NOTES),
            ('miss_holds',    self._draw_miss_holds,   None),
            ('ghost_taps',    self._draw_ghost_taps,   Stage.AFTER_GHOSTS),
            ('hud',           self._draw_hud,          Stage.HUD),
        )

    def draw(self, player, painter, t_now):
        ctx = self.build_context(player, painter, t_now)
        self.plugins.draw(Stage.PRE_FRAME, ctx)
        visibility = ctx.plugin_data.get('layer_visibility') or {}
        for name, fn, stage in self._layers:
            if visibility.get(name, True):
                fn(ctx, painter)
            if stage is not None:
                self.plugins.draw(stage, ctx)
        self.plugins.draw(Stage.POST_FRAME, ctx)

    @staticmethod
    def _draw_background(ctx, painter):
        p = ctx.player
        painter.fillRect(QRectF(0, 0, p.W, p.H), _qcolor(_BG_BASE))

    def _draw_lanes(self, ctx, painter):
        p = ctx.player
        for c in range(p.keycount):
            x = ctx.x0 + c * ctx.lane_w
            painter.fillRect(QRectF(x, 0, ctx.lane_w, p.H), _qcolor(_LANE_BG))
            _line(painter, _LANE_LINE, (x, 0), (x, p.H))
        x = ctx.x0 + p.keycount * ctx.lane_w
        _line(painter, _LANE_LINE, (x, 0), (x, p.H))

    def _draw_judgment(self, ctx, painter):
        p = ctx.player
        for name, w in reversed(p.windows):
            top = ctx.judge_y - w * p.scroll_speed
            bot = ctx.judge_y + w * p.scroll_speed
            color = p.judge_colors[name]
            painter.fillRect(QRectF(ctx.x0, top, p.keycount * ctx.lane_w,
                                    bot - top),
                             QColor(color[0], color[1], color[2], 24))
        _line(painter, (255, 255, 255), (ctx.x0, ctx.judge_y),
              (ctx.x0 + p.keycount * ctx.lane_w, ctx.judge_y), 2)

    def _draw_notes(self, ctx, painter):
        for i in ctx.candidates:
            n = self._build_note_view(ctx, i)
            if n is None:
                continue
            if n.is_ln:
                self._draw_ln_parts(ctx, painter, n)
            head_visible = self._draw_note_head_if_visible(ctx, painter, n)
            if head_visible:
                self._draw_press_mark(ctx, painter, n)
                if n.miss:
                    self._draw_miss_x(ctx, painter, n)

    def _build_note_view(self, ctx, i):
        """Gather the per-note values the draw helpers need. Returns
        `None` for notes that should be skipped entirely (off-lane,
        or a miss whose head is subsumed by an earlier miss's hold)."""
        p = ctx.player
        col = p._columns_list[i]
        if col >= p.keycount:
            return None
        if p.misses[i] and i < len(p._miss_head_suppressed) \
                and p._miss_head_suppressed[i]:
            return None

        note_t = p.times[i]
        end_t = p._ln_tail_times[i]
        is_ln = not math.isnan(end_t)
        off = p.offsets[i]
        press_t = note_t + off
        if is_ln:
            rel_off = p.hold_release_offsets.get((p._noterows_list[i], col))
            release_t = end_t + (rel_off or 0.0)
            y_end = ctx.time_to_y(end_t)
        else:
            rel_off = None
            release_t = None
            end_t = None
            y_end = 0  # unused when not an LN

        return _NoteView(
            i=i, col=col,
            y=ctx.time_to_y(note_t),
            y_end=y_end,
            lx=int(ctx.lane_x(col)),
            off=off,
            press_t=press_t,
            release_t=release_t,
            rel_off=rel_off,
            end_t=end_t,
            is_ln=is_ln,
            is_roll=bool(is_ln and p._roll_head_keys
                         and (p._noterows_list[i], col) in p._roll_head_keys),
            miss=bool(p.misses[i]),
            ln_state=self._ln_state(ctx, press_t, release_t, is_ln, p.misses[i]),
            note_color=p.palette[col],
            jcolor=p.judge_colors[p.note_judges[i]],
        )

    @staticmethod
    def _ln_state(ctx, press_t, release_t, is_ln, miss):
        if miss:
            return 'missed' if is_ln else 'missed_note'
        if not is_ln:
            return 'tap'
        if ctx.t_now < press_t:
            return 'upcoming'
        if ctx.t_now < release_t:
            return 'held'
        return 'released'

    def _draw_ln_parts(self, ctx, painter, n):
        """Body fill, tail sprite, and the release-guide line."""
        self._draw_ln_body_fill(ctx, painter, n)
        self._draw_ln_tail_sprite(ctx, painter, n)
        self._draw_ln_release_guide(ctx, painter, n)

    def _draw_ln_body_fill(self, ctx, painter, n):
        body = self._ln_body_span(ctx, n)
        if body is None:
            return
        top, bot, color = body
        if bot <= top:
            return
        ctx.drawers['ln_body'](painter, ctx.player.skin, n.lx, top, bot,
                               ctx.lane_w, color)

    def _draw_ln_tail_sprite(self, ctx, painter, n):
        hidden_by_press_hide = (ctx.player.press_hide
                                and n.ln_state == 'released' and not n.miss)
        on_screen = -ctx.screen_margin <= n.y_end <= ctx.player.H + ctx.screen_margin
        if hidden_by_press_hide or not on_screen:
            return
        ctx.drawers['ln_tail'](painter, ctx.player.skin, n.lx, n.y_end,
                               ctx.lane_w, ctx.note_h, self._ln_tail_color(n))

    def _draw_ln_release_guide(self, ctx, painter, n):
        """White tick + line from tail to where the player released the
        key. Only meaningful for successfully-held LNs in visible mode."""
        p = ctx.player
        has_release_offset = n.rel_off is not None
        eligible_state = n.ln_state != 'released' and not n.miss
        if not (has_release_offset and eligible_state and not p.press_hide):
            return
        rel_y = n.y_end + n.rel_off * p.scroll_speed
        ctx.drawers['ln_release_guide'](painter, n.lx, ctx.lane_w,
                                         n.y_end, rel_y)

    def _default_ln_release_guide(self, painter, lx, lane_w, y_tail, y_release):
        self._draw_lane_line(painter, _RELEASE_GUIDE, lx, lane_w,
                             y_tail, y_release)
        self._draw_tick(painter, _RELEASE_GUIDE, lx, y_release, lane_w)

    def _ln_body_span(self, ctx, n):
        """Return `(top_y, bot_y, color)` for the LN body fill, or None
        when no body should draw (e.g. released + press_hide)."""
        p = ctx.player
        held_color = _ROLL_BODY if n.is_roll else n.note_color
        released_color = _ROLL_TAIL if n.is_roll else _dim(n.note_color)

        if n.miss:
            return n.y_end, n.y, _MISS_LN_BODY
        if n.ln_state == 'upcoming':
            return n.y_end, n.y, held_color
        if n.ln_state == 'held':
            bot = ctx.judge_y if p.press_hide else n.y
            return n.y_end, bot, held_color
        if n.ln_state == 'released' and not p.press_hide:
            return n.y_end, ctx.judge_y, released_color
        return None

    @staticmethod
    def _ln_tail_color(n):
        if n.miss:
            return _MISS_LN_BODY
        if n.is_roll:
            return _ROLL_TAIL
        return _dim(n.note_color)

    def _draw_note_head_if_visible(self, ctx, painter, n):
        """Draw the note head sprite when appropriate; return whether it
        was drawn (the press-mark and miss-X both key off this)."""
        p = ctx.player
        head_visible, head_color, head_y = self._head_style(ctx, n)
        if head_visible:
            ctx.drawers['tap_head'](painter, p.skin, n.lx, head_y, ctx.lane_w,
                                     ctx.note_h, head_color)
        return head_visible

    @staticmethod
    def _head_style(ctx, n):
        """`(visible, color, y)` — the head is hidden during held LNs in
        press_hide mode (the held-body already covers the judge line)."""
        p = ctx.player
        if n.miss:
            color = _MISS_LN_BODY if n.is_ln else _MISS_TAP_BODY
            return True, color, n.y
        if p.press_hide:
            if n.is_ln:
                visible = n.ln_state in ('upcoming', 'held')
                y = ctx.judge_y if n.ln_state == 'held' else n.y
            else:
                visible = n.ln_state == 'tap' and ctx.t_now < n.press_t
                y = n.y
            return visible, n.note_color, y
        return n.ln_state in ('upcoming', 'tap', 'held'), n.note_color, n.y

    def _draw_press_mark(self, ctx, painter, n):
        """Thin vertical line + tick showing where the player hit relative
        to the note head. Skipped for missed LNs entirely for readability,
        and for any miss where the player never actually pressed — the
        parser writes a 1.0s sentinel offset for those, which would
        otherwise draw a full-second line that crosses unrelated notes."""
        p = ctx.player
        if n.miss and n.is_ln:
            return
        if n.miss and not p.miss_pressed[n.i]:
            return
        if n.is_ln and n.ln_state == 'held' and p.press_hide:
            return

        press_y = n.y + n.off * p.scroll_speed
        color = p.judge_colors['miss'] if n.miss else n.jcolor
        ctx.drawers['press_mark'](painter, n.lx, ctx.lane_w, n.y, press_y,
                                   color)

    def _default_press_mark(self, painter, lx, lane_w, y_head, y_press, color):
        self._draw_lane_line(painter, color, lx, lane_w, y_head, y_press)
        self._draw_tick(painter, color, lx, y_press, lane_w)

    def _draw_miss_x(self, ctx, painter, n):
        """Red outline box + X through the note head."""
        ctx.drawers['miss_x'](painter, n.lx, n.y, ctx.lane_w, ctx.note_h,
                               n.jcolor)

    @staticmethod
    def _default_miss_x(painter, lx, y, lane_w, note_h, jcolor):
        pad = 4
        hx, hy, hw, hh = _head_rect(lx, y, lane_w, note_h)
        _rect_outline(painter, _MISS_X_OUTLINE,
                      (hx - pad, hy - pad, hw + pad * 2, hh + pad * 2), 3)
        cx = lx + lane_w / 2
        _line(painter, jcolor, (cx - 10, y - 10), (cx + 10, y + 10), 2)
        _line(painter, jcolor, (cx - 10, y + 10), (cx + 10, y - 10), 2)

    def _draw_chart_extras(self, ctx, painter):
        """Mines, lifts, fakes — chart-only notes that never hit the
        replay stream. Each sweep is a pair of bisects into the
        time-sorted arrays."""
        p = ctx.player
        d = ctx.drawers

        def draw_mine(col, y):
            d['mine'](painter, ctx.lane_x(col), y, ctx.lane_w)

        def draw_lift(col, y):
            d['lift'](painter, p.skin, ctx.lane_x(col), y,
                      ctx.lane_w, ctx.note_h, p.palette[col])

        def draw_fake(col, y):
            d['fake'](painter, p.skin, ctx.lane_x(col), y,
                      ctx.lane_w, ctx.note_h, p.palette[col])

        self._sweep_chart_notes(ctx, p._mine_times, p._mine_cols, draw_mine)
        self._sweep_chart_notes(ctx, p._lift_times, p._lift_cols, draw_lift)
        self._sweep_chart_notes(ctx, p._fake_times, p._fake_cols, draw_fake)

    @staticmethod
    def _sweep_chart_notes(ctx, times, cols, draw_fn):
        if not times.size:
            return
        p = ctx.player
        lo = bisect.bisect_left(times, ctx.target_lo)
        hi = bisect.bisect_right(times, ctx.target_hi)
        for k in range(lo, hi):
            col = int(cols[k])
            if col >= p.keycount:
                continue
            draw_fn(col, ctx.time_to_y(float(times[k])))

    def _draw_miss_holds(self, ctx, painter):
        p = ctx.player
        ctx.visible_miss_holds = []
        if not p._miss_hold_press.size:
            return

        red = p.judge_colors['miss']
        draw = ctx.drawers['miss_hold_stroke']
        for k in self._visible_miss_hold_indices(ctx):
            col = int(p._miss_hold_cols[k])
            if col >= p.keycount:
                continue
            y_press = ctx.time_to_y(float(p._miss_hold_press[k]))
            y_release = ctx.time_to_y(float(p._miss_hold_release[k]))
            clipped = self._clip_to_screen(y_press, y_release, p.H)
            if clipped is None:
                continue
            top, bot = clipped
            draw(painter, int(ctx.lane_x(col)), ctx.lane_w, top, bot,
                 y_press, y_release, red)
            ctx.visible_miss_holds.append(k)

    def _default_miss_hold_stroke(self, painter, lx, lane_w, y_top, y_bot,
                                   y_press, y_release, color):
        self._draw_lane_line(painter, color, lx, lane_w, y_top, y_bot, width=2)
        self._draw_tick(painter, color, lx, y_press, lane_w)
        self._draw_tick(painter, color, lx, y_release, lane_w)

    @staticmethod
    def _visible_miss_hold_indices(ctx):
        """Indices of miss-holds whose press→release span could touch the
        screen, in the correct key space for the current scroll mode."""
        p = ctx.player
        if ctx.use_sv_space:
            press_key, release_key = p._miss_hold_press_sv, p._miss_hold_release_sv
            max_dur = p._miss_hold_max_sv_dur
        else:
            press_key, release_key = p._miss_hold_press, p._miss_hold_release
            max_dur = p._miss_hold_max_dur
        lo = bisect.bisect_left(press_key, ctx.target_lo - max_dur)
        hi = bisect.bisect_right(press_key, ctx.target_hi)
        return [k for k in range(lo, hi)
                if float(release_key[k]) >= ctx.target_lo]

    @staticmethod
    def _clip_to_screen(y_a, y_b, screen_h):
        """Order `(y_a, y_b)` top-to-bottom and clip to `[0, screen_h]`.
        Returns `None` when the span is fully off-screen."""
        top, bot = min(y_a, y_b), max(y_a, y_b)
        if bot < 0 or top > screen_h:
            return None
        return max(0, top), min(screen_h, bot)

    @staticmethod
    def _draw_tick(painter, color, lane_x, y, lane_w):
        """Short horizontal bar centered in the lane at `y`. Used to mark
        a discrete timing event (press, release, guide point)."""
        _rect(painter, color, (lane_x + 8, y - 2, lane_w - 16, 4))

    @staticmethod
    def _draw_lane_line(painter, color, lane_x, lane_w, y0, y1, width=1):
        """Vertical line down the lane's centerline from `y0` to `y1`.
        Used to connect two timing events (head→press, press→release)."""
        cx = lane_x + lane_w / 2
        _line(painter, color, (cx, y0), (cx, y1), width)

    def _draw_ghost_taps(self, ctx, painter):
        p = ctx.player
        ctx.visible_ghost_taps = []
        if not p._ghost_times.size:
            return
        ghost_key = p._ghost_sv_times if ctx.use_sv_space else p._ghost_times
        g_lo = bisect.bisect_left(ghost_key, ctx.target_lo)
        g_hi = bisect.bisect_right(ghost_key, ctx.target_hi)
        for k in range(g_lo, g_hi):
            gc = int(p._ghost_cols[k])
            if gc >= p.keycount:
                continue
            gy = ctx.time_to_y(float(p._ghost_times[k]))
            ctx.drawers['ghost_tap'](painter, ctx.lane_x(gc), gy,
                                      ctx.lane_w, ctx.note_h)
            ctx.visible_ghost_taps.append(k)

    def _draw_hud(self, ctx, painter):
        p = ctx.player
        sidebar_x = p.W - theme.SIDEBAR_WIDTH
        p.hud.clear_hitboxes()
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

        # Measure bottom first — top viewport ends where bottom starts, so
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
        paint their collapsed header here (regardless of open state) —
        the expanded panel is drawn separately by ``_draw_flyout`` and
        anchored next to the header, so the header must stay visible as
        the anchor point and re-click target. When a flyout is open,
        record the header's rect in ``plugin_data['flyout_anchors']``
        so ``_draw_flyout`` can position the panel against it."""
        p = sctx.player
        open_flyout = p.hud.open_flyout
        anchors = sctx.render_ctx.plugin_data.setdefault(
            'flyout_anchors', {})
        for section in sections:
            try:
                y_before = sctx.y
                section.draw(sctx)
                if (section.draw_expanded is not None
                        and section.key == open_flyout):
                    anchors[section.key] = (
                        sctx.sidebar_x, y_before,
                        sctx.sidebar_w, sctx.y - y_before,
                    )
            except Exception as exc:
                src = f' ({section.module})' if section.module else ''
                print(f'sidebar section failed: {section.name}{src}: {exc}')

    @staticmethod
    def _draw_note_head(painter, skin, lx, y, lane_w, note_h, color):
        if skin == 'circle':
            r = _circle_head_radius(lane_w)
            cx = lx + lane_w / 2
            _ellipse(painter, color, cx, y, r, r)
            _ellipse_outline(painter, (255, 255, 255), cx, y, r, r)
        else:
            rect = _head_rect(lx, y, lane_w, note_h)
            _rect(painter, color, rect)
            _rect_outline(painter, (255, 255, 255), rect)

    @staticmethod
    def _draw_ln_body(painter, skin, lx, y_top, y_bot, lane_w, color):
        if y_bot <= y_top:
            return
        if skin == 'circle':
            body_w = max(6, int(lane_w * 0.32))
            bx = lx + (lane_w - body_w) / 2
            _rect(painter, color, (bx, y_top, body_w, y_bot - y_top))
        else:
            _rect(painter, color, (lx + 6, y_top, lane_w - 12, y_bot - y_top))

    def _draw_ln_tail(self, painter, skin, lx, y, lane_w, note_h, color):
        self._draw_note_head(painter, skin, lx, y, lane_w, note_h, color)

    @staticmethod
    def _draw_ghost_tap(painter, lx, y, lane_w, _note_h):
        r = max(4, int(lane_w * 0.25))
        cx = lx + lane_w / 2
        _ellipse_outline(painter, (255, 255, 255), cx, y, r, r)
        _ellipse(painter, (255, 255, 255), cx, y, 2, 2)

    @staticmethod
    def _draw_mine(painter, lx, y, lane_w):
        """Per user spec: radius = col/4 for the whitish-gray outer
        shell, radius = col/8 for the red inner dot. Drawn filled so
        they stand out against the note palette."""
        cx = lx + lane_w / 2
        r_outer = max(4, int(lane_w / 4))
        r_inner = max(2, int(lane_w / 8))
        _ellipse(painter, (210, 210, 210), cx, y, r_outer, r_outer)
        _ellipse(painter, (220, 60, 60), cx, y, r_inner, r_inner)

    @staticmethod
    def _draw_lift(painter, skin, lx, y, lane_w, note_h, color):
        """Lifts score on release, not press. Draw as a hollow ring /
        rect outline so the player can tell them from filled taps at
        a glance. Keeps the lane color for column identity."""
        if skin == 'circle':
            r = _circle_head_radius(lane_w)
            cx = lx + lane_w / 2
            _ellipse_outline(painter, color, cx, y, r, r, 2)
            _ellipse_outline(painter, (255, 255, 255), cx, y,
                             max(2, r // 3), max(2, r // 3))
        else:
            rect = _head_rect(lx, y, lane_w, note_h)
            _rect_outline(painter, color, rect, 2)
            _rect_outline(painter, (255, 255, 255),
                          (lx + 8, y - note_h // 4,
                           lane_w - 16, note_h // 2), 1)

    @staticmethod
    def _draw_fake(painter, skin, lx, y, lane_w, note_h, color):
        """Fakes never judge; Etterna renders them at 25% opacity. We
        match that by dimming the lane color to ~1/4 and drawing a
        regular tap shape so the player still sees the gimmick."""
        dim = _dim(color, factor=4)
        if skin == 'circle':
            r = _circle_head_radius(lane_w)
            cx = lx + lane_w / 2
            _ellipse(painter, dim, cx, y, r, r)
            _ellipse_outline(painter, (90, 90, 90), cx, y, r, r)
        else:
            rect = _head_rect(lx, y, lane_w, note_h)
            _rect(painter, dim, rect)
            _rect_outline(painter, (90, 90, 90), rect)


def _fmt_num(x, decimals=2):
    """Render a scroll-speed scalar as an int when it's near-integer,
    else fixed-width decimal. Cross-mode translations produce values
    like 35.0212 that collapse to '35' but remain distinguishable from
    an actual 35 after the user nudges."""
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f'{x:.{decimals}f}'


def _line(painter, color, start, end, width=1):
    painter.setPen(QPen(_qcolor(color), int(width)))
    painter.drawLine(QPointF(float(start[0]), float(start[1])),
                     QPointF(float(end[0]), float(end[1])))


def _rect(painter, color, rect):
    x, y, w, h = _rect_tuple(rect)
    painter.setPen(QPen(QColor(0, 0, 0, 0)))
    painter.setBrush(QBrush(_qcolor(color)))
    painter.drawRect(QRectF(float(x), float(y), float(w), float(h)))


def _rect_outline(painter, color, rect, width=1):
    x, y, w, h = _rect_tuple(rect)
    painter.setPen(QPen(_qcolor(color), int(width)))
    painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
    painter.drawRect(QRectF(float(x), float(y), float(w), float(h)))


def _ellipse(painter, color, cx, cy, rx, ry):
    painter.setPen(QPen(QColor(0, 0, 0, 0)))
    painter.setBrush(QBrush(_qcolor(color)))
    painter.drawEllipse(QPointF(float(cx), float(cy)), float(rx), float(ry))


def _ellipse_outline(painter, color, cx, cy, rx, ry, width=1):
    painter.setPen(QPen(_qcolor(color), int(width)))
    painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
    painter.drawEllipse(QPointF(float(cx), float(cy)), float(rx), float(ry))


def _text(painter, text, color, x, baseline):
    painter.setPen(QPen(_qcolor(color), 1))
    painter.drawText(QPointF(float(x), float(baseline)), str(text))
