"""Native Qt renderer for the embedded replay player."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter, QPen)

from analysis.player import culling
from analysis.player.plugin_api import Stage
from analysis.player.render_context import RenderContext


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
        culling.prepare_time_window(ctx)
        ctx.candidates = culling.select_note_candidates(ctx)
        return ctx

    def draw(self, player, painter, t_now):
        painter.fillRect(QRectF(0, 0, player.W, player.H),
                         QColor(14, 14, 16))
        ctx = self.build_context(player, painter, t_now)

        self._draw_lanes(ctx, painter)
        self.plugins.draw(Stage.AFTER_LANES, ctx)

        self._draw_judgment(ctx, painter)
        self.plugins.draw(Stage.AFTER_JUDGMENT, ctx)

        self._draw_notes(ctx, painter)
        self.plugins.draw(Stage.AFTER_NOTES, ctx)

        self._draw_ghost_holds(ctx, painter)
        self._draw_ghost_taps(ctx, painter)
        self.plugins.draw(Stage.AFTER_GHOSTS, ctx)

        self._draw_hud(ctx, painter)
        self.plugins.draw(Stage.HUD, ctx)
        self.plugins.draw(Stage.POST_FRAME, ctx)

    def _draw_lanes(self, ctx, painter):
        p = ctx.player
        for c in range(p.keycount):
            x = ctx.x0 + c * ctx.lane_w
            painter.fillRect(QRectF(x, 0, ctx.lane_w, p.H),
                             QColor(22, 22, 24))
            _line(painter, (40, 40, 44), (x, 0), (x, p.H))
        x = ctx.x0 + p.keycount * ctx.lane_w
        _line(painter, (40, 40, 44), (x, 0), (x, p.H))

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
        p = ctx.player
        for i in ctx.candidates:
            note_t = p.times[i]
            c = p._columns_list[i]
            if c >= p.keycount:
                continue
            off = p.offsets[i]
            miss = p.misses[i]
            y = ctx.time_to_y(note_t)
            lx = int(ctx.x0 + c * ctx.lane_w)
            note_color = p.palette[c]

            end_t = p._ln_tail_times[i]
            is_ln = not math.isnan(end_t)
            if is_ln:
                rel_off = p.hold_release_offsets.get((p._noterows_list[i], c))
            else:
                rel_off = None
                end_t = None
            press_t = note_t + off
            release_t = (end_t + (rel_off or 0.0)) if is_ln else None

            if miss:
                ln_state = 'missed' if is_ln else 'missed_note'
            elif is_ln:
                if ctx.t_now < press_t:
                    ln_state = 'upcoming'
                elif ctx.t_now < release_t:
                    ln_state = 'held'
                else:
                    ln_state = 'released'
            else:
                ln_state = 'tap'

            jcolor = p.judge_colors[p.note_judges[i]]
            miss_red = p.judge_colors['miss']
            dim_color = tuple(v // 2 for v in note_color)
            miss_tap_color = (77, 77, 77)
            miss_ln_color = (38, 38, 38)

            if is_ln:
                y_end = ctx.time_to_y(end_t)
                if miss:
                    body_top, body_bot, body_color = y_end, y, miss_ln_color
                elif ln_state == 'upcoming':
                    body_top, body_bot, body_color = y_end, y, note_color
                elif ln_state == 'held':
                    if p.press_hide:
                        body_top, body_bot, body_color = y_end, ctx.judge_y, note_color
                    else:
                        body_top, body_bot, body_color = y_end, y, note_color
                elif ln_state == 'released':
                    if p.press_hide:
                        body_top = body_bot = None
                        body_color = None
                    else:
                        body_top, body_bot, body_color = y_end, ctx.judge_y, dim_color
                else:
                    body_top = body_bot = None
                    body_color = None

                if body_color is not None and body_bot > body_top:
                    self._draw_ln_body(painter, p.skin, lx, body_top,
                                       body_bot, ctx.lane_w, body_color)

                tail_visible = not (p.press_hide and ln_state == 'released'
                                    and not miss)
                tail_on_screen = (-ctx.screen_margin <= y_end
                                  <= p.H + ctx.screen_margin)
                if tail_visible and tail_on_screen:
                    tail_color = miss_ln_color if miss else dim_color
                    self._draw_ln_tail(painter, p.skin, lx, y_end,
                                       ctx.lane_w, ctx.note_h, tail_color)

                if (rel_off is not None and ln_state != 'released'
                        and not miss and not p.press_hide):
                    rel_y = y_end + rel_off * p.scroll_speed
                    cx = lx + ctx.lane_w / 2
                    _line(painter, (220, 220, 220), (cx, y_end), (cx, rel_y))
                    _rect(painter, (220, 220, 220),
                          (lx + 8, rel_y - 2, ctx.lane_w - 16, 4))

            head_y = y
            if is_ln and ln_state == 'held' and p.press_hide and not miss:
                head_y = ctx.judge_y

            if miss:
                head_visible = True
                head_color = miss_tap_color if not is_ln else miss_ln_color
            elif p.press_hide:
                if is_ln:
                    head_visible = ln_state in ('upcoming', 'held')
                else:
                    head_visible = ln_state == 'tap' and ctx.t_now < press_t
                head_color = note_color
            else:
                head_visible = ln_state in ('upcoming', 'tap', 'held')
                head_color = note_color

            if head_visible:
                self._draw_note_head(painter, p.skin, lx, head_y, ctx.lane_w,
                                     ctx.note_h, head_color)

            miss_has_press = bool(miss and p.miss_pressed[i])
            show_press_mark = ((not miss or miss_has_press) and head_visible
                               and not (is_ln and ln_state == 'held'
                                        and p.press_hide))
            if show_press_mark:
                joins_ghost_hold = bool(
                    miss_has_press and is_ln and p._miss_first_ghost_hold[i] >= 0)
                press_y = (ctx.time_to_y(press_t) if joins_ghost_hold
                           else y + off * p.scroll_speed)
                line_color = miss_red if miss else jcolor
                cx = lx + ctx.lane_w / 2
                _line(painter, line_color, (cx, y), (cx, press_y),
                      2 if joins_ghost_hold else 1)
                if not joins_ghost_hold:
                    _rect(painter, line_color,
                          (lx + 8, press_y - 2, ctx.lane_w - 16, 4))

            if miss and head_visible:
                pad = 4
                _rect_outline(painter, (255, 60, 60, 110),
                              (lx + 4 - pad, y - ctx.note_h // 2 - pad,
                               ctx.lane_w - 8 + pad * 2,
                               ctx.note_h + pad * 2), 3)
                cx = lx + ctx.lane_w / 2
                _line(painter, jcolor, (cx - 10, y - 10),
                      (cx + 10, y + 10), 2)
                _line(painter, jcolor, (cx - 10, y + 10),
                      (cx + 10, y - 10), 2)

    def _draw_ghost_holds(self, ctx, painter):
        import bisect

        p = ctx.player
        ctx.visible_ghost_holds = []
        if not p._ghost_hold_press.size:
            return
        if ctx.use_sv_space:
            gh_press_key = p._ghost_hold_press_sv
            gh_release_key = p._ghost_hold_release_sv
            gh_max_dur = p._ghost_hold_max_sv_dur
        else:
            gh_press_key = p._ghost_hold_press
            gh_release_key = p._ghost_hold_release
            gh_max_dur = p._ghost_hold_max_dur
        gh_hi = bisect.bisect_right(gh_press_key, ctx.target_hi)
        gh_lo = bisect.bisect_left(gh_press_key, ctx.target_lo - gh_max_dur)
        gh_red = p.judge_colors['miss']
        for k in range(gh_lo, gh_hi):
            if float(gh_release_key[k]) < ctx.target_lo:
                continue
            gc = int(p._ghost_hold_cols[k])
            if gc >= p.keycount:
                continue
            y_press = ctx.time_to_y(float(p._ghost_hold_press[k]))
            y_release = ctx.time_to_y(float(p._ghost_hold_release[k]))
            glx = int(ctx.x0 + gc * ctx.lane_w)
            cx = glx + ctx.lane_w / 2
            y_top = min(y_press, y_release)
            y_bot = max(y_press, y_release)
            if y_bot < 0 or y_top > p.H:
                continue
            y_top = max(0, y_top)
            y_bot = min(p.H, y_bot)
            _line(painter, gh_red, (cx, y_top), (cx, y_bot), 2)
            if not p._ghost_hold_extends_miss[k]:
                _rect(painter, gh_red,
                      (glx + 8, y_press - 2, ctx.lane_w - 16, 4))
            _rect(painter, gh_red,
                  (glx + 8, y_release - 2, ctx.lane_w - 16, 4))
            ctx.visible_ghost_holds.append(k)

    def _draw_ghost_taps(self, ctx, painter):
        import bisect

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
            self._draw_ghost_tap(painter, ctx.lane_x(gc), gy,
                                 ctx.lane_w, ctx.note_h)
            ctx.visible_ghost_taps.append(k)

    def _draw_hud(self, ctx, painter):
        p = ctx.player
        sidebar_x = p.W - 210
        p._hud_hitboxes = []
        _rect(painter, (20, 20, 22), (sidebar_x, 0, 210, p.H))
        painter.setFont(self.font)
        y = 14
        if not p.sv_sections or p.sv_suspended():
            sv_line = 'SV: n/a'
        else:
            sv_line = 'SV: on' if p.sv_enabled else 'SV: off'
        lines = [
            f't = {ctx.t_now:+7.3f}s',
            f'speed = {p.play_rate:.2f}x',
            f'notes = {len(p.times)}',
            f'keycount = {p.keycount}',
            sv_line,
            f'{"PAUSED" if p.paused else "PLAYING"}',
        ]
        for line in lines:
            _text(painter, line, (220, 220, 220), sidebar_x + 8, y + 13)
            y += 18

        y += 12
        painter.setFont(self.big_font)
        _text(painter, 'Judgments', (255, 171, 145), sidebar_x + 8, y + 18)
        painter.setFont(self.font)
        y += 26
        counts = {n: 0 for n, _ in p.windows}
        counts['miss'] = 0
        for j in p.note_judges:
            counts[j] = counts.get(j, 0) + 1
        for name, w in p.windows:
            line = f'{name:<6}  ±{w*1000:5.1f}ms  n={counts[name]}'
            _text(painter, line, p.judge_colors[name], sidebar_x + 8, y + 13)
            y += 18
        _text(painter, f'miss             n={counts["miss"]}',
              p.judge_colors['miss'], sidebar_x + 8, y + 13)
        y += 30

        for h in [
            'Space: pause', 'L/R: seek', 'Sh+L/R: seek10',
            'Up/Dn: scrollspd', '+/-: playspd',
            'M: mute', 'R: restart', 'Q: quit',
        ]:
            _text(painter, h, (120, 120, 130), sidebar_x + 8, y + 13)
            y += 16

        y += 12
        y = self._draw_plugin_controls(ctx, painter, sidebar_x, y)
        # Scroll controls: pinned near the bottom of the sidebar so they land
        # where the empty space is, regardless of how far the plugin list
        # extends.
        self._draw_scroll_controls(ctx, painter, sidebar_x)
        ctx.plugin_data['hud_y'] = y
        ctx.plugin_data['sidebar_x'] = sidebar_x

    def _draw_scroll_controls(self, ctx, painter, sidebar_x):
        p = ctx.player
        from analysis.player import scroll as scroll_registry
        mode = scroll_registry.get(p.scroll_mode)
        # Block: title + value + (− +) + scroll-type + rate-value + (− +) + game.
        block_h = 156
        y = p.H - block_h - 12
        col_x = sidebar_x + 8
        col_w = 190
        half_w = (col_w - 4) // 2
        btn_left = (col_x, 0, half_w, 20)
        btn_right = (col_x + half_w + 4, 0, col_w - half_w - 4, 20)

        _text(painter, 'Scroll', (255, 171, 145), col_x, y + 13)
        y += 20
        val_str = (mode.format_value(p._current_mode_value())
                   if mode and mode.format_value
                   else f'{p._current_mode_value():.2f}')
        _text(painter, f'{val_str} ({int(p.effective_scroll_ms)}ms)',
              (220, 220, 220), col_x + 2, y + 13)
        y += 20
        for (rx, _ry, rw, rh), label, factor in (
            (btn_left, 'scroll −', 1 / 1.15),
            (btn_right, 'scroll +', 1.15),
        ):
            rect = (rx, y, rw, rh)
            _rect(painter, (32, 32, 36), rect)
            _rect_outline(painter, (68, 68, 76), rect)
            _text(painter, label, (220, 220, 220), rect[0] + 8, rect[1] + 14)
            p._hud_hitboxes.append((rect, 'scroll_nudge', factor))
        y += 24
        mode_label = f'scroll type: {mode.label if mode else p.scroll_mode}'
        rect = (col_x, y, col_w, 20)
        _rect(painter, (32, 32, 36), rect)
        _rect_outline(painter, (68, 68, 76), rect)
        _text(painter, mode_label, (220, 220, 220), col_x + 8, y + 14)
        p._hud_hitboxes.append((rect, 'cycle_scroll_mode', None))
        y += 24
        # Rate row: `- {rate}x +` (rate readout is the left column, "+" on right,
        # "-" on... per spec: "- button {rate}x + button". Put − and + on the
        # outside, rate text centered between them.
        minus_rect = (col_x, y, 28, 20)
        plus_rect = (col_x + col_w - 28, y, 28, 20)
        _rect(painter, (32, 32, 36), minus_rect)
        _rect_outline(painter, (68, 68, 76), minus_rect)
        _text(painter, '−', (220, 220, 220), minus_rect[0] + 10,
              minus_rect[1] + 14)
        p._hud_hitboxes.append((minus_rect, 'rate_nudge', -0.1))
        _rect(painter, (32, 32, 36), plus_rect)
        _rect_outline(painter, (68, 68, 76), plus_rect)
        _text(painter, '+', (220, 220, 220), plus_rect[0] + 10,
              plus_rect[1] + 14)
        p._hud_hitboxes.append((plus_rect, 'rate_nudge', 0.1))
        rate_txt = f'{p.play_rate:.2f}x'
        # crude center: col_w mid minus ~half of text width (6px/char).
        tx = col_x + (col_w - len(rate_txt) * 6) // 2
        _text(painter, rate_txt, (220, 220, 220), tx, y + 14)
        y += 24
        # Game cycle.
        try:
            from analysis.core import game as game_mod
            games = list(game_mod.all_games().keys())
        except Exception:
            games = []
        game_label = f'game: {p.game}' if p.game in games or not games else f'game: {p.game}'
        rect = (col_x, y, col_w, 20)
        _rect(painter, (32, 32, 36), rect)
        _rect_outline(painter, (68, 68, 76), rect)
        _text(painter, game_label, (220, 220, 220), col_x + 8, y + 14)
        if len(games) > 1:
            p._hud_hitboxes.append((rect, 'cycle_game', None))

    def _draw_plugin_controls(self, ctx, painter, sidebar_x, y):
        p = ctx.player
        plugins = self.plugins.all_plugins()
        enabled = self.plugins.enabled_count()
        total = len(plugins)
        header = (sidebar_x + 8, y, 190, 20)
        _rect(painter, (32, 32, 36), header)
        _rect_outline(painter, (68, 68, 76), header)
        p._hud_hitboxes.append((header, 'toggle_plugin_panel', None))
        marker = '[-]' if p.plugin_panel_open else '[+]'
        _text(painter, f'{marker} Plugins {enabled}/{total}',
              (220, 220, 220), sidebar_x + 14, y + 14)
        y += 24
        if not p.plugin_panel_open:
            return y
        if not plugins:
            _text(painter, 'no plugins found', (120, 120, 130),
                  sidebar_x + 18, y + 13)
            return y + 18
        row_h = 18
        max_rows = max(0, (p.H - y - 8) // row_h)
        for plugin in plugins[:max_rows]:
            row = (sidebar_x + 10, y, 188, row_h)
            p._hud_hitboxes.append((row, 'toggle_plugin', plugin.key))
            box = (sidebar_x + 14, y + 3, 10, 10)
            _rect(painter, (16, 16, 18), box)
            _rect_outline(painter, (110, 110, 120), box)
            if plugin.enabled:
                _line(painter, (160, 230, 160),
                      (box[0] + 2, box[1] + 5),
                      (box[0] + 4, box[1] + 8), 2)
                _line(painter, (160, 230, 160),
                      (box[0] + 4, box[1] + 8),
                      (box[0] + 9, box[1] + 2), 2)
            color = (210, 210, 215) if plugin.enabled else (110, 110, 116)
            _text(painter, _shorten(plugin.name, 22), color,
                  sidebar_x + 30, y + 13)
            y += row_h
        if len(plugins) > max_rows:
            _text(painter, f'+{len(plugins) - max_rows} more',
                  (120, 120, 130), sidebar_x + 18, y + 13)
            y += row_h
        return y

    @staticmethod
    def _draw_note_head(painter, skin, lx, y, lane_w, note_h, color):
        if skin == 'circle':
            r = max(6, int((lane_w - 4) * 0.46))
            cx = lx + lane_w / 2
            _ellipse(painter, color, cx, y, r, r)
            _ellipse_outline(painter, (255, 255, 255), cx, y, r, r)
        else:
            rect = (lx + 4, y - note_h // 2, lane_w - 8, note_h)
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


def _fmt_num(x, decimals=2):
    """Render a scroll-speed scalar as an int when it's near-integer,
    else fixed-width decimal. Cross-mode translations produce values
    like 35.0212 that collapse to '35' but remain distinguishable from
    an actual 35 after the user nudges."""
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f'{x:.{decimals}f}'


def _shorten(text, max_chars):
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max(0, max_chars - 1)] + '~'


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
