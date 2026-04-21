"""Note, long-note, press-marker, and miss-marker render layer."""
from __future__ import annotations

import math


def draw(ctx):
    p = ctx.player
    pygame = ctx.pygame
    times_ = p.times
    offsets_ = p.offsets
    misses_ = p.misses
    columns_ = p._columns_list
    noterows_ = p._noterows_list
    ln_tails_ = p._ln_tail_times
    rel_offsets_ = p.hold_release_offsets
    palette_ = p.palette
    keycount_ = p.keycount
    time_to_y = ctx.time_to_y
    colors = ctx.colors

    for i in ctx.candidates:
        note_t = times_[i]
        c = columns_[i]
        if c >= keycount_:
            continue
        off = offsets_[i]
        miss = misses_[i]
        y = time_to_y(note_t)
        lx = int(ctx.x0 + c * ctx.lane_w)
        note_color = palette_[c]

        end_t = ln_tails_[i]
        is_ln = not math.isnan(end_t)
        if is_ln:
            rel_off = rel_offsets_.get((noterows_[i], c))
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

        jname = p.note_judges[i]
        jcolor = colors[jname]
        miss_red = colors['miss']
        dim_color = (note_color[0] // 2, note_color[1] // 2,
                     note_color[2] // 2)
        miss_tap_color = (77, 77, 77)
        miss_ln_color = (38, 38, 38)

        if is_ln:
            y_end = time_to_y(end_t)
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
                p.skin_obj.draw_ln_body(ctx.screen, lx, body_top, body_bot,
                                        ctx.lane_w, ctx.note_h, body_color)

            tail_visible = not (p.press_hide and ln_state == 'released'
                                and not miss)
            tail_on_screen = (-ctx.screen_margin <= y_end
                              <= p.H + ctx.screen_margin)
            if tail_visible and tail_on_screen:
                tail_color = miss_ln_color if miss else dim_color
                p.skin_obj.draw_ln_tail(ctx.screen, lx, y_end, ctx.lane_w,
                                        ctx.note_h, tail_color)

            if (rel_off is not None and ln_state != 'released'
                    and not miss and not p.press_hide):
                rel_y = y_end + rel_off * p.scroll_speed
                pygame.draw.line(ctx.screen, (220, 220, 220),
                                 (int(lx + ctx.lane_w / 2), int(y_end)),
                                 (int(lx + ctx.lane_w / 2), int(rel_y)), 1)
                pygame.draw.rect(ctx.screen, (220, 220, 220),
                                 (lx + 8, int(rel_y) - 2,
                                  int(ctx.lane_w - 16), 4))

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
            p.skin_obj.draw_note_head(ctx.screen, lx, head_y, ctx.lane_w,
                                      ctx.note_h, head_color)

        miss_has_press = bool(miss and p.miss_pressed[i])
        show_press_mark = ((not miss or miss_has_press) and head_visible
                           and not (is_ln and ln_state == 'held'
                                    and p.press_hide))
        if show_press_mark:
            joins_ghost_hold = bool(
                miss_has_press and is_ln and p._miss_first_ghost_hold[i] >= 0)
            press_y = (time_to_y(press_t) if joins_ghost_hold
                       else y + off * p.scroll_speed)
            line_color = miss_red if miss else jcolor
            pygame.draw.line(ctx.screen, line_color,
                             (int(lx + ctx.lane_w / 2), int(y)),
                             (int(lx + ctx.lane_w / 2), int(press_y)),
                             2 if joins_ghost_hold else 1)
            if not joins_ghost_hold:
                pygame.draw.rect(ctx.screen, line_color,
                                 (lx + 8, int(press_y) - 2,
                                  int(ctx.lane_w - 16), 4))

        if miss and head_visible:
            pad = 4
            ow = int(ctx.lane_w - 8) + pad * 2
            oh = ctx.note_h + pad * 2
            halo = pygame.Surface((ow, oh), pygame.SRCALPHA)
            pygame.draw.rect(halo, (255, 60, 60, 110), halo.get_rect(),
                             width=3)
            ctx.screen.blit(halo, (lx + 4 - pad,
                                   int(y) - ctx.note_h // 2 - pad))
            cx = lx + ctx.lane_w / 2
            pygame.draw.line(ctx.screen, jcolor,
                             (cx - 10, y - 10), (cx + 10, y + 10), 2)
            pygame.draw.line(ctx.screen, jcolor,
                             (cx - 10, y + 10), (cx + 10, y - 10), 2)
