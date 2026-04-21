"""Ghost tap and ghost hold overlay render layers."""
from __future__ import annotations

import bisect


def draw_ghost_holds(ctx):
    p = ctx.player
    pygame = ctx.pygame
    ctx.visible_ghost_holds = []
    gh_red = ctx.colors['miss']
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
    for k in range(gh_lo, gh_hi):
        pt = float(p._ghost_hold_press[k])
        rt = float(p._ghost_hold_release[k])
        if float(gh_release_key[k]) < ctx.target_lo:
            continue
        gc = int(p._ghost_hold_cols[k])
        if gc >= p.keycount:
            continue
        y_press = ctx.time_to_y(pt)
        y_release = ctx.time_to_y(rt)
        glx = int(ctx.x0 + gc * ctx.lane_w)
        cx = int(glx + ctx.lane_w / 2)
        y_top = min(y_press, y_release)
        y_bot = max(y_press, y_release)
        if y_bot < 0 or y_top > p.H:
            continue
        y_top = int(max(0, y_top))
        y_bot = int(min(p.H, y_bot))
        pygame.draw.line(ctx.screen, gh_red, (cx, y_top), (cx, y_bot), 2)
        if not p._ghost_hold_extends_miss[k]:
            pygame.draw.rect(ctx.screen, gh_red,
                             (glx + 8, int(y_press) - 2,
                              int(ctx.lane_w - 16), 4))
        pygame.draw.rect(ctx.screen, gh_red,
                         (glx + 8, int(y_release) - 2,
                          int(ctx.lane_w - 16), 4))
        ctx.visible_ghost_holds.append(k)


def draw_ghost_taps(ctx):
    p = ctx.player
    ctx.visible_ghost_taps = []
    if not p._ghost_times.size:
        return

    ghost_key = p._ghost_sv_times if ctx.use_sv_space else p._ghost_times
    g_lo = bisect.bisect_left(ghost_key, ctx.target_lo)
    g_hi = bisect.bisect_right(ghost_key, ctx.target_hi)
    for k in range(g_lo, g_hi):
        gt = float(p._ghost_times[k])
        gc = int(p._ghost_cols[k])
        if gc >= p.keycount:
            continue
        gy = ctx.time_to_y(gt)
        glx = int(ctx.x0 + gc * ctx.lane_w)
        p.skin_obj.draw_ghost_tap(ctx.screen, glx, gy, ctx.lane_w,
                                  ctx.note_h)
        ctx.visible_ghost_taps.append(k)
