"""Visible-window selection for replay-player drawing."""
from __future__ import annotations

import bisect

import numpy as np


def prepare_time_window(ctx):
    p = ctx.player
    sps = max(1e-3, ctx.scroll_speed)
    sv_hi = (ctx.judge_y + ctx.screen_margin) / sps
    sv_lo = (ctx.judge_y - (p.H + ctx.screen_margin)) / sps
    ctx.use_sv_space = bool(p.sv_enabled and p._sv_engine.enabled)
    if ctx.use_sv_space:
        cum_now = p._cumulative_sv_at(float(ctx.t_now))
        ctx.target_lo = cum_now + sv_lo
        ctx.target_hi = cum_now + sv_hi
    else:
        ctx.target_lo = ctx.t_now + sv_lo
        ctx.target_hi = ctx.t_now + sv_hi


def select_note_candidates(ctx):
    p = ctx.player
    # Pad the window by the largest press/release offset so notes whose
    # head has scrolled off but whose hit-line extension is still on
    # screen stay in the candidate set.
    pad = getattr(p, 'max_draw_pad_sec', 0.0)
    lo_t, hi_t = ctx.target_lo - pad, ctx.target_hi + pad
    if ctx.use_sv_space:
        lo = int(np.searchsorted(p._note_sv_cum, lo_t, side='left'))
        hi = int(np.searchsorted(p._note_sv_cum, hi_t, side='right'))
    else:
        lo = bisect.bisect_left(p.times, lo_t)
        hi = bisect.bisect_right(p.times, hi_t)
    candidates = list(range(lo, hi))
    seen = set(candidates)

    if p._ln_indices:
        ln_idx = p._ln_indices
        ln_lo = bisect.bisect_left(ln_idx, lo)
        ln_hi = bisect.bisect_right(ln_idx, hi)
        for k in range(ln_lo - 1, -1, -1):
            i = ln_idx[k]
            if i in seen:
                continue
            if p.times[i] < ctx.t_now - 60.0:
                break
            if _ln_intersects_screen(ctx, i):
                candidates.append(i)
                seen.add(i)
        for k in range(ln_hi, len(ln_idx)):
            i = ln_idx[k]
            if i in seen:
                continue
            if p.times[i] > ctx.t_now + 60.0:
                break
            if _ln_intersects_screen(ctx, i):
                candidates.append(i)
                seen.add(i)

    return candidates


def _ln_intersects_screen(ctx, i):
    p = ctx.player
    y_head = p._time_to_y(p.times[i], ctx.t_now)
    y_tail = p._time_to_y(p._ln_tail_times[i], ctx.t_now)
    top_y = min(y_head, y_tail)
    bot_y = max(y_head, y_tail)
    return (bot_y >= -ctx.screen_margin
            and top_y <= p.H + ctx.screen_margin)
