"""Visible-window selection for replay-player drawing."""
from __future__ import annotations

import bisect

import numpy as np


# When `px_per_cum` falls below this threshold the chart is effectively
# frozen on screen (a stop, a scrolls=0 region, or both). In that
# regime the SV-space window blows up to cover most of the chart,
# producing thousand-fold candidate explosions and frame-time spikes.
# Fall back to a time-domain window: clamp candidate selection to
# notes within +/- 5 s of the playhead. The visual result is identical
# (nothing's moving, so the window's exact extent past the screen
# doesn't matter) but the candidate count stays bounded.
_FROZEN_PX_PER_CUM = 1e-3
_FROZEN_TIME_LOOKBEHIND = 1.0
_FROZEN_TIME_LOOKAHEAD = 5.0


def prepare_time_window(ctx):
    p = ctx.player
    ctx.frame = p.render_frame_state(float(ctx.t_now))
    raw_px_per_cum = abs(ctx.frame.px_per_cum)
    # Detect "frozen" regime: stop / scrolls=0 / extreme zoom-out where
    # the cumulative-space window would otherwise expand past the
    # entire chart and force us to consider every note.
    frozen = raw_px_per_cum < _FROZEN_PX_PER_CUM
    px_per_cum = max(_FROZEN_PX_PER_CUM, raw_px_per_cum)
    sv_hi = (ctx.judge_y + ctx.screen_margin) / px_per_cum
    sv_lo = (ctx.judge_y - (p.H + ctx.screen_margin)) / px_per_cum
    use_sv_engine = bool(ctx.frame.use_sv)
    # `use_sv_space` toggles the candidate-selection bisect's index
    # (SV-cumulative array vs note-time array). When the engine is
    # frozen we override it to time so the bisect is bounded.
    ctx.use_sv_space = use_sv_engine and not frozen
    if frozen:
        # Time-domain clamp around the playhead.
        ctx.visual_cum_now = ctx.frame.visual_cum_now
        ctx.target_lo = ctx.t_now - _FROZEN_TIME_LOOKBEHIND
        ctx.target_hi = ctx.t_now + _FROZEN_TIME_LOOKAHEAD
    elif use_sv_engine:
        ctx.visual_cum_now = ctx.frame.visual_cum_now
        ctx.target_lo = ctx.visual_cum_now + sv_lo
        ctx.target_hi = ctx.visual_cum_now + sv_hi
        # Engine-specific real-time upper bound (Etterna caps at ~20 beats)

        max_t = p._sv_engine.max_visible_t_from(float(ctx.t_now))
        if max_t != float('inf'):
            cap_sv = p._cumulative_sv_at(max_t)
            if cap_sv < ctx.target_hi:
                ctx.target_hi = cap_sv
    else:
        ctx.visual_cum_now = ctx.frame.visual_cum_now
        ctx.target_lo = ctx.t_now + sv_lo
        ctx.target_hi = ctx.t_now + sv_hi


def select_note_candidates(ctx):
    p = ctx.player
    # Pad the window by the largest press/release offset so notes whose
    # head has scrolled off but whose hit-line extension is still on
    # screen stay in the candidate set.
    pad = _window_pad(ctx)
    lo_t, hi_t = ctx.target_lo - pad, ctx.target_hi + pad
    if ctx.use_sv_space:
        lo = int(np.searchsorted(p._note_sv_cum, lo_t, side='left'))
        hi = int(np.searchsorted(p._note_sv_cum, hi_t, side='right'))
    else:
        lo = bisect.bisect_left(p.times, lo_t)
        hi = bisect.bisect_right(p.times, hi_t)
    candidates = list(range(lo, hi))
    seen = set(candidates)

    if p.notes.ln_indices:
        ln_idx = p.notes.ln_indices
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


def _window_pad(ctx):
    p = ctx.player
    pad_sec = float(getattr(p, 'max_draw_pad_sec', 0.0) or 0.0)
    if pad_sec <= 0.0 or not ctx.use_sv_space:
        return pad_sec

    now_t = float(ctx.t_now)
    now_cum = float(ctx.visual_cum_now)
    lo_cum = float(p._cumulative_sv_at(now_t - pad_sec))
    hi_cum = float(p._cumulative_sv_at(now_t + pad_sec))
    return max(abs(now_cum - lo_cum), abs(hi_cum - now_cum))


def _ln_intersects_screen(ctx, i):
    p = ctx.player
    y_head = p._time_to_y(p.times[i], ctx.t_now, ctx.frame)
    y_tail = p._time_to_y(p.notes.ln_tail_times[i], ctx.t_now, ctx.frame)
    top_y = min(y_head, y_tail)
    bot_y = max(y_head, y_tail)
    return (bot_y >= -ctx.screen_margin
            and top_y <= p.H + ctx.screen_margin)
