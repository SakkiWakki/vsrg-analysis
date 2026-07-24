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
    # Quaver-style engines allow negative SV; their _note_sv_cum is not
    # sorted in chart-time order, so the bisect would silently miss
    # notes. Fall back to a chart-time linear filter on the cumulative
    # array in that regime -- still O(N) per frame but correct, and N is
    # bounded by visible-window padding upstream.
    sv_monotonic = getattr(getattr(p, '_sv_engine', None),
                           'cumulative_monotonic', True)
    if ctx.use_sv_space and sv_monotonic:
        lo = int(np.searchsorted(p._note_sv_cum, lo_t, side='left'))
        hi = int(np.searchsorted(p._note_sv_cum, hi_t, side='right'))
        candidates = list(range(lo, hi))
    elif ctx.use_sv_space:
        # Non-monotonic cum: the visible set isn't a contiguous index
        # range, so a linear mask over the whole array is the simplest
        # correct fallback. Bounded by len(p.times); still O(N) per
        # frame but the constant is one numpy mask + flatnonzero.
        cum = p._note_sv_cum
        mask = (cum >= lo_t) & (cum <= hi_t)
        candidates = np.flatnonzero(mask).tolist()
        # For the LN-extension scan below: bracket by the candidates'
        # min/max chart-time so LN bodies starting before / ending after
        # the visible set still get their on-screen intersection check.
        if candidates:
            lo = min(candidates)
            hi = max(candidates) + 1
        else:
            lo = hi = 0
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


def select_stream_candidates(ctx) -> None:
    """Visible-window selection over the unified chart-stream table
    (mines/lifts/fakes). Fills `ctx.stream_candidates` (ascending
    indices into `player.notes.stream_*`) and the parallel
    `ctx.stream_head_in_window` flags (the record's head sprite is
    inside the cull window and not expired).

    Finite-span records (hold mines) are candidates whenever they
    exist, flagged head-out when the window misses their head: the
    span body can cross the screen while the head is far away, they
    are sparse, and the drawer still clips by screen y."""
    p = ctx.player
    n = p.notes
    times = n.stream_times
    if not times.size:
        ctx.stream_candidates = np.empty(0, dtype=np.int64)
        ctx.stream_head_in_window = np.empty(0, dtype=bool)
        return

    search = n.stream_sv if (ctx.use_sv_space and n.stream_sv.size) else times
    lo = int(np.searchsorted(search, ctx.target_lo, side='left'))
    hi = int(np.searchsorted(search, ctx.target_hi, side='right'))
    idx = np.arange(lo, hi, dtype=np.int64)
    idx = idx[n.stream_cols[idx] < p.keycount]
    idx = idx[float(ctx.t_now) < n.stream_until[idx]]

    spans = np.flatnonzero(np.isfinite(n.stream_end_times))
    spans = spans[n.stream_cols[spans] < p.keycount]
    if spans.size:
        cand = np.union1d(idx, spans)
        head = np.isin(cand, idx)
    else:
        cand = idx
        head = np.ones(idx.shape, dtype=bool)
    ctx.stream_candidates = cand
    ctx.stream_head_in_window = head


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
