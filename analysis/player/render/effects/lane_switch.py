"""Lane-switch collapse as an effect.

Unlike the affine transform effects, lane switching is per-column
non-affine geometry (each lane's width animates independently and the
row re-centers), so it doesn't fold into the shared `QTransform`.
Instead it writes `ctx.lane_xs` / `ctx.lane_ws` -- which `ctx.lane_x`
and the field layer already consult -- and returns an identity
`EffectFrame`. Keeping it in the effects list means it composes in the
same ordered pipeline as playfield transforms (a rotate applies on top
of the collapsed field, matching fluXis).
"""
from __future__ import annotations

from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.lane_layout import column_layout


class LaneSwitchEffect:
    def __init__(self, timeline):
        self._timeline = timeline

    def __bool__(self):
        return bool(self._timeline)

    def at(self, ctx) -> EffectFrame | None:
        layout = column_layout(self._timeline, ctx.player.keycount,
                               float(ctx.t_now), ctx.x0, ctx.lane_w)
        if layout is None:
            return None
        ctx.lane_xs, ctx.lane_ws = layout
        return EffectFrame()   # geometry written to ctx; no transform
