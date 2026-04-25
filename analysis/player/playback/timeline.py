"""Render timeline: samples chart time into render-space cumulative.

Frame state contract:

    visual position = (cumulative_at(note_t) - visual_cum_now)
                      * render_multiplier_at(raw_t) * scroll_speed

`visual_cum_now` is computed by `CullSpacePredictor`: anchored on audio
callbacks (and discontinuities), linearly extrapolated between using the
engine's local rate, with breakpoint crossings handled exactly. This is
mathematically equivalent to `cumulative_at(raw_t)` on a constant-rate
segment, but reads the noisy stream-clock once per callback rather than
once per render frame, so per-frame jitter is suppressed.

`target_cum` -- the exact engine sample -- is also exposed for callers
that want the "ground truth" reading (e.g. the cull-space window's lo/hi
bounds, where stability is more important than smoothness).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from analysis.player.playback.cull_predictor import CullSpacePredictor


class _RenderTimelineEngine(Protocol):
    enabled: bool

    def cumulative_at(self, t: float) -> float: ...
    def cumulative_velocity_at(self, t: float) -> float: ...
    def render_multiplier_at(self, t: float) -> float: ...


@dataclass(frozen=True, slots=True)
class RenderFrameState:
    raw_t: float
    target_cum: float
    visual_cum_now: float
    render_multiplier: float
    px_per_cum: float
    use_sv: bool


def _engine_breakpoints(engine):
    if engine is None:
        return None
    fn = getattr(engine, 'breakpoints', None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


class RenderTimeline:
    """Sampler from chart time to render-space, backed by a per-frame
    cull-space predictor."""

    def __init__(self, engine: _RenderTimelineEngine | None) -> None:
        self._engine = engine
        self._predictor = CullSpacePredictor(
            engine, breakpoints=_engine_breakpoints(engine))

    def set_engine(self, engine: _RenderTimelineEngine | None) -> None:
        self._engine = engine
        self._predictor = CullSpacePredictor(
            engine, breakpoints=_engine_breakpoints(engine))

    def reset(self, raw_t: float) -> None:
        """Force the predictor to re-anchor at `raw_t`. Call after seek,
        rate change, engine swap, or any discontinuity that doesn't
        produce a raw_t jump on its own."""
        self._predictor.reset(float(raw_t))

    def render_at(self, *, raw_t: float, scroll_speed: float,
                  use_sv: bool, play_rate: float = 1.0) -> RenderFrameState:
        raw_t = float(raw_t)
        scroll_speed = float(scroll_speed)
        play_rate = float(play_rate)
        engine = self._engine
        if not use_sv or engine is None or not getattr(engine, 'enabled', False):
            return RenderFrameState(
                raw_t=raw_t,
                target_cum=raw_t,
                visual_cum_now=raw_t,
                render_multiplier=1.0,
                px_per_cum=scroll_speed,
                use_sv=False,
            )

        target_cum = float(engine.cumulative_at(raw_t))
        visual_cum_now = float(self._predictor.cumulative_now(raw_t, play_rate))
        render_multiplier = float(engine.render_multiplier_at(raw_t))
        return RenderFrameState(
            raw_t=raw_t,
            target_cum=target_cum,
            visual_cum_now=visual_cum_now,
            render_multiplier=render_multiplier,
            px_per_cum=render_multiplier * scroll_speed,
            use_sv=True,
        )

    def frame(self, *, raw_t: float, scroll_speed: float,
              use_sv: bool, play_rate: float = 1.0) -> RenderFrameState:
        """Back-compat alias for older call sites/tests."""
        return self.render_at(
            raw_t=raw_t,
            scroll_speed=scroll_speed,
            use_sv=use_sv,
            play_rate=play_rate,
        )


# Back-compat name while the rest of the codebase migrates.
RenderPlayhead = RenderTimeline
