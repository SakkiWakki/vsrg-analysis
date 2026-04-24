"""Pure sampled render timeline.

Audio/chart time is the only playhead. Rendering samples that chart time and
projects it into the active SV engine's render space:

    visual position = (cumulative_at(note_t) - cumulative_at(raw_t))
                      * render_multiplier_at(raw_t) * scroll_speed

That makes rendering a pure function of chart time. There is no frame-to-frame
prediction state here, so `render_at(t)` always means "draw the chart exactly
as it should look at chart time t".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _RenderTimelineEngine(Protocol):
    enabled: bool

    def cumulative_at(self, t: float) -> float: ...
    def render_multiplier_at(self, t: float) -> float: ...


@dataclass(frozen=True)
class RenderFrameState:
    raw_t: float
    target_cum: float
    visual_cum_now: float
    render_multiplier: float
    px_per_cum: float
    use_sv: bool


class RenderTimeline:
    """Stateless sampler from chart time to render-space."""

    def __init__(self, engine: _RenderTimelineEngine | None) -> None:
        self._engine = engine

    def set_engine(self, engine: _RenderTimelineEngine | None) -> None:
        self._engine = engine

    def reset(self, raw_t: float) -> None:
        # Kept for compatibility with older callers. The sampler is stateless.
        del raw_t

    def render_at(self, *, raw_t: float, scroll_speed: float,
                  use_sv: bool) -> RenderFrameState:
        raw_t = float(raw_t)
        scroll_speed = float(scroll_speed)
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
        render_multiplier = float(engine.render_multiplier_at(raw_t))
        return RenderFrameState(
            raw_t=raw_t,
            target_cum=target_cum,
            visual_cum_now=target_cum,
            render_multiplier=render_multiplier,
            px_per_cum=render_multiplier * scroll_speed,
            use_sv=True,
        )

    def frame(self, *, raw_t: float, scroll_speed: float,
              use_sv: bool) -> RenderFrameState:
        """Back-compat alias for older call sites/tests."""
        return self.render_at(
            raw_t=raw_t,
            scroll_speed=scroll_speed,
            use_sv=use_sv,
        )


# Back-compat name while the rest of the codebase migrates.
RenderPlayhead = RenderTimeline
