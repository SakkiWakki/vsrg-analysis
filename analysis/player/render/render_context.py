"""Per-frame drawing context shared by the Qt renderer and user plugins."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RenderContext:
    player: object
    colors: dict
    t_now: float
    x0: float
    lane_w: float
    judge_y: int
    painter: object | None = None
    note_h: int = 14
    screen_margin: int = 80
    target_lo: float = 0.0
    target_hi: float = 0.0
    visual_cum_now: float = 0.0
    frame: object | None = None
    use_sv_space: bool = False
    candidates: list[int] = field(default_factory=list)
    # Unified chart-stream culling output (mines/lifts/fakes; see
    # culling.select_stream_candidates): indices into the NotesModel
    # stream table + parallel head-in-window flags. Their records ride
    # the candidate y/mod arrays at positions len(candidates) onward.
    stream_candidates: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))
    stream_head_in_window: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool))
    # Per-frame stream views built by layers/notes.prepare.
    stream_views: list = field(default_factory=list)
    visible_miss_holds: list[int] = field(default_factory=list)
    visible_ghost_taps: list[int] = field(default_factory=list)
    plugin_data: dict = field(default_factory=dict)
    _scroll_speed: float = 0.0
    # Animated per-column geometry for lane-switch charts; None keeps
    # the uniform layout. Filled by build_context from the player's
    # lane-mask timeline (see analysis/player/render/lane_layout.py).
    lane_xs: tuple | None = None
    lane_ws: tuple | None = None
    # Per-field-layer opacity (our-layer-name -> alpha) for layerfade
    # effects; None keeps every layer opaque. Written by LayerFadeEffect
    # in build_context (like lane_xs), applied per layer in the draw loop.
    layer_opacities: dict | None = None
    # Composited effect frame for this paint (transform + z-ordered
    # overlay draws); None when no effect is active. Set by the renderer.
    effect_frame: object | None = None

    @property
    def width(self):
        return self.player.W

    @property
    def height(self):
        return self.player.H

    @property
    def keycount(self):
        return self.player.keycount

    @property
    def scroll_speed(self):
        return self._scroll_speed

    def time_to_y(self, t):
        judge_y = self.player.judge_y_px()
        frame = self.frame if self.frame is not None else self.player.render_frame_state(self.t_now)
        return judge_y - self.player._visual_sv_distance_from_frame(
            frame,
            float(t),
        ) * self._scroll_speed

    @property
    def chart_rect(self):
        """The replay viewport `(x, y, w, h)`: window minus sidebar.
        Effects, backgrounds, and clips reference this, never the raw
        window, so nothing replay-visual leaks into the HUD."""
        return self.player.chart_rect

    def lane_x(self, col):
        if self.lane_xs is not None:
            return self.lane_xs[int(col)]
        return self.x0 + int(col) * self.lane_w

    def lane_width(self, col):
        """Current width of `col`'s lane: animated during lane switches,
        `lane_w` otherwise. Anything drawn to a lane's width (note
        sprites, LN bodies, lane strokes) sizes with this."""
        if self.lane_ws is not None:
            return self.lane_ws[int(col)]
        return self.lane_w

    def lane_center(self, col):
        return self.lane_x(col) + self.lane_width(col) / 2
