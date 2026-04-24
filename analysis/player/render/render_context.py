"""Per-frame drawing context shared by the Qt renderer and user plugins."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RenderContext:
    player: object
    screen: object
    pygame: object
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
    visible_miss_holds: list[int] = field(default_factory=list)
    visible_ghost_taps: list[int] = field(default_factory=list)
    plugin_data: dict = field(default_factory=dict)
    _scroll_speed: float = 0.0

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
        judge_y = self.player.H * self.player.hit_line_y_frac
        frame = self.frame if self.frame is not None else self.player.render_frame_state(self.t_now)
        return judge_y - self.player._visual_sv_distance_from_frame(
            frame,
            float(t),
        ) * self._scroll_speed

    def lane_x(self, col):
        return self.x0 + int(col) * self.lane_w

    def lane_center(self, col):
        return self.lane_x(col) + self.lane_w / 2
