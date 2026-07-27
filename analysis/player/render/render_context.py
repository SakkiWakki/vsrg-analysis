"""Per-frame drawing context shared by the Qt renderer and user plugins."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis.player.render import lane_path


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
    # Per-column receptor visibility, or None for fully visible. Its own
    # field rather than a term of the lane curve because a receptor's
    # visibility is not the lane's: NotITG's comes from the dark family
    # alone, and the stealth gradients an arrow at the same point picks up
    # never apply to it (see lane_path).
    receptor_alpha: object | None = None
    _lane_path: object | None = None
    _receptor_marks: object | None = None

    @property
    def lane_path(self):
        """This frame's lane curve (render/lane_path.py): the one answer
        to what a column is like, for receptors, heads, hold bodies, tail
        caps and travel paths alike.

        Defaults to the straight lane every game without note mods has -
        a scroll offset is pixels up the lane from the hit line - so a
        consumer may always ask. A game whose lane bends replaces it
        (NotITG's `note_mods.apply`)."""
        if self._lane_path is None:
            self._lane_path = lane_path.straight(
                self.lane_center, lambda offsets: self.judge_y - offsets)
        return self._lane_path

    @lane_path.setter
    def lane_path(self, path):
        self._lane_path = path
        self._receptor_marks = None

    @property
    def receptor_marks(self):
        """The lane curve at offset 0 for every column, sampled once per
        frame. Consumers ask for this rather than sampling themselves
        because receptors are read PER NOTE - a hold clamps its body at
        its own column's receptor line - and the curve costs per call."""
        if self._receptor_marks is None:
            columns = np.arange(self.keycount, dtype=np.int64)
            self._receptor_marks = self.lane_path.sample(
                columns, np.zeros(len(columns)))
        return self._receptor_marks

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
