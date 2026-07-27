"""The curve an arrow travels down a lane, as one sampleable object.

Everything drawn in a lane sits somewhere on this curve: the receptor at
y_offset 0, a note head at its own offset, a hold BODY as the span between
its head and tail offsets, a hold TAIL at the end of that span, and a
travelpath as the whole visible range. Today each of those is derived
separately - `ctx.receptor_offsets`, a `_NoteView`'s `lx`/`y`,
`ctx.hold_body_samples`, a tangent recomputed from that polyline - which is
four spellings of one question.

GAME-AGNOSTIC ON PURPOSE. Most games' lanes are a straight line, which is
this curve with no displacement, so the general form lives here and a game
with note mods supplies the bend. `straight` is that degenerate case and
needs nothing from a game but its lane geometry.

GEOMETRY ONLY - appearance does not ride the path. The engine genuinely
diverges there: NotITG's receptors take their visibility from the dark
family alone, and the stealth/glow terms the y-offset-0 pipeline computes
never apply to them (ReceptorArrowRow), even though their POSITION comes
through the same GetXPos as an arrow's. So a sample carries where a thing
is and how it is turned, and every consumer keeps its own alpha rules.

`y_offset` is the caller's own scroll space - whatever unit its notes are
already positioned in - and only `displace` gives it meaning. `straight`
treats it as pixels below the receptor line.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LaneSamples:
    """Where a batch of (column, y_offset) pairs sits, in screen px.

    Arrays, not one sample per call: a field asks for every receptor, head,
    tail and body subdivision it needs in ONE go, because the displacement
    behind them is a vectorized pipeline whose per-call cost dwarfs the
    per-row cost (`note_mods._stash_hold_body_samples` already batches every
    sample of every visible hold for exactly this reason).

    `rotation_deg` and `zoom` are the per-sample turn and scale the path
    imposes; a straight lane leaves them at 0 and 1."""

    x: np.ndarray
    y: np.ndarray
    rotation_deg: np.ndarray
    zoom: np.ndarray

    def __len__(self) -> int:
        return len(self.x)

    def at(self, index: int) -> tuple:
        return (float(self.x[index]), float(self.y[index]),
                float(self.rotation_deg[index]), float(self.zoom[index]))

    def tangent_deg(self, first: int, last: int) -> float:
        """The heading in degrees from sample `first` to sample `last`, for
        a cap that has to follow the curve rather than the lane.

        A folded hold's end segment runs back UP the lane, and taking the
        heading rather than the lane's own direction is what makes the cap
        flip without a special case (`note_feed._emit_ln_tail` derives its
        rotation this way today)."""
        dx = float(self.x[last]) - float(self.x[first])
        dy = float(self.y[last]) - float(self.y[first])
        if dx == 0.0 and dy == 0.0:
            return 0.0
        return float(np.degrees(np.arctan2(dy, dx)))


class LanePath:
    """The lane curve for one frame, sampled in batches.

    `lane_center(col) -> x` and `note_y(y_offset) -> y` place an UNDISPLACED
    arrow; `displace` is the optional game hook that bends it. Construct one
    per frame and sample it as many times as needed - it holds no per-sample
    state.

    `displace(cols, y_offsets) -> (dx, dy, rotation_deg, zoom)` takes two
    equal-length arrays and returns four, in screen px / degrees / unitless.
    None means a straight lane."""

    __slots__ = ('_lane_center', '_note_y', '_displace')

    def __init__(self, lane_center, note_y, displace=None):
        self._lane_center = lane_center
        self._note_y = note_y
        self._displace = displace

    def sample(self, cols, y_offsets) -> LaneSamples:
        """Where each (column, y_offset) pair lands."""
        cols = np.asarray(cols, dtype=np.int64)
        y_offsets = np.asarray(y_offsets, dtype=np.float64)
        x = np.array([self._lane_center(int(c)) for c in cols],
                     dtype=np.float64)
        y = np.asarray(self._note_y(y_offsets), dtype=np.float64)
        rotation = np.zeros(len(cols), dtype=np.float64)
        zoom = np.ones(len(cols), dtype=np.float64)
        if self._displace is None:
            return LaneSamples(x, y, rotation, zoom)
        dx, dy, d_rot, d_zoom = self._displace(cols, y_offsets)
        return LaneSamples(x + np.asarray(dx, dtype=np.float64),
                           y + np.asarray(dy, dtype=np.float64),
                           rotation + np.asarray(d_rot, dtype=np.float64),
                           zoom * np.asarray(d_zoom, dtype=np.float64))

    def at(self, col: int, y_offset: float) -> tuple:
        """One point, for a caller that genuinely wants one (a receptor).
        Prefer `sample` wherever a batch exists."""
        return self.sample((col,), (y_offset,)).at(0)

    def between(self, col: int, start: float, end: float,
                samples: int) -> LaneSamples:
        """The span from `start` to `end` as `samples` points, endpoints
        included - a hold body, or a travelpath over the visible range.

        The endpoints land exactly on `start` and `end`, so a body stays
        attached to the head and tail that were placed independently."""
        count = max(2, int(samples))
        offsets = np.linspace(float(start), float(end), count)
        return self.sample(np.full(count, int(col), dtype=np.int64), offsets)


def straight(lane_center, note_y) -> LanePath:
    """A lane with no displacement - the case every game without note mods
    has, and the one NotITG's bent path degenerates to when its mods are
    all at zero."""
    return LanePath(lane_center, note_y)
