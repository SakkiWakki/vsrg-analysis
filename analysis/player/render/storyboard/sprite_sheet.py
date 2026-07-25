"""Sprite-sheet frame math: pure sampling the renderer needs.

A sheet sprite draws ONE cell of a grid texture per frame. Frame index
runs across a row then down (`index = col + row * cols`), and a state
list of `(frame_index, delay_seconds)` pairs drives which frame shows
over time. These two functions are the game-agnostic runtime math (a
clean rust-port boundary); the StepMania-specific decode of a filename
grid and `.sprite` manifest into this shape lives in the game frontend.
"""
from __future__ import annotations

import math


def frame_source_rect(frame: int, sheet_w: float, sheet_h: float,
                      cols: int, rows: int) -> tuple:
    """(x, y, w, h) of one frame's cell in the sheet, in sheet pixels.
    Frame index runs across a row then down; a frame past the grid clamps
    to the last cell."""
    count = cols * rows
    frame = min(max(0, frame), count - 1)
    col = frame % cols
    row = frame // cols
    fw = sheet_w / cols
    fh = sheet_h / rows
    return (col * fw, row * fh, fw, fh)


def frame_at_time(states: tuple, t: float) -> int:
    """The frame a state list shows `t` seconds into its animation,
    looping over the total length (each state holds for its delay, then
    the next; wraps mod the total). A single state (or a zero-length
    total) holds its one frame."""
    if not states:
        return 0
    if len(states) == 1:
        return states[0][0]
    total = sum(delay for _frame, delay in states)
    if total <= 0.0:
        return states[0][0]
    into = t % total
    for frame, delay in states:
        if into < delay:
            return frame
        into -= delay
    return states[-1][0]


def frame_steps(states: tuple, t_start: float, t0: float, t1: float,
                limit: int):
    """The frames `frame_at_time` shows across `[t0, t1]` as `(ts, frames)`
    step points - the frame at `t0`, then one per change - or None when the
    animation would need more than `limit` of them.

    The lane is a periodic step function: each state holds for its delay and
    the list wraps modulo the total, so the changes are exactly the cumulative
    state offsets repeated every cycle. A consumer that would otherwise sample
    at a fixed cadence to rediscover them gets them outright, and without the
    aliasing that cadence imposes on any state shorter than it. A sheet with
    one state (or none) never changes and needs no steps."""
    if not states or t1 <= t0:
        return [], []
    total = sum(delay for _frame, delay in states)
    if len(states) == 1 or total <= 0.0:
        return [t0], [states[0][0]]
    if len(states) * (t1 - t0) / total > limit:
        return None

    ts: list = []
    frames: list = []
    # Walk from the cycle containing `t0`; the state spanning it is emitted
    # AT t0, so the steps are read from the animation alone - seeding the
    # first value by sampling would disagree with the walk at a cycle
    # boundary, where the two round apart.
    at = t_start + ((t0 - t_start) // total) * total
    while at < t1:
        for frame, delay in states:
            start, at = at, at + delay
            if at <= t0 or (frames and frame == frames[-1]):
                continue
            if start >= t1:
                return ts, frames
            ts.append(max(start, t0))
            frames.append(frame)
    return ts, frames


def frame_at_time_anchored(states: tuple, anchors: tuple, t: float,
                           t_start: float = 0.0) -> int:
    """`frame_at_time` with runtime re-anchoring: `anchors` is a sorted
    tuple of `(at, state_index, animating)` events (a `setstate` jumps
    the animation to that state and it KEEPS playing; an `animate(off)`
    freezes it there). The frame at `t` plays the state list from the
    latest anchor at or before `t`; before any anchor the sheet
    auto-animates from `t_start`."""
    latest = None
    for anchor in anchors:
        if anchor[0] > t:
            break
        latest = anchor
    if latest is None:
        return frame_at_time(states, t - t_start)
    at, state, animating = latest
    if not states:
        return int(state)
    state = _clamp_state(states, state)
    if not animating:
        return states[state][0]
    return frame_at_time(states, t - _anchor_start(states, at, state))


def _clamp_state(states: tuple, state) -> int:
    """A state index clamped into `states` (an anchor may name one the sheet
    does not have)."""
    return min(max(0, int(state)), len(states) - 1)


def _anchor_start(states: tuple, at: float, state: int) -> float:
    """The `t_start` an anchored-but-still-animating span plays from: the
    anchor time pulled back by the state's own offset into the list, so
    `frame_at_time(states, t - start)` resumes AT that state."""
    return at - sum(delay for _frame, delay in states[:state])


class StateAnchors:
    """A `sample(t) -> (frame,)` sampler over `frame_at_time_anchored`,
    shaped like an EventTimeline so the renderer and lint tooling stay
    agnostic of where the frame index comes from."""

    def __init__(self, anchors: tuple, states: tuple, t_start: float = 0.0):
        self.anchors = tuple(anchors)
        self.states = tuple(states)
        self.t_start = float(t_start)

    def steps(self, t0: float, t1: float, limit: int):
        """The frames shown across `[t0, t1]` as `(ts, frames)` step points,
        or None when they would exceed `limit` (see `frame_steps`).

        Each anchor governs until the next one, and within its span the sheet
        either sits frozen on one state or plays the plain animation from that
        state's offset - so the whole lane is a concatenation of `frame_steps`
        walks, one per span, and never needs sampling. Before the first anchor
        the sheet auto-animates from `t_start`, exactly as
        `frame_at_time_anchored` reads it."""
        ts: list = []
        frames: list = []
        for start, end, anchor in self._spans(t0, t1):
            span = self._span_steps(start, end, anchor, limit)
            if span is None:
                return None
            ts.extend(span[0])
            frames.extend(span[1])
            if len(ts) > limit:
                return None
        return ts, frames

    def _spans(self, t0: float, t1: float):
        """`(start, end, anchor)` covering `[t0, t1)` - one span per anchor,
        preceded by the pre-anchor span (anchor None), each clipped to the
        window and skipped when it falls outside."""
        edges = [(-math.inf, None), *((a[0], a) for a in self.anchors)]
        for i, (at, anchor) in enumerate(edges):
            end = edges[i + 1][0] if i + 1 < len(edges) else math.inf
            if end > t0 and at < t1:
                yield max(at, t0), min(end, t1), anchor

    def _span_steps(self, start: float, end: float, anchor, limit: int):
        """One span's steps: the pre-anchor auto-animation, a frozen anchor's
        single held frame, or the animation resumed from the anchor's state."""
        if anchor is None:
            return frame_steps(self.states, self.t_start, start, end, limit)
        if not self.states:
            return [start], [int(anchor[1])]
        state = _clamp_state(self.states, anchor[1])
        if not anchor[2]:
            return [start], [self.states[state][0]]
        return frame_steps(self.states,
                           _anchor_start(self.states, anchor[0], state),
                           start, end, limit)

    def sample(self, t: float) -> tuple:
        return (float(frame_at_time_anchored(self.states, self.anchors,
                                             t, self.t_start)),)
