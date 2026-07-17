"""Sprite-sheet frame math: pure sampling the renderer needs.

A sheet sprite draws ONE cell of a grid texture per frame. Frame index
runs across a row then down (`index = col + row * cols`), and a state
list of `(frame_index, delay_seconds)` pairs drives which frame shows
over time. These two functions are the game-agnostic runtime math (a
clean rust-port boundary); the StepMania-specific decode of a filename
grid and `.sprite` manifest into this shape lives in the game frontend.
"""
from __future__ import annotations


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
    state = min(max(0, int(state)), len(states) - 1)
    if not animating:
        return states[state][0]
    into_list = sum(delay for _frame, delay in states[:state])
    return frame_at_time(states, into_list + (t - at))


class StateAnchors:
    """A `sample(t) -> (frame,)` sampler over `frame_at_time_anchored`,
    shaped like an EventTimeline so the renderer and lint tooling stay
    agnostic of where the frame index comes from."""

    def __init__(self, anchors: tuple, states: tuple, t_start: float = 0.0):
        self.anchors = tuple(anchors)
        self.states = tuple(states)
        self.t_start = float(t_start)

    def sample(self, t: float) -> tuple:
        return (float(frame_at_time_anchored(self.states, self.anchors,
                                             t, self.t_start)),)
