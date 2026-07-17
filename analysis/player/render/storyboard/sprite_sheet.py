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
