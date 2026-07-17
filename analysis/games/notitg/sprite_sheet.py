"""StepMania 3.95 sprite-sheet semantics (grid decode + state lists).

A StepMania Sprite treats its texture as an N-column x M-row grid of
animation frames when the filename ends in a `%ux%u` token (RageTexture
`GetFrameDimensionsFromFileName`, e.g. `shame_idle 2x1.png` -> 2 wide,
1 high). Frame index runs across a row first, then down
(`index = col + row * cols`, RageTexture `CreateFrameRects`), and a
frame's source rect is that cell of the image (RageTexture.h
`GetSourceFrameWidth = width / cols`).

A Sprite also carries a STATE LIST - `(frame_index, delay_seconds)`
pairs it steps through. Two sources set it (Sprite.cpp):

- default (`LoadFromTexture`): one state per grid frame, in order, each
  0.1s - a plain sequential auto-animation.
- a `.sprite` manifest (`LoadFromNode`): `Frame%04d=`/`Delay%04d=` pairs
  replace the defaults, letting a chart pin one frame (delay 999 =
  effectively static), oscillate (`0,1,0,1`), or play a sub-range.

`animate(false)`/`setstate(i)` control playback at runtime; those are
recorded as a pin timeline by the caller, not modeled here. This module
is the load-time decode: filename -> grid, manifest -> state list. The
game-agnostic runtime sampling (frame-at-time, source-rect) lives in
render/storyboard/sprite_sheet, re-exported here for callers that want
one import.
"""
from __future__ import annotations

import re
from pathlib import Path

from analysis.player.render.storyboard.sprite_sheet import (frame_at_time,
                                                            frame_source_rect)

__all__ = ['grid_from_filename', 'default_states', 'parse_sprite_states',
           'frame_at_time', 'frame_source_rect']

_GRID_TOKEN = re.compile(r'(?:^|\s)(\d+)x(\d+)(?=\.|$|\s)', re.IGNORECASE)

# LoadFromTexture's per-frame default when no .sprite overrides it.
_DEFAULT_DELAY = 0.1


def grid_from_filename(name: str) -> tuple:
    """(cols, rows) encoded in a sprite-sheet filename, or (1, 1) when
    none is present. SM scans space-separated tokens for the LAST `%ux%u`
    match (`GetFrameDimensionsFromFileName`), so `--fuck 1x24.bmp` is
    1 col x 24 rows and a plain `bg.png` is a single frame."""
    stem = Path(name).stem
    matches = _GRID_TOKEN.findall(stem)
    if not matches:
        return (1, 1)
    cols, rows = matches[-1]
    return (max(1, int(cols)), max(1, int(rows)))


def default_states(frame_count: int) -> tuple:
    """The sequential auto-animation SM builds in `LoadFromTexture`: one
    state per frame, in order, each 0.1s. `((frame, delay), ...)`."""
    return tuple((i, _DEFAULT_DELAY) for i in range(frame_count))


def parse_sprite_states(text: str, frame_count: int) -> tuple:
    """State list from a `.sprite` manifest body, or () when it defines
    none (the caller then keeps the default sequence). Reads paired
    `Frame%04d=`/`Delay%04d=` keys in order, stopping at the first gap
    exactly as `Sprite::LoadFromNode` does; a frame index past the sheet
    is clamped in (SM throws, we survive community files). Returns
    `((frame, delay), ...)`."""
    values = _ini_values(text)
    states = []
    for i in range(len(values)):
        frame = _as_int(values.get(f'frame{i:04d}'))
        delay = _as_float(values.get(f'delay{i:04d}'))
        if frame is None or delay is None:
            break
        states.append((min(max(0, frame), max(0, frame_count - 1)), delay))
    return tuple(states)


def _ini_values(text: str) -> dict:
    values = {}
    for line in text.splitlines():
        key, sep, value = line.partition('=')
        if sep:
            values[key.strip().lower()] = value.strip()
    return values


def _as_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
