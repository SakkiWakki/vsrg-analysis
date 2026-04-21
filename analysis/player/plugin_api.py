"""Public API for replay-player draw plugins.

Player draw plugins are regular Python modules with a top-level
``register(add)`` function. ``add`` accepts a display name, draw callable,
stage list, and optional priority. Draw callables receive ``(ctx, stage)``.
"""
from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    AFTER_LANES = 'after_lanes'
    AFTER_JUDGMENT = 'after_judgment'
    AFTER_NOTES = 'after_notes'
    AFTER_GHOSTS = 'after_ghosts'
    HUD = 'hud'
    POST_FRAME = 'post_frame'


def normalize_stage(stage):
    if isinstance(stage, Stage):
        return stage
    return Stage(str(stage))
