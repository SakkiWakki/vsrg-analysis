"""Column-space visual effects: transforms + overlay draws applied to
the chart layers as a group.

Everything drawn in column-space (lanes, notes, judgment line, mines,
ticks, press marks) goes through the frame's painter, so wrapping that
group in one `QTransform` moves all of it together -- no per-layer
awareness. An `Effect` answers, per frame, "what transform and what
extra draws does the playfield have right now"; the compositor folds
every active effect into a single transform + a z-ordered draw list and
applies it around the chart layers.

This is the substrate for modchart playback: lane switches, playfield
rotate/scale/move, camera, storyboards (draws below the field), and --
eventually, on a GL/wgpu backend -- shaders. Adapters attach effects
via `GameAdapter.effects(replay)`.
"""
from __future__ import annotations

from analysis.player.render.effects.base import (Effect, EffectFrame,
                                                  composite)

__all__ = ['Effect', 'EffectFrame', 'composite']
