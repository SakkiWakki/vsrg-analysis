"""Game-agnostic storyboard playback.

- model.py   IR: a Storyboard is a design-space canvas of timed
             Elements whose visual properties are per-property eased
             keyframe timelines. Per-game compilers (fluXis .fsb,
             osu .osb/.osu events) produce this shape.
- render.py  StoryboardEffect: draws the active elements each frame
             through the effects pipeline's z-ordered draw slots.
"""
from analysis.player.render.storyboard.model import Element, Storyboard
from analysis.player.render.storyboard.render import StoryboardEffect

__all__ = ['Element', 'Storyboard', 'StoryboardEffect']
