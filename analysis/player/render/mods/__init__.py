"""StepMania / NotITG per-note mod math (pure numpy, port-boundary).

`channels` compiles `ApplyModifiers`-style mod events (with `*S` approach
speeds) into piecewise-linear value curves. `arrow_effects` consumes those
percentages plus host-supplied y_offsets to produce per-note position /
rotation / alpha / zoom contributions, faithfully porting OpenITG's
ArrowEffects.cpp with NotITG extensions. No renderer or engine coupling.
"""
from analysis.player.render.mods.arrow_effects import (
    NoteOffsets, note_offsets, receptor_offsets)
from analysis.player.render.mods.channels import ModChannels, ModEvent

__all__ = [
    'ModChannels', 'ModEvent',
    'NoteOffsets', 'note_offsets', 'receptor_offsets',
]
