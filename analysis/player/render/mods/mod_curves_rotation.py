"""Prototype: the rot_z rotation mods expressed as spatial curves.

The rotation family of arrow-effects, ported to the curve algebra the
same way `mod_curves.confusionx_rot` ports the rot_x tilt. Each builder
returns an axis Curve (see `curves.Curve`) reproducing the hardcoded
kernel in `arrow_effects`, in DEGREES. Validated byte-equal in
tests/test_mod_curves_rotation.py.

  confusion -> rot_z : a y-independent whole-field beat spin (degrees)
  dizzy     -> rot_z : a per-NOTE spin proportional to (note_beat - beat)

`confusion` is the rot_z sibling of `confusionx_rot`'s rot_x tilt: the
identical `_confusion_axis_degrees` angle broadcast to every note, only
the axis it rotates about differs. `dizzy` is the odd one of the family:
its angle is y-INDEPENDENT but depends on each note's OWN beat, so the
per-note variation rides in the note_beats array, not in y_offset.

Ctx extension (proposed, NOT applied -- see report): dizzy needs a
per-note beat array. `curves.Ctx` today carries `cols` (per-note column)
but no per-note beat. This module reads `ctx.note_beats` when present,
so `Ctx` should gain a `note_beats: np.ndarray | None = None` field
aligned with the y_offset/cols batch. Until then a caller passes it via
a Ctx set with that attribute (the tests do exactly this).
"""
from __future__ import annotations

import numpy as np

from analysis.player.render.mods import curves as cv
from analysis.player.render.mods.arrow_effects import PI


def confusion_rot(percent, offset=0.0) -> cv.Curve:
    """confusion / confusionoffset ReceptorGetRotationZ (the in-plane Z
    spin), in DEGREES: a whole-field angle broadcast to every note,
    independent of y_offset. The rot_z sibling of `confusionx_rot`; same
    `_confusion_axis_degrees` formula, different axis.
        (beat*percent mod 2pi) * -180/pi + offset*180/pi"""
    def angle(c):
        spin = np.mod(c.beat * percent, 2.0 * PI) * -180.0 / PI
        return spin + offset * 180.0 / PI
    return cv.const(angle)


def dizzy_rot(percent) -> cv.Curve:
    """dizzy GetRotationZ (ArrowEffects.cpp:364-378), in DEGREES: a
    per-note spin proportional to beats-until-step, wrapped to a full
    turn. y-INDEPENDENT, but per-note through the note's OWN beat:
        ((note_beat - beat) * percent mod 2pi) * 180/pi.
    Reads the per-note beat from `ctx.note_beats` (proposed Ctx field;
    see module header)."""
    def curve(y, c):
        note_beats = np.asarray(c.note_beats, dtype=np.float64)
        rot = np.mod((note_beats - c.beat) * percent, 2.0 * PI)
        return np.broadcast_to(rot * 180.0 / PI, np.shape(y)).astype(np.float64)
    return curve
