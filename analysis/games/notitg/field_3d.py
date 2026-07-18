"""3D notefield perspective for the base player field.

The field tilts in real 3D from two producers, both sampled through
`field_projection.FieldTilt`:

- Recorded actor pokes: gat's crossup/rotator section (chart t~74-130s)
  pokes `P1`/`P2` (the real NoteFields) with per-frame `rotationx`/
  `rotationy`/`rotationz`/`skewx` from its UpdateCommand. Those channels
  are recorded onto the named actors and compiled into per-player
  keyframe streams.
- Scalar confusion tilt mods: `confusionx`/`confusiony` (with their
  *offset companions) are ReceptorGetRotationX/Y - a whole-field tilt.
  Routing them here renders them as true perspective; the 2D
  foreshortening kernels in arrow_effects remain only as the per-column
  (numbered variant) degradation and the deferral fallback below.

Each frame this effect samples the summed tilt, composes the SM actor
model matrix for the notefield plane, projects it with SM's
`LoadMenuPerspective` (fov 45; the recorded per-player SetVanishPoint
stream when the compiled dict carries `field_vanish`, else the centered
default), extracts the exact planar homography, and emits it on
`EffectFrame.transform` so QPainter warps the notefield layers in true
perspective. At rest (all channels zero) it emits nothing, so unmodded
charts and non-3D sections pay only the sample.

Player choice: the replay renders player 0's field as the base capture;
a dual-player chart's second field rides the field-instances path with
its own channels. When the chart hides the base field and lets copies
stand in (`base_field_hidden`), this effect defers: a field-space
transform would bake into the shared field capture and leak into every
copy (copies carry their own recorded 2D transforms). While deferred,
`tilt_active` reports False, so the 2D confusion kernels stay live as
the documented fallback - the capture keeps a flat approximation of the
tilt rather than losing it.

Double-apply guard: `tilt_active(t)` reports whether this projection
owns the X/Y tilt at t (either producer, not deferred). note_mods reads
it to zero the scalar confusionx/y (and hallway) kernels while the real
projection executes - one guard, both producers.
"""
from __future__ import annotations

from analysis.games.notitg import field_projection
from analysis.games.notitg.field_instances import _design_map
from analysis.player.render import transform3d
from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.storyboard.model import build_timelines

# The player-actor 3D channels this effect consumes, with their rest
# states (transform3d identity). `rotation` is the in-plane z spin;
# rotation_x/y are the out-of-plane tilts; skew_x is SM's shear.
_CHANNEL_RESTS = {
    'rotation_x': 0.0, 'rotation_y': 0.0, 'rotation': 0.0, 'skew_x': 0.0,
}

def _actor_3d_timelines(frames):
    """3D-channel EventTimelines for one recorded actor's poke streams, or
    None when it carries no 3D pokes."""
    keyframes = {prop: frames[prop] for prop in _CHANNEL_RESTS
                 if frames and frames.get(prop)}
    if not keyframes:
        return None
    return build_timelines(rests=_CHANNEL_RESTS, keyframes=keyframes)


def _player_actor_timelines(named, player):
    """EventTimelines for one player's 3D actor channels, or None when
    that player's actor carries no 3D poke (the common no-3D case). The
    compiler-supplied P1/P2 streams are the sole source. `player` is
    0-based; extra proxy-source players (P3+) have no compiled
    player-actor stream, so they resolve to None (no field-3D tilt -
    their transform comes from the proxy's own pokes)."""
    if not named:
        return None
    return _actor_3d_timelines(named.get(f'P{player + 1}'))


def notitg_field_3d(base_hidden=None, player_keyframes=None,
                    channels=None, beat_at=None, field_vanish=None,
                    player=0):
    """The field-3D effect for a chart, or None when neither producer
    drives a tilt (no actor 3D pokes AND no scalar confusion tilt mods).

    `player_keyframes` is the compiled dict's P1/P2 poke streams (the
    engine-loop compiler supplies them); absent, this player gets no
    actor tilt source. `channels`/`beat_at` supply the scalar confusion
    tilt mods; `field_vanish` is the compiled per-player SetVanishPoint
    stream dict (1-based players). `base_hidden`
    is the compiled `base_field_hidden` timeline (the same one the
    field-instances effect reads): while it is set the copies own the
    field and this effect defers, so the field capture stays flat and
    copies never inherit the base tilt."""
    actor_tl = _player_actor_timelines(player_keyframes, player)
    mod_tilt = field_projection.has_mod_tilt(channels, player)
    if actor_tl is None and not mod_tilt:
        return None
    tilt = field_projection.FieldTilt(
        actor_timelines=actor_tl,
        channels=channels if mod_tilt else None,
        player=player, beat_at=beat_at,
        vanish=(field_vanish or {}).get(player + 1))
    return NotitgField3D(tilt, base_hidden=base_hidden)


class NotitgField3D:
    """Effect emitting the notefield's 3D perspective as an
    `EffectFrame.transform` (column-space, the base playfield layers).

    Samples the summed field tilt each frame, projects the field plane
    about its centre (LoadMenuPerspective, per-frame vanish when
    recorded), and extracts the exact planar homography, conjugated by
    the design map so the whole 640x480 field tilts about its mapped
    centre - in lockstep with the field copies and the storyboard actors
    drawn over it."""

    def __init__(self, tilt, base_hidden=None):
        self._tilt = tilt
        self._base_hidden = base_hidden

    def at(self, ctx) -> EffectFrame | None:
        t = float(ctx.t_now)
        if self._deferring(t):
            return None
        rot = self._tilt.sample(t)
        if rot is None:
            return None

        verdict, H = field_projection.field_homography(
            *rot, vanish=self._tilt.vanish_at(t))
        if verdict == 'gone':
            # The field tilted fully through the eye plane: nothing to
            # draw. Hide it rather than render a meaningless warp.
            return EffectFrame(opacity=0.0)
        # 'clipped' still yields a usable homography for the front-facing
        # majority; QPainter clips the field to chart_rect anyway, so a
        # partial tilt renders its visible portion (the alternative -
        # affine fallback - loses the perspective the section is about).
        return EffectFrame(transform=_screen_transform(ctx.chart_rect, H))

    def tilt_active(self, t) -> bool:
        """Whether a real out-of-plane field tilt (rotation_x or
        rotation_y, from either producer) is executing at t: non-rest AND
        not deferred to copies. The note-mod consumer reads this to
        suppress the 2D confusion-tilt approximation of the SAME axes
        while the projection owns the tilt (double-apply guard)."""
        if self._deferring(float(t)):
            return False
        return self._tilt.tilt_active(float(t))

    def _deferring(self, t) -> bool:
        return (self._base_hidden is not None
                and self._base_hidden.sample(t)[0] >= 0.5)


def _screen_transform(chart_rect, H):
    """The screen-space transform tilting the notefield: the design-space
    homography conjugated by the design map, `M^-1 . H . M`.

    The field layers draw in SCREEN pixels, so a screen point must map
    back to 640x480 design space (M^-1), warp by the perspective H there,
    then return to screen (M) - the same conjugation the field copies and
    the screen camera use, so the tilt pivots on the mapped design centre
    in lockstep with them. Qt is row-vector (`p * T`), so pre-multiplying
    reads left-to-right: `M_inv * H_qt * M` applies M^-1 first."""
    from PySide6.QtGui import QTransform

    kx, ky, ox, oy = _design_map(chart_rect)
    to_design = QTransform()
    to_design.scale(1.0 / kx, 1.0 / ky)
    to_design.translate(-ox, -oy)
    to_screen = QTransform()
    to_screen.translate(ox, oy)
    to_screen.scale(kx, ky)
    return to_design * transform3d.qtransform_from_h(H) * to_screen
