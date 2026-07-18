"""One projection authority for the NotITG field.

Owns the design-space constants, the LoadMenuPerspective projection
(including the recorded per-player SetVanishPoint streams the compiler
already emits under `field_vanish`), and the field tilt summed from BOTH
producers:

- actor pokes: the recorded PlayerP{n} rotation_x/rotation_y/rotation/
  skew_x channels (gat's crossup/rotator UpdateCommand drives these), and
- mod channels: the scalar confusionx/confusiony percents with their
  *offset companions (ReceptorGetRotationX/Y) - the whole-field tilt the
  2D kernels otherwise approximate as zoom/dx. TRANSFORM3D.md proves that
  approximation is exactly the fov->0 limit of this projection, so
  routing the mods here upgrades it to true perspective.

Consumers: field_3d.NotitgField3D samples `FieldTilt` and
`field_homography` for the base-field effect; field_compose imports
`design_projection()` for the instance channels (one construction point,
no duplicate fov/vanish hardcodes); note_mods' double-apply guard reads
`FieldTilt.tilt_active` through the effect, so the 2D kernels drop the
scalar confusion axes whenever this projection owns them - regardless of
which producer drives the tilt.

confusion (the Z-axis spin) is NOT summed here: it is the in-plane
per-note receptor spin, a distinct engine mechanism that legitimately
coexists with the actor Z rotation (see the note_mods module doc).
Numbered per-column confusionx0.. variants also stay with the kernels:
a per-column tilt is per-note content, and TRANSFORM3D.md sanctions the
center-plane 2D degradation for per-note 3D.
"""
from __future__ import annotations

from analysis.player.render import transform3d
from analysis.player.render.mods.arrow_effects import _confusion_axis_degrees

# SM design space; the projection maps the z=0 design plane 1:1 to design
# pixels under the centered vanish, so untilted content is untouched.
DESIGN_W = 640.0
DESIGN_H = 480.0
DESIGN_CX = DESIGN_W / 2.0
DESIGN_CY = DESIGN_H / 2.0
PLANE_CORNERS = (
    (0.0, 0.0),
    (DESIGN_W, 0.0),
    (DESIGN_W, DESIGN_H),
    (0.0, DESIGN_H),
)

# RageDisplay LoadMenuPerspective default field of view, and the eye
# distance it implies: the center-plane scale of a +z push is
# EYE_DISTANCE / (EYE_DISTANCE - z).
FOV = 45.0
EYE_DISTANCE = transform3d.eye_distance(FOV, DESIGN_W)

_CENTERED = transform3d.projection(FOV, DESIGN_W, DESIGN_H, vanish=None)
_REST_EPS = 1e-4

# The scalar mod channels that drive each out-of-plane axis, with their
# offset companions (ReceptorGetRotationX / ReceptorGetRotationY).
_MOD_TILT = (('confusionx', 'confusionxoffset'),
             ('confusiony', 'confusionyoffset'))


def design_projection(vanish=None):
    """world -> design-pixel projection for a vanish point (design px).

    None or the design centre return the shared centered matrix - the
    at-rest case every frame of an unmodded chart hits."""
    if vanish is None:
        return _CENTERED
    vx, vy = vanish
    if abs(vx - DESIGN_CX) < _REST_EPS and abs(vy - DESIGN_CY) < _REST_EPS:
        return _CENTERED
    return transform3d.projection(FOV, DESIGN_W, DESIGN_H, vanish=(vx, vy))


def field_model(rx, ry, rz, skewx):
    """The SM actor model matrix for the notefield plane, in design-pixel
    content coords, tilting about the field centre (320, 240).

    SM rotates/skews an actor about its own origin; the field plane's
    content is expressed in design pixels, so we translate the centre to
    the origin, apply SkewX then the fused Rxyz (Actor::BeginDraw pushes
    translate/scale, rotation, then skew, and the matrix stack's local
    multiplies apply to content in reverse push order - skew acts before
    rotation), and translate back. A row point maps
    `v @ (T(-c) @ SkewX @ Rxyz @ T(c))`."""
    to_origin = transform3d.translate(-DESIGN_CX, -DESIGN_CY)
    back = transform3d.translate(DESIGN_CX, DESIGN_CY)
    model = to_origin
    if skewx:
        model = model @ transform3d.skew_x(skewx)
    return model @ transform3d.rotate_xyz(rx, ry, rz) @ back


def field_homography(rx, ry, rz, skewx, vanish=None):
    """(verdict, H) for the tilted field plane about the design centre
    under the (possibly off-centre) LoadMenuPerspective. verdict is
    project_with_verdict's: 'ok' | 'clipped' | 'gone'."""
    model = field_model(rx, ry, rz, skewx)
    verdict, H, _clip = transform3d.project_with_verdict(
        model, design_projection(vanish), PLANE_CORNERS)
    return verdict, H


def has_mod_tilt(channels, player=0) -> bool:
    """Whether the compiled mod channels drive the scalar confusion tilt
    for `player` (any tilt mod or offset companion has events)."""
    if channels is None:
        return False
    mods = set(channels.mods(player))
    return any(name in mods for pair in _MOD_TILT for name in pair)


class FieldTilt:
    """Per-player field tilt sampler over both producers.

    `actor_timelines` is the recorded-poke timeline dict (rotation_x /
    rotation_y / rotation / skew_x EventTimelines) or None; `channels` /
    `player` / `beat_at` supply the scalar confusion mods (channels may
    be None when no tilt mod has events - see `has_mod_tilt`); `vanish`
    is the player's compiled {'vanish_x'/'vanish_y': EventTimeline} dict
    or None for the centered default."""

    def __init__(self, actor_timelines=None, channels=None, player=0,
                 beat_at=None, vanish=None):
        self._actor = actor_timelines
        self._channels = channels
        self._player = int(player)
        self._beat_at = beat_at
        self._vanish = vanish or None

    def sample(self, t):
        """(rx, ry, rz, skewx) at t, or None when all rest (identity -
        no transform to emit, the zero-cost path)."""
        axes = self._axes(t)
        if all(abs(v) < _REST_EPS for v in axes):
            return None
        return axes

    def tilt_active(self, t) -> bool:
        """Whether an out-of-plane tilt (X or Y) is live at t, from
        either producer. Z spin and skew do not count - only the axes
        the 2D confusionx/y kernels would otherwise approximate."""
        rx, ry, _rz, _skewx = self._axes(t)
        return abs(rx) >= _REST_EPS or abs(ry) >= _REST_EPS

    def vanish_at(self, t):
        """(vx, vy) design-px vanish point at t, or None (centered)."""
        v = self._vanish
        if v is None:
            return None
        vx = v['vanish_x'].sample(t)[0] if 'vanish_x' in v else DESIGN_CX
        vy = v['vanish_y'].sample(t)[0] if 'vanish_y' in v else DESIGN_CY
        return vx, vy

    def _axes(self, t):
        rx = ry = rz = skewx = 0.0
        actor = self._actor
        if actor is not None:
            rx = actor['rotation_x'].sample(t)[0]
            ry = actor['rotation_y'].sample(t)[0]
            rz = actor['rotation'].sample(t)[0]
            skewx = actor['skew_x'].sample(t)[0]
        mod_rx, mod_ry = self._mod_degrees(t)
        return rx + mod_rx, ry + mod_ry, rz, skewx

    def _mod_degrees(self, t):
        """The (rx, ry) contribution of the scalar confusion tilt mods,
        in degrees. The song beat is sampled once, lazily - only when a
        tilt channel is actually away from rest at t."""
        channels = self._channels
        if channels is None:
            return 0.0, 0.0
        beat = None
        out = []
        for mod, companion in _MOD_TILT:
            percent = channels.value(mod, t, self._player)
            offset = channels.value(companion, t, self._player)
            if abs(percent) < _REST_EPS and abs(offset) < _REST_EPS:
                out.append(0.0)
                continue
            if beat is None:
                beat = self._beat_at(t)
            out.append(_confusion_axis_degrees(percent, beat, offset))
        return out[0], out[1]
