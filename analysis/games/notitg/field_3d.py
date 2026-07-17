"""3D notefield perspective from the recorded player-actor channels.

gat's crossup/rotator section (chart t~74-130s, audit item #1) tilts the
whole notefield in real 3D: the UpdateCommand pokes `P1`/`P2`
(`GetChild('PlayerP1'/'PlayerP2')`, the real NoteFields) with per-frame
`rotationx`/`rotationy`/`rotationz`/`skewx`, driven by the data-holder
quads `gat_g_rot_intro`/`gat_rotator`/`gat_crossup`/`gat_g_skewer`. Those
channels are recorded onto the named actors (`recording_actor`
`_SCALAR_SETTERS`) but no executor consumed them - the field stayed flat.

This effect samples the player-0 (`P1`) 3D channels each frame, composes
the SM actor model matrix for the notefield plane (transform3d
`local_matrix` semantics), projects it with SM's `LoadMenuPerspective`
defaults (centered vanish, fov 45 - see the recorder gap below), extracts
the exact planar homography, and emits it on `EffectFrame.transform` so
QPainter warps the notefield layers in true perspective. At rest (all
channels zero) it emits nothing, so unmodded charts and non-3D sections
pay only the sample.

Player choice: the replay is single-player, so one notefield is drawn.
P1 and P2 receive the same rotation magnitudes (only skew mirrors), so
P1's channels faithfully tilt the single field. When the chart hides the
base field and lets copies stand in (`base_field_hidden`), this effect
defers: a field-space transform would bake into the shared field capture
and leak into every copy (copies carry their own recorded 2D transforms).
The dominant 3D section (t~74-130) has the base field visible and zero
copies live, so the deferral costs nothing there.

Recorder gap - vanish point: gat drives `SetVanishPoint(GetX(), GetY())`
on the Proxy actors per frame (lua/default.xml ~L3922), but
`recording_actor` records no vanish channel (no `SetVanishPoint` setter),
so it is unavailable to compile. We therefore project with SM's
`LoadMenuPerspective` default vanish (screen centre) and fov 45
(RageDisplay default). Wiring a `vanish_x`/`vanish_y` channel through the
recorder is a follow-up owned by the recorder file; this effect reads it
the moment the compiled data carries it (see `_vanish`).

Double-apply note: the per-note `confusionx/yoffset` kernels
(arrow_effects) are a SEPARATE mechanism (per-note 2D foreshortening the
chart also uses) and stay fully live - they are quiet across t~74-130
(measured), so no section drives the same visual through both paths at
once. See the module doc in the project memory for the evidence.
"""
from __future__ import annotations

from functools import lru_cache

from analysis.games.notitg.field_instances import _design_map
from analysis.player.render import transform3d
from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.storyboard.model import build_timelines

# SM design space and the notefield plane's corners in it, in DESIGN-PIXEL
# coords (0..640, 0..480) - the same frame the projection maps 1:1 at z=0,
# so an untilted field is the identity. The whole plane tilts about its
# centre (SM SetVanishPoint defaults and the actor pivot both sit there).
_DESIGN_W = 640.0
_DESIGN_H = 480.0
_DESIGN_CX = _DESIGN_W / 2.0
_DESIGN_CY = _DESIGN_H / 2.0
_PLANE_CORNERS = (
    (0.0, 0.0),
    (_DESIGN_W, 0.0),
    (_DESIGN_W, _DESIGN_H),
    (0.0, _DESIGN_H),
)

# RageDisplay's LoadMenuPerspective default field of view; vanish None =
# screen centre (see the recorder gap in the module doc).
_DEFAULT_FOV = 45.0

# The player-actor 3D channels this effect consumes, with their rest
# states (transform3d identity). `rotation` is the in-plane z spin;
# rotation_x/y are the out-of-plane tilts; skew_x is SM's shear.
_CHANNEL_RESTS = {
    'rotation_x': 0.0, 'rotation_y': 0.0, 'rotation': 0.0, 'skew_x': 0.0,
}
_EPS = 1e-4

# The recorded player actors, in player order. P1 is player 0's real
# NoteField (the single field we render); P2 is kept for a future
# dual-field path.
_PLAYER_ACTORS = ('P1', 'P2')


@lru_cache(maxsize=8)
def _player_field_keyframes(sm_path):
    """Harvest the P1/P2 named-actor poke streams for a chart.

    Runs the modfile compile pipeline (chunks + mod-action replay +
    update integrator) purely to recover the player-actor keyframes;
    the crossup/rotator pokes live inside the per-frame UpdateCommand,
    so the integrator must run. Memoized per chart (a full compile), so
    the load-time cost is paid once. Returns {} on any failure - a chart
    with no 3D pokes (or no modfile) simply gets no field-3D effect."""
    from analysis.games.etterna import sm_chart
    from analysis.games.notitg import modfile, update_integrator
    from pathlib import Path

    try:
        entries = modfile.parse_fgchanges(sm_path)
        lua_dir = modfile._resolve_lua_dir(sm_path, entries)
        if lua_dir is None:
            return {}
        sm_data = sm_chart.parse_sm(sm_path)
        bg_stem = Path(modfile._sm_background_name(sm_path)).stem.casefold()
        root, _chunks, _classic = modfile._load_document(lua_dir, bg_stem)
        _bpms, _offset, chart = modfile._timing(sm_data)
        to_seconds = modfile._beat_to_seconds(sm_data, chart)
        start_beat = min((b for b, _n, k in entries if k == 'FGCHANGES'),
                         default=0.0)
        env, _warn = modfile._run_chunks(root, start_beat, to_seconds)
        env.replay_mod_actions()
        update_integrator.integrate_update(env, root, to_seconds)
        return env.named_actor_keyframes()
    except Exception:
        return {}


def _actor_3d_timelines(frames):
    """3D-channel EventTimelines for one recorded actor's poke streams, or
    None when it carries no 3D pokes."""
    keyframes = {prop: frames[prop] for prop in _CHANNEL_RESTS
                 if frames and frames.get(prop)}
    if not keyframes:
        return None
    return build_timelines(rests=_CHANNEL_RESTS, keyframes=keyframes)


def _player_field_timelines(sm_path, named=None):
    """Per-player EventTimelines for the 3D channels, or () when no player
    actor carries any 3D poke (the common no-3D-chart case). When the
    compiler supplies the P1/P2 streams they are used directly; otherwise
    a private harvest compile is run (and cached per chart)."""
    if named is None:
        named = _player_field_keyframes(sm_path)
    out = tuple(_actor_3d_timelines(named.get(actor))
                for actor in _PLAYER_ACTORS)
    return out if any(tl is not None for tl in out) else ()


def notitg_field_3d(sm_path, base_hidden=None, player_keyframes=None):
    """The field-3D effect for a chart, or None when it has no 3D pokes.

    `player_keyframes` is the compiled dict's P1/P2 poke streams when the
    compiler provides them (the engine-loop path does); without it this
    falls back to a private harvest compile of the chart. `base_hidden`
    is the compiled `base_field_hidden` timeline (the same one the
    field-instances effect reads): while it is set the copies own the
    field and this effect defers, so the field capture stays flat and
    copies never inherit the base tilt."""
    timelines = _player_field_timelines(sm_path, player_keyframes)
    if not timelines:
        return None
    return NotitgField3D(timelines[0], base_hidden=base_hidden)


class NotitgField3D:
    """Effect emitting the notefield's 3D perspective as an
    `EffectFrame.transform` (column-space, the base playfield layers).

    Samples player-0's recorded rotation_x/rotation_y/rotation/skew_x each
    frame, builds the SM model matrix for the field plane about its
    centre, projects with LoadMenuPerspective (fov 45, centred vanish),
    and extracts the exact planar homography, conjugated by the design map
    so the whole 640x480 field tilts about its mapped centre - in lockstep
    with the field copies and the storyboard actors drawn over it."""

    def __init__(self, timelines, base_hidden=None):
        self._tl = timelines
        self._base_hidden = base_hidden

    def __bool__(self):
        return self._tl is not None

    def at(self, ctx) -> EffectFrame | None:
        t = float(ctx.t_now)
        if self._deferring(t):
            return None
        rot = self._sample(t)
        if rot is None:
            return None

        rx, ry, rz, skewx = rot
        model = _field_model(rx, ry, rz, skewx)
        # vanish=None -> LoadMenuPerspective's centred default: gat's
        # per-frame SetVanishPoint is unrecorded (see the module doc), so
        # the projection uses the default centre until a vanish channel
        # exists in the compiled data.
        proj = transform3d.projection(_DEFAULT_FOV, _DESIGN_W, _DESIGN_H,
                                      vanish=None)
        verdict, H, _clip = transform3d.project_with_verdict(
            model, proj, _PLANE_CORNERS)
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
        """Whether a real out-of-plane field tilt (rotation_x or rotation_y)
        is executing at t: non-rest AND not deferred to copies. The note-mod
        consumer reads this to suppress the 2D confusion-tilt approximation
        of the SAME axes while the projection owns the tilt (double-apply
        guard). Z spin and skew do not count - only the X/Y tilt that the
        confusionx/y kernels approximate."""
        if self._deferring(float(t)):
            return False
        rx = self._tl['rotation_x'].sample(float(t))[0]
        ry = self._tl['rotation_y'].sample(float(t))[0]
        return abs(rx) >= _EPS or abs(ry) >= _EPS

    def _deferring(self, t) -> bool:
        return (self._base_hidden is not None
                and self._base_hidden.sample(t)[0] >= 0.5)

    def _sample(self, t):
        """(rx, ry, rz, skewx) at t, or None when all rest (identity - no
        transform to emit, the zero-cost path)."""
        rx = self._tl['rotation_x'].sample(t)[0]
        ry = self._tl['rotation_y'].sample(t)[0]
        rz = self._tl['rotation'].sample(t)[0]
        skewx = self._tl['skew_x'].sample(t)[0]
        if (abs(rx) < _EPS and abs(ry) < _EPS and abs(rz) < _EPS
                and abs(skewx) < _EPS):
            return None
        return rx, ry, rz, skewx


def _field_model(rx, ry, rz, skewx):
    """The SM actor model matrix for the notefield plane, in design-pixel
    content coords, tilting about the field centre (320, 240).

    SM rotates/skews an actor about its own origin; the field plane's
    content is expressed in design pixels, so we translate the centre to
    the origin, apply SkewX then the fused Rxyz (Actor::BeginDraw pushes
    translate/scale, rotation, then skew, and the matrix stack's local
    multiplies apply to content in reverse push order - skew acts before
    rotation), and translate back. A row point maps
    `v @ (T(-c) @ SkewX @ Rxyz @ T(c))`."""
    to_origin = transform3d.translate(-_DESIGN_CX, -_DESIGN_CY)
    back = transform3d.translate(_DESIGN_CX, _DESIGN_CY)
    model = to_origin
    if skewx:
        model = model @ transform3d.skew_x(skewx)
    return model @ transform3d.rotate_xyz(rx, ry, rz) @ back


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

    k, ox, oy = _design_map(chart_rect)
    to_design = QTransform()
    to_design.scale(1.0 / k, 1.0 / k)
    to_design.translate(-ox, -oy)
    to_screen = QTransform()
    to_screen.translate(ox, oy)
    to_screen.scale(k, k)
    return to_design * transform3d.qtransform_from_h(H) * to_screen
