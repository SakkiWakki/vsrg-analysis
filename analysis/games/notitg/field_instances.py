"""Field-instance producer: NotITG playfield copies as EffectFrame.fields.

gat's fullscreen tiling is built from ActorFrameTexture captures and
notefield Proxies redrawn as sprites. Each such copy is a sprite whose
texture IS the whole captured 640x480 SM screen (the AFT) or a player's
notefield (a Proxy), transformed - moved, rotated, zoomed - about the
screen centre. The compiler records each copy's full transform timeline
(games/notitg/modfile._field_copies); this effect turns those timelines
into per-frame field instances.

The renderer's field pipeline (effects/base.py, qt_renderer._blit_field_
instances) renders the field layer group once into a window-sized
pixmap, then blits it once per `(transform, opacity, scope)` in SCREEN
space, clipped to the chart region. `(None, 1.0, 'field')` is the
untouched original. So each visible copy contributes one screen-space
transform placing that capture where the copy actor sits, and exactly one
identity original is present unless the chart hides the base field.

Per-copy capture scope: gat has two copy SOURCES with different content.
ActorProxy copies (P1p..) re-render only a player's NoteField, so their
content is notes + receptors, never the background - always the 'field'
(transparent notefield) capture, one shared background showing through.
ActorFrameTexture copies (gat_aft) capture the WHOLE COMPOSED SCREEN as
of the PREVIOUS frame - so they carry 'screen' scope: the renderer blits
last frame's chart-area composite (backdrop + field + the copy blits
themselves) under the copy transform. This is engine-exact: during gat's
grid section the base NoteField is hidden and the AFT content is the
scattered proxy grid (flipped by the copy's basezoomy=-1), never a clean
centered receptor row; and the one-frame-delayed self-reference is SM's
AFT feedback (the ending echo/DelayFrame trails). It also makes the old
`ShowAFT`/`ShowAFTBG` bg-in-capture distinction moot: the previous-frame
composite already contains the background exactly when the chart drew it.

Design-space mapping: a copy's placement is authored in SM's 640x480
screen space. A copy centred at (320, 240) with unit scale IS the
identity field, so its screen transform is the conjugation
    M . T_copy . M^-1
where M maps SM design space onto the chart region (the same 'height'
fit as the storyboard renderer's _design_transform, duplicated here as
_design_map with an inverse) and T_copy is the copy's own move/rotate/
scale about the screen centre. Transitions come free from the recorded
tween keyframes - sampling the timelines at t interpolates them.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QRectF
from PySide6.QtGui import QTransform

from analysis.player.render.effects.base import EffectFrame

_DESIGN_W = 640.0
_DESIGN_H = 480.0
_SCREEN_CX = _DESIGN_W / 2.0
_SCREEN_CY = _DESIGN_H / 2.0
_MIN_VISIBLE_ALPHA = 1.0 / 255.0

# Proxy copies re-render only the NoteField (notes + receptors), so they
# never carry the background - always the field-only capture. AFT copies
# capture the whole composed screen as of the previous frame - the
# 'screen' scope (previous-frame chart-area composite, feedback semantics).
_PROXY_SCOPE = 'field'
_AFT_SCOPE = 'screen'
# The second-player field capture scope (see NotitgDualField): copies
# whose source is a player-2 proxy blit from the independently-modded
# player-1-channel capture instead of the primary one.
_FIELD2_SCOPE = 'field2'
_PROXY_SOURCES = frozenset({'P1p', 'P2p', 'P3p', 'P4p'})
# Proxy sources that re-render player 2's NoteField (the second capture).
# ScreenGameplay names players PlayerP1.. (1-based); P2p/P4p target the
# second player. P1p/P3p target player 1 -> the primary capture.
_P2_PROXY_SOURCES = frozenset({'P2p', 'P4p'})

# Theme P1/P2 field X in SM 640x480 design space. OpenITG/ITG fallback
# metrics.ini: PlayerP1TwoPlayersTwoSidesX = SCREEN_CENTER_X-160,
# PlayerP2..X = SCREEN_CENTER_X+160 (research item 73). A dual-player
# chart's two fields rest here; the base identity capture for each side
# is translated by (theme_x - design_center) so P1 sits left, P2 right.
# gat repositions its fields via pokes/proxies, but this is the baseline.
_P1_FIELD_X = _SCREEN_CX - 160.0
_P2_FIELD_X = _SCREEN_CX + 160.0


class SecondFieldSpec:
    """Instruction to the renderer to render the field layers a SECOND
    time for a dual-player NotITG chart.

    `note_mods` is the player-1 mod consumer: while the second capture is
    rendered, the renderer swaps it in for the player's primary one so the
    field's notes, receptor offsets, and reverse baseline all evaluate
    against player 2's channels (gat applies different mods per side).
    Everything else about the two captures is identical (same chart, same
    candidate set) - only the sampled (mod, player) channels diverge.

    The two captures are each rendered centered in the design box (like
    the single-field capture); the per-side X placement lives in the blit
    transforms the field producer emits, not in the capture itself."""

    def __init__(self, note_mods, p1_timelines=None, p2_timelines=None,
                 p1_osc=None, p2_osc=None):
        self.note_mods = note_mods
        # Recorded transform streams (x/y/rotation/scale/hidden) for each
        # player group, from the compiled dict, or None when the chart
        # never moved that player (it keeps the versus-split rest).
        self.p1_timelines = p1_timelines
        self.p2_timelines = p2_timelines
        # Effect-oscillator DELTAS (x/y/rotation) per player - the
        # vibrate 'spazz' (per-frame random teleport mirage), wag, bob -
        # synthesized from the recorded spans; added on top of the
        # transform stream.
        self.p1_osc = p1_osc
        self.p2_osc = p2_osc


def player_placement(player_timelines, rest_dx_design, t, k, scope,
                     osc=None):
    """A field-instance entry seating a player's capture where the chart
    positions that player's group.

    NotITG P1/P2 are two real players, each a group the chart moves via
    recorded pokes on the `PlayerP1`/`PlayerP2` actors (position, scale,
    rotation, hidden). `player_timelines` is that recorded transform
    (x/y/rotation/scale_x/scale_y/hidden EventTimelines) from the
    compiled dict; `rest_dx_design` is the versus-split X the field sits
    at before the chart moves it (StepMania's PlayerP{n}X metric,
    center-relative). Returns None when the player's recorded `hidden`
    bit is set (the chart hid that side), so the capture is not blitted.

    The capture pixmap is centered in the design box at scale `k`, so the
    recorded transform (design-space, centered on 320,240) conjugates
    into screen space the same way a field copy does."""
    if player_timelines is not None and \
            player_timelines['hidden'].sample(t)[0] >= 0.5:
        return None
    rx = rest_dx_design
    ry = 0.0
    rotation = 0.0
    skew = 0.0
    sx = sy = 1.0
    if player_timelines is not None:
        # Recorded x/y are absolute design-space and the player recorder
        # is SEEDED with the engine's starting position, so the stream
        # is authoritative from t=0: a chart that never moves the player
        # samples the seed (= the metric rest), the intro bounce eases
        # from it, and both-at-center is the overlap gimmick.
        def sample(prop, rest):
            tl = player_timelines.get(prop)
            return tl.sample(t)[0] if tl is not None else rest
        rx = sample('x', _SCREEN_CX) - _SCREEN_CX
        ry = sample('y', _SCREEN_CY) - _SCREEN_CY
        rotation = sample('rotation', 0.0)
        skew = sample('skew_x', 0.0)
        sx = sample('scale_x', 1.0)
        sy = sample('scale_y', 1.0)
        # rotation_y foreshortening: a y-axis turn narrows the field by
        # |cos| about its own centre - the standing 2D approximation
        # until the true 3D projection tier consumes the recorded
        # rotation_y directly (same trick as the confusionx kernel).
        rot_y = sample('rotation_y', 0.0)
        if rot_y:
            sx *= abs(math.cos(math.radians(rot_y)))
    if osc is not None:
        # Oscillator jitter deltas ride on top: vibrate's per-frame
        # random offsets are the receptor 'spazz' mirage.
        rx += osc['x'].sample(t)[0]
        ry += osc['y'].sample(t)[0]
        rotation += osc['rotation'].sample(t)[0]
    if sx == 0.0 or sy == 0.0:
        return None
    if rx == 0.0 and ry == 0.0 and rotation == 0.0 and skew == 0.0 \
            and sx == 1.0 and sy == 1.0:
        return (None, 1.0, scope)
    t_screen = QTransform()
    t_screen.translate(rx * k, ry * k)
    if rotation or skew or sx != 1.0 or sy != 1.0:
        t_screen.translate(_SCREEN_CX * k, _SCREEN_CY * k)
        if rotation:
            t_screen.rotate(rotation)
        if skew:
            t_screen.shear(skew, 0.0)
        t_screen.scale(sx, sy)
        t_screen.translate(-_SCREEN_CX * k, -_SCREEN_CY * k)
    return (t_screen, 1.0, scope)


def _design_map(chart_rect):
    """(k, ox, oy) mapping SM 640x480 design space to the chart region,
    'min' letterbox fit (the exact 640x480 box, centered) - kept in
    lockstep with the notitg storyboard's _design_transform so a field
    copy lines up with the storyboard actors drawn over it and with the
    notefield centered in the same box."""
    x, y, w, h = chart_rect
    k = min(w / _DESIGN_W, h / _DESIGN_H)
    ox = x + (w - _DESIGN_W * k) / 2.0
    oy = y + (h - _DESIGN_H * k) / 2.0
    return k, ox, oy


def design_box(chart_rect) -> QRectF:
    """The mapped SM 640x480 box in screen space (the hard crop region),
    matching _design_map. Used by the renderer to clip AFT copies to the
    design box so offscreen content never bleeds into a copy."""
    k, ox, oy = _design_map(chart_rect)
    return QRectF(ox, oy, _DESIGN_W * k, _DESIGN_H * k)


class NotitgFieldInstances:
    """Effect turning compiled field-copy timelines into per-frame
    `EffectFrame.fields`. Copies with alpha ~0 at t are dropped; the
    identity original is emitted (field-only scope) unless the base field
    is hidden at t (the chart moves the real NoteField away and lets the
    copies stand in) - then the copies replace it and no identity draws.

    `base_hidden` samples 1.0 while the real NoteField is hidden.

    `aft_bg_timeline` is accepted for compatibility with the compiled-data
    producer (games/notitg/modfile.py still emits `aft_bg_visible`) but is
    UNUSED: AFT copies now carry 'screen' scope, so background presence in
    a copy is automatic - the previous-frame composite contains the
    background exactly when the chart drew it, with no separate toggle.

    `second_field` (SecondFieldSpec | None): when a dual-player chart
    supplies it, the effect additionally
      - routes every player-2 proxy copy (P2p/P4p) to the 'field2' scope
        so it blits the second, independently-modded capture,
      - places the two identity originals at the theme P1/P2 X offsets
        (P1 left, P2 right) instead of one centered original, and
      - forwards the spec on the EffectFrame so the renderer renders the
        second capture.
    Single-player charts leave it None and behave exactly as before."""

    def __init__(self, field_copies, aft_bg_timeline=None,
                 base_hidden=None, second_field=None):
        self._copies = tuple(field_copies)
        self._base_hidden = base_hidden
        self._second_field = second_field

    def __bool__(self):
        return bool(self._copies) or self._second_field is not None

    def at(self, ctx) -> EffectFrame | None:
        if not self._copies and self._second_field is None:
            return None
        t = float(ctx.t_now)
        base_hidden = self._base_field_hidden(t)
        k, ox, oy = _design_map(ctx.chart_rect)
        copies = [entry for copy in self._copies
                  if (entry := self._instance(copy, t, k, ox, oy)) is not None]
        if self._second_field is not None:
            return self._dual_frame(base_hidden, copies, k, t)
        return self._single_frame(base_hidden, copies)

    def _single_frame(self, base_hidden, copies):
        """Single-player: one centered identity original, present only
        alongside copies (or a placeholder when the base is hidden).
        Base visible with no copies -> None (the renderer's fast path
        draws the base directly)."""
        if not base_hidden and not copies:
            return None
        if base_hidden:
            # Base suppressed (copies replace it). A zero-opacity
            # placeholder keeps `fields` non-empty even with no visible
            # copies, so the renderer takes the capture path.
            instances = copies or [(None, 0.0, _PROXY_SCOPE)]
        else:
            instances = [(None, 1.0, _PROXY_SCOPE), *copies]
        return EffectFrame(fields=tuple(instances))

    def _dual_frame(self, base_hidden, copies, k, t):
        """Dual-player: always the capture path. Each player's capture is
        seated where the chart positions that player's group - the
        recorded PlayerP1/P2 transform (rest = the versus-split X), so
        the fields split on GotoSides and overlap on the intro gimmick as
        the chart drives them. `base_hidden` (P1's hidden bit) drops both
        originals when the proxy wall owns the field; each player's own
        recorded hidden additionally drops that side. A zero-opacity
        placeholder keeps `fields` non-empty on a fully-hidden frame."""
        spec = self._second_field
        if base_hidden:
            originals = []
        else:
            originals = [
                player_placement(spec.p1_timelines, _P1_FIELD_X - _SCREEN_CX,
                                 t, k, _PROXY_SCOPE, osc=spec.p1_osc),
                player_placement(spec.p2_timelines, _P2_FIELD_X - _SCREEN_CX,
                                 t, k, _FIELD2_SCOPE, osc=spec.p2_osc),
            ]
            originals = [o for o in originals if o is not None]
        instances = [*originals, *copies] or [(None, 0.0, _PROXY_SCOPE)]
        return EffectFrame(fields=tuple(instances),
                           second_field=spec)

    def _base_field_hidden(self, t) -> bool:
        return (self._base_hidden is not None
                and self._base_hidden.sample(t)[0] >= 0.5)

    def _instance(self, copy, t, k, ox, oy):
        tl = copy['timelines']
        if tl['hidden'].sample(t)[0] >= 0.5:
            return None
        alpha = tl['alpha'].sample(t)[0]
        if alpha < _MIN_VISIBLE_ALPHA:
            return None

        x = tl['x'].sample(t)[0]
        y = tl['y'].sample(t)[0]
        rotation = tl['rotation'].sample(t)[0]
        scope = self._scope(copy)
        base_sy = tl['base_scale_y'].sample(t)[0]
        if scope == _AFT_SCOPE:
            # Charts set basezoomy(-1) on every AFT sampler purely to
            # compensate the engine's bottom-up GL captures. Our capture
            # is already top-down, so honoring the compensation flips a
            # correct image upside down over the live scene - neutralize
            # it. A deliberate extra mirror (scale_y, or a chart leaving
            # basezoomy at +1 to WANT the raw flipped texture) still
            # lands: only the sign convention is translated.
            base_sy = -base_sy
        sx = tl['scale_x'].sample(t)[0] * tl['base_scale_x'].sample(t)[0]
        sy = tl['scale_y'].sample(t)[0] * base_sy
        if sx == 0.0 or sy == 0.0:
            return None
        return (_copy_transform(x, y, rotation, sx, sy, k, ox, oy),
                min(1.0, alpha), scope)

    def _scope(self, copy) -> str:
        """The capture a copy blits from. An AFT copy blits the previous-
        frame screen composite ('screen'). A proxy copy blits a NoteField
        capture: player 2's proxies (P2p/P4p) blit the second capture
        ('field2') when this is a dual-player chart, all others the
        primary ('field')."""
        source = copy['source']
        if source not in _PROXY_SOURCES:
            return _AFT_SCOPE
        if self._second_field is not None and source in _P2_PROXY_SOURCES:
            return _FIELD2_SCOPE
        return _PROXY_SCOPE


class NotitgScreenCamera:
    """Whole-scene camera from the per-frame update's top-screen pokes.

    gat_updateproxies drives `SCREENMAN:GetTopScreen()` as a screen-zoom
    camera: it scales the top screen and offsets it by `(1-zoom)*center`
    so the zoom pivots on the design centre (the t=42 pull-back and the
    t=42/t=383 push-in). Those pokes are compiled to a screen-transform
    timeline (x/y/scale in 640x480 design space); this effect maps them
    onto the chart region and emits them on the scene_transform channel,
    the same whole-scene slot as the fluXis camera.

    The design-space transform SM applies is scale-about-origin then
    translate (`zoom(z); x((1-z)*cx); y((1-z)*cy)`), i.e. scale `z` about
    the design centre; we conjugate it by the design map so it pivots on
    the mapped centre in screen space."""

    def __init__(self, timelines):
        self._tl = timelines

    def __bool__(self):
        return self._tl is not None

    def at(self, ctx) -> EffectFrame | None:
        if self._tl is None:
            return None
        t = float(ctx.t_now)
        sx = self._tl['scale_x'].sample(t)[0]
        sy = self._tl['scale_y'].sample(t)[0]
        tx = self._tl['x'].sample(t)[0]
        ty = self._tl['y'].sample(t)[0]
        if sx == 1.0 and sy == 1.0 and tx == 0.0 and ty == 0.0:
            return None
        k, ox, oy = _design_map(ctx.chart_rect)
        transform = QTransform()
        transform.translate(ox, oy)
        transform.scale(k, k)
        transform.translate(tx, ty)
        transform.scale(sx, sy)
        transform.scale(1.0 / k, 1.0 / k)
        transform.translate(-ox, -oy)
        return EffectFrame(scene_transform=transform)


def _copy_transform(x, y, rotation, sx, sy, k, ox, oy) -> QTransform:
    """Screen-space transform placing the captured field where a copy
    actor sits: M . T_copy . M^-1. Qt post-multiplies, so the calls read
    outermost-first (M applied last to a point) and the copy's own
    move/rotate/scale about the screen centre sits in the middle."""
    t = QTransform()
    t.translate(ox, oy)
    t.scale(k, k)
    t.translate(x, y)
    t.rotate(rotation)
    t.scale(sx, sy)
    t.translate(-_SCREEN_CX, -_SCREEN_CY)
    t.scale(1.0 / k, 1.0 / k)
    t.translate(-ox, -oy)
    return t
