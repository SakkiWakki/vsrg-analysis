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

    def __init__(self, note_mods):
        self.note_mods = note_mods


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
            return self._dual_frame(base_hidden, copies, k)
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

    def _dual_frame(self, base_hidden, copies, k):
        """Dual-player: always the capture path. Two identity originals at
        the theme P1/P2 X offsets (P1 left from the primary capture, P2
        right from the second capture) unless the base is hidden, plus the
        routed copies, plus the second_field spec so the renderer renders
        the second capture. A zero-opacity placeholder keeps `fields`
        non-empty on a fully-hidden frame."""
        if base_hidden:
            originals = []
        else:
            originals = [
                _field_placement(_P1_FIELD_X - _SCREEN_CX, k, _PROXY_SCOPE),
                _field_placement(_P2_FIELD_X - _SCREEN_CX, k, _FIELD2_SCOPE),
            ]
        instances = [*originals, *copies] or [(None, 0.0, _PROXY_SCOPE)]
        return EffectFrame(fields=tuple(instances),
                           second_field=self._second_field)

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


def _field_placement(dx_design, k, scope):
    """A field-instance entry that blits `scope`'s capture translated by
    `dx_design` design-px horizontally. The capture pixmap is already in
    screen space at the design map's scale `k`, so a design-space X shift
    is `dx_design * k` screen px; a zero shift is the untouched identity
    (None transform, the renderer's direct blit). Used to seat the two
    dual-player originals at the theme P1/P2 X offsets."""
    if dx_design == 0.0:
        return (None, 1.0, scope)
    t = QTransform()
    t.translate(dx_design * k, 0.0)
    return (t, 1.0, scope)


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
