"""Field-instance consumer: NotITG playfield instances as
EffectFrame.fields.

A NotITG frame draws playfield captures: the two player fields (real
tournament players, each an always-rendered independently-modded group),
ActorProxy copies re-rendering a player's notefield, and
ActorFrameTexture copies replaying the previous frame's composed screen.
The compile step emits each as a generic instance carrying ONE composed
transform channel (games/notitg/field_compose); this effect samples
`instance['transform'].at(t)`, conjugates the result by the chart-region
design map, and assigns the capture scope. It holds no per-property, no
per-player, and no per-chart knowledge.

The renderer's field pipeline (effects/base.py,
qt_renderer._blit_field_instances) renders the field layer group once
into a window-sized pixmap (twice for dual-player charts - the second
capture samples player 2's mod channels, see `SecondFieldSpec`), then
blits it once per `(transform, opacity, scope)` in SCREEN space, clipped
to the chart region. `(None, 1.0, 'field')` is the untouched original.

Scopes: proxy/player instances blit a notefield capture - player 2's
instances the independently-modded second capture ('field2') when a
`second_field` spec is present, everything else the primary ('field').
AFT instances carry 'screen': the renderer blits last frame's chart-area
composite under the copy transform, which is engine-exact capture
feedback (the freeze/echo sections) and makes bg-in-capture automatic.

Design-space mapping: a sampled homography is authored in SM's 640x480
screen space; the screen transform is the conjugation M . H . M^-1
where M maps SM design space onto the chart region (kept in lockstep
with the storyboard renderer's _design_transform)."""
from __future__ import annotations

import numpy as np

from PySide6.QtCore import QRectF
from PySide6.QtGui import QTransform

from analysis.player.render import transform3d
from analysis.player.render.effects.base import EffectFrame

_DESIGN_W = 640.0
_DESIGN_H = 480.0

_PROXY_SCOPE = 'field'
_AFT_SCOPE = 'screen'
# The second-player capture scope: instances whose source is player 2
# blit from the independently-modded second capture.
_FIELD2_SCOPE = 'field2'

_IDENTITY_H = np.eye(3)


class SecondFieldSpec:
    """Instruction to the renderer to render the field layers a SECOND
    time for a dual-player NotITG chart.

    `note_mods` is the player-2 mod consumer: while the second capture
    is rendered, the renderer swaps it in for the player's primary one
    so the field's notes, receptor offsets, and reverse baseline all
    evaluate against player 2's channels (charts apply different mods
    per side). Everything else about the two captures is identical -
    only the sampled (mod, player) channels diverge."""

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


def _screen_transform(H, k, ox, oy) -> QTransform | None:
    """Window-space QTransform for a sampled capture->design homography:
    None for the identity (the untransformed-blit fast path), else the
    design-map conjugation M^-1 . H . M (Qt row-vector composition:
    the leftmost factor applies to a point first)."""
    if np.allclose(H, _IDENTITY_H, atol=1e-9):
        return None
    to_design = QTransform(1.0 / k, 0.0, 0.0, 1.0 / k, -ox / k, -oy / k)
    to_screen = QTransform(k, 0.0, 0.0, k, ox, oy)
    return to_design * transform3d.qtransform_from_h(H) * to_screen


class NotitgFieldInstances:
    """Effect turning compiled field instances into per-frame
    `EffectFrame.fields`.

    `base_hidden` samples 1.0 while the chart hides the real NoteField
    (`P1:hidden(1)`) to let copies stand in. Single-player: the identity
    original draws unless base-hidden, alongside any live copies (no
    instances live and base visible -> None, the renderer's direct-draw
    fast path). Dual-player (`second_field` present): the two player
    instances ARE the originals - base-hidden drops them while copies
    own the field - and the frame always takes the capture path (a
    zero-opacity placeholder keeps `fields` non-empty) so the second
    capture renders."""

    def __init__(self, instances, base_hidden=None, second_field=None):
        self._instances = tuple(instances)
        self._base_hidden = base_hidden
        self._second_field = second_field

    def __bool__(self):
        return bool(self._instances)

    def at(self, ctx) -> EffectFrame | None:
        if not self._instances:
            return None
        t = float(ctx.t_now)
        base_hidden = self._base_field_hidden(t)
        k, ox, oy = _design_map(ctx.chart_rect)

        entries = []
        for inst in self._instances:
            if inst['kind'] == 'player' and base_hidden:
                continue
            sampled = inst['transform'].at(t)
            if sampled is None:
                continue
            H, alpha = sampled
            entries.append((_screen_transform(H, k, ox, oy),
                            min(1.0, alpha), self._scope(inst)))

        if self._second_field is not None:
            return EffectFrame(
                fields=tuple(entries or [(None, 0.0, _PROXY_SCOPE)]),
                second_field=self._second_field)
        return self._single_frame(base_hidden, entries)

    def _single_frame(self, base_hidden, entries) -> EffectFrame | None:
        """Single-player: one centered identity original, present only
        alongside copies (or a placeholder when the base is hidden).
        Base visible with no copies -> None (the renderer's fast path
        draws the base directly)."""
        if not base_hidden and not entries:
            return None
        if base_hidden:
            # Base suppressed (copies replace it). A zero-opacity
            # placeholder keeps `fields` non-empty even with no visible
            # copies, so the renderer takes the capture path.
            entries = entries or [(None, 0.0, _PROXY_SCOPE)]
        else:
            entries = [(None, 1.0, _PROXY_SCOPE), *entries]
        return EffectFrame(fields=tuple(entries))

    def _base_field_hidden(self, t) -> bool:
        return (self._base_hidden is not None
                and self._base_hidden.sample(t)[0] >= 0.5)

    def _scope(self, inst) -> str:
        if inst['kind'] == 'aft':
            return _AFT_SCOPE
        if inst['player'] == 2 and self._second_field is not None:
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
