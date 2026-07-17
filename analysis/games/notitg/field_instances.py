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
content is notes + receptors, never the background. ActorFrameTexture
copies (gat_aft) grab the whole SM screen, but only some sections include
the background (the chart toggles a fullscreen bg.png between its
`ShowAFT` and `ShowAFTBG` states). Each instance therefore carries a
`scope`: 'field' blits the transparent notefield capture (one shared
background shows through every copy); 'full' blits the whole-screen
capture (background baked in). Proxy copies are always 'field'; AFT
copies sample a compiled bg-in-capture timeline per frame.

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
# capture the whole screen and follow the chart's bg-in-capture timeline.
_PROXY_SCOPE = 'field'
_PROXY_SOURCES = frozenset({'P1p', 'P2p', 'P3p', 'P4p'})


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

    `aft_bg_timeline` samples 1.0 while an AFT capture includes the
    background (gat's `ShowAFTBG`), 0.0 otherwise; AFT copies pick their
    scope from it per frame. `base_hidden` samples 1.0 while the real
    NoteField is hidden."""

    def __init__(self, field_copies, aft_bg_timeline=None,
                 base_hidden=None):
        self._copies = tuple(field_copies)
        self._aft_bg = aft_bg_timeline
        self._base_hidden = base_hidden

    def __bool__(self):
        return bool(self._copies)

    def at(self, ctx) -> EffectFrame | None:
        if not self._copies:
            return None
        t = float(ctx.t_now)
        base_hidden = self._base_field_hidden(t)
        k, ox, oy = _design_map(ctx.chart_rect)
        copies = [entry for copy in self._copies
                  if (entry := self._instance(copy, t, k, ox, oy)) is not None]
        if not base_hidden and not copies:
            # Base visible, no copies: nothing to replicate - let the
            # renderer's fast path draw the base field directly.
            return None
        if base_hidden:
            # The base field is suppressed (the copies replace it). A
            # zero-opacity placeholder keeps `fields` non-empty even with
            # no visible copies this frame, so the renderer takes the
            # capture path (which never draws the base directly) instead of
            # the fast path.
            instances = copies or [(None, 0.0, 'field')]
        else:
            instances = [(None, 1.0, 'field'), *copies]
        return EffectFrame(fields=tuple(instances))

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
        sx = tl['scale_x'].sample(t)[0] * tl['base_scale_x'].sample(t)[0]
        sy = tl['scale_y'].sample(t)[0] * tl['base_scale_y'].sample(t)[0]
        if sx == 0.0 or sy == 0.0:
            return None
        return (_copy_transform(x, y, rotation, sx, sy, k, ox, oy),
                min(1.0, alpha), self._scope(copy, t))

    def _scope(self, copy, t) -> str:
        """A proxy copy is always field-only; an AFT copy carries the
        background only while the bg-in-capture timeline is set."""
        if copy['source'] in _PROXY_SOURCES:
            return _PROXY_SCOPE
        if self._aft_bg is not None and self._aft_bg.sample(t)[0] >= 0.5:
            return 'full'
        return 'field'


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
