"""StoryboardEffect: draws a compiled Storyboard through the effects
pipeline.

Each storyboard layer becomes one z-slot draw (below or above the
chart depending on the layer's z), so playfield transforms, flashes,
and shaders compose around storyboard content exactly like any other
effect; draws are clipped to the chart region by the existing effect
machinery.

Design-space mapping: the design rect scales uniformly into the chart
region ('min' fit letterboxes, 'height' fit matches osu's 480-tall
convention where wide regions extend sideways) and everything else is
painted in design units under that transform. Element state is
sampled per frame from the IR timelines, so scrubbing is stateless.

Assets load lazily into a per-effect pixmap cache; a missing file
skips its element with one warning. Sprite tinting (color != white)
uses a cache of pre-tinted pixmaps keyed by quantized color so common
strobe tints don't re-rasterize every frame.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QImage,
                           QLinearGradient, QPainter, QPainterPath, QPen,
                           QPixmap)

from analysis.player.render import transform3d as _t3d
from analysis.player.render.effects.base import EffectFrame
from analysis.player.render.storyboard.asset_size import AssetSizeSpec, resolve
from analysis.player.render.storyboard.sprite_sheet import (frame_at_time,
                                                            frame_source_rect)

_MIN_VISIBLE_ALPHA = 1.0 / 255.0
_TINT_QUANT_LEVELS = 32

# Perspective camera default: fov 45 (SM LoadMenuPerspective). An
# element whose whole chain rests here with no out-of-plane tilt is
# affine and paints through the plain QPainter bracket (no projection).
_DEFAULT_FOV = 45.0
_EPS = 1e-6


@lru_cache(maxsize=32)
def _design_projection(design_w, design_h, fov):
    """A perspective camera's world -> design-pixel projection, cached
    per (design size, fov). General scene projection (any game's
    storyboard): fov is the frame camera's field of view in degrees,
    the design rect its viewport. A chart uses a handful of fovs."""
    return _t3d.projection(fov, design_w, design_h)


def _logical_size(el, pm):
    """The frame's LOGICAL size + grid for `el` drawn from pixmap `pm`,
    resolved through the one asset-size funnel. An element without a
    `size_spec` (fluXis/osu plain sprites, hand-built test elements) uses
    a bare grid spec, so the funnel still owns the pixels->logical map."""
    spec = el.size_spec or AssetSizeSpec(cols=el.sheet_cols,
                                         rows=el.sheet_rows)
    return resolve(pm.width(), pm.height(), spec)


_SIZE_UNSET = -1.0


def _draw_size(el, t, natural) -> tuple:
    """The element's size basis in design units before `scale_x/y` -
    SM's UNZOOMED size. It is the natural (logical) frame size unless a
    `zoomto`/`setsize` set an absolute size on that axis (`size_x`/
    `size_y` >= 0), which replaces the natural basis (so the FUCK bars'
    `zoomto(20, SCREEN_HEIGHT)` fill full height regardless of the tiny
    4px frame). `scale` still multiplies on top, exactly as SM applies
    zoom after zoomto. A group carries no size timelines, so it passes
    through untouched."""
    nat_w, nat_h = natural
    fit_mode = el.sample('fit_mode', t)[0] if 'fit_mode' in el.timelines \
        else 0.0
    if fit_mode >= 0.5:
        return _fit_size(el, t, nat_w, nat_h, fit_mode)
    (size_x,) = el.sample('size_x', t)
    (size_y,) = el.sample('size_y', t)
    w = size_x if size_x >= 0.0 else nat_w
    h = size_y if size_y >= 0.0 else nat_h
    return (w, h)


_FIT_COVER = 1.0


def _fit_size(el, t, nat_w, nat_h, mode) -> tuple:
    """Natural size scaled by the UNIFORM zoom ScaleToCover/ScaleToFitInside
    picks for the recorded rect (Actor.cpp:685-698): the per-axis ratios
    rect/natural, then the LARGER (cover, fills the rect) or SMALLER
    (fit-inside, letterboxes) of the two, applied to both axes. Returns the
    fitted UNZOOMED size; `scale_x/y` still multiplies on top, exactly as
    the engine applies zoom after SetZoom. Natural (1,1) - a sprite whose
    texture size the sim could not carry - degrades to filling the rect."""
    left = el.sample('fit_left', t)[0]
    top = el.sample('fit_top', t)[0]
    right = el.sample('fit_right', t)[0]
    bottom = el.sample('fit_bottom', t)[0]
    ratio_x = abs((right - left) / nat_w) if nat_w else 0.0
    ratio_y = abs((bottom - top) / nat_h) if nat_h else 0.0
    zoom = max(ratio_x, ratio_y) if mode == _FIT_COVER \
        else min(ratio_x, ratio_y)
    return (nat_w * zoom, nat_h * zoom)


_CROP_PROPS = ('crop_left', 'crop_top', 'crop_right', 'crop_bottom')


def _crop_fractions(el, t) -> tuple:
    """(left, top, right, bottom) crop fractions (0..1) for `el` at `t`,
    each the share of the actor hidden from that edge (SM SetCrop*). An
    element with no crop timelines (fluXis/osu sprites, plain test
    elements never poked with a crop verb) reads all-zero, so the draw is
    byte-identical to the uncropped path."""
    return tuple(el.timelines[prop].sample(t)[0] if prop in el.timelines
                 else 0.0 for prop in _CROP_PROPS)


def _inset_rect(rect: QRectF, crop) -> QRectF:
    """`rect` with each edge pulled in by its crop fraction of the rect's
    own size - the drawn sub-region left after cropping."""
    left, top, right, bottom = crop
    w, h = rect.width(), rect.height()
    return QRectF(rect.left() + left * w, rect.top() + top * h,
                  w * max(0.0, 1.0 - left - right),
                  h * max(0.0, 1.0 - top - bottom))


def _texcoord_offset(el, t):
    """The live UV scroll offset (u, v) in [0, 1), or None when the
    element never scrolls (the fast path). `texcoord_scroll` is the
    closed-form anchor (t0, offset_u, offset_v, vel_u, vel_v) recorded
    from SM's SetTexCoordVelocity; the offset at `t` is
    offset + vel * (t - t0), wrapped mod 1 exactly as the engine floors
    its accumulated coords (Sprite.cpp:364-374)."""
    timeline = el.timelines.get('texcoord_scroll')
    if timeline is None:
        return None
    t0, ou, ov, vu, vv = timeline.sample(t)
    if ou == 0.0 and ov == 0.0 and vu == 0.0 and vv == 0.0:
        return None
    return ((ou + vu * (t - t0)) % 1.0, (ov + vv * (t - t0)) % 1.0)


def _draw_scrolled(painter, rect, pm, src, uv) -> None:
    """Draw the source cell sampled at a wrapped UV offset: GL REPEAT
    under QPainter, which cannot wrap, so the shifted cell is drawn as
    up to four sub-rects (the offset tail, then the wrapped head, per
    axis) mapped onto the matching sub-rects of the dest."""
    u, v = uv
    du, dv = u * src.width(), v * src.height()
    fx = rect.width() / src.width() if src.width() else 0.0
    fy = rect.height() / src.height() if src.height() else 0.0
    cols = ((du, src.width() - du, 0.0), (0.0, du, src.width() - du))
    rows = ((dv, src.height() - dv, 0.0), (0.0, dv, src.height() - dv))
    for sx, sw, dx in cols:
        if sw <= 0.0:
            continue
        for sy, sh, dy in rows:
            if sh <= 0.0:
                continue
            dest = QRectF(rect.left() + dx * fx, rect.top() + dy * fy,
                          sw * fx, sh * fy)
            piece = QRectF(src.left() + sx, src.top() + sy, sw, sh)
            painter.drawPixmap(dest, pm, piece)


def _is_sheet(el) -> bool:
    """A sprite whose asset is an SM NxM grid (more than one frame)."""
    return el.sheet_cols * el.sheet_rows > 1


def _sheet_frame(el, t: float) -> int:
    """The frame index a sheet sprite shows at time `t`: the recorded
    setstate/animate sampler when present (anchored restarts of the
    state list), otherwise the auto-animated frame stepped through
    `sheet_states` on the effect clock (relative to the element's own
    start)."""
    if el.state_pin is not None:
        return int(round(el.state_pin.sample(t)[0]))
    return frame_at_time(el.sheet_states, t - el.t_start)


def _design_transform(storyboard, chart_rect):
    """(scale_x, scale_y, offset_x, offset_y) mapping design space to
    screen: 'min' letterboxes the whole design rect, 'height' scales by
    height alone, 'stretch' scales each axis independently to fill the
    region (NotITG: the engine stretches its 640x480 design screen to
    the window)."""
    x, y, w, h = chart_rect
    match storyboard.fit:
        case 'stretch':
            kx, ky = w / storyboard.design_w, h / storyboard.design_h
        case 'height':
            kx = ky = h / storyboard.design_h
        case _:
            kx = ky = min(w / storyboard.design_w, h / storyboard.design_h)
    ox = x + (w - storyboard.design_w * kx) / 2.0
    oy = y + (h - storyboard.design_h * ky) / 2.0
    return kx, ky, ox, oy


def _design_box_rect(storyboard, kx, ky, ox, oy) -> QRectF:
    """The mapped design rect in screen space - NotITG's hard crop box.
    Content painted past its edges is clipped away."""
    return QRectF(ox, oy, storyboard.design_w * kx, storyboard.design_h * ky)


def _bitmaptext_width(font, text: str) -> float:
    """Total pen advance for `text` in an SM bitmap font (design px)."""
    return sum(font.advance(ord(char)) for char in text)


def _is_white_texture(path: str) -> bool:
    """SM's built-in solid-white texture. Quads/Sprites that fill a
    flat color reference the virtual asset name 'white' (no real file on
    disk), so it is synthesized rather than warned as missing."""
    return isinstance(path, str) and path.strip().lower() == 'white'


def _white_pixmap() -> QPixmap:
    """A 1x1 opaque-white pixmap; drawPixmap stretches it to the
    element's size, and the sprite tint path recolors it like any
    texture (a white base multiplies cleanly to the target color)."""
    pm = QPixmap(1, 1)
    pm.fill(Qt.GlobalColor.white)
    return pm


def _quantize_color(r, g, b) -> tuple:
    q = _TINT_QUANT_LEVELS - 1
    return (round(r * q), round(g * q), round(b * q))


def _is_white(quantized) -> bool:
    q = _TINT_QUANT_LEVELS - 1
    return quantized == (q, q, q)


# -- per-corner diffuse gradient / glow / edge fade ---------------------------
# The color channels rest at sentinels the flat draw ignores: a corner < 0
# is "unset" (SetDiffuseUpperLeft never called), glow alpha 0 is "no glow"
# (Actor.cpp:1008), fade 0 is "hard edge". So an element poked with none of
# these paints the exact flat quad the base path always drew.
_CORNER_PROPS = ('color_ul', 'color_ur', 'color_lr', 'color_ll')
_FADE_PROPS = ('fade_left', 'fade_right', 'fade_top', 'fade_bottom')
_GLOW_MIN_ALPHA = 1.0 / 255.0


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _corner_qcolor(rgba, flat: QColor, alpha: float) -> QColor:
    """One gradient corner as a QColor: the individually-set corner color
    (Actor.h:190), or the flat diffuse when this corner is unset. `alpha`
    (the element's sampled diffusealpha * inherited) multiplies in, since
    SM folds the actor alpha into the vertex color."""
    if rgba[0] >= 0.0:
        r, g, b, a = rgba
    else:
        r, g, b, a = flat.redF(), flat.greenF(), flat.blueF(), 1.0
    return QColor.fromRgbF(_clamp01(r), _clamp01(g), _clamp01(b),
                           _clamp01(a * alpha))


def _gradient_pixmap(corners) -> QPixmap:
    """A 2x2 pixmap of the four corner colors (UL, UR, LL, LR). Drawn
    stretched with smooth (bilinear) scaling it reproduces SM's per-vertex
    quad gradient (Sprite.cpp:526-530) - the GPU interpolates vertex
    colors across the quad, and a smoothly-upscaled 2x2 is that exact
    bilinear field."""
    img = QImage(2, 2, QImage.Format.Format_ARGB32_Premultiplied)
    ul, ur, lr, ll = corners
    img.setPixelColor(0, 0, ul)
    img.setPixelColor(1, 0, ur)
    img.setPixelColor(0, 1, ll)
    img.setPixelColor(1, 1, lr)
    return QPixmap.fromImage(img)


def _fade_alpha_mask(w: float, h: float, fades) -> QImage | None:
    """An alpha mask (white, per-pixel alpha) ramping from 0 at each faded
    edge up to 1 at the fade distance inward (SM SetFade*, Sprite.cpp:560).
    `fades` is (left, right, top, bottom) as fractions of the w x h rect.
    None when no edge fades (the caller then skips the mask entirely)."""
    left, right, top, bottom = fades
    if max(left, right, top, bottom) <= 0.0:
        return None
    img = QImage(max(1, int(round(w))), max(1, int(round(h))),
                 QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.white)
    painter = QPainter(img)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
    for edge, dist in zip(('left', 'right', 'top', 'bottom'), fades):
        if dist > 0.0:
            _paint_edge_ramp(painter, img.width(), img.height(), edge, dist)
    painter.end()
    return img


def _paint_edge_ramp(painter, w, h, edge, dist) -> None:
    """Multiply into the mask a linear alpha ramp 0->1 over `dist` of the
    rect from `edge`, holding 1 across the rest (a per-edge fade the
    DestinationIn composition intersects with the others)."""
    horizontal = edge in ('left', 'right')
    span = (w if horizontal else h) * _clamp01(dist)
    if edge == 'left':
        grad = QLinearGradient(0, 0, span, 0)
        stops = ((0.0, 0.0), (1.0, 1.0))
    elif edge == 'right':
        grad = QLinearGradient(w - span, 0, w, 0)
        stops = ((0.0, 1.0), (1.0, 0.0))
    elif edge == 'top':
        grad = QLinearGradient(0, 0, 0, span)
        stops = ((0.0, 0.0), (1.0, 1.0))
    else:
        grad = QLinearGradient(0, h - span, 0, h)
        stops = ((0.0, 1.0), (1.0, 0.0))
    for pos, a in stops:
        grad.setColorAt(pos, QColor.fromRgbF(1.0, 1.0, 1.0, a))
    # Outside the ramp band the gradient's Pad spread holds the end alpha
    # (1 on the interior side), so only the edge band is attenuated.
    painter.fillRect(QRectF(0, 0, w, h), grad)


class _ElementWalk:
    """The group-descent source for the direct Element-tree walk: a
    group's live children are its in-window `el.children`, carrying no
    node handle. `_paint_children` calls `children(group_el, node, t)`;
    the document renderer swaps in a walker that reads the node tree
    instead, keeping the paint math untouched."""

    def children(self, el, node, t):
        return tuple((child, None) for child in el.children
                     if child.t_start <= t < child.t_end)


_ELEMENT_WALK = _ElementWalk()

# Phase-3 consolidation switch. When True, StoryboardEffect renders by
# walking the compiled-document node tree (group edges compose transforms,
# layer slots band draws, the REQUIRED visibility timeline gates) instead
# of the parallel Element arrays -- the document becomes the single source
# the player draws from. Both paths are proven byte-identical
# (tests/test_document_equivalence.py) and oracle-stable, so the default
# is the document path; flip to False to fall back to the direct Element
# walk (kept as the equivalence reference and a safety valve).
USE_DOCUMENT_PATH = True


class StoryboardEffect:
    def __init__(self, storyboard):
        # Only TOP-LEVEL elements own z-slots; a group draws its whole
        # subtree inside its own slot bracket, so nested children never
        # appear in the layer table.
        elements = sorted(storyboard.elements,
                          key=lambda e: (e.z, e.z_index, e.t_start))
        self._sb = storyboard
        self._elements = tuple(elements)
        self._starts = np.array([e.t_start for e in elements],
                                dtype=np.float64)
        self._ends = np.array([e.t_end for e in elements], dtype=np.float64)
        zs = sorted({e.z for e in elements})
        self._layer_indices = tuple(
            (z, tuple(i for i, e in enumerate(elements) if e.z == z))
            for z in zs)
        self._pixmaps = {}
        self._tinted = {}
        self._text_metrics = {}
        self._swapped = {}
        self._document_renderer = (self._build_document_renderer(storyboard)
                                   if USE_DOCUMENT_PATH else None)

    def _build_document_renderer(self, storyboard):
        """Build the node-tree renderer this effect delegates `at` to when
        the document path is on. Local imports break the render.document <->
        render.storyboard construction cycle (document.render imports this
        module for the paint helpers); this runs once per effect, not on
        the hot path."""
        from analysis.player.render.document.builder import storyboard_document
        from analysis.player.render.document.design_space import DesignSpace
        from analysis.player.render.document.render import (
            DocumentStoryboardRenderer)

        design = DesignSpace(width=storyboard.design_w,
                             height=storyboard.design_h,
                             fit=storyboard.fit, clip=storyboard.clip_design_box)
        document, index = storyboard_document(storyboard, design)
        return DocumentStoryboardRenderer(document, index, storyboard, self)

    def __bool__(self):
        return bool(self._elements)

    def at(self, ctx) -> EffectFrame | None:
        if self._document_renderer is not None:
            return self._document_renderer.at(ctx)
        return self._at_element_walk(ctx)

    def _at_element_walk(self, ctx) -> EffectFrame | None:
        t = float(ctx.t_now)
        active = (self._starts <= t) & (t < self._ends)
        if not active.any():
            return None

        draws = []
        for z, indices in self._layer_indices:
            live = tuple(i for i in indices if active[i])
            if live:
                draws.append((z, self._layer_draw(live, t)))
        if not draws:
            return None
        return EffectFrame(draws=tuple(draws))

    def _layer_draw(self, indices, t):
        def draw(ctx, painter):
            kx, ky, ox, oy = _design_transform(self._sb, ctx.chart_rect)
            painter.save()
            if self._sb.clip_design_box:
                painter.setClipRect(
                    _design_box_rect(self._sb, kx, ky, ox, oy),
                    Qt.ClipOperation.IntersectClip)
            # One world transform maps design space to screen; elements
            # paint in raw design coordinates (per-axis scales shear
            # rotated content exactly as the engine's stretch does).
            painter.translate(ox, oy)
            painter.scale(kx, ky)
            for i in indices:
                self._paint_element(painter, self._elements[i], t, 1.0,
                                    0.0, 0.0,
                                    self._sb.design_w, self._sb.design_h)
            painter.restore()
        return draw

    # -- element painting -------------------------------------------------

    def _paint_element(self, painter, el, t, k, ox, oy,
                       ref_w, ref_h, inherited_alpha=1.0,
                       walker=_ELEMENT_WALK, node=None,
                       world3d=None, fov=None) -> None:
        el = self._swap_view(el, t)
        # SM's `hidden` bit hard-gates the draw independently of alpha, so
        # an actor carrying a diffusealpha crossfade stays dark while
        # hidden (the ShowAFTBG capture sprite sits `hidden,1` until its
        # message shows it).
        if el.sample('hidden', t)[0] >= 0.5:
            return
        alpha = el.sample('alpha', t)[0] * inherited_alpha
        if alpha < _MIN_VISIBLE_ALPHA:
            return
        # A group has no natural size; it rotates/scales about its own
        # anchor + position point (zero-size origin), then draws children.
        size = (0.0, 0.0) if el.kind == 'group' else self._element_size(el, t)
        if size is None:
            return
        w, h = _draw_size(el, t, size)

        (x,) = el.sample('x', t)
        (y,) = el.sample('y', t)
        (rotation,) = el.sample('rotation', t)
        (sx,) = el.sample('scale_x', t)
        (sy,) = el.sample('scale_y', t)
        if el.flip_h:
            sx = -sx
        if el.flip_v:
            sy = -sy
        if sx == 0.0 or sy == 0.0:
            return

        # 3D scene channels. A frame's `fov` sets the perspective camera
        # for its whole subtree (the innermost that set one wins); the
        # out-of-plane rotations / z push / skew tilt the actor plane. As
        # long as nothing in the chain is 3D-active the element paints
        # through the exact 2D QPainter bracket below - the flat-chart
        # no-op path.
        (rx,) = el.sample('rotation_x', t)
        (ry,) = el.sample('rotation_y', t)
        (z,) = el.sample('z', t)
        (sz,) = el.sample('scale_z', t)
        (skx,) = el.sample('skew_x', t)
        (sky,) = el.sample('skew_y', t)
        (el_fov,) = el.sample('fov', t)
        fov = el_fov if abs(el_fov - _DEFAULT_FOV) > _EPS else fov
        active_3d = (world3d is not None
                     or (fov is not None and abs(fov - _DEFAULT_FOV) > _EPS)
                     or any(abs(v) > _EPS for v in (rx, ry, z, skx, sky))
                     or abs(sz - 1.0) > _EPS)

        if active_3d:
            self._paint_element_3d(
                painter, el, t, k, ox, oy, ref_w, ref_h, alpha, walker,
                node, world3d, fov, w, h, x, y, rotation, sx, sy, rx, ry,
                z, sz, skx, sky)
            return

        ax, ay = el.anchor
        painter.save()
        painter.translate(ox + (ax * ref_w + x) * k,
                          oy + (ay * ref_h + y) * k)
        painter.scale(k, k)
        if rotation:
            painter.rotate(rotation)
        painter.scale(sx, sy)
        painter.translate(-el.origin[0] * w, -el.origin[1] * h)
        painter.setOpacity(min(1.0, alpha))
        if el.additive:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Plus)
        if el.kind == 'group':
            self._paint_children(painter, el, t, alpha, walker, node)
        else:
            self._paint_kind(painter, el, t, w, h)
        painter.restore()

    def _paint_element_3d(self, painter, el, t, k, ox, oy, ref_w, ref_h,
                          alpha, walker, node, world3d, fov, w, h, x, y,
                          rotation, sx, sy, rx, ry, z, sz, skx, sky) -> None:
        """Paint `el` through its frame chain's perspective camera.

        The element's local model matrix (design space: translate to its
        anchored position incl. z, rotate about all three axes, scale,
        skew) composes onto the inherited group world; the result is
        projected by the chain's fov camera (LoadMenuPerspective in
        design pixels). A group recurses with the composed world so its
        children inherit the tilt; a leaf draws its quad through the
        planar homography (QPainter executes the projective transform).

        Design pixels ARE the space the layer's painter maps to screen
        (the design->screen translate+scale is already on the painter),
        so the projection targets design pixels and the existing mapping
        carries it to the window - a game-agnostic scene projection, not
        NotITG-specific."""
        ax, ay = el.anchor
        # Local model in design space: rotate/scale/skew about the
        # element origin, translate to its anchored design position.
        px = ax * ref_w + x
        py = ay * ref_h + y
        local = _t3d.local_matrix(
            pos=(px, py, z), rot=(rx, ry, rotation), scl=(sx, sy, sz),
            skewx=skx, skewy=sky)
        world = local if world3d is None else _t3d.compose(world3d, local)
        cam_fov = _DEFAULT_FOV if fov is None else fov

        if el.additive:
            painter.save()
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Plus)
        painter.setOpacity(min(1.0, alpha))

        if el.kind == 'group':
            for child_el, child_node in walker.children(el, node, t):
                self._paint_element(painter, child_el, t, 1.0, 0.0, 0.0,
                                    0.0, 0.0, alpha, walker, child_node,
                                    world3d=world, fov=cam_fov)
        else:
            self._draw_quad_3d(painter, el, t, w, h, world, cam_fov)
        if el.additive:
            painter.restore()

    def _draw_quad_3d(self, painter, el, t, w, h, world, fov) -> None:
        """Draw a leaf element's w x h quad through the projected world.

        The quad's own corners sit at [-origin*size, +(-origin+1)*size]
        (the origin-relative box the 2D path draws via the final
        `translate(-origin*w, -origin*h)`). Its content->design-pixel
        homography is `origin_box @ world @ projection`; setting it on
        the painter (combined with the layer's design->screen map already
        in place) draws the element's local rect (0,0,w,h) exactly where
        the perspective puts it."""
        ox_, oy_ = el.origin
        content_to_world = _t3d.translate(-ox_ * w, -oy_ * h) @ world
        projection = _design_projection(self._sb.design_w,
                                        self._sb.design_h, fov)
        corners = ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h))
        verdict, H, _clip = _t3d.project_with_verdict(
            content_to_world, projection, corners)
        if verdict == 'gone':
            return
        saved = painter.transform()
        painter.setTransform(_t3d.qtransform_from_h(H), combine=True)
        self._paint_kind(painter, el, t, w, h)
        painter.setTransform(saved)

    def _paint_children(self, painter, el, t, group_alpha,
                        walker=_ELEMENT_WALK, node=None) -> None:
        """Draw a group's subtree in the group's own transformed space.
        The group bracket already applied its translate/rotate/scale, so
        the painter origin is now the frame's local (0, 0): children
        position by raw (x, y) relative to it (SM ActorFrame semantics,
        a zero-size anchor box) at k=1. Each child re-checks its own
        window, so one outside [t_start, t_end) is skipped while siblings
        still draw; if the whole group is outside its window the caller
        never reaches here.

        `walker` supplies the child sequence: the default walks the
        Element tree (`el.children`); the document renderer passes a walker
        that walks the compiled-document node tree instead, so the SAME
        transform/paint math runs but the group descent and the window
        gate come from the node model (see render/document/render.py)."""
        for child_el, child_node in walker.children(el, node, t):
            self._paint_element(painter, child_el, t, 1.0, 0.0, 0.0,
                                0.0, 0.0, group_alpha, walker, child_node)

    def _paint_kind(self, painter, el, t, w, h) -> None:
        """Draw the element's content, then the additive glow pass. Content
        is the flat diffuse (the fast path) unless the actor set a
        per-corner diffuse gradient or an edge fade, in which case the quad
        is drawn through a gradient/fade compositor. glow (SetGlow) rides on
        top as an additive-tinted overlay (Sprite.cpp's glow pass)."""
        color = self._qcolor(el.sample('color', t))
        corners = self._corners(el, t)
        fades = tuple(el.sample(p, t)[0] for p in _FADE_PROPS)
        if corners is not None or max(fades) > 0.0:
            self._paint_color_composited(painter, el, t, w, h, color,
                                         corners, fades)
        else:
            self._paint_flat(painter, el, t, w, h, color)
        self._paint_glow(painter, el, t, w, h)

    def _corners(self, el, t):
        """The four gradient corner QColors (UL, UR, LR, LL) when any corner
        was individually set, else None (the flat-diffuse fast path). Each
        unset corner falls back to the flat diffuse color."""
        rgba = tuple(el.sample(p, t) for p in _CORNER_PROPS)
        if all(c[0] < 0.0 for c in rgba):
            return None
        flat = self._qcolor(el.sample('color', t))
        return tuple(_corner_qcolor(c, flat, 1.0) for c in rgba)

    def _paint_flat(self, painter, el, t, w, h, color) -> None:
        rect = QRectF(0.0, 0.0, w, h)
        crop = _crop_fractions(el, t)
        if crop != (0.0, 0.0, 0.0, 0.0):
            rect = _inset_rect(rect, crop)
        match el.kind:
            case 'sprite' | 'frames':
                pm = self._tinted_pixmap(self._asset_at(el, t), color)
                src = _inset_rect(self._source_rect(el, t, pm), crop)
                uv = _texcoord_offset(el, t)
                if uv is None:
                    painter.drawPixmap(rect, pm, src)
                else:
                    _draw_scrolled(painter, rect, pm, src, uv)
            case 'rect':
                painter.fillRect(rect, color)
            case 'ellipse':
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(rect)
            case 'outline_rect' | 'outline_ellipse':
                (border,) = el.sample('border', t)
                painter.setPen(QPen(color, max(0.1, border)))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                if el.kind == 'outline_rect':
                    painter.drawRect(rect)
                else:
                    painter.drawEllipse(rect)
            case 'text':
                font, metrics = self._font_for(el)
                painter.setFont(font)
                painter.setPen(color)
                painter.drawText(QPointF(0.0, metrics.ascent()), el.text)
            case 'bitmaptext':
                self._paint_bitmaptext(painter, el, color)

    def _paint_color_composited(self, painter, el, t, w, h, color,
                                corners, fades) -> None:
        """Render the quad with a per-corner diffuse gradient and/or edge
        fades into an offscreen buffer, then blit it once.

        The gradient is a smoothly-upscaled 2x2 of the corner colors -
        SM's per-vertex quad interpolation (Sprite.cpp:526-530). For a
        sprite the gradient MULTIPLIES the texture (the diffuse tint the
        vertex colors are); for a rect/ellipse it IS the fill. Edge fades
        multiply an alpha ramp mask over the result (Sprite.cpp:560). Kinds
        without a quad body (outline/text) have no gradient analogue, so
        they fall back to the flat draw under the fade mask only."""
        iw, ih = max(1, int(round(w))), max(1, int(round(h)))
        buf = QPixmap(iw, ih)
        buf.fill(Qt.GlobalColor.transparent)
        bp = QPainter(buf)
        bp.scale(iw / w if w else 1.0, ih / h if h else 1.0)
        if corners is not None and el.kind in ('sprite', 'frames', 'rect',
                                               'ellipse'):
            self._paint_gradient_body(bp, el, t, w, h, corners)
        else:
            self._paint_flat(bp, el, t, w, h, color)
        mask = _fade_alpha_mask(iw, ih, fades)
        if mask is not None:
            bp.resetTransform()
            bp.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_DestinationIn)
            bp.drawImage(0, 0, mask)
        bp.end()
        painter.drawPixmap(QRectF(0.0, 0.0, w, h), buf, QRectF(buf.rect()))

    def _paint_gradient_body(self, painter, el, t, w, h, corners) -> None:
        """The gradient diffuse pass into `painter`'s buffer. A sprite's
        texture is multiplied by the gradient (vertex-color tint); a solid
        rect/ellipse is filled with the gradient directly."""
        rect = QRectF(0.0, 0.0, w, h)
        crop = _crop_fractions(el, t)
        if crop != (0.0, 0.0, 0.0, 0.0):
            rect = _inset_rect(rect, crop)
        grad = _gradient_pixmap(corners)
        match el.kind:
            case 'sprite' | 'frames':
                pm = self._pixmap(self._asset_at(el, t))
                if pm is None:
                    return
                src = _inset_rect(self._source_rect(el, t, pm), crop)
                painter.drawPixmap(rect, pm, src)
                painter.setCompositionMode(
                    QPainter.CompositionMode.CompositionMode_Multiply)
                painter.drawPixmap(rect, grad, QRectF(grad.rect()))
                painter.setCompositionMode(
                    QPainter.CompositionMode.CompositionMode_DestinationIn)
                painter.drawPixmap(rect, pm, src)
            case 'rect':
                painter.drawPixmap(rect, grad, QRectF(grad.rect()))
            case 'ellipse':
                # Clip the gradient to the ellipse shape so the gradient
                # fill matches the flat ellipse's silhouette.
                path = QPainterPath()
                path.addEllipse(rect)
                painter.save()
                painter.setClipPath(path)
                painter.drawPixmap(rect, grad, QRectF(grad.rect()))
                painter.restore()

    def _paint_glow(self, painter, el, t, w, h) -> None:
        """SM's additive glow pass (Sprite.cpp:536-541): a quad the size of
        the element, filled with the glow color, drawn additively at the
        glow alpha. glow rests at alpha 0 so an un-glowed actor draws
        nothing here."""
        r, g, b, a = el.sample('glow', t)
        if a <= _GLOW_MIN_ALPHA:
            return
        painter.save()
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_Plus)
        painter.setOpacity(_clamp01(a))
        glow = QColor.fromRgbF(_clamp01(r), _clamp01(g), _clamp01(b), 1.0)
        crop = _crop_fractions(el, t)
        rect = _inset_rect(QRectF(0.0, 0.0, w, h), crop) \
            if crop != (0.0, 0.0, 0.0, 0.0) else QRectF(0.0, 0.0, w, h)
        if el.kind in ('sprite', 'frames'):
            pm = self._tinted_pixmap(self._asset_at(el, t), glow)
            src = _inset_rect(self._source_rect(el, t, pm), crop)
            painter.drawPixmap(rect, pm, src)
        else:
            painter.fillRect(rect, glow)
        painter.restore()

    def _paint_bitmaptext(self, painter, el, color) -> None:
        """Composite `el.text` from its SM bitmap-font atlas: one glyph
        cell per character, each drawn centred on the advancing pen so
        ink lines up like StepMania's own drawing, tinted by `color`.
        The whole string is centred on the element origin (SM BitmapText
        default), matching the (0.5, 0.5) origin the compiler assigns."""
        atlas = self._pixmap(el.font.texture_path)
        if atlas is None:
            return
        glyphs = self._tinted_pixmap(el.font.texture_path, color)
        pen = -_bitmaptext_width(el.font, el.text) / 2.0
        cell_w, cell_h = el.font.cell_logical(atlas.width(), atlas.height())
        top = -cell_h / 2.0
        for char in el.text:
            codepoint = ord(char)
            advance = el.font.advance(codepoint)
            cell = el.font.cell(codepoint, atlas.width(), atlas.height())
            if cell is not None:
                cx, cy, cw, ch = cell
                dest = QRectF(pen + (advance - cell_w) / 2.0, top,
                              cell_w, cell_h)
                painter.drawPixmap(dest, glyphs, QRectF(cx, cy, cw, ch))
            pen += advance

    def _source_rect(self, el, t, pm) -> QRectF:
        """The region of `pm` this sprite draws: one grid cell for an SM
        NxM sheet (the current frame), else the whole pixmap."""
        if not _is_sheet(el):
            return QRectF(pm.rect())
        frame = _sheet_frame(el, t)
        grid = _logical_size(el, pm)
        x, y, w, h = frame_source_rect(frame, pm.width(), pm.height(),
                                       grid.cols, grid.rows)
        return QRectF(x, y, w, h)

    def _element_size(self, el, t):
        """Natural (w, h) in design units, or None when undrawable. A
        sheet sprite's natural size is ONE frame, not the whole sheet."""
        match el.kind:
            case 'sprite' | 'frames':
                pm = self._pixmap(self._asset_at(el, t))
                if pm is None:
                    return None
                return _logical_size(el, pm).natural
            case 'text':
                _font, metrics = self._font_for(el)
                bounds = metrics.boundingRect(el.text)
                return (bounds.width(), metrics.height())
            case 'bitmaptext':
                atlas = self._pixmap(el.font.texture_path)
                if atlas is None:
                    return None
                _cell_w, cell_h = el.font.cell_logical(atlas.width(),
                                                       atlas.height())
                return (_bitmaptext_width(el.font, el.text), cell_h)
            case _:
                (w,) = el.sample('w', t)
                (h,) = el.sample('h', t)
                return (w, h) if w > 0 and h > 0 else None

    # -- asset caches -------------------------------------------------------

    def _swap_view(self, el, t):
        """The element with any recorded runtime texture swap applied
        (SM Sprite:Load - the `asset_swap` channel carries (path, cols,
        rows) from the game frontend). Same element object until a swap
        is live, then a cached clone with the new asset + sheet grid and
        SM's default sequential animation states (Sprite.cpp:246-285
        rebuilds m_States on load; a recorded state_pin still overrides
        them). The clone is cached per (element, path) so steady frames
        never re-allocate."""
        timeline = el.timelines.get('asset_swap')
        if timeline is None or el.kind not in ('sprite', 'frames'):
            return el
        path, cols, rows = timeline.sample(t)
        if not path or path == el.asset:
            return el

        key = (id(el), path)
        view = self._swapped.get(key)
        if view is None:
            cols, rows = max(1, int(cols)), max(1, int(rows))
            view = replace(
                el, kind='sprite', asset=path, frames=(),
                sheet_cols=cols, sheet_rows=rows,
                sheet_states=tuple((i, 0.1) for i in range(cols * rows)),
                size_spec=None)
            self._swapped[key] = view
        return view

    def _asset_at(self, el, t) -> str | None:
        if el.kind != 'frames':
            return el.asset
        if not el.frames or el.frame_delay <= 0:
            return el.frames[0] if el.frames else None
        i = int((t - el.t_start) / el.frame_delay)
        if el.loop_forever:
            i %= len(el.frames)
        return el.frames[min(i, len(el.frames) - 1)]

    def _pixmap(self, path) -> QPixmap | None:
        if path is None:
            return None
        if path not in self._pixmaps:
            pm = (_white_pixmap() if _is_white_texture(path)
                  else QPixmap(path))
            if pm.isNull():
                print(f'storyboard asset missing/unreadable: {path}')
            self._pixmaps[path] = None if pm.isNull() else pm
        return self._pixmaps[path]

    def _tinted_pixmap(self, path, color: QColor) -> QPixmap:
        pm = self._pixmap(path)
        quantized = _quantize_color(color.redF(), color.greenF(),
                                    color.blueF())
        if _is_white(quantized):
            return pm

        key = (path, quantized)
        if key not in self._tinted:
            tinted = QPixmap(pm.size())
            tinted.fill(Qt.GlobalColor.transparent)
            p = QPainter(tinted)
            p.drawPixmap(0, 0, pm)
            p.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Multiply)
            p.fillRect(tinted.rect(), color)
            p.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_DestinationIn)
            p.drawPixmap(0, 0, pm)
            p.end()
            self._tinted[key] = tinted
        return self._tinted[key]

    def _font_for(self, el):
        key = el.font_px
        if key not in self._text_metrics:
            font = QFont()
            font.setPixelSize(max(1, int(el.font_px or 32)))
            self._text_metrics[key] = (font, QFontMetricsF(font))
        return self._text_metrics[key]

    def _qcolor(self, rgb) -> QColor:
        r, g, b = rgb
        return QColor.fromRgbF(max(0.0, min(1.0, r)),
                               max(0.0, min(1.0, g)),
                               max(0.0, min(1.0, b)))
