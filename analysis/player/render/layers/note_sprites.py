"""Default sprite specs for the standard rect/circle skins.

Each entry is a `SpriteSpec` that can rasterize one note-part into a
pixmap. Adapters compose this with their own overrides via
`GameAdapter.note_sprites(replay)`:

    def note_sprites(self, replay):
        specs = default_note_sprites()
        specs['tap_head'] = my_custom_spec
        return specs

All rasterize callbacks paint into a painter whose origin is the
pixmap's top-left. They never know about screen coordinates; the
drawer blits the result with `drawPixmap` at the right `(x, y)`.

State keys (shared across head-like sprites):
  'normal'    ; upcoming / tap / held
  'miss_tap'  ; missed non-LN
  'miss_ln'   ; missed LN (head or tail)
  'roll'      ; roll-headed LN tail
  'released'  ; successfully released LN tail

LN bodies are 1-row tile pixmaps meant for `drawTiledPixmap`; they get
their height from the draw-time rect, not from `size()`.

Head/tail/body/lift/fake sprites read their column color from
`ctx.player.palette[col]`. That palette is normally static, but games
with animated theming (fluXis `colorfade`) rewrite it per frame; the
sprite cache is keyed by `(col, state)`, so the animating consumer
(PaletteFadeEffect) invalidates the cache only when its QUANTIZED palette
changes, bounding re-rasterization. These callbacks stay palette-agnostic
-- they just read the current color.
"""
from __future__ import annotations

from analysis.player.render.layers.sprite_cache import SpriteSpec
from analysis.player.render.primitives import (
    _ellipse, _ellipse_outline, _rect, _rect_outline,
)


# ── constants (match notes.py) ───────────────────────────────────

_ROLL_BODY      = (90, 210, 90)
_ROLL_TAIL      = (60, 160, 60)
_TICK_BODY      = (255, 230, 40) # Bright yellow for convention
_MISS_TAP_BODY  = (77, 77, 77)
_MISS_LN_BODY   = (38, 38, 38)


def _dim(color, factor=2):
    return tuple(v // factor for v in color)


def _circle_r(lane_w):
    return max(6, int((lane_w - 4) * 0.46))


def _head_state_color(state, color):
    match state:
        case 'miss_tap':
            return _MISS_TAP_BODY
        case 'miss_ln':
            return _MISS_LN_BODY
        case 'tick':
            return _TICK_BODY
        case _:
            return color


def _tail_state_color(state, color):
    match state:
        case 'miss_ln':
            return _MISS_LN_BODY
        case 'roll':
            return _ROLL_TAIL
        case 'released':
            return _dim(color)
        case _:
            return color


def _body_state_color(state, color, is_roll):
    if state == 'miss_ln':
        return _MISS_LN_BODY
    if state == 'released':
        return _ROLL_TAIL if is_roll else _dim(color)
    return _ROLL_BODY if is_roll else color


# ── sprite size callbacks (evaluated on cache miss) ──────────────

# Vertical pad for head-shaped sprites so outlines, antialiased strokes,
# and oversized glyphs (the ghost-tap ring at 25% of lane_w) don't clip
# at the pixmap edges. The bar skin's note-head area is centered at
# `(lane_w/2, note_h/2 + HEAD_PAD)` inside a `(lane_w, note_h + 2*HEAD_PAD)`
# pixmap. Blit sites use `pm.height() / 2` as the y-offset so the visual
# center lines up with the note's `y` regardless of the actual pixmap
# height -- this lets the circle skin allocate a square pixmap sized to
# its diameter without breaking the bar skin's geometry.
HEAD_PAD = 6


def _head_size(ctx):
    """Pixmap size for head-shaped sprites.

    Bar: `(lane_w, note_h + 2*HEAD_PAD)` -- a flat rect tall enough to
    show the bar plus its outline pad.

    Circle: `(lane_w, lane_w)` -- a square sized to the lane so the
    full disc (~0.92 * lane_w diameter) renders without clipping
    regardless of `note_h`. Decoupling the head pixmap from `note_h`
    means the circle's apparent size tracks the lane geometry, not
    the bar-skin's scroll-speed-driven thickness.
    """
    if ctx.player.skin == 'circle':
        return ctx.lane_w, ctx.lane_w
    return ctx.lane_w, ctx.note_h + 2 * HEAD_PAD


def _mine_size(ctx):
    return ctx.lane_w, ctx.lane_w  # square box wrapping the outer radius


def _body_size(ctx):
    # 1-row pixmap for drawTiledPixmap; height forced to 1 at raster time.
    return ctx.lane_w, 1


# ── rasterize callbacks ──────────────────────────────────────────

def _head_cy(ctx):
    """Vertical center of the head area inside the head pixmap. The
    rasterize site paints into pixmap-local coords; this is the y to
    aim at so the visual center lands at `pm.height() / 2`, which is
    where every blit site anchors the note's `y`."""
    if ctx.player.skin == 'circle':
        return ctx.lane_w / 2
    return ctx.note_h / 2 + HEAD_PAD


def _rect_head_rect(lane_w, note_h):
    """Head rect in pixmap coords, inset 1 px on each side so the
    outline stroke stays fully inside the allocated space."""
    return (5, HEAD_PAD + 1, lane_w - 10, note_h - 2)


def _rasterize_tap_head(painter, key, ctx):
    """Head sprite painted into a `(lane_w, note_h + 2*HEAD_PAD)`
    pixmap, note-head centered at `y = note_h/2 + HEAD_PAD`."""
    skin = ctx.player.skin
    lane_w, note_h = ctx.lane_w, ctx.note_h
    col = key['col']
    state = key['state']
    palette_color = ctx.player.palette[col]
    color = _head_state_color(state, palette_color)

    cx = lane_w / 2
    cy = _head_cy(ctx)

    if skin == 'circle':
        r = _circle_r(lane_w)
        _ellipse(painter, color, cx, cy, r, r)
        _ellipse_outline(painter, (255, 255, 255), cx, cy, r, r)
    else:
        rect = _rect_head_rect(lane_w, note_h)
        _rect(painter, color, rect)
        _rect_outline(painter, (255, 255, 255), rect)


def _rasterize_ln_tail(painter, key, ctx):
    """LN-tail sprite. Same shape as tap_head but its state palette
    (`miss_ln`, `roll`, `released`) differs so the buckets stay distinct."""
    skin = ctx.player.skin
    lane_w, note_h = ctx.lane_w, ctx.note_h
    col = key['col']
    state = key['state']
    color = _tail_state_color(state, ctx.player.palette[col])

    cx = lane_w / 2
    cy = _head_cy(ctx)

    if skin == 'circle':
        r = _circle_r(lane_w)
        _ellipse(painter, color, cx, cy, r, r)
        _ellipse_outline(painter, (255, 255, 255), cx, cy, r, r)
    else:
        rect = _rect_head_rect(lane_w, note_h)
        _rect(painter, color, rect)
        _rect_outline(painter, (255, 255, 255), rect)


def ln_body_width(skin, lane_w) -> float:
    """Visible width of the LN body strip within a `lane_w` lane. Shared
    by the tile rasterizer and the body-path stroker (layers/notes.py) so
    straight and mod-bent bodies read as the same noodle."""
    match skin:
        case 'circle':
            return max(6, int(lane_w * 0.32))
        case _:
            return max(2, lane_w - 12)


def _rasterize_ln_body(painter, key, ctx):
    """1-row tile for `drawTiledPixmap`. Width matches the body strip;
    vertical repetition covers any LN height."""
    lane_w = ctx.lane_w
    col = key['col']
    state = key['state']
    is_roll = bool(key.get('is_roll', False))
    color = _body_state_color(state, ctx.player.palette[col], is_roll)

    body_w = ln_body_width(ctx.player.skin, lane_w)
    bx = (lane_w - body_w) / 2
    _rect(painter, color, (bx, 0, body_w, 1))


def _rasterize_mine(painter, key, ctx):
    """Concentric discs ; red inner + gray outer."""
    lane_w = ctx.lane_w
    cx = lane_w / 2
    cy = lane_w / 2
    r_outer = max(4, int(lane_w / 4))
    r_inner = max(2, int(lane_w / 8))
    _ellipse(painter, (210, 210, 210), cx, cy, r_outer, r_outer)
    _ellipse(painter, (220, 60, 60), cx, cy, r_inner, r_inner)


def _rasterize_lift(painter, key, ctx):
    """Hollow ring/rect so lifts read distinctly from filled taps."""
    skin = ctx.player.skin
    lane_w, note_h = ctx.lane_w, ctx.note_h
    col = key['col']
    color = ctx.player.palette[col]

    cx = lane_w / 2
    cy = _head_cy(ctx)

    if skin == 'circle':
        r = _circle_r(lane_w)
        _ellipse_outline(painter, color, cx, cy, r, r, 2)
        _ellipse_outline(painter, (255, 255, 255), cx, cy,
                         max(2, r // 3), max(2, r // 3))
    else:
        rect = _rect_head_rect(lane_w, note_h)
        _rect_outline(painter, color, rect, 2)
        _rect_outline(painter, (255, 255, 255),
                      (8, HEAD_PAD + note_h // 4,
                       lane_w - 16, note_h // 2), 1)


def _rasterize_fake(painter, key, ctx):
    """Dimmed tap shape ; fakes never judge."""
    skin = ctx.player.skin
    lane_w, note_h = ctx.lane_w, ctx.note_h
    col = key['col']
    color = _dim(ctx.player.palette[col], factor=4)

    cx = lane_w / 2
    cy = _head_cy(ctx)

    if skin == 'circle':
        r = _circle_r(lane_w)
        _ellipse(painter, color, cx, cy, r, r)
        _ellipse_outline(painter, (90, 90, 90), cx, cy, r, r)
    else:
        rect = _rect_head_rect(lane_w, note_h)
        _rect(painter, color, rect)
        _rect_outline(painter, (90, 90, 90), rect)


GHOST_TAP_PAD = 4


def _ghost_tap_radius(ctx):
    return max(4, int(ctx.lane_w * 0.25))


def _ghost_tap_size(ctx):
    r = _ghost_tap_radius(ctx)
    d = 2 * r + 2 * GHOST_TAP_PAD
    return d, d


def _rasterize_ghost_tap(painter, key, ctx):
    r = _ghost_tap_radius(ctx)
    w, h = _ghost_tap_size(ctx)

    cx = w / 2
    cy = h / 2

    _ellipse_outline(painter, (255, 255, 255), cx, cy, r, r)
    _ellipse(painter, (255, 255, 255), cx, cy, 2, 2)


def _tick_size(ctx):
    # Tick sprite: full lane width (inset applied at rasterize time),
    # 4 px tall. Blit offset nudges the caller to y - 2 so `y` stays
    # the tick centerline.
    return ctx.lane_w, 4


def _rasterize_tick(painter, key, ctx):
    """Small horizontal marker used by press-marks, release guides, and
    miss-hold strokes. Keyed by color so one sprite handles every
    distinct judge color."""
    lane_w = ctx.lane_w
    color = key['color']
    _rect(painter, color, (8, 0, lane_w - 16, 4))


MISS_X_PAD = 6  # room for the 3-px outline + antialias + safety
_MISS_X_PAD = MISS_X_PAD


def _miss_x_size(ctx):
    """Miss-X pixmap dimensions.

    Bar: `(lane_w, note_h + 2*pad)` -- the red outline rect extends
    `pad` above and below the head rect so the 3-px outline stroke
    stays fully inside the pixmap.

    Circle: `(lane_w, lane_w)` -- a square sized to the lane, mirroring
    the head pixmap. The red outline becomes a ring around the disc
    rather than a rectangle around a (now nonexistent) bar.
    """
    if ctx.player.skin == 'circle':
        return ctx.lane_w, ctx.lane_w
    return ctx.lane_w, ctx.note_h + 2 * _MISS_X_PAD


def _rasterize_miss_x(painter, key, ctx):
    """Skin-aware miss overlay: red outline wrapping the head shape +
    an X through the center, painted in the per-judgment color. Pixmap
    origin is set so the head's visual center lands at `pm.height()/2`
    -- the blit site centers the pixmap on the note's `y`."""
    from analysis.player.render.primitives import _line  # local - one call
    jcolor = key['jcolor']
    miss_outline = (255, 60, 60, 110)

    lane_w = ctx.lane_w
    cx = lane_w / 2

    if ctx.player.skin == 'circle':
        # Pixmap is `(lane_w, lane_w)`; head center sits at `(lane_w/2,
        # lane_w/2)`. Outline ring sits just outside the disc.
        cy = lane_w / 2
        head_r = _circle_r(lane_w)
        ring_r = head_r + _MISS_X_PAD - 1
        _ellipse_outline(painter, miss_outline, cx, cy, ring_r, ring_r, 3)
    else:
        # Pixmap is `(lane_w, note_h + 2*pad)`; head rect lives at
        # `(4, pad, lane_w-8, note_h)` so the visual center is
        # `(lane_w/2, pad + note_h/2) == (lane_w/2, pm.height()/2)`.
        note_h = ctx.note_h
        pad = _MISS_X_PAD
        cy = pad + note_h / 2
        hx, hy, hw, hh = (4, pad, lane_w - 8, note_h)
        _rect_outline(painter, miss_outline,
                      (hx - pad + 2, hy - pad + 2,
                       hw + pad * 2 - 4, hh + pad * 2 - 4), 3)

    _line(painter, jcolor, (cx - 10, cy - 10), (cx + 10, cy + 10), 2)
    _line(painter, jcolor, (cx - 10, cy + 10), (cx + 10, cy - 10), 2)


# ── public entry point ──────────────────────────────────────────

def default_note_sprites() -> dict[str, SpriteSpec]:
    """Baseline sprite set every adapter gets for free. Override any
    entry to reskin without touching the others."""
    return {
        'tap_head':   SpriteSpec(size=_head_size, rasterize=_rasterize_tap_head,
                                 key_fields=('col', 'state')),
        'ln_head':    SpriteSpec(size=_head_size, rasterize=_rasterize_tap_head,
                                 key_fields=('col', 'state')),
        'ln_tail':    SpriteSpec(size=_head_size, rasterize=_rasterize_ln_tail,
                                 key_fields=('col', 'state')),
        'ln_body':    SpriteSpec(size=_body_size, rasterize=_rasterize_ln_body,
                                 key_fields=('col', 'state', 'is_roll'),
                                 tiled=True),
        'mine':       SpriteSpec(size=_mine_size, rasterize=_rasterize_mine,
                                 key_fields=()),
        'lift':       SpriteSpec(size=_head_size, rasterize=_rasterize_lift,
                                 key_fields=('col',)),
        'fake':       SpriteSpec(size=_head_size, rasterize=_rasterize_fake,
                                 key_fields=('col',)),
        'ghost_tap':  SpriteSpec(size=_ghost_tap_size, rasterize=_rasterize_ghost_tap,
                                 key_fields=()),
        'miss_x':     SpriteSpec(size=_miss_x_size, rasterize=_rasterize_miss_x,
                                 key_fields=('jcolor',)),
        'tick':       SpriteSpec(size=_tick_size, rasterize=_rasterize_tick,
                                 key_fields=('color',)),
    }
