"""Per-frame note-palette consumer for animated fluXis color theming.

The note sprite cache rasterizes one pixmap per `(col, state)` and reads
the column color from `ctx.player.palette[col]` at raster time. A
`colorfade` timeline changes those colors continuously, so a naive
consumer would re-rasterize every note sprite every frame -- unacceptable
(the cache exists precisely to avoid per-frame rasterization).

QUANTIZATION TRADEOFF (mirrors the storyboard tint cache in
render/storyboard/render.py): we snap each sampled channel to one of
`_QUANT_LEVELS` steps and only push a new palette + invalidate the sprite
cache when the quantized palette actually changes. A smooth 0..255 fade
that would otherwise trigger a raster on every frame instead re-rasters
at most `_QUANT_LEVELS` times per channel over its whole span. The cost
is banding: the note color steps rather than fading pixel-smoothly. At 32
levels the steps are ~8/255 apart -- imperceptible on note fills at
gameplay speed, and the fade still reads as continuous.

This runs as an Effect (per-frame `at(ctx)`) purely for its side effect
on `ctx.player.palette` + the sprite cache; it draws nothing, so it
returns None (an inactive frame) and adds zero compositing cost.
"""
from __future__ import annotations

# Match the storyboard tint cache granularity (render/storyboard/render.py).
_QUANT_LEVELS = 32


def _quantize(rgb):
    q = _QUANT_LEVELS - 1
    return tuple(round(c / 255 * q) * 255 // q for c in rgb)


class PaletteFadeEffect:
    """Samples an animated `NotePalette` each frame; when the quantized
    palette changes it writes the new colors to `player.palette` and
    invalidates the note sprite cache so the next draw re-rasterizes at
    the new colors."""

    def __init__(self, palette):
        self._palette = palette
        self._last_quantized = None

    def __bool__(self):
        # Only worth wiring when the palette actually animates; a static
        # palette is already baked into `player.palette` at init.
        return bool(self._palette) and self._palette.animated

    def at(self, ctx):
        player = getattr(ctx, 'player', None)
        if player is None:
            return None
        sampled = self._palette.sample(ctx.t_now)
        quantized = tuple(_quantize(c) for c in sampled)
        if quantized != self._last_quantized:
            self._last_quantized = quantized
            player.palette = [tuple(c) for c in quantized]
            cache = getattr(ctx, 'sprite_cache', None)
            if cache is not None:
                cache.invalidate()
        return None
