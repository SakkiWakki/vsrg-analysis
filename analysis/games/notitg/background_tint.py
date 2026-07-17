"""Background tint/blackout from gat's `dark` (and `cover`) channels.

gat darkens the whole scene in stretches - the reference's cover art
fades to red-tinted (t~134) then near-black (t~280) while the notes and
receptors stay bright. In NotITG this is the `dark` mod (and, when sane,
`cover`) driven from the mod tables. We realise it as a black overlay
drawn just above the background storyboard and below the notes, with
alpha = the sampled channel: 0 = untouched, 1 = fully black.

`cover` in gat is authored as a nonsense `-10000000` value (a utility
poke, not a real blackout), so it is clamped to [0, 1] and contributes
nothing here; `dark` carries the real signal. Sampling the compiled
channels keeps the tint scrub-exact and needs no per-frame mod state.
"""
from __future__ import annotations

from PySide6.QtGui import QColor

from analysis.games.notitg.field_instances import design_box
from analysis.player.render.effects.base import EffectFrame

# Between the background art (hoisted to z=-100, behind the notes) and
# the notes (drawn in the field pass, above the below-band): a below-draw
# just over the background darkens the art toward the reference's
# blackout while the notes and receptors stay bright, matching the
# reference (dark cover art, bright notes).
_TINT_Z = -50
_MAX_TINT = 1.0
_MIN_VISIBLE = 1.0 / 255.0
_BLACK = QColor(0, 0, 0)


class NotitgBackgroundTint:
    """Effect sampling the `dark`/`cover` channels into a black overlay."""

    def __init__(self, channels):
        self._channels = channels

    def __bool__(self):
        return self._channels is not None

    def at(self, ctx) -> EffectFrame | None:
        alpha = self._alpha(float(ctx.t_now))
        if alpha < _MIN_VISIBLE:
            return None
        return EffectFrame(draws=((_TINT_Z, self._draw(alpha)),))

    def _alpha(self, t) -> float:
        values = self._channels.values_at(t)
        dark = _clamp01(values.get('dark', 0.0))
        cover = _clamp01(values.get('cover', 0.0))
        return min(_MAX_TINT, max(dark, cover))

    def _draw(self, alpha):
        def draw(ctx, painter):
            painter.save()
            painter.setOpacity(alpha)
            painter.fillRect(design_box(ctx.chart_rect), _BLACK)
            painter.restore()
        return draw


def _clamp01(value) -> float:
    return max(0.0, min(1.0, float(value)))
