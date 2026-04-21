"""Skin abstraction for the pygame replay player.

Each Skin subclass owns how to render a note head, an LN body, and an LN
tail marker. Player calls into these methods instead of branching on skin
name inline — same pattern as pset6's SkinData, just pygame-native.

`OsuSkin` (loading a native osu! skin from skin.ini + per-column PNGs) is
intentionally not implemented yet; it's the motivation for the abstraction
but it needs its own parser.
"""
from __future__ import annotations
import pygame


class Skin:
    name = ''

    def draw_note_head(self, surf, lx, y, lane_w, note_h, color):
        raise NotImplementedError

    def draw_ln_body(self, surf, lx, y_top, y_bot, lane_w, note_h, color):
        raise NotImplementedError

    def draw_ln_tail(self, surf, lx, y, lane_w, note_h, color):
        raise NotImplementedError

    def draw_ghost_tap(self, surf, lx, y, lane_w, note_h):
        """Indicator for a press that didn't land on any note (osu only).

        Transparent body, white outline, small centered dot — sized to half
        the column width so it reads as 'a tap happened here' without
        overpowering the actual notes. Shared across all skins since it's
        purely a playback overlay, not a note-style choice."""
        r = max(4, int(lane_w * 0.25))   # half the column size = radius ≈ lane_w/4
        cx = int(lx + lane_w / 2)
        cy = int(y)
        pygame.draw.circle(surf, (255, 255, 255), (cx, cy), r, 1)
        pygame.draw.circle(surf, (255, 255, 255), (cx, cy), 2)


class BarSkin(Skin):
    """Current default: flat rectangles with a white outline."""
    name = 'bar'

    def draw_note_head(self, surf, lx, y, lane_w, note_h, color):
        rect = (lx + 4, int(y) - note_h // 2, int(lane_w - 8), note_h)
        pygame.draw.rect(surf, color, rect)
        pygame.draw.rect(surf, (255, 255, 255), rect, 1)

    def draw_ln_body(self, surf, lx, y_top, y_bot, lane_w, note_h, color):
        if y_bot <= y_top:
            return
        pygame.draw.rect(surf, color, (lx + 6, int(y_top),
                                       int(lane_w - 12),
                                       int(y_bot - y_top)))

    def draw_ln_tail(self, surf, lx, y, lane_w, note_h, color):
        pygame.draw.rect(surf, color, (lx + 4, int(y) - note_h // 2,
                                       int(lane_w - 8), note_h))


class CircleSkin(Skin):
    """Circular heads/tails with a thin centered body stripe."""
    name = 'circle'

    @staticmethod
    def _radius(lane_w, note_h):
        # Size to the lane, not to note_h — note_h is a legacy rectangle-
        # height knob that keeps circles tiny on wide lanes. Leave a small
        # gap (~8% of lane) so adjacent-column circles don't touch.
        return max(6, int((lane_w - 4) * 0.46))

    def draw_note_head(self, surf, lx, y, lane_w, note_h, color):
        r = self._radius(lane_w, note_h)
        cx = int(lx + lane_w / 2)
        pygame.draw.circle(surf, color, (cx, int(y)), r)
        pygame.draw.circle(surf, (255, 255, 255), (cx, int(y)), r, 1)

    def draw_ln_body(self, surf, lx, y_top, y_bot, lane_w, note_h, color):
        if y_bot <= y_top:
            return
        body_w = max(6, int(lane_w * 0.32))
        bx = int(lx + (lane_w - body_w) / 2)
        pygame.draw.rect(surf, color, (bx, int(y_top), body_w,
                                       int(y_bot - y_top)))

    def draw_ln_tail(self, surf, lx, y, lane_w, note_h, color):
        r = self._radius(lane_w, note_h)
        cx = int(lx + lane_w / 2)
        pygame.draw.circle(surf, color, (cx, int(y)), r)
        pygame.draw.circle(surf, (255, 255, 255), (cx, int(y)), r, 1)


_REGISTRY: dict[str, Skin] = {
    BarSkin.name: BarSkin(),
    CircleSkin.name: CircleSkin(),
}


def get(name: str) -> Skin:
    return _REGISTRY.get(name, _REGISTRY['bar'])


def names() -> list[str]:
    return list(_REGISTRY.keys())
