"""Per-replay sprite cache: one QPixmap per distinct note-part render.

Notes in VSRG are drawn thousands of times per second. With vector
primitives (`drawRect`, `drawEllipse`) every frame re-runs stroke + fill
geometry for each visible note. Caching the rasterized result into a
QPixmap and calling `drawPixmap` once per note cuts that to a single
blit per sprite.

Design constraints (from the refactor brief):

1. **One buffer per note-part.** Tap head, LN head, LN tail, LN body,
   mine, lift, fake — each owns an independent buffer namespace so a
   game can skin them independently (e.g. LN heads different from taps).
2. **Per-replay allocation.** Only the sprites the active game's
   adapter declares get pre-allocated. An LN-only game never rasterizes
   a tap-head pixmap. Allocation happens at replay load; the cache dies
   when the Player goes out of scope.
3. **Resize-invalidating.** Pixmaps are keyed by the geometry they
   were rasterized at (`note_h`, `lane_w`). The renderer calls
   `check_geometry(W, H)` each frame; a change clears every buffer and
   rasterization resumes on next `get()`.

LN bodies get special treatment: rasterized as a 1-row pixmap per
`(col, state)` and tiled vertically via `painter.drawTiledPixmap`, so
one allocation covers every height the note can hit during play.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap


class SpriteSpec(NamedTuple):
    """One entry in the adapter's sprite declaration.

    - `size(ctx) -> (w, h)`
      Pixmap dimensions this sprite rasterizes at, given the current
      render context. Evaluated lazily per cache miss so a window-
      resize just invalidates the cache and the next `get` picks up
      the new dimensions.
    - `rasterize(painter, key_dict, ctx) -> None`
      Paints the sprite into `painter` (already bound to a fresh
      pixmap translated to origin). `key_dict` carries the concrete
      values for the `key_fields`.
    - `key_fields`
      Tuple of kwarg names `get()` expects. Used to build the cache
      key tuple in a stable order.
    - `tiled`
      If True the sprite is a 1-row tile meant for `drawTiledPixmap`
      (used for variable-height LN bodies). Height is forced to 1.
    """
    size: Callable
    rasterize: Callable
    key_fields: tuple
    tiled: bool = False


@dataclass
class NoteSpriteCache:
    """Per-Player sprite cache. Bound once after the adapter declares
    `note_sprites(replay)`; pixmaps allocate lazily on first `get`.

    Attributes are plain state so tests can poke at `_buffers` without
    trying to reach through property indirection. The cache is meant
    to be treated as an opaque object from the drawer's perspective.
    """
    _specs: dict[str, SpriteSpec] = field(default_factory=dict)
    _buffers: dict[str, dict[tuple, QPixmap]] = field(default_factory=dict)
    _geom: tuple = (0, 0)

    def bind(self, specs: dict[str, SpriteSpec]) -> None:
        """Replace the declared sprite set. Called once per replay load.
        Drops every cached pixmap so the new replay starts from empty
        buckets even when some names overlap with a prior replay."""
        self._specs = dict(specs)
        self._buffers = {name: {} for name in specs}
        self._geom = (0, 0)

    def check_geometry(self, w: int, h: int) -> None:
        """Invalidate every pixmap when the play window resizes. The
        renderer calls this once per frame in `build_context`; a
        `(w, h)` match is a no-op."""
        geom = (int(w), int(h))
        if geom == self._geom:
            return
        for bucket in self._buffers.values():
            bucket.clear()
        self._geom = geom

    def get(self, name: str, ctx, **key_kwargs) -> QPixmap:
        """Return a rasterized pixmap for `name` at the given key.
        Misses rasterize on the spot via the spec's callback; hits
        return the cached pixmap directly.

        Raises `KeyError` if `name` wasn't declared at bind time -
        adapters need to declare every sprite their drawers reach for.
        """
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f'sprite {name!r} not declared by this game '
                           f'adapter; known: {sorted(self._specs)}')
        key = tuple(key_kwargs[f] for f in spec.key_fields)
        bucket = self._buffers[name]
        cached = bucket.get(key)
        if cached is not None:
            return cached
        bucket[key] = self._rasterize(spec, key_kwargs, ctx)
        return bucket[key]

    @staticmethod
    def _rasterize(spec, key_kwargs, ctx) -> QPixmap:
        w, h = spec.size(ctx)
        if spec.tiled:
            h = 1
        pm = QPixmap(int(max(1, w)), int(max(1, h)))
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            spec.rasterize(painter, key_kwargs, ctx)
        finally:
            painter.end()
        return pm
