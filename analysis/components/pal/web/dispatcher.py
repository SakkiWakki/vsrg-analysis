"""Backend dispatcher + selection policy.

``WebTexturePAL`` is a tiny registry-plus-picker. It holds a list of
:class:`WebTextureBackend` instances; callers ask for a WebTexture
against a target surface (local CPU, local GL, cross-process GL/VK) and
the PAL picks the best available backend.

Selection policy, in order:
    1. Drop backends whose capabilities don't include the caller's
       surface.
    2. Drop backends whose ``is_available()`` returns False (missing
       dependencies, wrong thread, etc.).
    3. Prefer ``zero_copy=True`` over false -- readback is the last
       resort.
    4. Break ties by registration order (earlier = higher priority;
       callers register platform-specific backends before the
       universal qpixmap fallback).
"""
from __future__ import annotations

from typing import Callable

from analysis.components.pal.web.base import (
    SURFACE_LOCAL_CPU,
    WebTexture,
    WebTextureBackend,
    WebTextureBackendCaps,
)


class NoBackendError(RuntimeError):
    """Raised when no registered backend can serve the requested surface."""


class WebTexturePAL:
    """Web-texture backend dispatcher.

    Not a singleton -- tests build their own PAL with mock backends;
    production code uses :meth:`default` which lazily populates the
    global registry from ``web.qpixmap_backend`` and (future) GL/Vk
    backends.
    """

    _default: 'WebTexturePAL | None' = None

    def __init__(self) -> None:
        self._backends: list[WebTextureBackend] = []

    # ── Registration ────────────────────────────────────────────────

    def register(self, backend: WebTextureBackend) -> None:
        """Append a backend. Earlier registrations win ties, so register
        high-capability backends (GL, Vulkan) before falling back to
        CPU (``qpixmap``)."""
        self._backends.append(backend)

    def backends(self) -> tuple[WebTextureBackend, ...]:
        return tuple(self._backends)

    # ── Selection ───────────────────────────────────────────────────

    def select(self, surface: str) -> WebTextureBackend:
        """Return the best available backend for ``surface``.

        Raises :class:`NoBackendError` if none qualify. Callers
        typically pass a ``SURFACE_*`` constant; custom strings work
        too as long as the relevant backend declares them in its
        capabilities.
        """
        candidates = []
        for b in self._backends:
            caps: WebTextureBackendCaps = b.capabilities()
            if surface not in caps.surfaces:
                continue
            if not b.is_available():
                continue
            candidates.append((b, caps))

        if not candidates:
            raise NoBackendError(
                f'no web-texture backend available for surface {surface!r} '
                f'(registered: {[b.name for b in self._backends]})')

        # zero-copy first; else registration order.
        candidates.sort(key=lambda bc: (not bc[1].zero_copy,))
        return candidates[0][0]

    def create(self, *, surface: str = SURFACE_LOCAL_CPU,
               width: int, height: int) -> WebTexture:
        """Convenience wrapper: pick backend + construct texture."""
        return self.select(surface).create(width=width, height=height)

    # ── Default (lazy, process-wide) ────────────────────────────────

    @classmethod
    def default(cls, factory: 'Callable[[WebTexturePAL], None] | None' = None
                ) -> 'WebTexturePAL':
        """Return (and lazily populate) the process-global PAL.

        ``factory`` is called once on first access with the freshly
        constructed PAL, so callers can customize the registration
        order without monkeypatching. Passing ``None`` (the default)
        registers the built-in backends in the canonical order.
        """
        if cls._default is not None:
            return cls._default

        pal = cls()
        if factory is None:
            _register_builtin(pal)
        else:
            factory(pal)
        cls._default = pal
        return pal

    @classmethod
    def reset_default_for_tests(cls) -> None:
        cls._default = None


def _register_builtin(pal: WebTexturePAL) -> None:
    """Register the backends that ship in-tree.

    Ordering matters (see class docstring): higher-capability backends
    first so ``select`` picks them when their surface matches, falling
    through to the universal CPU path last.
    """
    # Phase 1 -- CPU-only. Phase 2 will prepend a scene-graph backend;
    # Phase 3 will prepend dmabuf / vk_image cross-process backends.
    from analysis.components.pal.web.qpixmap_backend import QPixmapBackend
    pal.register(QPixmapBackend())
