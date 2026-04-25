"""Linux/Gamescope overlay platform.

Wraps the existing :class:`analysis.overlay.publisher.OverlayPublisher`
; the shm + seqlock + C-side drag handshake all stay exactly as they
were. The only job of this adapter is to translate the neutral
:class:`~analysis.components.pal.base.OverlayFrame` command list into
the publisher's ``_FrameBuilder`` calls inside an atomic frame.

One publisher per overlay key. Handles ``(key → publisher)`` are
remembered inside the platform so ``submit_frame`` doesn't need the
caller to track plumbing.
"""
from __future__ import annotations

from analysis.components.pal.base import (
    CMD_GROUP_BEGIN,
    CMD_GROUP_END,
    CMD_RECT,
    CMD_TEXT,
    OverlayFrame,
    OverlayHandle,
    OverlayPlatformCapabilities,
)


class GamescopeOverlayPlatform:
    """Platform adapter for Linux + gamescope.

    Lazy publisher creation: ``setup`` allocates a publisher and opens
    shm; ``teardown`` closes it. ``submit_frame`` opens one ``frame``
    context per call so each submission is atomic from the C-side
    renderer's perspective.
    """

    def __init__(self, *, config_store=None):
        self._config = config_store
        # key → OverlayPublisher
        self._publishers: dict[str, object] = {}

    def is_available(self) -> bool:
        # detect() already gated on env vars; by the time this runs we
        # trust the session is gamescope. Still, return True so callers
        # have a uniform probe.
        return True

    def capabilities(self) -> OverlayPlatformCapabilities:
        # Today the C renderer handles drag edits but not general
        # button input. Drag is a separate path via the seqlock, not
        # an 'action' name dispatched back to Python.
        return OverlayPlatformCapabilities(
            supports_input=False,
            supports_drag_edit=True,
            native_resolution=(2560, 1440),
        )

    def setup(self, key: str, *, width: int, height: int) -> OverlayHandle:
        from analysis.overlay.publisher import OverlayPublisher
        cfg = self._config
        if cfg is None:
            from analysis.config import get_config
            cfg = get_config()
        pub = OverlayPublisher(
            str(key), width=int(width), height=int(height),
            config_store=cfg)
        pub.start()
        self._publishers[str(key)] = pub
        return OverlayHandle(key=str(key), width=int(width),
                             height=int(height), impl=pub)

    def submit_frame(self, handle: OverlayHandle,
                     frame: OverlayFrame) -> None:
        pub = handle.impl
        if pub is None:
            return
        with pub.frame() as builder:
            self._replay(frame, builder)

    def teardown(self, handle: OverlayHandle) -> None:
        pub = self._publishers.pop(handle.key, None)
        if pub is None:
            pub = handle.impl
        if pub is not None:
            try:
                pub.stop()
            except Exception as exc:
                print(f'[pal.gamescope] teardown stop failed: {exc}')

    # ── Command replay ────────────────────────────────────────────

    @staticmethod
    def _replay(frame: OverlayFrame, builder) -> None:
        """Walk ``frame.records`` and replay into the publisher's
        ``_FrameBuilder``. ``begin_group/end_group`` are translated
        into nested ``with builder.group(...)`` via a manual stack so
        we don't rely on recursion."""
        # Manually manage the group context managers. ``_FrameBuilder.group``
        # is a contextmanager; we can __enter__/__exit__ them directly
        # off a stack of open names, mirroring the record structure.
        open_ctxs: list = []
        try:
            for kind, args in frame.records:
                if kind == CMD_GROUP_BEGIN:
                    (name,) = args
                    cm = builder.group(str(name))
                    cm.__enter__()
                    open_ctxs.append(cm)
                elif kind == CMD_GROUP_END:
                    if open_ctxs:
                        cm = open_ctxs.pop()
                        cm.__exit__(None, None, None)
                elif kind == CMD_RECT:
                    (id_str, x, y, w, h, color, anchor) = args
                    builder.rect(str(id_str), float(x), float(y),
                                 float(w), float(h),
                                 color=int(color), anchor=int(anchor))
                elif kind == CMD_TEXT:
                    (id_str, s, x, y, px_scale, color, anchor) = args
                    builder.text(str(id_str), str(s), float(x), float(y),
                                 px_scale=float(px_scale),
                                 color=int(color), anchor=int(anchor))
                # Unknown kinds are silently dropped ; forward-compat.
        finally:
            # Close any groups the frame forgot to end so we don't
            # leak state across frames.
            while open_ctxs:
                cm = open_ctxs.pop()
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass
