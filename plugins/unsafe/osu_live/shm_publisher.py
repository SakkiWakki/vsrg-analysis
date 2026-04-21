"""Compatibility wrapper for the osu! live overlay publisher.

New overlay plugins should live under ``overlay/`` and expose
``register_overlay(add)``. This module remains for older call sites that
imported ``get_publisher`` directly.
"""
from __future__ import annotations

import threading

from analysis.overlay.publisher import OverlayPublisher
from plugins.unsafe.osu_live.overlay.live_hud import (OVERLAY_KEY, PUBLISH_HZ,
                                                      draw)


_publisher: OverlayPublisher | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def get_publisher(config_store=None,
                  width: int = 2560,
                  height: int = 1440) -> OverlayPublisher:
    global _publisher, _thread
    with _lock:
        if _publisher is None:
            _publisher = OverlayPublisher(
                OVERLAY_KEY, width=width, height=height,
                config_store=config_store)
            _publisher.start()
            _thread = _publisher.run_draw_thread(
                draw, hz=PUBLISH_HZ, name='OsuLiveOverlayPublisher')
        return _publisher
