"""Port of the osu! live HUD to the generic overlay plugin API.

Everything osu-specific lives here: we read ``LiveSnapshot`` from the
shared :class:`OsuLiveClient`, shape it as a widget list, and hand
that to :class:`plugins.overlay_api.OverlayPublisher`. The overlay
C binary knows nothing about osu! — it just draws the widgets.

This file used to own a fixed POD struct and a bespoke seqlock; that
logic now lives in ``plugins/overlay_api.py`` so any plugin can
publish to the overlay the same way.
"""
from __future__ import annotations

import threading

import numpy as np

from plugins.overlay_api import (ANCHOR_TL, BLACK_DIM, BLUE_ACCENT, HIST_BAR,
                                 WARN_AMBER, WHITE, OverlayPublisher, rgba)
from plugins.unsafe.osu_live.client import LiveSnapshot, get_client


_PLUGIN_KEY = 'osu_live'
_HIST_BINS  = 41            # ±100 ms, 5 ms per bin
_PUBLISH_HZ = 30.0


# ── Offset helpers (same logic the old native publisher used) ──────

def _offsets_to_histogram(offsets: np.ndarray) -> list[int]:
    bins = [0] * _HIST_BINS
    if offsets is None or len(offsets) == 0:
        return bins
    ms = np.asarray(offsets, dtype=np.float64) * 1000.0
    ms = np.clip(ms, -100.0, 100.0)
    idx = ((ms + 100.0) / 5.0).astype(np.int32)
    idx = np.clip(idx, 0, _HIST_BINS - 1)
    counts = np.bincount(idx, minlength=_HIST_BINS)
    return [int(counts[i]) for i in range(_HIST_BINS)]


def _unstable_rate_ms(snap: LiveSnapshot) -> float:
    if snap.unstable_rate and snap.unstable_rate > 0:
        return float(snap.unstable_rate)
    if len(snap.offsets) < 2:
        return 0.0
    ms = np.asarray(snap.offsets, dtype=np.float64) * 1000.0
    return float(10.0 * np.std(ms))


# ── HUD builder: LiveSnapshot → widgets ────────────────────────────

# Layout constants expressed in normalized canvas units (0..1).
# These are the defaults; the user can shift+tab + drag to override,
# and the override is persisted in the app's ConfigStore.
#
# Panel at top-left: combo, accuracy + UR, hit counts, histogram.
_PANEL_X  = 0.012
_PANEL_Y  = 0.022
_PANEL_W  = 0.27
_PANEL_H  = 0.21

_HIST_X   = _PANEL_X + 0.010
_HIST_Y   = _PANEL_Y + _PANEL_H - 0.058
_HIST_W   = _PANEL_W - 0.020
_HIST_H   = 0.048


def _build_hud(pub: OverlayPublisher) -> None:
    snap: LiveSnapshot = get_client().snapshot()

    # Outside of gameplay the HUD is noise. Emit zero widgets so
    # the overlay renders a clean, transparent frame (and the
    # content-hash skips after the first post so we don't spam
    # commits).
    if not snap.connected or not snap.in_gameplay:
        with pub.frame() as f:
            pass
        return

    hist = _offsets_to_histogram(snap.offsets)
    ur   = _unstable_rate_ms(snap)
    peak = max(1, max(hist))

    with pub.frame() as f:
        # Everything inside this block shares one group_id so dragging
        # any part of the HUD moves the whole thing together. The
        # persisted delta is shared across histogram bars too, which
        # means new bars appearing mid-play inherit the user's layout.
        with f.group('osu_live.panel'):
            f.rect('panel_bg', _PANEL_X, _PANEL_Y, _PANEL_W, _PANEL_H,
                   color=BLACK_DIM, anchor=ANCHOR_TL)
            f.rect('panel_accent', _PANEL_X, _PANEL_Y, _PANEL_W, 0.003,
                   color=BLUE_ACCENT, anchor=ANCHOR_TL)

            # Combo, big.
            f.text('combo', f'{snap.combo}X',
                   _PANEL_X + 0.010, _PANEL_Y + 0.018,
                   px_scale=2.5, color=WHITE, anchor=ANCHOR_TL)

            # Accuracy + UR line.
            line2 = f'{snap.accuracy:.2f}%  UR {ur:.1f}'
            f.text('acc_ur', line2,
                   _PANEL_X + 0.010, _PANEL_Y + 0.065,
                   px_scale=2.0, color=WHITE, anchor=ANCHOR_TL)

            # Hit counts: 300:100:50:miss.
            hits_line = (f'{snap.hits_300}:{snap.hits_100}:'
                         f'{snap.hits_50}:{snap.hits_miss}')
            f.text('hit_counts', hits_line,
                   _PANEL_X + 0.010, _PANEL_Y + 0.105,
                   px_scale=1.6, color=WHITE, anchor=ANCHOR_TL)

            # Histogram baseline.
            f.rect('hist_base', _HIST_X, _HIST_Y + _HIST_H - 0.001,
                   _HIST_W, 0.001,
                   color=rgba(64, 64, 76, 230), anchor=ANCHOR_TL)

            bin_w = _HIST_W / _HIST_BINS
            for i, count in enumerate(hist):
                h_norm = (count / peak) * _HIST_H
                if h_norm <= 0:
                    continue
                bar_x = _HIST_X + i * bin_w
                bar_y = _HIST_Y + (_HIST_H - h_norm)
                f.rect(f'hist_bar_{i}', bar_x, bar_y,
                       max(0.0005, bin_w - 0.0008), h_norm,
                       color=HIST_BAR, anchor=ANCHOR_TL)

            # Zero marker.
            f.rect('hist_zero', _HIST_X + _HIST_W * 0.5 - 0.0004,
                   _HIST_Y, 0.0008, _HIST_H,
                   color=rgba(255, 255, 255, 100), anchor=ANCHOR_TL)


# ── Singleton ──────────────────────────────────────────────────────

_publisher: OverlayPublisher | None = None
_lock = threading.Lock()
_thread: threading.Thread | None = None


def get_publisher(config_store=None,
                  width: int = 2560,
                  height: int = 1440) -> OverlayPublisher:
    """Return (and lazily start) the osu_live overlay publisher.

    ``config_store`` persists drag positions per (width × height)
    bucket; pass the app-wide ``ConfigStore`` if you want
    user-positioned layouts to survive restarts. Omitting it still
    lets runtime drags work — they just don't persist.
    """
    global _publisher, _thread
    with _lock:
        if _publisher is None:
            _publisher = OverlayPublisher(
                _PLUGIN_KEY, width=width, height=height,
                config_store=config_store)
            _publisher.start()
            _thread = _publisher.run_thread(
                _build_hud, hz=_PUBLISH_HZ, name='OsuLiveOverlayPublisher')
        return _publisher
