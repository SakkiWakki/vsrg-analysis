"""In-game osu! HUD published through the generic overlay registry."""
from __future__ import annotations

import numpy as np

from analysis.overlay.api import (ANCHOR_TL, BLACK_DIM, BLUE_ACCENT, HIST_BAR,
                                  WHITE, OverlayGameState, rgba)
from plugins.unsafe.osu_live.client import get_client
from plugins.unsafe.osu_live.state import snapshot_to_overlay_state


_HIST_BINS = 41            # +/-100 ms, 5 ms per bin
PUBLISH_HZ = 30.0
OVERLAY_KEY = 'osu_live'


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


def _unstable_rate_ms(state: OverlayGameState) -> float:
    if state.unstable_rate and state.unstable_rate > 0:
        return float(state.unstable_rate)
    if len(state.hit_offsets_s) < 2:
        return 0.0
    ms = np.asarray(state.hit_offsets_s, dtype=np.float64) * 1000.0
    return float(10.0 * np.std(ms))


# Layout constants in normalized canvas units.
_PANEL_X = 0.012
_PANEL_Y = 0.022
_PANEL_W = 0.27
_PANEL_H = 0.21

_HIST_X = _PANEL_X + 0.010
_HIST_Y = _PANEL_Y + _PANEL_H - 0.058
_HIST_W = _PANEL_W - 0.020
_HIST_H = 0.048


def draw(frame) -> None:
    """Draw one overlay frame.

    The registry owns ``OverlayPublisher.frame()`` and commits after this
    returns. Emitting no widgets clears the overlay on the next frame.
    """
    state = snapshot_to_overlay_state(get_client().snapshot())

    if not state.is_playing:
        return

    draw_state(frame, state)


def draw_state(frame, state: OverlayGameState) -> None:
    """Render the reusable HUD from game-agnostic live state."""
    hist = _offsets_to_histogram(np.asarray(state.hit_offsets_s))
    ur = _unstable_rate_ms(state)
    peak = max(1, max(hist))

    group_name = 'osu_live.panel' if state.game == 'osu' \
        else f'{state.game}.live.panel'
    with frame.group(group_name):
        frame.rect('panel_bg', _PANEL_X, _PANEL_Y, _PANEL_W, _PANEL_H,
                   color=BLACK_DIM, anchor=ANCHOR_TL)
        frame.rect('panel_accent', _PANEL_X, _PANEL_Y, _PANEL_W, 0.003,
                   color=BLUE_ACCENT, anchor=ANCHOR_TL)

        frame.text('combo', f'{state.combo}X',
                   _PANEL_X + 0.010, _PANEL_Y + 0.018,
                   px_scale=2.5, color=WHITE, anchor=ANCHOR_TL)

        line2 = f'{state.accuracy:.2f}%  UR {ur:.1f}'
        frame.text('acc_ur', line2,
                   _PANEL_X + 0.010, _PANEL_Y + 0.065,
                   px_scale=2.0, color=WHITE, anchor=ANCHOR_TL)

        hits_line = (f'{state.judgment("300")}:{state.judgment("100")}:'
                     f'{state.judgment("50")}:{state.judgment("miss")}')
        frame.text('hit_counts', hits_line,
                   _PANEL_X + 0.010, _PANEL_Y + 0.105,
                   px_scale=1.6, color=WHITE, anchor=ANCHOR_TL)

        frame.rect('hist_base', _HIST_X, _HIST_Y + _HIST_H - 0.001,
                   _HIST_W, 0.001,
                   color=rgba(64, 64, 76, 230), anchor=ANCHOR_TL)

        bin_w = _HIST_W / _HIST_BINS
        for i, count in enumerate(hist):
            h_norm = (count / peak) * _HIST_H
            if h_norm <= 0:
                continue
            bar_x = _HIST_X + i * bin_w
            bar_y = _HIST_Y + (_HIST_H - h_norm)
            frame.rect(f'hist_bar_{i}', bar_x, bar_y,
                       max(0.0005, bin_w - 0.0008), h_norm,
                       color=HIST_BAR, anchor=ANCHOR_TL)

        frame.rect('hist_zero', _HIST_X + _HIST_W * 0.5 - 0.0004,
                   _HIST_Y, 0.0008, _HIST_H,
                   color=rgba(255, 255, 255, 100), anchor=ANCHOR_TL)


_PROVIDER_DIAG = {'last_present': None, 'last_phase': None}


def _live_state_provider():
    """Provider for unified overlay components (see analysis.components.provider).

    Unified components ; e.g. the ported judgments plugin ; read live
    game state through ``current_game_state()`` on the overlay render
    thread. They're surface-agnostic, so they can't reach into osu!'s
    client directly. We install this shim once at registration time.
    """
    from analysis import diag
    try:
        snap = get_client().snapshot()
    except Exception as exc:
        if _PROVIDER_DIAG['last_present'] is not False:
            _PROVIDER_DIAG['last_present'] = False
            diag.log('osu_live.provider', f'client snapshot raised: {exc}')
        return None
    state = snapshot_to_overlay_state(snap)
    phase = getattr(state, 'phase', '?')
    if not state.is_playing:
        if _PROVIDER_DIAG['last_present'] is not False \
                or _PROVIDER_DIAG['last_phase'] != phase:
            _PROVIDER_DIAG['last_present'] = False
            _PROVIDER_DIAG['last_phase'] = phase
            diag.log('osu_live.provider',
                     f'not playing ; phase={phase} '
                     f'connected={snap.connected}')
        return None
    if _PROVIDER_DIAG['last_present'] is not True \
            or _PROVIDER_DIAG['last_phase'] != phase:
        _PROVIDER_DIAG['last_present'] = True
        _PROVIDER_DIAG['last_phase'] = phase
        diag.log('osu_live.provider',
                 f'state available ; phase={phase} '
                 f'keycount={state.keycount} '
                 f'judgments={state.judgments}')
    return state


def register_overlay(add):
    from analysis import diag
    from analysis.components.provider import set_game_state_provider
    set_game_state_provider(_live_state_provider)
    diag.log('osu_live.live_hud', 'register_overlay: provider installed')
    add('osu! live HUD', draw, key=OVERLAY_KEY, hz=PUBLISH_HZ)
