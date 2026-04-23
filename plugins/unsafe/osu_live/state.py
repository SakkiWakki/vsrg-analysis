"""Adapter from native osu! game memory to the generic overlay state model."""
from __future__ import annotations

from analysis.components.api import GameMemoryState
from analysis.overlay.api import (PHASE_DISCONNECTED, PHASE_IDLE,
                                  PHASE_PLAYING, OverlayGameState)


def snapshot_to_overlay_state(snap: GameMemoryState | None) -> OverlayGameState:
    """Translate a native memory snapshot into game-agnostic overlay state.
    Returns a disconnected state when snap is None."""
    if snap is None:
        return OverlayGameState(game='osu', phase=PHASE_DISCONNECTED,
                                keycount=0)

    phase = PHASE_PLAYING if snap.in_gameplay else PHASE_IDLE
    offsets = tuple(e / 1000.0 for e in snap.hit_errors_ms)
    judgments = (
        ('300', snap.hit_300),
        ('geki', snap.hit_geki),
        ('200', snap.hit_katu),
        ('100', snap.hit_100),
        ('50', snap.hit_50),
        ('miss', snap.hit_miss),
    )

    return OverlayGameState(
        game='osu',
        phase=phase,
        song_id=snap.map_md5,
        song_title=snap.map_title,
        keycount=4,
        combo=snap.combo,
        max_combo=snap.max_combo,
        accuracy=snap.accuracy,
        judgments=judgments,
        hit_offsets_s=offsets,
        hit_lanes=(),
        pressed_lanes=(),
    )
