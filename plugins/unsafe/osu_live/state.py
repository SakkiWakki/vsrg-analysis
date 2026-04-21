"""Adapter from osu live snapshots to the generic overlay state model."""
from __future__ import annotations

from analysis.overlay.api import (PHASE_DISCONNECTED, PHASE_IDLE,
                                  PHASE_PLAYING, OverlayGameState)

from plugins.unsafe.osu_live.client import LiveSnapshot


def snapshot_to_overlay_state(snap: LiveSnapshot) -> OverlayGameState:
    """Translate osu's live snapshot into game-agnostic overlay state."""
    if not snap.connected:
        phase = PHASE_DISCONNECTED
    elif snap.in_gameplay:
        phase = PHASE_PLAYING
    else:
        phase = PHASE_IDLE

    keycount = max(0, int(snap.keycount))
    hit_lanes = tuple(_valid_lanes(snap.columns, keycount))
    offsets = tuple(float(x) for x in snap.offsets)

    judgments = (
        ('300', int(snap.hits_300)),
        ('100', int(snap.hits_100)),
        ('50', int(snap.hits_50)),
        ('miss', int(snap.hits_miss)),
    )

    return OverlayGameState(
        game='osu',
        phase=phase,
        song_id=str(snap.map_title or ''),
        song_title=str(snap.map_title or ''),
        keycount=keycount,
        combo=int(snap.combo),
        max_combo=int(snap.max_combo),
        accuracy=float(snap.accuracy),
        unstable_rate=float(snap.unstable_rate),
        judgments=judgments,
        hit_offsets_s=offsets,
        hit_lanes=hit_lanes,
        pressed_lanes=(),
    )


def _valid_lanes(values, keycount: int) -> tuple[int, ...]:
    if keycount <= 0:
        return ()
    out = []
    for v in values:
        lane = int(v)
        if 0 <= lane < keycount:
            out.append(lane)
    return tuple(out)
