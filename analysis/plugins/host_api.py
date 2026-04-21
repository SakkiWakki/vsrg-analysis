"""Narrow host API exposed to sandboxed plugins.

Sandboxed plugins cannot directly access ``player``, ``renderer``, or any
Qt/matplotlib objects. Instead they receive a ``PlayerState`` view through
``SidebarContext.player_state`` (v1) for read-only observation, and can
register UI via ``SidebarContext`` as before.

Future work: expose read-only chart/replay snapshots, a scoped FS API
("load replay by id"), etc., without giving plugins arbitrary path access.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerState:
    """Read-only snapshot of the current player state.

    Fields are deliberately minimal for v1 — add more as concrete plugin
    needs appear. All arrays are provided as tuples so plugins can't
    mutate the live game state.
    """
    t: float                # current playback time (s)
    play_rate: float
    paused: bool
    keycount: int
    note_count: int
    judge_counts: dict[str, int]  # {judgment_name: n}
    windows: tuple[tuple[str, float], ...]  # ((name, width_s), ...)

    @classmethod
    def from_player(cls, p, t_now: float):
        counts = {n: 0 for n, _ in p.windows}
        counts['miss'] = 0
        for j in p.note_judges:
            counts[j] = counts.get(j, 0) + 1
        return cls(
            t=float(t_now),
            play_rate=float(p.play_rate),
            paused=bool(p.paused),
            keycount=int(p.keycount),
            note_count=int(len(p.times)),
            judge_counts=dict(counts),
            windows=tuple((n, float(w)) for n, w in p.windows),
        )
