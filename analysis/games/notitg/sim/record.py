"""Recorded-stream shaping: the sim's flat call streams -> windows.

The chart applies mods by calling `ApplyModifiers`/`ApplyGameCommand`
every frame while a window is live (the classic template's clearall +
reapply reader), and per-frame drivers inject more calls in the same
frame. The engine's effective target per (mod, player) each frame is
the LAST call of that frame - the table reader's `no movey0` loses to
the walker's ramp applied after it. `coalesce_applied` therefore
explodes each row into per-mod applications, keeps the last application
per (mod, player) within each frame, and folds the survivors into
contiguous windows split on value change or call gap. This replaces the
harvest path's mods-table normalization AND its clearall decoding: the
window edges are where the chart actually started/stopped applying,
which IS the engine truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from analysis.games.notitg.mod_channels import parse_modstring

# Calls further apart than this many seconds belong to different
# windows. The template reapplies every Update (~0.02s); 2.5 ticks of
# slack tolerates tick jitter without merging real gaps.
_WINDOW_GAP_S = 2.5 / 60.0

# Applications closer together than half a tick are the same frame:
# the later call is the frame's winner for that (mod, player).
_SAME_FRAME_S = 0.5 / 60.0


@dataclass
class ModWindow:
    """One contiguous application run of a single mod: `value` is the
    engine fraction (percent / 100) chased at approach `speed`."""
    name: str
    value: float
    speed: float
    player: int | None
    t_start: float
    t_end: float
    beat_start: float
    beat_end: float
    calls: int = 1

    @property
    def modstring(self) -> str:
        """The window as a single-mod ApplyModifiers string."""
        return f'*{self.speed:g} {self.value * 100.0:g} {self.name}'


@lru_cache(maxsize=4096)
def _parsed(modstring: str) -> tuple:
    return tuple(parse_modstring(modstring))


def coalesce_applied(applied) -> list:
    """(t, beat, modstring, player) rows -> per-mod ModWindows.
    Rows arrive in call order (sim time is monotonic), so within one
    frame a later application simply overwrites the pending frame entry
    before it is folded into a window."""
    open_windows: dict = {}
    pending: dict = {}
    out: list = []

    def fold(key, entry) -> None:
        t, beat, value, speed = entry
        window = open_windows.get(key)
        if window is not None and window.value == value \
                and window.speed == speed \
                and t - window.t_end <= _WINDOW_GAP_S:
            window.t_end = t
            window.beat_end = beat
            window.calls += 1
            return
        if window is not None:
            out.append(window)
        name, player = key
        open_windows[key] = ModWindow(name, value, speed, player,
                                      t, t, beat, beat)

    for t, beat, modstring, player in applied:
        for value, speed, name in _parsed(modstring):
            key = (name, player)
            held = pending.get(key)
            if held is not None and t - held[0] >= _SAME_FRAME_S:
                fold(key, held)
                held = None
            pending[key] = (t, beat, value, speed)
    for key, held in pending.items():
        fold(key, held)
    out.extend(open_windows.values())
    out.sort(key=lambda w: (w.t_start, w.name))
    return out


def summarize(result) -> dict:
    """Loop-run statistics for smoke reports and the parity harness."""
    actors = result.actors
    keyframed = {rec_id: a for rec_id, a in actors.items() if a.keyframes()}
    total_keyframes = sum(
        len(kfs) for a in keyframed.values() for kfs in a.keyframes().values())
    windows = coalesce_applied(result.applied_mods)
    return {
        'ticks': result.ticks,
        'faults': result.faults,
        'actors': len(actors),
        'actors_with_keyframes': len(keyframed),
        'recorded_keyframes': total_keyframes,
        'applied_calls': len(result.applied_mods),
        'mod_windows': len(windows),
        'shader_flags': len(result.shader_flags),
        'warnings': len(result.warnings),
    }
