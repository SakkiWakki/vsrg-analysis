"""Recorded-stream shaping: the sim's flat call streams -> windows.

The chart applies mods by calling `ApplyModifiers`/`ApplyGameCommand`
every frame while a window is live (the classic template's clearall +
reapply reader). The sim records each call as one (t, beat, modstring,
player) row; `coalesce_applied` folds those into contiguous windows -
one per (mod name, player) run of adjacent-in-time calls - splitting
when the modstring's value changes or the calls stop. This replaces the
harvest path's mods-table normalization AND its clearall decoding: the
window edges here are where the chart actually started/stopped calling,
which IS the engine truth.
"""
from __future__ import annotations

from dataclasses import dataclass

# Calls further apart than this many seconds belong to different
# windows. The template reapplies every Update (~0.02s); 2.5 ticks of
# slack tolerates tick jitter without merging real gaps.
_WINDOW_GAP_S = 2.5 / 60.0


@dataclass
class ModWindow:
    """One contiguous application run of a mod."""
    modstring: str
    player: int | None
    t_start: float
    t_end: float
    beat_start: float
    beat_end: float
    calls: int = 1

    def merge_call(self, t, beat) -> None:
        self.t_end = t
        self.beat_end = beat
        self.calls += 1


def mod_name(modstring: str) -> str:
    """The mod a modstring targets: its last whitespace token, lowered
    ('*5 40 drunk' -> 'drunk', 'no dark' -> 'dark')."""
    tokens = modstring.strip().lower().split()
    while tokens and tokens[-1] == 'no':
        tokens.pop()
    return tokens[-1] if tokens else ''


def coalesce_applied(applied) -> list:
    """(t, beat, modstring, player) rows -> ModWindows, grouped by
    (mod name, player), split on value change or time gap. Rows arrive
    in call order (sim time is monotonic)."""
    open_windows: dict = {}
    out: list = []
    for t, beat, modstring, player in applied:
        key = (mod_name(modstring), player)
        window = open_windows.get(key)
        if window is not None and window.modstring == modstring \
                and t - window.t_end <= _WINDOW_GAP_S:
            window.merge_call(t, beat)
            continue
        if window is not None:
            out.append(window)
        open_windows[key] = ModWindow(modstring, player, t, t, beat, beat)
    out.extend(open_windows.values())
    out.sort(key=lambda w: (w.t_start, w.modstring))
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
