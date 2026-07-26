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
from operator import itemgetter

from analysis.games.notitg.mod_channels import (
    MOD_INIT_DEFAULTS, parse_modstring, parse_speed_mods)

# Calls further apart than this many seconds belong to different
# windows. The template reapplies every Update (~0.02s); 2.5 ticks of
# slack tolerates tick jitter without merging real gaps.
_WINDOW_GAP_S = 2.5 / 60.0

# Applications closer together than half a tick are the same frame:
# the later call is the frame's winner for that (mod, player).
_SAME_FRAME_S = 0.5 / 60.0


# Speed-mod pseudo-channels: x/c/m tokens are not percent mods (the
# channel compiler drops them) but the scroll-multiplier consumer reads
# them from window rows, so they coalesce like any mod and render back
# as their own token forms. Excluded from chase channels and clearall.
_SPEED_CHANNELS = {'x': 'xmod', 'c': 'cmod', 'm': 'mmod'}
_SPEED_TOKENS = {'xmod': '{value:g}x', 'cmod': 'c{value:g}',
                 'mmod': 'm{value:g}'}


@dataclass
class ModWindow:
    """One contiguous application run of a single mod: `value` is the
    engine fraction (percent / 100) chased at approach `speed` - or the
    multiplier/rate for the speed pseudo-channels."""
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
        token = _SPEED_TOKENS.get(self.name)
        if token is not None:
            return f'*{self.speed:g} ' + token.format(value=self.value)
        return f'*{self.speed:g} {self.value * 100.0:g} {self.name}'


@lru_cache(maxsize=4096)
def _row_plan(modstring: str) -> tuple:
    """Everything one ApplyModifiers string contributes to a frame, derived
    ONCE: whether it clears every channel, its (value, speed, name) mods, its
    speed-token mods, and the mod names to union into `seen`. A per-frame
    driver re-applies the SAME string every tick, so the cache collapses the
    whole per-row parse - including the `clearall` scan, which used to run a
    fresh `.lower()` over every one of the 350K rows - into one lookup."""
    mods = tuple(parse_modstring(modstring))
    return ('clearall' in modstring.lower(), mods,
            tuple(parse_speed_mods(modstring)),
            frozenset(name for _value, _speed, name in mods))


# The template's per-frame `clearall` retargets every mod to its ENGINE
# DEFAULT at this approach speed before the live windows reapply (same
# constant the harvest decode used); within one frame the reapply wins
# per-channel. The default is 0 for all but the scale family, where it is
# 100% - clearing `zoomx` to 0 collapses the field to a point and holds it
# there for as long as the chart leaves the mod alone.
_CLEARALL_SPEED = 1.0
_clearall_target = MOD_INIT_DEFAULTS.get


# The resolved-row sort key (row[2] is `t`). A C-level itemgetter, not a
# lambda: the sort runs over 2.6M rows on a heavy chart.
_row_t = itemgetter(2)


def _player_indexes(raw) -> tuple:
    """A row's raw player (1/2 from the chart, None = both) as engine
    channel indexes. Expansion happens at INGESTION so a per-player
    clearall and a both-players window meet on the same key inside one
    frame (last call wins there, as in the engine)."""
    return (0, 1) if raw is None else (max(0, int(raw) - 1),)


def _frame_resolved(applied) -> list:
    """(t, beat, modstring, player) rows -> per-frame effective targets
    [(name, player_index, t, beat, value, speed)], last call per
    (mod, player) within a frame winning, with `clearall` expanded to a
    0-target for every channel that player has ever applied.

    This is the single hottest pass of the mod-channel compile (350K+ rows x
    ~9 channels each on a heavy chart), so the flush is INLINED at its three
    sites (a closure call per key was ~3M of them) and the hot names are
    locals. Flushing does not `del pending[key]`: every site overwrites the
    key on the next statement, so the delete only ever undid itself."""
    pending: dict = {}
    seen: dict = {}
    out: list = []
    pending_get = pending.get
    out_append = out.append
    same_frame = _SAME_FRAME_S

    for t, beat, modstring, player in applied:
        indexes = _player_indexes(player)
        clears, mods, speed_mods, names = _row_plan(modstring)

        if clears:
            # The classic reader's row is `clearall, <live windows>`: the
            # clearall retargets every seen mod to 0, then the SAME row's
            # trailing tokens re-apply theirs (parsed below, overwriting
            # the 0 per channel exactly as the engine's in-frame order).
            cleared = (t, beat, 0.0, _CLEARALL_SPEED)
            for index in indexes:
                for name in seen.get(index, ()):
                    key = (name, index)
                    held = pending_get(key)
                    if held is not None and t - held[0] >= same_frame:
                        out_append((name, index, *held))
                    default = _clearall_target(name)
                    pending[key] = cleared if default is None else \
                        (t, beat, default, _CLEARALL_SPEED)

        for value, speed, name in mods:
            for index in indexes:
                key = (name, index)
                held = pending_get(key)
                if held is not None and t - held[0] >= same_frame:
                    out_append((name, index, *held))
                pending[key] = (t, beat, value, speed)

        for speed, kind, value in speed_mods:
            name = _SPEED_CHANNELS[kind]
            for index in indexes:
                key = (name, index)
                held = pending_get(key)
                if held is not None and t - held[0] >= same_frame:
                    out_append((name, index, *held))
                pending[key] = (t, beat, value, speed)

        # `seen` grows by the WHOLE row at once (the clearall above already
        # read it, so a per-name update inside the loop above bought
        # nothing but 650K set operations).
        for index in indexes:
            known = seen.get(index)
            if known is None:
                seen[index] = set(names)
            else:
                known |= names

    for (name, index), held in pending.items():
        out_append((name, index, *held))
    out.sort(key=_row_t)
    return out


def coalesce_applied(applied) -> list:
    """Frame-resolved targets -> per-mod ModWindows (contiguous runs of
    one value, split on value/speed change or call gap)."""
    open_windows: dict = {}
    out: list = []
    for name, player, t, beat, value, speed in _frame_resolved(applied):
        key = (name, player)
        window = open_windows.get(key)
        if window is not None and window.value == value \
                and window.speed == speed \
                and t - window.t_end <= _WINDOW_GAP_S:
            window.t_end = t
            window.beat_end = beat
            window.calls += 1
            continue
        if window is not None:
            out.append(window)
        open_windows[key] = ModWindow(name, value, speed, player,
                                      t, t, beat, beat)
    out.extend(open_windows.values())
    out.sort(key=lambda w: (w.t_start, w.name))
    return out


def chase_events(applied) -> list:
    """Frame-resolved targets -> ModEvent retargets for
    `ModChannels.compile`, which runs the exact fapproach chase
    (RageUtil.cpp:51: value moves linearly toward the target at
    speed/sec, snapping on arrival - PlayerOptions::Approach applies it
    per mod with dt * approach speed). One event per (mod, player)
    target-or-speed CHANGE; a raw player of 1/2 maps to channel index
    0/1, None applies to both."""
    from analysis.player.render.mods.channels import ModEvent

    last: dict = {}
    events: list = []
    for name, index, t, _beat, value, speed in _frame_resolved(applied):
        if name in _SPEED_TOKENS:
            continue
        key = (name, index)
        if last.get(key) == (value, speed):
            continue
        last[key] = (value, speed)
        events.append(ModEvent(t, value, speed, name, index))
    return events


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
