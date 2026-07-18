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

from analysis.games.notitg.mod_channels import (
    parse_modstring, parse_speed_mods)

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
def _parsed(modstring: str) -> tuple:
    return tuple(parse_modstring(modstring))


@lru_cache(maxsize=4096)
def _parsed_speed(modstring: str) -> tuple:
    return tuple(parse_speed_mods(modstring))


# The template's per-frame `clearall` retargets every mod to 0 at this
# approach speed before the live windows reapply (same constant the
# harvest decode used); within one frame the reapply wins per-channel.
_CLEARALL_SPEED = 1.0


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
    0-target for every channel that player has ever applied."""
    pending: dict = {}
    seen: dict = {}
    out: list = []

    def flush(key, next_t) -> None:
        held = pending.get(key)
        if held is not None and next_t - held[0] >= _SAME_FRAME_S:
            name, index = key
            out.append((name, index, *held))
            del pending[key]

    for t, beat, modstring, player in applied:
        indexes = _player_indexes(player)
        if 'clearall' in modstring.lower():
            for index in indexes:
                for name in seen.get(index, ()):
                    key = (name, index)
                    flush(key, t)
                    pending[key] = (t, beat, 0.0, _CLEARALL_SPEED)
            continue
        for value, speed, name in _parsed(modstring):
            for index in indexes:
                seen.setdefault(index, set()).add(name)
                key = (name, index)
                flush(key, t)
                pending[key] = (t, beat, value, speed)
        for speed, kind, value in _parsed_speed(modstring):
            for index in indexes:
                key = (_SPEED_CHANNELS[kind], index)
                flush(key, t)
                pending[key] = (t, beat, value, speed)
    for (name, index), held in pending.items():
        out.append((name, index, *held))
    out.sort(key=lambda row: row[2])
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


# A per-frame driver re-fires its mod densely, its per-frame value
# tracking a function of the song beat: the fire SEQUENCE is the intended
# continuous value (whether each fire snaps or eases is irrelevant once
# we sample the fires directly - the approach speed only mattered to the
# chase we are bypassing). A curve needs enough closely-spaced fires that
# resampling it beats the chase; a handful of sparse windows are a step
# sequence the chase already renders exactly.
_CURVE_MIN_FIRES = 16
# The stream must be DENSE: at least this fraction of consecutive fires
# sit within `_CURVE_DENSE_GAP` beats. A mod with a few dense runs buried
# in sparse declarative windows is not a per-frame curve.
_CURVE_DENSE_GAP = 0.5
_CURVE_DENSE_FRAC = 0.75
# Two fires within this beat epsilon are the "same" beat: if their values
# disagree the stream depends on frame history (an accumulator), not beat
# alone, so it cannot be a beat-keyed curve.
_CURVE_SAME_BEAT = 1e-4


def _fires_by_channel(applied) -> dict:
    """{(name, index): [(beat, value), ...]} of the per-frame resolved
    fires, in time order, excluding the speed pseudo-channels. The
    approach speed is dropped: a beat curve samples the fire values
    directly, it does not chase them."""
    out: dict = {}
    for name, index, _t, beat, value, _speed in _frame_resolved(applied):
        if name in _SPEED_TOKENS:
            continue
        out.setdefault((name, index), []).append((beat, value))
    return out


def _pure_in_beat(fires) -> bool:
    """Whether the fires are a well-defined function of beat: no two
    same-beat fires carry different values. An accumulator (value depends
    on frame history, not beat alone) fails this and keeps the chase."""
    return all(not (b1 - b0 <= _CURVE_SAME_BEAT and v0 != v1)
               for (b0, v0), (b1, v1) in zip(fires, fires[1:]))


def _is_perframe_curve(fires) -> bool:
    """Whether a channel's fires are a per-frame driver painting a
    beat-keyed curve: many fires, densely spaced for most of the run,
    with changing values, and pure in beat."""
    if len(fires) < _CURVE_MIN_FIRES:
        return False
    beats = [b for b, _v in fires]
    dense = sum(1 for b0, b1 in zip(beats, beats[1:])
                if b1 - b0 <= _CURVE_DENSE_GAP)
    if dense < _CURVE_DENSE_FRAC * (len(beats) - 1):
        return False
    values = [v for _b, v in fires]
    if max(values) == min(values):
        return False
    return _pure_in_beat(fires)


def perframe_curves(applied) -> dict:
    """{(name, index): (beats, values)} for the channels whose fires are
    a per-frame beat-keyed curve (see `_is_perframe_curve`). These sample
    at beat_at(t) live instead of compiling through the time-keyed
    fapproach chase, which distorts a dense fire sequence into a jagged
    staircase. Consecutive duplicate-beat fires collapse to the last
    value so the beat axis is strictly increasing for interpolation."""
    curves: dict = {}
    for key, fires in _fires_by_channel(applied).items():
        if not _is_perframe_curve(fires):
            continue
        beats: list = []
        values: list = []
        for beat, value in fires:
            if beats and beat - beats[-1] <= _CURVE_SAME_BEAT:
                values[-1] = value
                continue
            beats.append(beat)
            values.append(value)
        curves[key] = (beats, values)
    return curves


def chase_events(applied, skip=None) -> list:
    """Frame-resolved targets -> ModEvent retargets for
    `ModChannels.compile`, which runs the exact fapproach chase
    (RageUtil.cpp:51: value moves linearly toward the target at
    speed/sec, snapping on arrival - PlayerOptions::Approach applies it
    per mod with dt * approach speed). One event per (mod, player)
    target-or-speed CHANGE; a raw player of 1/2 maps to channel index
    0/1, None applies to both. `skip` is the set of (name, index) keys
    handled as beat curves (`perframe_curves`), which must not ALSO
    chase - each mod lives in exactly one representation."""
    from analysis.player.render.mods.channels import ModEvent

    skip = skip or set()
    last: dict = {}
    events: list = []
    for name, index, t, _beat, value, speed in _frame_resolved(applied):
        if name in _SPEED_TOKENS or (name, index) in skip:
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
