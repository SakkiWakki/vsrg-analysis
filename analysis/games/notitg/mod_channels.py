"""Bridge: harvested modfile windows -> compiled mod channels.

gat-style templates run a per-frame reader: `mod,clearall`, then every
entry whose window contains the current beat/time re-applies its
modstring (later table entries win overlaps). Compiled equivalent:
each window emits target=percent at its start and target=0 at its end
(the clearall revert), same approach speed both ways; overlaps resolve
by emit order at equal times. The channels module then solves the
linear approach into piecewise segments.

`compile_modfile` already resolves every window to seconds
(`t_start`/`t_end`), so `ModEvent.beat` carries seconds here and
`ModChannels.compile` runs on its identity clock.

Modstring grammar (PlayerOptions::FromString subset): comma-separated
tokens, each `[*S] [P% | P | no] name`; `no name` = 0 percent; a bare
name = 100%. `*S` = approach speed in fraction/second, absent = the
engine default 1.0, `*-1` (any S <= 0) = snap. Unknown mod names
still become channels (the pipeline ignores names it has no formula
for); engine view controls and x/C/M speed mods are dropped here.

Players: rows without a player apply to everyone -> player 0; an
explicit pn maps to pn - 1 for the future multi-field split.
"""
from __future__ import annotations

import re
from bisect import bisect_right

from analysis.player.render.mods.channels import ModChannels, ModEvent

_TOKEN = re.compile(
    r'^(?:\*(?P<speed>-?\d+(?:\.\d+)?)\s+)?'
    r'(?:(?P<no>no)\s+|(?P<percent>-?\d+(?:\.\d+)?)%?\s+)?'
    r'(?P<name>[a-z][a-z0-9 ]*?)$')

_ENGINE_CONTROLS = {'clearall', 'overhead', 'incoming', 'space',
                    'hallway', 'distant'}
_SPEED_MOD = re.compile(r'^(?:\d+(?:\.\d+)?x|c\d+|m\d+)$')

# A speed token, optionally approach-prefixed: `*0.45 1.5x`, `2x`, `c400`,
# `m550`. xmod carries a scroll multiplier; C/M-mods pin an absolute
# scroll rate (constant / max-BPM) that our user-scroll model does not
# express, so they are extracted but skipped (documented).
_SPEED_TOKEN = re.compile(
    r'^(?:\*(?P<speed>-?\d+(?:\.\d+)?)\s+)?'
    r'(?P<xmod>\d+(?:\.\d+)?)x$'
    r'|^(?:\*(?P<cspeed>-?\d+(?:\.\d+)?)\s+)?(?P<cmod>[cm]\d+)$')

_DEFAULT_SPEED = 1.0

# gat's persistent baseline scroll (the always-on `{0,9000,'2x,...'}`
# window). Dynamic xmod changes are expressed as multipliers RELATIVE to
# this base, so at rest the field scrolls at the user's chosen speed and
# only the modchart's bursts/slows rescale it.
_DEFAULT_BASE_XMOD = 2.0


def parse_modstring(modstring: str) -> list:
    """[(percent_fraction, speed, name), ...] for one ApplyGameCommand
    payload; engine controls and speed mods are dropped."""
    out = []
    for token in str(modstring).lower().split(','):
        token = ' '.join(token.split())
        if not token:
            continue
        match = _TOKEN.match(token)
        if match is None:
            continue
        name = ' '.join(match['name'].split())
        if name in _ENGINE_CONTROLS or _SPEED_MOD.match(name):
            continue
        if match['no']:
            percent = 0.0
        elif match['percent'] is not None:
            percent = float(match['percent']) / 100.0
        else:
            percent = 1.0
        speed = (float(match['speed']) if match['speed'] is not None
                 else _DEFAULT_SPEED)
        out.append((percent, speed, name.replace(' ', '')))
    return out


def parse_speed_mods(modstring: str) -> list:
    """[(speed, kind, value), ...] speed tokens in one modstring.

    kind is 'x' (xmod: value is the multiplier), or 'c'/'m' (CMod/MMod:
    value is the pinned rate). `speed` is the `*S` approach prefix
    (default 1.0, <= 0 snaps). Non-speed tokens are ignored here."""
    out = []
    for token in str(modstring).lower().split(','):
        token = ' '.join(token.split())
        if not token:
            continue
        match = _SPEED_TOKEN.match(token)
        if match is None:
            continue
        if match['xmod'] is not None:
            speed = (float(match['speed']) if match['speed'] is not None
                     else _DEFAULT_SPEED)
            out.append((speed, 'x', float(match['xmod'])))
        else:
            speed = (float(match['cspeed']) if match['cspeed'] is not None
                     else _DEFAULT_SPEED)
            out.append((speed, match['cmod'][0], float(match['cmod'][1:])))
    return out


def compile_scroll_multipliers(mod_events, base_xmod=_DEFAULT_BASE_XMOD):
    """Windowed xmod changes -> ffx-shaped scroll-multiplier events
    (`{time, duration, multiplier, ease}`, ms-keyed) for
    `GameAdapter.scroll_multipliers`.

    Each xmod window re-targets the field's scroll to `xmod / base_xmod`
    at its start (chasing at the `*S` approach speed) and reverts to base
    (1.0) at its end. The shared approach-chase compiler resolves the
    chain to piecewise-linear breakpoints (the base window is always the
    resting 1.0). Player-0 windows only, matching the note-mod consumer.
    C/M-mods pin an absolute rate our user-scroll model cannot express;
    they are skipped (their count is returned for the caller to log)."""
    events = []
    skipped_cm = 0
    for row in mod_events:
        if _row_player(row) != 0:
            continue
        start = float(row['t_start'])
        end = float(row['t_end'])
        if end < start:
            continue
        for speed, kind, value in parse_speed_mods(row['modstring']):
            if kind != 'x':
                skipped_cm += 1
                continue
            mult = value / base_xmod if base_xmod else 1.0
            events.append(ModEvent(start, mult, speed, 'xmod', 0))
            events.append(ModEvent(end, 1.0, speed, 'xmod', 0))

    breakpoints = _xmod_breakpoints(events)
    return _breakpoints_to_scroll_events(breakpoints), skipped_cm


def _row_player(row) -> int:
    raw = row.get('player')
    return 0 if raw is None else max(0, int(raw) - 1)


def _xmod_breakpoints(events):
    """(times, values) of the resolved xmod-multiplier curve, resting at
    1.0 (the always-on base window).

    Walks the events in time order applying the same approach semantics
    as the mod channels (snap `*-1` vs constant-rate chase) but emits a
    HOLD point at each event's time before its ramp/jump, so a value that
    idles between two distant windows stays flat instead of interpolating
    across the gap. This flat-hold is why xmod cannot reuse
    `_compile_channel` directly (that layout suits densely-retargeted
    per-note mods, not sparse whole-field speed changes)."""
    times: list = []
    values: list = []

    def sample(t):
        return _piecewise_at(times, values, t, rest=1.0)

    for ev in sorted(events, key=lambda e: e.beat):
        t = ev.beat
        current = sample(t)
        if ev.speed <= 0.0:
            # A snap holds `current` up to t, then jumps: two points at
            # the same t (kept distinct) form the vertical step.
            times.append(t)
            values.append(current)
            times.append(t)
            values.append(ev.value)
        else:
            _append_point(times, values, t, current)
            gap = abs(ev.value - current)
            arrival = t if gap == 0.0 else t + gap / ev.speed
            _append_point(times, values, arrival, ev.value)
    return times, values


def _piecewise_at(times, values, t, rest):
    if not times:
        return rest
    idx = bisect_right(times, t) - 1
    if idx < 0:
        return rest
    if idx + 1 >= len(times):
        return values[idx]
    t0, t1 = times[idx], times[idx + 1]
    if t1 <= t0:
        return values[idx + 1]
    f = (t - t0) / (t1 - t0)
    return values[idx] + (values[idx + 1] - values[idx]) * f


def _append_point(times, values, t, v):
    if times and times[-1] == t:
        values[-1] = v
        return
    times.append(t)
    values.append(v)


def _breakpoints_to_scroll_events(breakpoints):
    """Piecewise-linear (times, values) -> ffx scroll-multiplier events.

    A linear segment from breakpoint (t0, v0) to (t1, v1) becomes a
    keyframe at t0 that eases to v1 over [t0, t1] - `EventTimeline` eases
    from the previous keyframe's target (v0) toward this keyframe's value
    across its own duration, so the emitted keyframe carries the segment
    END value and its duration is the segment length. The trailing
    breakpoint (final held value) is an instant keyframe."""
    times, values = breakpoints
    kept = [(t, v) for t, v in zip(times, values) if t >= 0.0]
    out = []
    for i, (t, v) in enumerate(kept):
        if i + 1 < len(kept):
            t_next, v_next = kept[i + 1]
            out.append({'time': t * 1000.0,
                        'duration': (t_next - t) * 1000.0,
                        'multiplier': v_next, 'ease': 0})
        else:
            out.append({'time': t * 1000.0, 'duration': 0.0,
                        'multiplier': v, 'ease': 0})
    return out


def compile_mod_channels(mod_events) -> ModChannels:
    """Compile `compile_modfile`'s normalized mod-window dicts
    (`t_start`/`t_end` seconds, `modstring`, `player`) into sampled
    channels."""
    events = []
    for row in mod_events:
        start = float(row['t_start'])
        end = float(row['t_end'])
        if end < start:
            continue
        raw_player = row.get('player')
        player = 0 if raw_player is None else max(0, int(raw_player) - 1)
        for percent, speed, name in parse_modstring(row['modstring']):
            events.append(ModEvent(start, percent, speed, name, player))
            events.append(ModEvent(end, 0.0, speed, name, player))
    return ModChannels.compile(events)
