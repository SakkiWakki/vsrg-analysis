"""Bridge: harvested modfile windows -> compiled mod channels.

gat-style templates run a per-frame reader (default.xml ~3999): every
frame it applies `mod,clearall` and then re-applies each window whose
`beat`/time is inside it, later table entries winning overlaps.

`clearall` runs PlayerOptions::Init (verified against ITGmania
PlayerOptions.cpp FromOneModString + Init): it resets every mod's TARGET
value to 0 AND every mod's approach SPEED to 1.0. A modstring token
`*S P name` sets both the target (P) and the approach speed (S) via
SET_FLOAT, so approach speed is per-mod TARGET state carried until the
next token touches that mod. The engine chases the current value toward
the target at the target's speed every frame (PlayerOptions::Approach:
`fapproach(current, target, dt * other.m_SpeedfFoo)`).

So the revert when a window ENDS is NOT at the window's own approach
speed. Nothing re-applies the mod, so clearall's target (0) at clearall's
speed (1.0) stands: the value floats back to 0 at speed 1.0 -- a smooth
~1s ease for a full mod, not the instant snap a `*10000` window's own
speed would give. Compiled equivalent: resolve each (mod, player) into a
piecewise target curve where, at every instant, the active target is the
last-emitted window covering it (0 at clearall speed where none is), then
emit a retarget event at each change; the channels module chases those
into linear segments.

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

import heapq
import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass

from analysis.player.render.mods.channels import ModChannels, ModEvent

_TOKEN = re.compile(
    r'^(?:\*(?P<speed>-?\d+(?:\.\d+)?)\s+)?'
    r'(?:(?P<no>no)\s+|(?P<percent>-?\d+(?:\.\d+)?)%?\s+)?'
    r'(?P<name>[a-z][a-z0-9 ]*?)$')

# Perspective/appearance controls with no consumer yet; hallway left
# this set once arrow_effects gained its pinhole-recede kernel.
_ENGINE_CONTROLS = {'clearall', 'overhead', 'incoming', 'space', 'distant'}
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

# The approach speed a mod reverts at once no window drives it: `clearall`
# runs PlayerOptions::Init, which resets every mod's target to 0 and its
# approach speed to 1.0. So the float-back-to-rest is always at speed 1.0,
# independent of the ended window's own `*S`.
_CLEARALL_SPEED = 1.0

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

    Each xmod window drives the field's scroll to `xmod / base_xmod`
    over its span (chasing at the `*S` approach speed); where no window
    is active the scroll rests at base (1.0), reverting at clearall speed
    (the float, so a `*100000` burst eases back over ~1s, it does not
    snap). Overlapping windows resolve exactly like the per-note mod
    channels through `_resolve_windows`: the highest-order active window
    wins and an end another window still covers never dips - so a
    persistent `{0, 9999, '2.5x'}` baseline (re-applied as per-frame
    bursts by the reader) holds a FLAT rate instead of sawtoothing to
    base between the bursts. Player-0 windows only, matching the note-mod
    consumer. C/M-mods pin an absolute rate our user-scroll model cannot
    express; they are skipped (their count is returned for the caller to
    log)."""
    windows = []
    skipped_cm = 0
    for order, row in enumerate(mod_events):
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
            windows.append(_Window(start, end, mult, speed, order))

    events = [ModEvent(t, value, speed, 'xmod', 0)
              for t, value, speed in _resolve_windows(windows)]
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
    per-note mods, not sparse whole-field speed changes).

    Each chase is clamped to end at the next event's time, carrying the
    value it actually reached: an unclamped ramp can arrive past the next
    event (a persistent `2.5x` window's slow chase overrunning the
    per-tick reapply bursts) and make the breakpoints non-monotonic,
    which corrupts `_piecewise_at`'s bisect - the classic scroll-mult
    sawtooth."""
    times: list = []
    values: list = []

    def sample(t):
        return _piecewise_at(times, values, t, rest=1.0)

    ordered = sorted(events, key=lambda e: e.beat)
    next_times = [ordered[j].beat for j in range(1, len(ordered))]
    next_times.append(float('inf'))
    for ev, until in zip(ordered, next_times):
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
            if arrival <= until:
                _append_point(times, values, arrival, ev.value)
            else:
                reached = current + (ev.value - current) * (until - t) \
                    / (arrival - t)
                _append_point(times, values, until, reached)
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
    windows = defaultdict(list)
    for order, row in enumerate(mod_events):
        start = float(row['t_start'])
        end = float(row['t_end'])
        if end < start:
            continue
        raw_player = row.get('player')
        # An absent player number means the classic reader applies the
        # mod to BOTH players (ApplyGameCommand without a pn arg), so
        # the window lands on both channels, not just player 0.
        players = ((0, 1) if raw_player is None
                   else (max(0, int(raw_player) - 1),))
        for percent, speed, name in parse_modstring(row['modstring']):
            for player in players:
                windows[(name, player)].append(
                    _Window(start, end, percent, speed, order))

    events = []
    for (name, player), chan_windows in windows.items():
        for beat, value, speed in _resolve_windows(chan_windows):
            events.append(ModEvent(beat, value, speed, name, player))
    return ModChannels.compile(events)


@dataclass(frozen=True)
class _Window:
    """One window's contribution to a single (mod, player): drives `value`
    at approach `speed` over its span (`_resolve_windows` treats it
    half-open, reverting at `end`). `order` is the row index, breaking
    overlap ties (later table entry wins, matching the reader's re-apply
    order)."""
    start: float
    end: float
    value: float
    speed: float
    order: int


_REST_TARGET = (0.0, _CLEARALL_SPEED)


def _resolve_windows(windows) -> list:
    """A channel's overlapping windows -> `[(time, value, speed), ...]`
    retarget events, one per change in the resolved target.

    At any instant the engine's target is the highest-order window
    covering it (each re-applies every frame; later table entries win), or
    `(0, clearall speed)` where none is active. We build that step
    function over the intervals cut by every window boundary and emit a
    retarget wherever the target changes. Consequences: an end that a
    still-active window overrides never dips to 0 (overlapped mods hold);
    an isolated window's end reverts at clearall speed -- the float -- not
    the window's own approach `*S`; a zero-length window (start == end)
    covers no interval, so it is a no-op just as the engine's next frame
    would already have `clearall`-ed the one-frame spike away.

    Windows are treated half-open `[start, end)`: the target reverts AT
    the window's end (the engine reverts the frame after), which is why
    the trailing interval past the last end resets to rest.

    Sweep line over the boundary events with a lazy-deletion max-heap on
    `order`: per-frame templates re-apply every mod every frame (a
    getfucked2-scale chart carries ~10k windows on ~90 channels), and
    rescanning every window at every boundary is quadratic enough to
    read as a hang."""
    marks = []
    for w in windows:
        if w.end > w.start:
            marks.append((w.start, 1, w))
            marks.append((w.end, 0, w))
    # Ends sort before starts at the same instant (half-open intervals).
    marks.sort(key=lambda m: (m[0], m[1]))

    events = []
    prev = _REST_TARGET
    heap = []
    ended: set = set()
    # Unique heap tiebreak: two windows share an `order` when one row's
    # modstring names the same mod twice, and _Window defines no
    # ordering. Later push wins the tie, matching the parse order
    # (the row's later token overwrites the earlier).
    push_seq = 0
    i, n = 0, len(marks)
    while i < n:
        t = marks[i][0]
        while i < n and marks[i][0] == t:
            _t, is_start, w = marks[i]
            if is_start:
                heapq.heappush(heap, (-w.order, -push_seq, w))
                push_seq += 1
            else:
                ended.add(id(w))
            i += 1
        while heap and id(heap[0][2]) in ended:
            heapq.heappop(heap)
        target = ((heap[0][2].value, heap[0][2].speed) if heap
                  else _REST_TARGET)
        if target != prev:
            events.append((t, *target))
            prev = target
    return events
