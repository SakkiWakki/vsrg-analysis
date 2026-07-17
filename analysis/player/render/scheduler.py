"""Events and live-evaluated curves - the render-side scheduler.

The compiled modchart is a set of EVENTS (discrete things that happen at
a time/beat) and CURVES (values that evolve continuously), NOT a dense
grid of per-frame samples. A per-frame driver that rotates a field is
one curve, not 3600 keyframes/minute; a mod window is two events, not a
sample per tick. This module is where the renderer holds that model and
evaluates it live at frame time.

Three pieces, each a small primitive:

- `Clock` maps song-time seconds to the coordinate a curve is authored
  in. Song time is the identity; `beat` is the bpm integral; `scroll` is
  the SV integral (`analysis/player/sv`); an effect `timer` loops. A
  curve names its clock, so "oscillate every 2 beats" and "every 2
  seconds" are the same curve over different clocks (the rate-mod /
  stop / warp reductions all become clock choice).

- `Channel` is a curve + its clock: `channel.at(t_seconds)` evaluates
  the curve at `clock.at(t)`. The curve can be analytic (an oscillator's
  `magnitude * shape(phase)`), piecewise (an `EventTimeline`), or any
  callable - all behind one `.at()`. Nothing is pre-sampled.

- `EventSchedule` is a time-sorted list of `Event`s (a tween start, a
  mod window opening, a message dispatch) with `due(t)` / `active(t)`
  queries. The discrete backbone the curves hang off of.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Callable, Protocol


class Clock(Protocol):
    """Maps song-time seconds to a curve's authoring coordinate."""

    def at(self, t_seconds: float) -> float:
        ...


class SongTimeClock:
    """The identity clock: a curve authored in song seconds."""

    def at(self, t_seconds: float) -> float:
        return t_seconds


class IntegralClock:
    """A clock backed by a monotonic integral of song time - beat (bpm
    integral) or scroll position (SV integral). `integral(t)` is any
    `f(seconds) -> coordinate`; the SV `CumulativeIntegrator.cumulative_at`
    and a bpm beat map both fit."""

    def __init__(self, integral: Callable[[float], float]):
        self._integral = integral

    def at(self, t_seconds: float) -> float:
        return self._integral(t_seconds)


class LoopClock:
    """A wrapping clock: coordinate advances with the base clock then
    wraps at `period` (+ `delay`), the effect-timer / stutter-loop case.
    The degenerate loop is `mu = dtau` on [0, period] plus a warp atom
    back to 0 (the user's LoopScheduler); a curve over this clock
    repeats every period."""

    def __init__(self, base: Clock, period: float, delay: float = 0.0):
        self._base = base
        self._span = max(1e-9, period + delay)

    def at(self, t_seconds: float) -> float:
        return self._base.at(t_seconds) % self._span


_SONG_TIME = SongTimeClock()


@dataclass(frozen=True)
class Channel:
    """A curve evaluated live over a clock. `curve(coord) -> value` is
    any callable in the clock's coordinate; `.at(t)` feeds it
    `clock.at(t)`. `rest` is returned when `curve` is None (an unset
    channel), so a consumer can hold one Channel per property and get the
    resting value for free."""
    curve: Callable[[float], object] | None
    clock: Clock = _SONG_TIME
    rest: object = 0.0

    def at(self, t_seconds: float):
        if self.curve is None:
            return self.rest
        return self.curve(self.clock.at(t_seconds))


def timeline_channel(timeline, clock: Clock = _SONG_TIME, index: int = 0):
    """A Channel over an existing `EventTimeline` (piecewise-eased
    keyframes), sampling component `index`. Bridges the current keyframe
    curves into the channel model without rebaking them."""
    if not timeline:
        return Channel(None, clock, 0.0)
    return Channel(lambda c: timeline.sample(c)[index], clock,
                   timeline.sample(float('-inf'))[index])


@dataclass(frozen=True)
class Event:
    """One scheduled thing. `t` is its song-time start; `duration` (0 for
    an instant) bounds when it is `active`; `payload` is caller-defined
    (a command name, a mod window, a tween spec)."""
    t: float
    payload: object
    duration: float = 0.0

    @property
    def t_end(self) -> float:
        return self.t + self.duration


@dataclass
class EventSchedule:
    """Time-sorted events with due/active queries. The discrete backbone;
    curves evaluate between events, events mark where behavior changes."""
    events: list = field(default_factory=list)
    _starts: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.events = sorted(self.events, key=lambda e: e.t)
        self._starts = [e.t for e in self.events]

    def add(self, event: Event) -> None:
        idx = bisect_right(self._starts, event.t)
        self.events.insert(idx, event)
        self._starts.insert(idx, event.t)

    def due(self, t0: float, t1: float) -> list:
        """Events starting in the half-open song-time span [t0, t1) - the
        ones a frame stepping from t0 to t1 should fire, in order."""
        lo = bisect_right(self._starts, t0) - 1
        lo = max(0, lo)
        while lo < len(self._starts) and self._starts[lo] < t0:
            lo += 1
        out = []
        i = lo
        while i < len(self._starts) and self._starts[i] < t1:
            out.append(self.events[i])
            i += 1
        return out

    def active(self, t: float) -> list:
        """Events whose [t, t_end) span covers song-time `t`."""
        idx = bisect_right(self._starts, t)
        return [e for e in self.events[:idx] if e.t <= t < e.t_end
                or (e.duration == 0.0 and e.t == t)]
