"""Pre-walked beat<->time map used by all beat-space-aware engines.

A `TimingMap` consumes the chart's BPMs, stops, delays, warps, and song
offset, and produces O(log n) `beat_to_time` / `time_to_beat` /
`bps_at_time` lookups. It encodes Etterna's same-row event precedence
(BPM -> DELAY -> marker -> STOP -> WARP), which has been the source of
multiple historical bugs and is hardened by the existing test suite.

The class is the canonical source of truth for beat<->time conversion in
the player; the reference `BeatSpaceSVEngine` and the new measure-based
engine both consume it.
"""
from __future__ import annotations

import bisect

import numpy as np


class TimingMap:
    """Pre-walked BPMs + STOPS + DELAYS + WARPS event stream, with monotonic
    bisect-indexable checkpoints for O(log n) beat<->time conversion.

    Walking the event stream on every `beat_to_time` / `time_to_beat` call
    was the render-hot-path bottleneck on Etterna charts with many BPM/stop
    segments (Undiscovered Colors has 310 BPMs + 591 stops; a linear walk
    costs ~34us per call, and `distance()` does two of those per visible
    note per frame). Walking once into checkpoint arrays drops the per-call
    cost to a bisect + constant-time arithmetic.
    """

    def __init__(self, bpms, sm_offset, stops, delays, warps):
        bpms = sorted(bpms or [(0.0, 120.0)])
        stops = sorted(stops or [])
        delays = sorted(delays or [])
        warps = sorted(warps or [])
        sm_offset = float(sm_offset)

        # Single unified event stream. Etterna's FindEvent same-row
        # precedence is BPM -> DELAY -> target marker -> STOP -> WARP.
        # There is no marker in this prewalk, so STOP must still precede WARP;
        # otherwise stop+warp gimmicks lose the warp collapse in lookups.
        events = []
        for b, v in bpms:   events.append((b, 0, v))
        for b, v in delays: events.append((b, 1, v))
        for b, v in stops:  events.append((b, 2, v))
        for b, v in warps:  events.append((b, 3, v))
        events.sort(key=lambda e: (e[0], e[1]))

        # Checkpoints: after processing each event, record
        #   beat_enter:   cur_beat at moment this event starts (pre-event)
        #   time_enter:   time at that moment
        #   beat_exit:    cur_beat after applying the event (==beat_enter for
        #                 BPM/DELAY/STOP; beat_enter+warp_len for WARP)
        #   time_exit:    time after event (adds delay/stop duration)
        #   bps_after:    sec-per-beat-inverse that applies from this event
        #                 to the next
        # The segment between events[i] and events[i+1] advances at bps_after
        # from beat_exit to events[i+1]'s beat.
        self._beat_enter: list[float] = []
        self._time_enter: list[float] = []
        self._beat_exit: list[float] = []
        self._time_exit: list[float] = []
        self._bps_after: list[float] = []
        self._event_kind: list[int] = []

        t = -sm_offset
        cur_beat = 0.0
        bps = bpms[0][1] / 60.0
        warp_end = None

        # TODO: Generalize this to non-Etterna
        for (eb, kind, val) in events:
            if eb < 0:
                continue
            # Advance from cur_beat to eb along current bps.
            if warp_end is not None and eb <= warp_end:
                cur_beat = eb
            elif warp_end is not None and cur_beat < warp_end <= eb:
                cur_beat = warp_end
                warp_end = None
                t += (eb - cur_beat) / bps
                cur_beat = eb
            else:
                t += (eb - cur_beat) / bps
                cur_beat = eb

            beat_enter = cur_beat
            time_enter = t

            if kind == 0:       # BPM change
                bps = float(val) / 60.0
            elif kind == 1:     # DELAY: pause before the row's notes
                t += float(val)
            elif kind == 2:     # STOP: pause after the row's notes
                t += float(val)
            elif kind == 3:     # WARP ; beat teleports forward, time doesn't
                warp_end = cur_beat + float(val)

            # beat_exit reflects the post-event beat cursor for bisect
            # lookups; for WARP, that's the warp landing; for others,
            # unchanged from beat_enter. The event loop's own cur_beat
            # stays at beat_enter for warps so subsequent in-warp events
            # still get skipped correctly by the warp_end state check.
            if kind == 3:
                beat_exit = warp_end
            else:
                beat_exit = cur_beat
            time_exit = t

            self._beat_enter.append(beat_enter)
            self._time_enter.append(time_enter)
            self._beat_exit.append(beat_exit)
            self._time_exit.append(time_exit)
            self._bps_after.append(bps)
            self._event_kind.append(kind)

        # Initial bps for beat lookups at time < first event.
        self._bps_initial = bpms[0][1] / 60.0
        self._t_at_beat_zero = -sm_offset
        self._trailing_warp_end = warp_end

        # Numpy mirrors for batched lookup paths (project_times). Kept
        # in sync with the list fields; rebuild would require recomputing
        # both if we ever mutated them, but TimingMap is constructed once
        # per engine and never edited.
        self._beat_enter_np = np.asarray(self._beat_enter, dtype=np.float64)
        self._time_enter_np = np.asarray(self._time_enter, dtype=np.float64)
        self._beat_exit_np = np.asarray(self._beat_exit, dtype=np.float64)
        self._time_exit_np = np.asarray(self._time_exit, dtype=np.float64)
        self._bps_after_np = np.asarray(self._bps_after, dtype=np.float64)

    def time_to_beat_array(self, times: np.ndarray) -> np.ndarray:
        """Vectorized `time_to_beat`. Same rules as the scalar path
        (frozen beats inside STOP/DELAY pauses, extrapolation before the
        first event and past the last), applied over an entire array at
        once.
        """
        t = np.asarray(times, dtype=np.float64)
        if not t.size:
            return np.empty(0, dtype=np.float64)
        if self._beat_enter_np.size == 0:
            return (t - self._t_at_beat_zero) * self._bps_initial

        out = np.empty_like(t)
        # Pre-first-event branch: extrapolate from beat 0 at initial bps.
        before = t < self._time_enter_np[0]
        if before.any():
            out[before] = (t[before] - self._t_at_beat_zero) * self._bps_initial
        remaining = ~before
        if not remaining.any():
            return out

        tr = t[remaining]
        # Pause window test: t lies between time_enter[idx] and time_exit[idx]
        # at the same idx; beat is frozen to beat_exit[idx].
        pause_idx = np.searchsorted(self._time_enter_np, tr, side='right') - 1
        pause_valid = pause_idx >= 0
        pause_in = np.zeros_like(tr, dtype=bool)
        if pause_valid.any():
            safe_p = np.clip(pause_idx, 0, self._time_exit_np.size - 1)
            pause_in = pause_valid & (tr < self._time_exit_np[safe_p])
        out_r = np.empty_like(tr)
        if pause_in.any():
            out_r[pause_in] = self._beat_exit_np[pause_idx[pause_in]]
        not_pause = ~pause_in
        if not_pause.any():
            tr2 = tr[not_pause]
            idx = np.searchsorted(self._time_exit_np, tr2, side='right') - 1
            idx = np.clip(idx, 0, self._time_exit_np.size - 1)
            out_r[not_pause] = (self._beat_exit_np[idx]
                                + (tr2 - self._time_exit_np[idx])
                                * self._bps_after_np[idx])
        out[remaining] = out_r
        return out

    def beat_to_time(self, beat: float) -> float:
        """Time at `beat`. Bisect the event list, then advance at
        post-event bps from that event to `beat`."""
        if not self._beat_enter:
            return self._t_at_beat_zero + beat / self._bps_initial
        # Find last event whose beat_enter <= target beat. (If target beat
        # lies in the event's own span [beat_enter, beat_exit], time is
        # mid-transition; return time_enter for beats before event
        # completion, time_exit for after. Effectively: if target_beat ==
        # beat_enter, the PRE-event time is correct.)
        idx = bisect.bisect_right(self._beat_enter, beat) - 1
        if idx < 0:
            return self._t_at_beat_zero + beat / self._bps_initial
        # If the target is exactly on an event row, Etterna's marker
        # precedence is after BPM/DELAY and before STOP/WARP. The prewalk has
        # no marker event, so recover that boundary explicitly.
        left = bisect.bisect_left(self._beat_enter, beat)
        if left <= idx and self._beat_enter[left] == beat:
            marker_time = self._time_enter[left]
            for j in range(left, idx + 1):
                if self._event_kind[j] <= 1:  # BPM or DELAY happen before marker
                    marker_time = self._time_exit[j]
                else:                         # STOP/WARP happen after marker
                    break
            return marker_time
        # Is the target beat inside a WARP span that this event opened?
        # A WARP event has beat_exit > beat_enter; beats strictly between
        # land at time_exit (no time passes).
        bx = self._beat_exit[idx]
        if beat < bx:
            return self._time_exit[idx]
        return self._time_exit[idx] + (beat - bx) / self._bps_after[idx]

    def time_to_beat(self, t: float) -> float:
        """Inverse of beat_to_time. Bisect on time_exit (the post-event
        times are monotonically non-decreasing)."""
        if not self._time_exit:
            return (t - self._t_at_beat_zero) * self._bps_initial
        if t < self._time_enter[0]:
            return (t - self._t_at_beat_zero) * self._bps_initial
        # If we're inside a STOP/DELAY window, chart beat is frozen at that
        # event's beat until the pause finishes.
        pause_idx = bisect.bisect_right(self._time_enter, t) - 1
        if pause_idx >= 0 and t < self._time_exit[pause_idx]:
            return self._beat_exit[pause_idx]
        idx = bisect.bisect_right(self._time_exit, t) - 1
        if idx < 0:
            return self._beat_exit[0]
        return self._beat_exit[idx] + (t - self._time_exit[idx]) * self._bps_after[idx]

    def bps_at_time(self, t: float) -> float:
        """Instantaneous beats-per-second at time t.

        Returns 0 inside STOP/DELAY windows where chart beat is frozen.
        """
        if not self._time_enter:
            return self._bps_initial
        pause_idx = bisect.bisect_right(self._time_enter, t) - 1
        if pause_idx >= 0 and t < self._time_exit[pause_idx]:
            return 0.0
        idx = bisect.bisect_right(self._time_exit, t) - 1
        if idx < 0:
            return self._bps_initial
        return self._bps_after[idx]


# Back-compat alias for the historical private name. Existing code that
# imports `_TimingMap` from engine.py continues to work.
_TimingMap = TimingMap
