"""Chart time singleton.

Single source of truth for "what second of the chart is the playhead at".
Both the render thread and whatever else wants chart time (plugins, the
sidebar, tests) read `ChartClock.now()`; the clock itself decides whether
that comes from the audio engine (when playing) or a wall-clock accumulator
(when paused, scrubbing, or audioless).

Why this exists: chart time used to be a plain float advanced from
`time.monotonic()` dt each tick. The audio callback produces samples at
its own hardware cadence, so the two drifted — every stall (long paint,
GC pause, OS scheduling) widened the gap. `AudioEngine.set_state` then
seek-resynced to close it, which flushes the phase vocoder's OLA buffer
and chops audible chunks out.

Design: when audio is ready and playing, read `source_pos_s` from the PV
directly. No drift can accumulate because there's only one clock now.
Wall-clock fallback stays for:

  * paused playback (t doesn't advance, but scrub edits it directly),
  * audioless replays,
  * scrubbing (tab freezes the clock, slider drives t).
"""
from __future__ import annotations

import threading
import time
from typing import Callable


class ChartClock:
    """Thread-safe chart time source.

    Exactly one `ChartClock` per Player. Writes (`seek`, `set_paused`,
    `set_audio_source`) lock; reads (`now`) lock briefly to pick the
    active source. Callers from any thread are fine.
    """

    def __init__(self, *, initial: float = 0.0,
                 t_min: float = -2.0, t_max: float | None = None) -> None:
        self._lock = threading.Lock()
        # Wall-clock anchor: `t = _wall_anchor + (monotonic() - _wall_mono) * rate`.
        # On seek/pause/rate-change we rebase the anchor so the formula above
        # keeps giving the current t.
        self._wall_anchor = float(initial)
        self._wall_mono = time.monotonic()
        self._rate = 1.0
        self._paused = True
        self._t_min = float(t_min)
        self._t_max = float(t_max) if t_max is not None else float('inf')
        # Audio time reader: callable returning source-file seconds, thread-
        # safe on the caller's side (the audio engine locks internally).
        # `None` means no audio — fall back to wall-clock.
        self._audio_getter: Callable[[], float] | None = None

    # -- configuration ---------------------------------------------------

    def set_audio_source(self, getter: Callable[[], float] | None) -> None:
        """Install or clear the audio-time reader. Call when the engine
        becomes ready (install) or is stopped/destroyed (clear)."""
        with self._lock:
            # Capture current t BEFORE swapping, so the wall-clock anchor
            # picks up from wherever the outgoing source left off
            cur = self._now_locked()
            self._audio_getter = getter
            self._rebase_wall_locked(cur)

    def set_bounds(self, t_min: float, t_max: float) -> None:
        """Called by Player during init / when t_max is extended by audio
        duration. Clamps out-of-range seeks."""
        with self._lock:
            self._t_min = float(t_min)
            self._t_max = float(t_max)

    # -- playhead control -----------------------------------------------

    def seek(self, t: float) -> None:
        """Jump the playhead. The audio engine does its own seek in
        response; this just updates the wall-clock anchor so the fallback
        reads consistently if audio isn't driving."""
        with self._lock:
            t = self._clamp_locked(float(t))
            self._rebase_wall_locked(t)

    def set_paused(self, paused: bool) -> None:
        """Pause/unpause. When pausing, freeze t to whatever `now()` says
        right now (wall-clock stops advancing); when unpausing, rebase so
        the next `now()` continues from the frozen value."""
        with self._lock:
            if bool(paused) == self._paused:
                return
            cur = self._now_locked()
            self._paused = bool(paused)
            self._rebase_wall_locked(cur)

    def set_rate(self, rate: float) -> None:
        """Change playback rate. Rebase so the formula stays continuous."""
        with self._lock:
            rate = max(0.05, float(rate))
            if abs(rate - self._rate) < 1e-9:
                return
            cur = self._now_locked()
            self._rate = rate
            self._rebase_wall_locked(cur)

    # -- read -------------------------------------------------------------

    def now(self) -> float:
        """Current chart time in seconds. Clamped to [t_min, t_max].
        Safe to call from any thread."""
        with self._lock:
            return self._clamp_locked(self._now_locked())

    def intended(self) -> float:
        """Chart time the Player has asked for — the wall-clock anchor,
        advanced forward if unpaused + not audio-driven. Differs from
        `now()` only when an audio source is attached: `now()` reads the
        PV's actual position; `intended()` reads what seek/setter calls
        have written. The audio engine uses this to decide whether to
        seek (drift > threshold means its PV is behind intent)."""
        with self._lock:
            if self._paused or self._audio_getter is not None:
                return self._clamp_locked(self._wall_anchor)
            return self._clamp_locked(
                self._wall_anchor
                + (time.monotonic() - self._wall_mono) * self._rate)

    # -- introspection (used by player_tab for end-of-chart checks) ------

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def rate(self) -> float:
        with self._lock:
            return self._rate

    @property
    def t_min(self) -> float:
        return self._t_min

    @property
    def t_max(self) -> float:
        return self._t_max

    # -- internal (all require self._lock held) --------------------------

    def _now_locked(self) -> float:
        if self._paused:
            return self._wall_anchor
        if self._audio_getter is not None:
            try:
                return float(self._audio_getter())
            except Exception:
                # Audio getter blew up: fall through to wall-clock so the
                # clock keeps advancing. Better to render stale than stall.
                pass
        return self._wall_anchor + \
            (time.monotonic() - self._wall_mono) * self._rate

    def _rebase_wall_locked(self, t: float) -> None:
        self._wall_anchor = float(t)
        self._wall_mono = time.monotonic()

    def _clamp_locked(self, t: float) -> float:
        if t < self._t_min:
            return self._t_min
        if t > self._t_max:
            return self._t_max
        return t
