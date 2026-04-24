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

Design: when audio is ready and playing, read chart time from the audio
engine's DAC-clock-anchored playhead directly. No drift can accumulate
because there's only one clock now.
Wall-clock fallback stays for:

  * paused playback (t doesn't advance, but scrub edits it directly),
  * audioless replays,
  * scrubbing (tab freezes the clock, slider drives t).

`CullSpaceSmoother` remains in this module as an experimental helper, but
the production clock path does not run audio time through it. The inverse
from cull-space back to chart-time is not single-valued in scroll=0
regions, so smoothing the master clock there can return an arbitrary time
inside a plateau. The audio engine instead exposes a DAC-clock-anchored
chart time directly, which is smooth enough to render without a secondary
SV-space correction.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Protocol

from analysis.player.sv.debug import LOGGER as _SV_DEBUG_LOGGER


class _CullSpaceEngine(Protocol):
    """Subset of SVEngine the smoother reads."""
    def cumulative_at(self, t: float) -> float: ...
    def inverse_cumulative_at(self, sv: float) -> float: ...


class CullSpaceSmoother:
    """First-order low-pass smoother operating in cull-space.

    Ported from osu!-framework's `InterpolatingFramedClock` (damp toward
    target with exponential half-life, snap if drift exceeds a threshold),
    but applied in the SV engine's cumulative-SV space instead of plain
    chart-time. This keeps visible corrections uniform across SV regions:

        target_sv = engine.cumulative_at(raw_target_t)
        delta     = target_sv - last_target_sv     # audio's sv-velocity
        advanced  = current_sv + delta             # follow audio's motion
        drift     = target_sv - advanced           # residual lag
        if |drift| > SNAP:  current_sv = target_sv
        else:               current_sv = advanced + drift * (1 - 0.5^(dt/H))
        return engine.inverse_cumulative_at(current_sv)

    The "advance" step tracks audio's own cull-space velocity (which bakes
    in local SV factor) so the smoother doesn't lag on high-SV sections.
    Only residual drift — not the whole velocity — is low-passed.
    """

    _DRIFT_HALF_LIFE = 0.05      # seconds (half of sv-drift closed per 50ms)
    _SNAP_THRESHOLD_SV = 0.0333  # cull-space units (≈2 frames @ 60Hz at sv-rate=1)

    def __init__(self, engine: _CullSpaceEngine) -> None:
        self._engine = engine
        self._current_sv: float | None = None
        self._last_target_sv: float | None = None
        self._last_read_wall: float = 0.0

    def reset(self, t: float) -> None:
        """Re-anchor at chart-time t (call on seek or paused->playing)."""
        sv = self._engine.cumulative_at(t)
        self._current_sv = sv
        self._last_target_sv = sv
        self._last_read_wall = time.monotonic()

    def now(self, raw_target_t: float) -> float:
        """Smooth-step from the last read to `raw_target_t` (chart-time
        the audio engine just returned). Returns a chart-time suitable
        for rendering. Safe to call from any thread."""
        target_sv = self._engine.cumulative_at(raw_target_t)
        now_wall = time.monotonic()
        if self._current_sv is None:
            # First call: no history, no damping possible. Start pinned.
            self._current_sv = target_sv
            self._last_target_sv = target_sv
            self._last_read_wall = now_wall
            return raw_target_t

        elapsed = max(0.0, now_wall - self._last_read_wall)
        # Advance current by audio's own cull-space delta since last read.
        # This captures local SV rate for free — no need to compute a
        # d(sv)/dt ourselves.
        advanced = self._current_sv + (target_sv - self._last_target_sv)
        drift = target_sv - advanced
        if abs(drift) > self._SNAP_THRESHOLD_SV:
            self._current_sv = target_sv
        else:
            alpha = 1.0 - 0.5 ** (elapsed / self._DRIFT_HALF_LIFE)
            self._current_sv = advanced + drift * alpha

        self._last_target_sv = target_sv
        self._last_read_wall = now_wall
        return self._engine.inverse_cumulative_at(self._current_sv)


class _CullSpaceVelocityEngine(_CullSpaceEngine, Protocol):
    def cumulative_velocity_at(self, t: float) -> float: ...


class VisualCullSpacePredictor:
    """Render-only playhead predictor in cull-space.

    This does NOT try to return a chart-time. It keeps a smoothed
    `cumulative_at(t)` directly, which avoids the ambiguity of inverting
    through scroll=0 plateaus. The predictor advances from local cull-space
    velocity and damps residual drift toward the raw audio target.

    Important invariant: when the raw chart time is moving forward, the
    predicted cull-space position is clamped not to outrun the raw target.
    That removes the visible "rubber-band back" when a steep SV spike makes
    a tiny timing overprediction obvious on the next frame.
    """

    _DRIFT_HALF_LIFE = 0.012
    _SNAP_THRESHOLD_PX = 18.0
    _DISCONTINUITY_THRESHOLD_T = 0.100
    _BACKTRACK_THRESHOLD_T = 0.005

    def __init__(self, engine: _CullSpaceVelocityEngine) -> None:
        self._engine = engine
        self._current_sv: float | None = None
        self._last_target_sv: float | None = None
        self._last_raw_t: float | None = None
        self._last_wall: float = 0.0

    def reset(self, raw_t: float) -> None:
        target_sv = self._engine.cumulative_at(raw_t)
        self._current_sv = target_sv
        self._last_target_sv = target_sv
        self._last_raw_t = float(raw_t)
        self._last_wall = time.monotonic()

    def cumulative_now(self, raw_t: float, visual_scale: float = 1.0) -> float:
        target_sv = self._engine.cumulative_at(raw_t)
        now_wall = time.monotonic()
        if (self._current_sv is None or self._last_raw_t is None
                or self._last_target_sv is None):
            self._current_sv = target_sv
            self._last_target_sv = target_sv
            self._last_raw_t = float(raw_t)
            self._last_wall = now_wall
            return target_sv

        elapsed = max(0.0, now_wall - self._last_wall)
        raw_dt = float(raw_t) - self._last_raw_t
        if (raw_dt < -self._BACKTRACK_THRESHOLD_T
                or abs(raw_dt) > self._DISCONTINUITY_THRESHOLD_T):
            self._current_sv = target_sv
        else:
            # Follow the exact integrated cull-space delta that the raw audio
            # target traversed since the last frame, even if it crossed
            # multiple SV sections. Only the residual lag is damped.
            advanced = self._current_sv + (target_sv - self._last_target_sv)
            drift = target_sv - advanced
            scale = abs(float(visual_scale))
            if scale <= 1e-9:
                # Invisible region (e.g. SPEEDS=0): pin immediately because
                # any correction is unobservable and delaying it only creates
                # a later jump when visibility returns.
                self._current_sv = target_sv
            else:
                drift_px = drift * scale
                if abs(drift_px) > self._SNAP_THRESHOLD_PX:
                    self._current_sv = target_sv
                else:
                    alpha = 1.0 - 0.5 ** (elapsed / self._DRIFT_HALF_LIFE)
                    self._current_sv = advanced + (drift_px * alpha) / scale
                    if raw_dt >= 0.0:
                        self._current_sv = min(self._current_sv, target_sv)

        self._last_target_sv = target_sv
        self._last_raw_t = float(raw_t)
        self._last_wall = now_wall
        return self._current_sv


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
        # Retained only for tests/experiments. Production reads return the
        # audio engine's DAC-clock-anchored chart time directly.
        self._smoother: CullSpaceSmoother | None = None

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
            if self._smoother is not None:
                self._smoother.reset(cur)
            self._debug_log_locked('chart_set_audio_source', {
                'attached': bool(getter is not None),
                'cur': float(cur),
            })

    def set_sv_engine(self, engine) -> None:
        """Accept the Player's SV engine for compatibility.

        The clock no longer smooths the master playhead in SV-space because
        `inverse_cumulative_at()` is ambiguous in scroll=0 regions. Keep the
        method so existing callers don't need to branch."""
        with self._lock:
            self._smoother = None

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
            if self._smoother is not None:
                self._smoother.reset(t)
            self._debug_log_locked('chart_seek', {'t': float(t)})

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
            if self._smoother is not None:
                # Re-anchor on unpause so the first render-read after
                # resume doesn't show a snap-back to stale smoother state.
                self._smoother.reset(cur)
            self._debug_log_locked('chart_set_paused', {
                'paused': bool(self._paused),
                'cur': float(cur),
            })

    def set_rate(self, rate: float) -> None:
        """Change playback rate. Rebase so the formula stays continuous."""
        with self._lock:
            rate = max(0.05, float(rate))
            if abs(rate - self._rate) < 1e-9:
                return
            cur = self._now_locked()
            self._rate = rate
            self._rebase_wall_locked(cur)
            self._debug_log_locked('chart_set_rate', {
                'rate': float(self._rate),
                'cur': float(cur),
            })

    # -- read -------------------------------------------------------------

    def now(self) -> float:
        """Current chart time in seconds. Clamped to [t_min, t_max].
        Safe to call from any thread."""
        with self._lock:
            t = self._clamp_locked(self._now_locked())
            self._debug_log_locked('chart_now', {
                't': float(t),
                'paused': bool(self._paused),
                'has_audio': bool(self._audio_getter is not None),
            })
            return t

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
                raw = float(self._audio_getter())
            except Exception:
                # Audio getter blew up: fall through to wall-clock so the
                # clock keeps advancing. Better to render stale than stall.
                pass
            else:
                return raw
        return self._wall_anchor + \
            (time.monotonic() - self._wall_mono) * self._rate

    def _rebase_wall_locked(self, t: float) -> None:
        self._wall_anchor = float(t)
        self._wall_mono = time.monotonic()

    def _debug_log_locked(self, subtype: str, payload: dict) -> None:
        if not _SV_DEBUG_LOGGER.enabled:
            return
        rec = {'type': 'chart_clock', 'subtype': subtype}
        rec.update(payload)
        _SV_DEBUG_LOGGER.log(rec)

    def _clamp_locked(self, t: float) -> float:
        if t < self._t_min:
            return self._t_min
        if t > self._t_max:
            return self._t_max
        return t
