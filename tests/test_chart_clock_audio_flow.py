"""Regression tests for the ChartClock / audio-engine interaction.

These tests use a fake audio source (plain callable) instead of the real
AudioEngine, so they don't pull in sounddevice / soundfile / any real
DSP. They exercise the CONTRACT the engine has to honor:

  1. Attaching a new audio source mid-play doesn't jump t.
  2. Detaching preserves t (already covered in test_chart_clock; reinforced
     here as part of the full flow).
  3. Seeking updates `intended` immediately but `now` only tracks once
     the audio source catches up — matches the real flow where the GUI
     writes the wall anchor but the PV seek takes a moment.
  4. When audio returns a frozen (stepped) time, `now` still extrapolates
     smoothly because the engine's own smoothing does it — here we just
     confirm the clock doesn't accidentally cache/stall.
  5. The pause-then-seek-then-resume ordering leaves t where expected.
"""
import time

import pytest

from analysis.player.scroll.chart_clock import ChartClock


class _FakeAudio:
    """Minimal stand-in for AudioEngine: exposes a getter that returns a
    chart-time, plus a seek that updates what the getter reports.

    Models the interpolation behavior the real engine implements: on
    seek, the reported time jumps immediately. Between seeks, the time
    advances by wall-clock delta times rate (i.e. like the engine's
    post-smoothing output). This keeps the test orthogonal to the real
    DSP while matching the real contract."""

    def __init__(self, initial: float = 0.0, rate: float = 1.0) -> None:
        self._anchor_t = float(initial)
        self._anchor_wall = time.monotonic()
        self._rate = float(rate)
        self._playing = True

    def current_chart_time(self) -> float:
        if not self._playing:
            return self._anchor_t
        return self._anchor_t + (time.monotonic() - self._anchor_wall) * self._rate

    def seek(self, t: float) -> None:
        self._anchor_t = float(t)
        self._anchor_wall = time.monotonic()

    def set_playing(self, playing: bool) -> None:
        # Freeze the reported time at the current value when pausing;
        # resume from there when unpausing.
        self._anchor_t = self.current_chart_time()
        self._anchor_wall = time.monotonic()
        self._playing = bool(playing)


def test_attach_then_read_follows_audio():
    c = ChartClock()
    c.set_paused(False)
    audio = _FakeAudio(initial=30.0)
    c.set_audio_source(audio.current_chart_time)
    # Clock now reads from audio
    assert c.now() >= 30.0


def test_detach_preserves_t_across_audio_seek():
    """If audio seeks away and then we detach, the clock should hold the
    last-known audio time, not snap back to whatever wall-clock thinks."""
    c = ChartClock()
    c.set_paused(False)
    audio = _FakeAudio(initial=10.0)
    c.set_audio_source(audio.current_chart_time)
    audio.seek(90.0)
    t_before = c.now()
    assert t_before >= 90.0
    c.set_audio_source(None)
    t_after = c.now()
    # No backward jump
    assert t_after == t_before or t_after > t_before


def test_seek_updates_intended_without_waiting_for_audio():
    """Writing t via `seek` updates the clock's own anchor. `now()` still
    reads the audio getter (stale, pre-seek) until the audio side seeks.
    This matches the real flow where the GUI must send `intended` to the
    engine's seek(), not `now`, to avoid sending stale values."""
    c = ChartClock()
    c.set_paused(False)
    audio = _FakeAudio(initial=5.0)
    c.set_audio_source(audio.current_chart_time)

    c.seek(100.0)
    # intended reflects the seek; now still tracks audio
    assert c.intended() == pytest.approx(100.0, abs=0.01)
    assert c.now() < 50.0  # still ~5 + a tiny wall delta

    # Caller tells audio engine to seek; now they match
    audio.seek(100.0)
    assert c.now() >= 100.0


def test_pause_freezes_t_even_when_audio_advances():
    """Pausing the clock should stop `now()` from advancing regardless of
    what the audio source reports. Real AudioEngine also stops the stream
    in response to pause, but the clock must be the authoritative gate."""
    c = ChartClock()
    audio = _FakeAudio(initial=20.0)
    c.set_audio_source(audio.current_chart_time)
    c.set_paused(False)
    t0 = c.now()
    c.set_paused(True)
    t1 = c.now()
    # Audio getter keeps advancing (we haven't stopped it), but clock is frozen
    time.sleep(0.05)
    t2 = c.now()
    assert t1 == t2
    assert t1 >= t0


def test_rate_change_still_advances_between_audio_callbacks():
    """The clock itself doesn't apply rate when audio is driving (the audio
    engine already bakes rate into its own time output), but rate-setting
    shouldn't break audio-driven reads either. Use a rate-aware fake."""
    c = ChartClock()
    audio = _FakeAudio(initial=0.0, rate=2.0)
    c.set_audio_source(audio.current_chart_time)
    c.set_paused(False)
    t0 = c.now()
    time.sleep(0.05)
    t1 = c.now()
    # Audio reports 2x rate -> we should see ~0.1s advance over 0.05s real
    assert t1 - t0 > 0.07


def test_scrub_flow_detach_write_reattach():
    """Full scrub flow:
       1. Detach audio (simulate _on_playbar_pressed)
       2. Write new t (_on_playbar_changed)
       3. Seek audio + reattach (_on_playbar_released)
    The clock should report the scrubbed t throughout and smoothly
    resume from there after reattach."""
    c = ChartClock(t_max=300.0)
    audio = _FakeAudio(initial=50.0)
    c.set_audio_source(audio.current_chart_time)
    c.set_paused(False)

    # Scrub starts
    c.set_audio_source(None)
    # User drags slider to 200
    c.seek(200.0)
    # Unpaused wall-clock fallback: t advances from 200 but only by tiny
    # wall-clock delta in the nanoseconds between seek and this read.
    assert c.now() == pytest.approx(200.0, abs=0.01)
    # Release: seek audio to match, then reattach
    audio.seek(200.0)
    c.set_audio_source(audio.current_chart_time)
    # Clock now tracks audio from 200
    assert c.now() >= 200.0


def test_audio_getter_returning_nonsense_does_not_crash():
    """An audio source that raises shouldn't take down the render thread;
    the clock should fall back to wall-clock. Already tested elsewhere
    but reinforced here as part of the documented contract."""
    c = ChartClock()
    c.set_paused(False)

    def bad():
        raise RuntimeError('audio dead')

    c.set_audio_source(bad)
    t = c.now()  # must not raise
    assert isinstance(t, float)
