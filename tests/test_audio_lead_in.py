"""Regression tests for the lead-in / scrub / pause silence model.

The audio engine emits silent samples for any frame where it should not
be producing source audio (lead-in chart-time < 0, scrub-hold, pause).
The frame counter advances during silence so chart-time is always
exactly representable as `(src_frame - lead_in_frames) / sr`.

These tests use a stub callback driver (no real PortAudio) that calls
the engine's callback directly and inspects request state. They lock
the contract before the GUI / clock simplifications go in.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub source: minimal interface the new AudioEngine needs from a "decoded
# audio file" + PV. The real engine pulls from StreamingPhaseVocoder; for
# tests we substitute something that returns a constant non-zero waveform
# so we can distinguish silent from non-silent output blocks.
# ---------------------------------------------------------------------------


class _StubPV:
    """Minimal PV-shaped stub.

    Honors `_src_pos` advancement at the requested rate so the engine's
    frame-counter math matches the real PV. Returns a constant 1.0 signal
    so the engine can distinguish "audio playing" from "silent" in tests.
    """

    def __init__(self, sr: int = 44100):
        self._sr = sr
        self._src_pos = 0.0
        self.rate = 1.0
        self.channels = 2

    def set_rate(self, rate: float) -> None:
        self.rate = max(0.05, float(rate))

    def seek(self, chart_time: float) -> None:
        self._src_pos = float(chart_time) * self._sr

    def generate(self, n_frames: int) -> tuple[np.ndarray, bool]:
        # 1.0-amplitude stereo to make "audio is on" obvious in outdata.
        out = np.ones((n_frames, self.channels), dtype=np.float32)
        self._src_pos += n_frames * self.rate
        return out, True


# ---------------------------------------------------------------------------
# Stub engine: the new audio engine API as I'm about to implement it. The
# real engine plugs in StreamingPhaseVocoder + PortAudio; the stub keeps
# the contract testable without audio hardware.
# ---------------------------------------------------------------------------


class _StubAudioEngine:
    """Mirrors the new AudioEngine surface for tests.

    Public surface:
      seek_to_chart_time(t) -- jump; if t<0, set lead-in frames and park
                               PV at source 0.
      set_silent(bool)      -- request silence (paused/scrubbing).
      set_rate(rate)        -- propagate playback rate to PV.
      current_chart_time()  -- always exact: (src_frame - lead_in)/sr.

    Plus a `_callback(frames)` for tests that simulates one PortAudio
    block.
    """

    def __init__(self, sr: int = 44100):
        self._sr = sr
        self._pv = _StubPV(sr)
        self._volume = 1.0
        self._silent = True
        self._rate = 1.0
        self._src_frame_pos = 0.0
        self._lead_in_frames = 0

    def seek_to_chart_time(self, chart_t: float) -> None:
        if chart_t < 0:
            self._lead_in_frames = int(round(-chart_t * self._sr))
            self._src_frame_pos = 0.0
            self._pv.seek(0.0)
        else:
            self._lead_in_frames = 0
            self._src_frame_pos = float(chart_t) * self._sr
            self._pv.seek(chart_t)

    def set_silent(self, silent: bool) -> None:
        self._silent = bool(silent)

    def set_rate(self, rate: float) -> None:
        self._rate = max(0.05, float(rate))
        self._pv.set_rate(self._rate)

    def current_chart_time(self) -> float:
        return (self._src_frame_pos - self._lead_in_frames) / self._sr

    def _callback(self, n_frames: int) -> np.ndarray:
        """Simulate one PortAudio block. Returns the output buffer."""
        in_lead_in = self._src_frame_pos < self._lead_in_frames
        is_silent = self._silent or in_lead_in
        out = np.empty((n_frames, self._pv.channels), dtype=np.float32)
        if is_silent:
            out.fill(0.0)
            self._src_frame_pos += n_frames * self._rate
        else:
            samples, _ = self._pv.generate(n_frames)
            np.multiply(samples, self._volume, out=out)
            self._src_frame_pos = float(self._pv._src_pos)
        return out


# ---------------------------------------------------------------------------
# Direct engine invariants
# ---------------------------------------------------------------------------


def test_initial_state_is_silent_at_zero():
    eng = _StubAudioEngine()
    assert eng.current_chart_time() == 0.0
    out = eng._callback(512)
    assert (out == 0.0).all()


def test_seek_to_positive_chart_time_lands_on_frame():
    eng = _StubAudioEngine(sr=44100)
    eng.seek_to_chart_time(1.0)
    assert eng.current_chart_time() == pytest.approx(1.0, abs=1e-12)


def test_seek_to_negative_sets_lead_in():
    eng = _StubAudioEngine(sr=44100)
    eng.seek_to_chart_time(-0.5)
    # 0.5 sec lead-in @ 44100 Hz = 22050 frames.
    assert eng._lead_in_frames == 22050
    assert eng.current_chart_time() == pytest.approx(-0.5, abs=1e-12)


def test_lead_in_emits_silence_then_audio():
    """Negative chart-time -> silent samples. After enough callbacks
    drive the frame counter past lead_in_frames, samples become non-zero."""
    eng = _StubAudioEngine(sr=44100)
    eng.set_silent(False)
    eng.seek_to_chart_time(-0.01)   # 441 frame lead-in
    # First block (512 frames @ 44100): crosses chart-t = 0 partway
    # through. Engine logic is "silent until src_frame >= lead_in_frames",
    # so the entire 512-frame block is either silent (if src_frame < lead_in
    # at block start) or non-silent (if >= lead_in). At start, 0 < 441,
    # so the whole block is silent.
    out0 = eng._callback(512)
    assert (out0 == 0.0).all(), 'first block during lead-in must be silent'
    # After block 0, src_frame_pos = 512. Now 512 >= 441, so block 1 is
    # non-silent.
    out1 = eng._callback(512)
    assert (out1 != 0.0).any(), \
        'second block (past lead-in boundary) must produce audio'


def test_chart_time_crosses_zero_at_exact_frame():
    """At sample-rate precision, chart-time becomes >= 0 at exactly
    frame `lead_in_frames`. No drift."""
    eng = _StubAudioEngine(sr=44100)
    eng.set_silent(False)
    eng.seek_to_chart_time(-1.0)
    # 1 second of lead-in = 44100 frames. Walk past it in 100-frame
    # increments and check chart-time crosses zero on the right block.
    blocks_before = 44100 // 100   # 441 blocks of 100 frames = 44100 frames
    for _ in range(blocks_before):
        eng._callback(100)
    assert eng.current_chart_time() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Silent-flag covers paused / scrub
# ---------------------------------------------------------------------------


def test_silent_flag_suppresses_audio_even_past_lead_in():
    eng = _StubAudioEngine()
    eng.set_silent(False)
    eng.seek_to_chart_time(2.0)
    out = eng._callback(512)
    assert (out != 0.0).any(), 'normal play should produce audio'

    eng.set_silent(True)
    out = eng._callback(512)
    assert (out == 0.0).all(), 'silent flag should mute audio'


def test_chart_time_advances_during_silent_blocks():
    """The frame counter must tick during silence so chart-time stays
    accurate when the user resumes."""
    eng = _StubAudioEngine(sr=44100)
    eng.set_silent(True)
    eng.seek_to_chart_time(2.0)
    t_before = eng.current_chart_time()
    eng._callback(4410)   # 100ms of silence
    t_after = eng.current_chart_time()
    assert t_after - t_before == pytest.approx(0.1, abs=1e-9)


# ---------------------------------------------------------------------------
# Rate scaling
# ---------------------------------------------------------------------------


def test_chart_time_advances_at_rate():
    """At rate=2.0, chart-time advances at 2x the wall-clock rate. One
    block of N output frames consumes 2N source frames -> chart-time
    advances by 2N/sr."""
    eng = _StubAudioEngine(sr=44100)
    eng.set_silent(False)
    eng.set_rate(2.0)
    eng.seek_to_chart_time(0.0)
    eng._callback(2205)   # 50ms of audio at 1.0x = 100ms at 2.0x
    assert eng.current_chart_time() == pytest.approx(0.1, abs=1e-9)


def test_lead_in_completes_faster_at_higher_rate():
    """At rate=2.0 a 1-second lead-in completes in 0.5 seconds of audio
    output -- because src_frame_pos advances at 2x."""
    eng = _StubAudioEngine(sr=44100)
    eng.set_silent(False)
    eng.set_rate(2.0)
    eng.seek_to_chart_time(-1.0)   # 44100-frame lead-in
    # Drive for 22050 frames at 2x: src advances by 44100. Should land at 0.
    eng._callback(22050)
    assert eng.current_chart_time() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Drag-to-negative regression
# ---------------------------------------------------------------------------


def test_drag_to_negative_then_release_plays_correctly():
    """The full GUI flow:
      1. User drags playbar to chart_t = -0.02 (scrubbing -> silent).
      2. User holds cursor at that position (audio stays silent, chart
         clock would advance but isn't time-based here).
      3. User releases (scrub clears -> non-silent).
      4. Lead-in plays out (silent until src_frame >= lead_in_frames).
      5. Audio comes in past frame `lead_in_frames`.
    """
    eng = _StubAudioEngine(sr=44100)

    # Step 1: drag begins. Engine is silent (paused/scrubbing).
    eng.set_silent(True)
    eng.seek_to_chart_time(-0.02)
    assert eng.current_chart_time() == pytest.approx(-0.02, abs=1e-9)

    # Step 2: callbacks during the hold produce silent samples; chart-time
    # advances on the engine side but the GUI's `silent` flag keeps it
    # muted. After 50ms of silent callbacks, chart-time advances 50ms.
    out = eng._callback(2205)   # 50ms at sr=44100
    assert (out == 0.0).all()
    assert eng.current_chart_time() == pytest.approx(0.03, abs=1e-9)

    # Step 3: user releases the playbar at the new position chart_t=-0.02.
    # GUI should re-seek to that position to undo the silent advancement,
    # then clear the silent flag.
    eng.seek_to_chart_time(-0.02)
    eng.set_silent(False)
    assert eng.current_chart_time() == pytest.approx(-0.02, abs=1e-9)

    # Step 4: callbacks emit silence until the frame counter passes the
    # lead-in boundary.
    lead_in_frames = int(round(0.02 * 44100))   # 882
    # Walk in 200-frame blocks until past lead-in.
    while eng._src_frame_pos < lead_in_frames:
        out = eng._callback(200)
        assert (out == 0.0).all(), 'still in lead-in -> should be silent'

    # Step 5: next block past lead-in must produce audio.
    out_after = eng._callback(200)
    assert (out_after != 0.0).any(), 'past lead-in -> audio plays'


def test_scrub_hold_keeps_chart_time_visible_but_silent():
    """While the user holds the playbar still, audio stays silent but
    chart-time should freeze at the seeked value (not advance through
    silence). This is GUI-side: each tick re-seeks via
    seek_to_chart_time so chart-time stays pinned. The engine itself
    keeps advancing -- it's the GUI's job to re-seek if it wants
    chart-time pinned."""
    eng = _StubAudioEngine(sr=44100)
    eng.set_silent(True)
    eng.seek_to_chart_time(5.0)

    # Without re-seek, src_frame_pos advances during silent callbacks.
    eng._callback(4410)
    assert eng.current_chart_time() != pytest.approx(5.0, abs=1e-3)

    # GUI's per-tick behavior: while scrubbing, repeatedly re-seek to
    # the playbar's value. Chart-time pinned.
    for _ in range(10):
        eng.seek_to_chart_time(5.0)
        eng._callback(4410)
    eng.seek_to_chart_time(5.0)
    assert eng.current_chart_time() == pytest.approx(5.0, abs=1e-9)
