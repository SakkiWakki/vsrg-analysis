"""Public `AudioEngine`: owns decoding, the PortAudio output stream, and
mediates between the GUI thread and the audio callback thread via a mutex.
The heavy DSP lives in StreamingPhaseVocoder.

Surface:

  seek_to_chart_time(t)   -- Jump to chart-time `t`. Negative t becomes
                             a lead-in: the engine emits silent frames
                             until the source-frame counter passes the
                             lead-in mark, then the PV starts producing
                             audio. The transition is exact at one
                             sample-rate frame boundary.
  set_silent(silent)      -- GUI-requested silence (paused / scrubbing).
                             Independent of lead-in. Either source of
                             silence makes the callback emit zeros.
  set_rate(rate)          -- Playback rate; propagated to the PV. Source
                             frames advance at `rate * frames` per
                             callback regardless of silence so chart-time
                             stays accurate when the user un-silences.
  set_volume(v)
  set_pitch_correct(bool)
  current_chart_time()    -- Source position in seconds, minus lead-in.
                             Negative during lead-in, exact across
                             callbacks via DAC-anchor refinement when
                             available, sample-grain accurate otherwise.
  stop()
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time

import numpy as np

from analysis.player.sv.debug import LOGGER as _SV_DEBUG_LOGGER

from .phase_vocoder import StreamingPhaseVocoder
from .source import WaveSource
from .stream_worker import StreamWorker

# sounddevice is the PortAudio binding we drive the output stream with.
# Imported lazily so the module still imports in test environments that
# don't have an audio device.
try:
    import sounddevice as _sd
except Exception:
    _sd = None

# Decoding stack: soundfile (libsndfile ; wav/flac/ogg/opus) as the primary,
# audioread (ffmpeg/gstreamer wrapper ; mp3/m4a/anything) as the fallback for
# formats libsndfile doesn't handle. Both are pure decoders that hand us
# float samples directly, no mixer roundtrip.
try:
    import soundfile as _sf
except Exception:
    _sf = None

try:
    import audioread as _audioread
except Exception:
    _audioread = None


# ===== public engine, same API as before =====

class AudioEngine:
    RESYNC_THRESHOLD_S = 0.15

    def __init__(self, audio_path: str | None, volume: float = 0.5,
                 pitch_correct: bool = True):
        self.ready = False
        self._volume = float(volume)
        self._pitch_correct = bool(pitch_correct)
        self._sr = 44100
        self._base_duration = 0.0
        self._ended = False
        # Engine state kept under a lock because the audio callback (on the
        # PortAudio thread) and the GUI thread both touch it.
        self._lock = threading.Lock()
        # Serialize all access to the phase vocoder/source state. The audio
        # callback calls `generate()` while the GUI thread may concurrently
        # call seek/set_rate/set_pitch_correct. Those mutate the same
        # internal buffers and cursors and must never race.
        self._pv_lock = threading.Lock()
        self._source: WaveSource | None = None
        self._pv: StreamingPhaseVocoder | None = None
        self._stream = None
        # Stream-owner: holds the OutputStream's lifecycle on its own
        # thread to keep audio off the Qt main thread. Always present so
        # `stop()` can be called unconditionally even when no audio file
        # is loaded.
        self._stream_worker = StreamWorker()
        # GUI-requested silence (paused / scrubbing). Independent of
        # lead-in: either source makes the callback emit zeros.
        self._silent = True
        self._rate = 1.0
        # Source position in seconds. Always >= 0 (audio domain). The
        # callback advances this on every block (silent or not) so
        # chart-time stays accurate during silence. Sub-frame refinement
        # comes from the DAC-anchor pair below when valid.
        self._chart_time = 0.0
        # PortAudio-backed anchor: `hw_pos` (in chart-time-seconds) is
        # audible at DAC clock time `hw_wall`. Valid after the first
        # post-discontinuity callback. Used to refine current_chart_time()
        # below sample-grain accuracy by extrapolating from the stream
        # clock.
        self._hw_pos = 0.0
        self._hw_wall = 0.0
        self._scheduled_chart_pos = 0.0
        self._dac_anchor_valid = False
        # Lead-in: the engine emits silent frames until source frames
        # accumulated since the negative seek pass this threshold (in
        # seconds). Exposed via current_chart_time() = _chart_time -
        # _lead_in_seconds, so chart-time is negative during lead-in and
        # crosses 0 on the exact source-frame boundary.
        self._lead_in_seconds = 0.0

        if not audio_path or not os.path.exists(audio_path):
            return
        if _sd is None:
            print('audio: sounddevice not installed ; no audio playback')
            return

        samples, sr = self._decode(audio_path)
        if samples is None:
            return
        self._sr = int(sr)
        self._source = WaveSource(samples, sr)
        self._base_duration = self._source.duration
        self._pv = StreamingPhaseVocoder(self._source, rate=1.0,
                                          pitch_correct=self._pitch_correct)
        self._open_stream()

    def _open_stream(self) -> None:
        """Create + start the PortAudio OutputStream on a dedicated
        worker thread so it doesn't share affinity / GIL pressure with
        the Qt main thread that constructed the engine. See
        `stream_worker.StreamWorker` for the lifecycle contract."""
        def make_stream():
            stream = _sd.OutputStream(
                samplerate=self._sr,
                channels=self._source.src_channels,
                dtype='float32',
                blocksize=512,
                callback=self._callback,
            )
            stream.start()
            return stream

        stream, err = self._stream_worker.open(make_stream)
        if err is not None:
            print(f'audio: failed to open output stream: {err}')
            self._stream = None
        else:
            self._stream = stream
            self.ready = True

    # --- decoding: soundfile first (fast, float-native for wav/flac/ogg),
    # audioread fallback (ffmpeg, handles mp3/m4a/webm/...). Both return
    # float32 samples ready to feed the phase vocoder without a mixer
    # roundtrip.
    def _decode(self, path: str):
        if _sf is not None:
            try:
                arr, sr = _sf.read(path, dtype='float32', always_2d=True)
                return arr, int(sr)
            except Exception as e:
                # Fall through to audioread ; libsndfile doesn't do mp3/m4a
                # on every build.
                sf_err = e
        else:
            sf_err = None

        audioread_err = None
        if _audioread is not None:
            try:
                with _audioread.audio_open(path) as f:
                    sr = int(f.samplerate)
                    channels = int(f.channels)
                    chunks = []
                    for buf in f:
                        # audioread yields little-endian int16 byte buffers.
                        chunks.append(np.frombuffer(buf, dtype='<i2'))
                    if not chunks:
                        return None, 0
                    pcm = np.concatenate(chunks)
                    arr = pcm.astype(np.float32) / 32768.0
                    if channels > 1:
                        arr = arr.reshape(-1, channels)
                    else:
                        arr = arr.reshape(-1, 1)
                    return arr, sr
            except Exception as e:
                audioread_err = e

        ffmpeg_err = None
        ffmpeg = shutil.which('ffmpeg')
        ffprobe = shutil.which('ffprobe')
        if ffmpeg and ffprobe:
            try:
                probe = subprocess.run(
                    [ffprobe, '-v', 'error', '-select_streams', 'a:0',
                     '-show_entries', 'stream=sample_rate,channels',
                     '-of', 'json', path],
                    check=True, capture_output=True, text=True)
                stream = json.loads(probe.stdout)['streams'][0]
                sr = int(stream['sample_rate'])
                channels = int(stream['channels'])
                raw = subprocess.run(
                    [ffmpeg, '-v', 'error', '-i', path, '-f', 'f32le',
                     '-acodec', 'pcm_f32le', '-'],
                    check=True, capture_output=True).stdout
                arr = np.frombuffer(raw, dtype='<f4')
                if channels > 1:
                    arr = arr.reshape(-1, channels)
                else:
                    arr = arr.reshape(-1, 1)
                return arr.copy(), sr
            except Exception as e:
                ffmpeg_err = e

        print('audio: no decoder available '
              f'(soundfile={sf_err!r}, audioread={audioread_err!r}, '
              f'ffmpeg={ffmpeg_err!r})')
        return None, 0

    # --- the PortAudio callback ---
    def _callback(self, outdata, frames, time_info, status):
        # Runs on the audio thread. Must be real-time-safe-ish: no I/O, no
        # allocations beyond what the PV does internally.
        with self._pv_lock:
            with self._lock:
                pv = self._pv
                gui_silent = self._silent
                volume = self._volume
                rate = self._rate
                block_chart_start = self._scheduled_chart_pos
                lead_in_seconds = self._lead_in_seconds
                prev_hw_pos = self._hw_pos
                prev_hw_wall = self._hw_wall
                prev_anchor_valid = self._dac_anchor_valid
            # Two sources of silence:
            #   1) GUI request (paused / scrubbing)        -> _silent=True
            #   2) Lead-in: source position hasn't reached the lead-in
            #      mark yet (block_chart_start < _lead_in_seconds).
            # Either way we emit zeros AND advance the source-frame
            # counter at `rate * frames` so chart-time stays accurate.
            in_lead_in = block_chart_start < lead_in_seconds
            silent = gui_silent or in_lead_in or pv is None
            cont = True
            if silent:
                outdata.fill(0.0)
                samples = None
            else:
                try:
                    samples, cont = pv.generate(frames)
                except Exception as e:
                    outdata.fill(0.0)
                    print(f'audio callback: {e}')
                    samples = None
        if samples is not None:
            if samples.shape[0] < frames:
                pad = np.zeros((frames - samples.shape[0], samples.shape[1]),
                               dtype=np.float32)
                samples = np.concatenate([samples, pad], axis=0)
            np.multiply(samples, volume, out=outdata)
        # Anchor the render clock to the PortAudio stream clock, not to
        # callback-invocation wall time. `outputBufferDacTime` is when the
        # FIRST sample of this block reaches the DAC; the end-of-block DAC
        # time is one output-block duration later. Pairing that with the chart
        # time represented by the end of this block yields a continuous clock
        # even though callbacks arrive only once per block.
        callback_now = float(getattr(time_info, 'currentTime',
                                     time_info.outputBufferDacTime))
        stream_time_now = self._safe_stream_time_locked()
        # Source-domain advance per block: `rate * frames / sr`. Same for
        # silent and non-silent blocks, so chart-time keeps ticking during
        # silence and the user resumes at the right place.
        block_chart_end = block_chart_start + (frames / self._sr) * rate
        dac_end = float(time_info.outputBufferDacTime) + (frames / self._sr)
        jump_at_callback_now = None
        if prev_anchor_valid:
            prev_now_t = prev_hw_pos + (callback_now - prev_hw_wall) * rate
            next_now_t = block_chart_end + (callback_now - dac_end) * rate
            jump_at_callback_now = next_now_t - prev_now_t
        with self._lock:
            self._scheduled_chart_pos = block_chart_end
            self._hw_pos = block_chart_end
            self._hw_wall = dac_end
            self._dac_anchor_valid = True
            self._chart_time = max(self._chart_time, block_chart_start)
            if not cont:
                self._ended = True
                self._chart_time = max(self._chart_time, block_chart_end)
            self._debug_log_locked('audio_callback', {
                'frames': int(frames),
                'block_chart_start': float(block_chart_start),
                'block_chart_end': float(block_chart_end),
                'callback_current_time': float(callback_now),
                'output_dac_time': float(time_info.outputBufferDacTime),
                'dac_end': float(dac_end),
                'stream_time': stream_time_now,
                'callback_to_dac_start': (
                    float(time_info.outputBufferDacTime) - float(callback_now)
                ),
                'callback_to_dac_end': float(dac_end - callback_now),
                'anchor_jump_at_callback_now': (
                    None if jump_at_callback_now is None
                    else float(jump_at_callback_now)
                ),
                'prev_hw_pos': float(prev_hw_pos),
                'prev_hw_wall': float(prev_hw_wall),
                'prev_anchor_valid': bool(prev_anchor_valid),
                'rate': float(rate),
                'silent': bool(silent),
                'gui_silent': bool(gui_silent),
                'in_lead_in': bool(in_lead_in),
                'cont': bool(cont),
                'hw_pos': float(self._hw_pos),
                'hw_wall': float(self._hw_wall),
                'anchor_valid': bool(self._dac_anchor_valid),
                'chart_time': float(self._chart_time),
            })

    # --- public API (matches the old engine) ---
    def set_volume(self, v: float) -> None:
        with self._lock:
            self._volume = float(v)

    def set_pitch_correct(self, on: bool) -> None:
        # Lock-free: `pv.set_pitch_correct` is a single bool write into a
        # PV attribute. Concurrent with `pv.generate()`, the worst case
        # is the in-flight block uses the old or new mode -- both are
        # legal output. Holding `_pv_lock` here makes the GUI thread wait
        # up to one full block of pv.generate() (~5-15 ms), which is the
        # entire reason rate / pitch nudges glitch the audio.
        with self._lock:
            self._pitch_correct = bool(on)
            pv = self._pv
        if pv is not None:
            pv.set_pitch_correct(on)

    @property
    def _pitch_correct_public(self) -> bool:
        return self._pitch_correct

    def current_chart_time(self) -> float:
        """Chart-time for the ChartClock to read.

        Source position in seconds, minus lead-in. Negative during lead-in
        (audio is emitting silence; source frame counter still advances so
        chart-time crosses 0 at an exact source-frame boundary). DAC-anchor
        refinement gives sub-frame accuracy when the stream is rolling.
        """
        with self._lock:
            audible = self._current_source_position_locked()
            t = audible - self._lead_in_seconds
            self._debug_log_locked('audio_current_chart_time', {
                'result': float(t),
                'audible_src': float(audible),
                'lead_in_seconds': float(self._lead_in_seconds),
                'silent': bool(self._silent),
                'rate': float(self._rate),
                'chart_time': float(self._chart_time),
                'scheduled_chart_pos': float(self._scheduled_chart_pos),
                'hw_pos': float(self._hw_pos),
                'hw_wall': float(self._hw_wall),
                'anchor_valid': bool(self._dac_anchor_valid),
                'stream_time': self._safe_stream_time_locked(),
            })
            return t

    def _current_source_position_locked(self) -> float:
        """Audible source-domain position in seconds (always >= 0).

        The callback-backed DAC anchor refines this below sample-grain
        when valid. When the anchor is stale (startup, just after seek),
        we fall back to the latest written `_chart_time`. The reading is
        clamped to be monotone non-decreasing -- the cull-space predictor
        absorbs sub-ms stream-clock jitter on its side, and `_chart_time`
        is the contract surface that downstream consumers (timing maps,
        seek logic) rely on being monotone.
        """
        if not self._dac_anchor_valid:
            return self._chart_time
        now = self._safe_stream_time_locked()
        if now is None:
            return self._chart_time
        t = float(self._hw_pos + (now - self._hw_wall) * self._rate)
        if t > self._chart_time:
            self._chart_time = t
        return self._chart_time

    # Back-compat: existing tests poke `_playing` to enable the
    # DAC-anchor math. In the new model the math is always live, so
    # _playing maps to "not silent".
    @property
    def _playing(self) -> bool:
        return not self._silent

    @_playing.setter
    def _playing(self, value: bool) -> None:
        self._silent = not bool(value)

    def _safe_stream_time_locked(self) -> float | None:
        stream = self._stream
        if stream is None:
            return None
        try:
            return float(stream.time)
        except Exception:
            return None

    def _debug_log_locked(self, subtype: str, payload: dict) -> None:
        if not _SV_DEBUG_LOGGER.enabled:
            return
        rec = {'type': 'audio', 'subtype': subtype}
        rec.update(payload)
        _SV_DEBUG_LOGGER.log(rec)

    # --- new surface: silence, seek, rate as orthogonal mutators ---

    def set_silent(self, silent: bool) -> None:
        """GUI-requested silence (paused, scrubbing). Independent of
        lead-in. While silent, the source-frame counter still advances
        so chart-time stays correct when the GUI un-silences."""
        with self._lock:
            self._silent = bool(silent)
            self._debug_log_locked('audio_set_silent', {
                'silent': bool(self._silent),
            })

    def set_rate(self, rate: float) -> None:
        """Update playback rate. Propagated to the PV so source frames
        per output frame stays in sync.

        Lock-free: `pv.set_rate` is a single float write. Concurrent with
        `pv.generate()`, the in-flight block reads either the old or new
        rate -- both are legal because `_step` recomputes the analysis
        hop from `self.rate` each iteration, so a mid-block transition
        is harmless. Holding `_pv_lock` here would serialize against
        `pv.generate()` (~5-15 ms per block), exactly the stall the user
        hears as choppiness on rate nudges.
        """
        rate = max(0.05, float(rate))
        with self._lock:
            rate_changed = abs(rate - self._rate) > 1e-3
            pv = self._pv
            if rate_changed:
                self._rate = rate
            self._debug_log_locked('audio_set_rate', {
                'rate': float(self._rate),
            })
        if rate_changed and pv is not None:
            pv.set_rate(rate)

    def seek_to_chart_time(self, chart_t: float) -> None:
        """Jump to chart-time `chart_t`. Negative values become a lead-in
        offset; the PV is parked at source-frame 0 and the engine emits
        silent frames until the source-frame counter passes the lead-in
        mark. Positive values seek the PV directly.
        """
        if not self.ready:
            return
        chart_t = float(chart_t)
        if chart_t < 0.0:
            lead_in_seconds = -chart_t
            audio_t = 0.0
        else:
            lead_in_seconds = 0.0
            audio_t = chart_t
        with self._pv_lock:
            with self._lock:
                pv = self._pv
                self._debug_log_locked('audio_seek_to_chart_time_enter', {
                    'chart_t': chart_t,
                    'audio_t': audio_t,
                    'lead_in_seconds': lead_in_seconds,
                })
            if pv is not None:
                pv.seek(audio_t)
            with self._lock:
                self._chart_time = audio_t
                self._scheduled_chart_pos = audio_t
                self._lead_in_seconds = lead_in_seconds
                self._dac_anchor_valid = False
                self._ended = False
                self._debug_log_locked('audio_seek_to_chart_time_exit', {
                    'chart_time': float(self._chart_time),
                    'scheduled_chart_pos': float(self._scheduled_chart_pos),
                    'lead_in_seconds': float(self._lead_in_seconds),
                })

    # --- back-compat wrappers for existing GUI / tests ----------------

    def set_state(self, t: float, rate: float, playing: bool) -> None:
        """Back-compat wrapper. New callers should use set_silent +
        set_rate + seek_to_chart_time directly. Maps the legacy
        (t, rate, playing) tuple onto the new orthogonal mutators:

          - playing=False  ->  silence requested.
          - playing=True   ->  no silence (lead-in still applies if
                               chart-time hasn't reached 0).
          - rate           ->  set_rate.

        Does NOT seek -- preserves the old semantic that per-tick
        set_state propagates pause/rate without flushing the PV. End-of-
        chart detection (t >= base_duration) still latches `_ended`.
        """
        if not self.ready:
            return
        self.set_rate(rate)
        if t >= self._base_duration:
            with self._lock:
                self._ended = True
                self._silent = True
        else:
            self.set_silent(not bool(playing))
            with self._lock:
                self._ended = False

    def seek(self, t: float) -> None:
        """Back-compat alias for seek_to_chart_time. New callers should
        use seek_to_chart_time directly."""
        self.seek_to_chart_time(t)

    def stop(self) -> None:
        with self._lock:
            self._silent = True
        self._stream_worker.close(self._stream)
        self._stream = None

    def prewarm_rates(self, rates) -> None:
        """No-op ; the streaming PV has no per-rate precompute."""
        return


def _monotonic() -> float:
    return time.monotonic()
