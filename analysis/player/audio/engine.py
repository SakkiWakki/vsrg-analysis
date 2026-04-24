"""Public `AudioEngine`: owns decoding, the PortAudio output stream, and
mediates between the GUI thread (set_state/seek/volume) and the audio
callback thread via a mutex. The heavy DSP lives in StreamingPhaseVocoder."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time

import numpy as np

from analysis.player.sv_debug import LOGGER as _SV_DEBUG_LOGGER

from .phase_vocoder import StreamingPhaseVocoder
from .source import WaveSource

# sounddevice is the PortAudio binding we drive the output stream with.
# Imported lazily so the module still imports in test environments that
# don't have an audio device.
try:
    import sounddevice as _sd
except Exception:
    _sd = None

# Decoding stack: soundfile (libsndfile — wav/flac/ogg/opus) as the primary,
# audioread (ffmpeg/gstreamer wrapper — mp3/m4a/anything) as the fallback for
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
    # Allow tiny negative corrections from the hardware clock to avoid
    # visible freeze-then-jump motion when callback/stream timing jitters.
    # Larger negative jumps are still clamped so seeks/resume never backstep.
    _SMALL_BACKSTEP_TOLERANCE_S = 0.003

    def __init__(self, audio_path: str | None, volume: float = 0.5,
                 pitch_correct: bool = True):
        self.ready = False
        self._volume = float(volume)
        self._pitch_correct = bool(pitch_correct)
        self._sr = 44100
        self._base_duration = 0.0
        self._ended = False
        self._playing = False
        # Engine state kept under a lock because the audio callback (on the
        # PortAudio thread) and set_state/seek (on the GUI thread) both
        # touch it.
        self._lock = threading.Lock()
        # Serialize all access to the phase vocoder/source state. The audio
        # callback calls `generate()` while the GUI thread may concurrently call
        # `seek()` / `set_rate()` / `set_pitch_correct()`. Those mutate the same
        # internal buffers and cursors and must never race.
        self._pv_lock = threading.Lock()
        self._source: WaveSource | None = None
        self._pv: StreamingPhaseVocoder | None = None
        self._stream = None
        # Single authoritative chart-time value. While paused, seeking, or
        # waiting for the first callback after a discontinuity, reads return
        # this value directly. Once a callback-backed DAC anchor exists, reads
        # advance this value from the hardware clock but never step backward.
        self._chart_time = 0.0
        self._rate = 1.0
        # PortAudio-backed anchor: `hw_pos` chart-time is audible at DAC clock
        # time `hw_wall`. This anchor is only valid after a callback has
        # provided real DAC timestamps for queued audio.
        self._hw_pos = 0.0
        self._hw_wall = 0.0
        self._scheduled_chart_pos = 0.0
        self._dac_anchor_valid = False

        if not audio_path or not os.path.exists(audio_path):
            return
        if _sd is None:
            print('audio: sounddevice not installed — no audio playback')
            return

        samples, sr = self._decode(audio_path)
        if samples is None:
            return
        self._sr = int(sr)
        self._source = WaveSource(samples, sr)
        self._base_duration = self._source.duration
        self._pv = StreamingPhaseVocoder(self._source, rate=1.0,
                                          pitch_correct=self._pitch_correct)
        try:
            self._stream = _sd.OutputStream(
                samplerate=self._sr,
                channels=self._source.src_channels,
                dtype='float32',
                blocksize=512,
                callback=self._callback,
            )
            self._stream.start()
            self.ready = True
        except Exception as e:
            print(f'audio: failed to open output stream: {e}')
            self._stream = None

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
                # Fall through to audioread — libsndfile doesn't do mp3/m4a
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
                playing = self._playing
                volume = self._volume
                rate = self._rate
                block_chart_start = self._scheduled_chart_pos
                prev_hw_pos = self._hw_pos
                prev_hw_wall = self._hw_wall
                prev_anchor_valid = self._dac_anchor_valid
            if pv is None or not playing:
                outdata.fill(0.0)
                if pv is None or not playing:
                    # report ended when we're past the end of the source
                    return
                return
            try:
                samples, cont = pv.generate(frames)
            except Exception as e:
                outdata.fill(0.0)
                print(f'audio callback: {e}')
                return
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
                'playing': bool(self._playing),
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
        with self._pv_lock:
            with self._lock:
                self._pitch_correct = bool(on)
                pv = self._pv
            if pv is not None:
                pv.set_pitch_correct(on)

    @property
    def _pitch_correct_public(self) -> bool:
        return self._pitch_correct

    def current_chart_time(self) -> float:
        """Audio-master chart time for the ChartClock to read.

        The only live clock is the callback-backed DAC anchor. Until a callback
        has provided that anchor (startup, immediately after seek, pause->play),
        reads return the last explicit chart-time written by the transport.
        """
        with self._lock:
            t = self._current_chart_time_locked()
            self._debug_log_locked('audio_current_chart_time', {
                'result': float(t),
                'playing': bool(self._playing),
                'rate': float(self._rate),
                'chart_time': float(self._chart_time),
                'scheduled_chart_pos': float(self._scheduled_chart_pos),
                'hw_pos': float(self._hw_pos),
                'hw_wall': float(self._hw_wall),
                'anchor_valid': bool(self._dac_anchor_valid),
                'stream_time': self._safe_stream_time_locked(),
            })
            return t

    def _current_chart_time_locked(self) -> float:
        if not self._playing:
            return self._chart_time
        if not self._dac_anchor_valid:
            return self._chart_time
        now = self._safe_stream_time_locked()
        if now is None:
            return self._chart_time
        t = self._hw_pos + (now - self._hw_wall) * self._rate
        t = float(t)
        if t >= self._chart_time:
            self._chart_time = t
            return self._chart_time

        # Tiny regressions happen around callback-anchor handoff and host-time
        # quantisation. Accepting a very small backstep avoids plateaus that
        # render as jitter. Large backsteps remain clamped away.
        backstep = self._chart_time - t
        if backstep <= self._SMALL_BACKSTEP_TOLERANCE_S:
            self._chart_time = t
        return self._chart_time

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

    def set_state(self, t: float, rate: float, playing: bool) -> None:
        """Apply chart-clock intent to the audio engine.

        Does NOT seek by itself: with audio as the chart clock master, the
        per-tick `set_state` call only propagates pause/rate changes and
        end-of-chart detection. The render clock comes from the stream's DAC
        timeline, so re-seeking the PV every tick would just flush its OLA
        buffer and chop the audio. Use `seek()` for explicit jumps (unpause,
        scrub release, restart)."""
        if not self.ready:
            return
        rate = max(0.05, float(rate))
        with self._pv_lock:
            with self._lock:
                cur = self._current_chart_time_locked()
                self._debug_log_locked('audio_set_state_enter', {
                    'requested_t': float(t),
                    'requested_rate': float(rate),
                    'requested_playing': bool(playing),
                    'current': float(cur),
                    'playing': bool(self._playing),
                    'rate': float(self._rate),
                    'chart_time': float(self._chart_time),
                    'scheduled_chart_pos': float(self._scheduled_chart_pos),
                    'hw_pos': float(self._hw_pos),
                    'hw_wall': float(self._hw_wall),
                    'anchor_valid': bool(self._dac_anchor_valid),
                    'stream_time': self._safe_stream_time_locked(),
                })
                pv = self._pv
                rate_changed = abs(rate - self._rate) > 1e-3
                was_playing = self._playing
            if rate_changed and pv is not None:
                pv.set_rate(rate)
            with self._lock:
                if rate_changed:
                    self._rate = rate
                if not playing:
                    self._playing = False
                    self._chart_time = cur
                    self._scheduled_chart_pos = cur
                    self._dac_anchor_valid = False
                    self._ended = False
                elif t >= self._base_duration:
                    self._playing = False
                    self._chart_time = float(t)
                    self._scheduled_chart_pos = float(t)
                    self._dac_anchor_valid = False
                    self._ended = True
                else:
                    if not was_playing:
                        self._chart_time = float(t)
                        self._scheduled_chart_pos = float(t)
                        self._dac_anchor_valid = False
                    self._playing = True
                    self._ended = False
                self._debug_log_locked('audio_set_state_exit', {
                    'chart_time': float(self._chart_time),
                    'playing': bool(self._playing),
                    'rate': float(self._rate),
                    'ended': bool(self._ended),
                    'scheduled_chart_pos': float(self._scheduled_chart_pos),
                    'hw_pos': float(self._hw_pos),
                    'hw_wall': float(self._hw_wall),
                    'anchor_valid': bool(self._dac_anchor_valid),
                    'stream_time': self._safe_stream_time_locked(),
                })

    def seek(self, t: float) -> None:
        """Explicitly seek the PV to chart time `t`. Called on scrub
        release, restart, or any user-visible jump — the OLA buffer flush
        is the price of a non-contiguous seek."""
        if not self.ready:
            return
        with self._pv_lock:
            with self._lock:
                self._debug_log_locked('audio_seek_enter', {
                    'requested_t': float(t),
                    'playing': bool(self._playing),
                    'rate': float(self._rate),
                    'chart_time': float(self._chart_time),
                    'scheduled_chart_pos': float(self._scheduled_chart_pos),
                    'hw_pos': float(self._hw_pos),
                    'hw_wall': float(self._hw_wall),
                    'anchor_valid': bool(self._dac_anchor_valid),
                    'stream_time': self._safe_stream_time_locked(),
                })
                pv = self._pv
            if pv is not None:
                pv.seek(float(t))
            with self._lock:
                self._chart_time = float(t)
                self._ended = False
                self._scheduled_chart_pos = float(t)
                self._dac_anchor_valid = False
                self._debug_log_locked('audio_seek_exit', {
                    'chart_time': float(self._chart_time),
                    'scheduled_chart_pos': float(self._scheduled_chart_pos),
                    'hw_pos': float(self._hw_pos),
                    'hw_wall': float(self._hw_wall),
                    'anchor_valid': bool(self._dac_anchor_valid),
                    'stream_time': self._safe_stream_time_locked(),
                })

    def stop(self) -> None:
        with self._lock:
            self._playing = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None

    def prewarm_rates(self, rates) -> None:
        """No-op — the streaming PV has no per-rate precompute."""
        return


def _monotonic() -> float:
    return time.monotonic()
