"""Public `AudioEngine`: owns decoding, the PortAudio output stream, and
mediates between the GUI thread (set_state/seek/volume) and the audio
callback thread via a mutex. The heavy DSP lives in StreamingPhaseVocoder."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
import time

import numpy as np

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
        self._source: WaveSource | None = None
        self._pv: StreamingPhaseVocoder | None = None
        self._stream = None
        self._chart_time = 0.0       # last known chart time at set_state
        self._rate = 1.0
        # Audio-position smoothing: `_pv.source_time` advances in hop-sized
        # jumps (the PV only updates it when the callback asks it to
        # generate samples). Raw reads give the render thread a stepped,
        # jittery t. We snapshot (pv_time, wall_time_at_snapshot) each time
        # the callback runs, then extrapolate between snapshots for smooth
        # chart-time output.
        self._anchor_src_time = 0.0
        self._anchor_wall = 0.0

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
        with self._lock:
            pv = self._pv
            playing = self._playing
            volume = self._volume
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
        # Snapshot the PV's fresh source position against wall-clock so the
        # render thread's `current_chart_time()` can interpolate smoothly
        # between callbacks instead of stepping in hop-sized jumps.
        with self._lock:
            self._anchor_src_time = float(pv.source_time)
            self._anchor_wall = time.monotonic()
            if not cont:
                self._ended = True

    # --- public API (matches the old engine) ---
    def set_volume(self, v: float) -> None:
        with self._lock:
            self._volume = float(v)

    def set_pitch_correct(self, on: bool) -> None:
        with self._lock:
            self._pitch_correct = bool(on)
            if self._pv is not None:
                self._pv.set_pitch_correct(on)

    @property
    def _pitch_correct_public(self) -> bool:
        return self._pitch_correct

    def current_chart_time(self) -> float:
        """Audio-master chart time for the ChartClock to read. Safe to call
        from the render thread; takes the engine lock briefly.

        Returns an interpolated time: the PV's source_time (stepped in
        hop-sized jumps) is anchored against wall-clock at each callback,
        and we add the wall-clock delta since that anchor. This gives the
        renderer a smoothly-advancing clock instead of a visibly stepped
        one, while still matching the PV's actual position on average."""
        with self._lock:
            if self._pv is None:
                if self._source is not None:
                    return float(self._source.pos) / float(self._source.sr)
                return self._chart_time
            if not self._playing:
                # When paused the callback isn't updating the anchor, so
                # return the anchor directly (or the raw PV time if we
                # never got a callback yet).
                if self._anchor_wall == 0.0:
                    return float(self._pv.source_time)
                return self._anchor_src_time
            rate = self._rate
            elapsed = time.monotonic() - self._anchor_wall
            extrapolated = self._anchor_src_time + elapsed * rate
            # Clamp to the raw PV time on the upper side so we never get
            # ahead of what's actually been produced — if the callback
            # stalls longer than expected, we'd otherwise drift forward.
            pv_now = float(self._pv.source_time)
            if extrapolated > pv_now + rate * 0.05:
                # More than one 50ms rate-step ahead of reality: cap.
                return pv_now + rate * 0.05
            return extrapolated

    def set_state(self, t: float, rate: float, playing: bool) -> None:
        """Apply chart-clock intent to the audio engine.

        Does NOT seek by itself: with audio as the chart clock master, the
        per-tick `set_state` call only propagates pause/rate changes and
        end-of-chart detection. The PV's source_time IS the playhead, so
        re-seeking it every tick would just undo its own progress. Use
        `seek()` for explicit jumps (unpause, scrub release, restart)."""
        if not self.ready:
            return
        rate = max(0.05, float(rate))
        with self._lock:
            if abs(rate - self._rate) > 1e-3:
                self._rate = rate
                if self._pv is not None:
                    self._pv.set_rate(rate)
            if not playing:
                self._playing = False
                self._chart_time = t
                self._ended = False
                return
            if t >= self._base_duration:
                self._playing = False
                self._ended = True
                return
            self._playing = True
            self._chart_time = t
            self._ended = False

    def seek(self, t: float) -> None:
        """Explicitly seek the PV to chart time `t`. Called on scrub
        release, restart, or any user-visible jump — the OLA buffer flush
        is the price of a non-contiguous seek."""
        if not self.ready:
            return
        with self._lock:
            if self._pv is not None:
                self._pv.seek(float(t))
            self._chart_time = float(t)
            self._ended = False
            # Rebase the interpolation anchor so current_chart_time() starts
            # extrapolating from the new position instead of the pre-seek one.
            self._anchor_src_time = float(t)
            self._anchor_wall = time.monotonic()

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
