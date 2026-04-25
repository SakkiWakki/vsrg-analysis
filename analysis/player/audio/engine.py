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

from .audio_producer import AudioProducer, AudioRing
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
        # Audio runs in a child process by default. The child has its
        # own GIL, so Qt paints in this (parent) process can't starve
        # the producer thread -- the in-process path was vulnerable to
        # ~47 ms paints during SV-heavy charts, producing audible
        # ring underflows.
        #
        # Set `VSRG_AUDIO_BACKEND=inprocess` to fall back to the
        # legacy in-process path -- useful for tests that poke engine
        # internals (`_pv`, `_source`, `_callback`) directly.
        self._proc_client = None
        backend = os.environ.get('VSRG_AUDIO_BACKEND', '').lower()
        if backend != 'inprocess':
            self._init_process_backend(audio_path, volume, pitch_correct)
            if self._proc_client is not None:
                return
            # Process backend failed to start (e.g. sounddevice missing
            # in the child or the audio file couldn't be decoded). Fall
            # through to the in-process path so the user still gets
            # whatever audio works.
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
        # Rate of the block whose end-position the DAC anchor pins.
        # Used for extrapolation in `_current_source_position_locked`.
        # MUST come from the producer's per-block stamp, not from the
        # engine's live `_rate`: with the ring decoupling, the live
        # rate can be ~186 ms ahead of what's audibly playing, so
        # extrapolating with the live rate during a rate change
        # introduces a transient position error that compounds with
        # frame drops.
        self._hw_rate = 1.0
        self._scheduled_chart_pos = 0.0
        self._dac_anchor_valid = False
        # Lead-in: the engine emits silent frames until source frames
        # accumulated since the negative seek pass this threshold (in
        # seconds). Exposed via current_chart_time() = _chart_time -
        # _lead_in_seconds, so chart-time is negative during lead-in and
        # crosses 0 on the exact source-frame boundary.
        self._lead_in_seconds = 0.0
        # Diagnostic counters for PortAudio callback status flags
        # (underflow / overflow / priming / etc). A non-zero count means
        # we're missing the audio deadline; the last-string lets callers
        # see which flag fired most recently without grepping the log.
        self._cb_status_count = 0
        self._cb_last_status = ''
        # Distinct from PortAudio's status-flag count: the number of
        # times the callback found the ring empty and had to emit
        # silence. Should be near-zero in steady state; non-zero means
        # the producer can't keep up (rare under XMOD with the producer
        # decoupled from the GUI thread).
        self._cb_ring_underflow_count = 0
        # Producer-thread + decoupling ring. Keeps `pv.generate()` off
        # the PortAudio callback's deadline. Created in `_open_stream`
        # once we know `sr` and the source channel count.
        self._ring: AudioRing | None = None
        self._producer: AudioProducer | None = None
        # Scratch buffer the callback writes into when popping a block
        # from the ring. Allocated once at stream-open so the callback
        # never allocates (which would risk GIL contention).
        self._cb_scratch: np.ndarray | None = None

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

    def _init_process_backend(self, audio_path: str | None, volume: float,
                              pitch_correct: bool) -> None:
        """Out-of-process audio path. The child owns PortAudio and the
        phase vocoder; this engine is a thin shim that forwards
        commands and reads chart-time / status from shared memory.

        Process-backed engines do NOT populate the in-process state
        (`_pv`, `_source`, `_stream`, locks) -- those exist only in
        the child. Tests that poke those internals must use the
        default in-process backend.
        """
        from .audio_process import AudioProcessClient

        # Defaults matching the in-process surface so callers that
        # `getattr` these don't trip up.
        self.ready = False
        self._silent = True
        self._rate = 1.0
        self._volume = float(volume)
        self._pitch_correct = bool(pitch_correct)
        self._sr = 44100
        self._base_duration = 0.0
        self._ended = False
        self._chart_time = 0.0
        self._lead_in_seconds = 0.0
        self._cb_status_count = 0
        self._cb_last_status = ''
        self._cb_ring_underflow_count = 0
        # Stub the in-process state so tests/back-compat that read
        # them don't AttributeError. None of these are functional under
        # the process backend.
        self._lock = threading.Lock()
        self._pv_lock = threading.Lock()
        self._source = None
        self._pv = None
        self._stream = None
        self._stream_worker = StreamWorker()
        self._ring = None
        self._producer = None
        self._cb_scratch = None
        self._dac_anchor_valid = False
        self._hw_pos = 0.0
        self._hw_wall = 0.0
        self._hw_rate = 1.0
        self._scheduled_chart_pos = 0.0

        if not audio_path or not os.path.exists(audio_path):
            return
        try:
            self._proc_client = AudioProcessClient(
                audio_path, pitch_correct=pitch_correct, volume=volume,
            )
        except Exception as e:
            print(f'audio: process backend failed to start: {e}')
            self._proc_client = None
            return
        self.ready = bool(self._proc_client.ready)
        if self.ready:
            self._base_duration = float(self._proc_client.base_duration)

    def _open_stream(self) -> None:
        """Create + start the PortAudio OutputStream on a dedicated
        worker thread so it doesn't share affinity / GIL pressure with
        the Qt main thread that constructed the engine. See
        `stream_worker.StreamWorker` for the lifecycle contract.

        Also spins up the producer thread + audio ring so the callback
        can be a pure memcpy. Capacity is 32768 frames (~743 ms at
        44.1 kHz / 64 blocks of 512). That's deliberately deep: the
        producer thread shares the GIL with the GUI thread, so a 47 ms
        Qt paint completely starves the producer for that window. With
        743 ms of pre-rendered audio in the ring, the consumer can
        survive multi-frame paint hitches without underrunning. Memory
        cost is `32768 * channels * 4` bytes (~256 KB stereo) -- a
        cheap insurance policy.

        The ring-depth-vs-rate-change trade: a deeper ring means the
        audible audio can lag the engine's live `_rate` by up to one
        ring depth. The `_hw_rate` field on the engine carries each
        block's rate-at-render-time so DAC-anchor extrapolation
        remains correct.
        """
        block_size = 512
        capacity = 32768  # power of two, multiple of block_size
        channels = self._source.src_channels
        self._ring = AudioRing(capacity, channels, block_size=block_size)
        self._cb_scratch = np.zeros((block_size, channels), dtype=np.float32)
        self._producer = AudioProducer(self._pv, self._ring, self._sr)
        # Mirror current engine state into the producer before starting,
        # so the first generated block carries the right rate/silent.
        self._producer.set_silent(self._silent)
        self._producer.set_rate(self._rate)
        self._producer.start()

        def make_stream():
            stream = _sd.OutputStream(
                samplerate=self._sr,
                channels=channels,
                dtype='float32',
                blocksize=block_size,
                callback=self._callback,
            )
            stream.start()
            return stream

        stream, err = self._stream_worker.open(make_stream)
        if err is not None:
            print(f'audio: failed to open output stream: {err}')
            self._stream = None
            self._producer.stop()
            self._producer = None
            self._ring = None
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
        # Runs on PortAudio's audio thread. Hard real-time deadline:
        # `frames / sr` seconds (~11.6 ms at 512/44.1k). The producer
        # thread does all `pv.generate()` work ahead of time and stamps
        # blocks into `self._ring`; this callback's job is to:
        #   1. memcpy one block of ring frames into `outdata`
        #   2. update the DAC-clock anchor from PortAudio's `time_info`
        #      and the producer's chart-end stamp
        #   3. emit silence + count an underflow if the ring drained
        # Steps 1+3 are a few microseconds of numpy. No PV, no locks
        # held during compute.
        if status:
            with self._lock:
                self._cb_status_count += 1
                self._cb_last_status = str(status)
                self._debug_log_locked('audio_status', {
                    'status': str(status),
                    'frames': int(frames),
                    'rate': float(self._rate),
                    'silent': bool(self._silent),
                    'count': int(self._cb_status_count),
                })

        ring = self._ring
        producer = self._producer
        scratch = self._cb_scratch
        # Pre-ring construction (very early init / failure paths) -- be
        # safe and emit silence.
        if ring is None or producer is None or scratch is None:
            outdata.fill(0.0)
            return

        with self._lock:
            volume = self._volume

        stamp = ring.read_block(scratch)
        if stamp is None:
            # Producer fell behind. Emit silence; the DAC anchor still
            # advances by one block at the engine's current rate so
            # chart-time keeps ticking (otherwise the playhead would
            # freeze for one frame and the renderer would see a jump).
            outdata.fill(0.0)
            with self._lock:
                self._cb_ring_underflow_count += 1
                rate_est = self._rate
                anchor_chart_end = self._scheduled_chart_pos \
                    + (frames / self._sr) * rate_est
        else:
            np.multiply(scratch, volume, out=outdata)
            rate_est = stamp.rate
            anchor_chart_end = stamp.chart_end
        # Wake the producer in case it was sleeping on the high-water
        # mark. Cheap; just sets an Event.
        producer.signal_drain()

        # DAC anchor: pair the chart-end of the just-played block with
        # the DAC time of its last sample. Same math as before, but the
        # chart-end now comes from the producer's stamp instead of
        # being recomputed locally from rate.
        callback_now = float(getattr(time_info, 'currentTime',
                                     time_info.outputBufferDacTime))
        dac_end = float(time_info.outputBufferDacTime) + (frames / self._sr)

        with self._lock:
            self._scheduled_chart_pos = anchor_chart_end
            self._hw_pos = anchor_chart_end
            self._hw_wall = dac_end
            self._hw_rate = rate_est
            self._dac_anchor_valid = True
            # Audible chart-time at the START of this block is anchor
            # minus the block's duration in source-time at the rate the
            # block was rendered with.
            block_dt = (frames / self._sr) * rate_est
            block_chart_start = anchor_chart_end - block_dt
            self._chart_time = max(self._chart_time, block_chart_start)
            if stamp is not None and stamp.ended:
                self._ended = True
                self._chart_time = max(self._chart_time, anchor_chart_end)
            self._debug_log_locked('audio_callback', {
                'frames': int(frames),
                'block_chart_end': float(anchor_chart_end),
                'callback_current_time': float(callback_now),
                'output_dac_time': float(time_info.outputBufferDacTime),
                'dac_end': float(dac_end),
                'rate': float(rate_est),
                'ring_underflow': stamp is None,
                'stamp_silent': bool(stamp.silent) if stamp else None,
                'hw_pos': float(self._hw_pos),
                'hw_wall': float(self._hw_wall),
                'chart_time': float(self._chart_time),
            })

    # --- public API (matches the old engine) ---
    def set_volume(self, v: float) -> None:
        if self._proc_client is not None:
            self._proc_client.set_volume(v)
            self._volume = float(v)
            return
        with self._lock:
            self._volume = float(v)

    def callback_status_snapshot(self) -> tuple[int, str]:
        """Return `(count, last_status)` of PortAudio status flags seen
        in `_callback`, summed with ring-underflow events. The string
        annotates which side reported -- `'underflow'` (PortAudio) or
        `'ring underflow'` (producer fell behind). Callers can poll
        this from the GUI thread (read-only), no lock needed because
        both fields are independent reads of plain Python objects.

        The string also carries a rough ring-fill gauge so we can
        diagnose at a glance: even when the totals are zero, a ring
        that's chronically near-empty means the producer is barely
        keeping up and the next GIL hitch will cause a fresh underflow."""
        if self._proc_client is not None:
            return self._proc_client.callback_status_snapshot()
        pa_n = self._cb_status_count
        ring_n = self._cb_ring_underflow_count
        ring = self._ring
        gauge = ''
        if ring is not None and ring._cap:
            fill_pct = int(100 * ring.readable_frames() / ring._cap)
            gauge = f' fill={fill_pct}%'
        if ring_n and not pa_n:
            return ring_n, f'ring underflow{gauge}'
        if pa_n and not ring_n:
            return pa_n, f'{self._cb_last_status}{gauge}'
        if pa_n or ring_n:
            total = pa_n + ring_n
            label = 'ring underflow' if ring_n else self._cb_last_status
            return total, f'{label}{gauge}'
        return 0, gauge.lstrip()

    def set_pitch_correct(self, on: bool) -> None:
        # Routed through the producer (which owns the PV). The producer
        # writes the bool directly into the PV without waiting for its
        # next tick, so the change takes effect on the very next
        # `pv.generate()` call.
        if self._proc_client is not None:
            self._proc_client.set_pitch_correct(on)
            self._pitch_correct = bool(on)
            return
        with self._lock:
            self._pitch_correct = bool(on)
        if self._producer is not None:
            self._producer.set_pitch_correct(on)

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
        if self._proc_client is not None:
            return self._proc_client.current_chart_time()
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

        Extrapolation uses `_hw_rate` (the rate of the block whose end
        the anchor pins) NOT the engine's live `_rate`. With the
        producer ring decoupled, `_rate` can run up to one ring depth
        (~186 ms) ahead of what's audibly playing, so extrapolating
        with the live rate during a rate change introduces a position
        error that compounds when frames drop.
        """
        if not self._dac_anchor_valid:
            return self._chart_time
        now = self._safe_stream_time_locked()
        if now is None:
            return self._chart_time
        t = float(self._hw_pos + (now - self._hw_wall) * self._hw_rate)
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
        if self._proc_client is not None:
            self._silent = bool(silent)
            self._proc_client.set_silent(silent)
            return
        with self._lock:
            self._silent = bool(silent)
            self._debug_log_locked('audio_set_silent', {
                'silent': bool(self._silent),
            })
        if self._producer is not None:
            self._producer.set_silent(silent)

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
        if self._proc_client is not None:
            if abs(rate - self._rate) > 1e-3:
                self._rate = rate
                self._proc_client.set_rate(rate)
            return
        with self._lock:
            rate_changed = abs(rate - self._rate) > 1e-3
            if rate_changed:
                self._rate = rate
            self._debug_log_locked('audio_set_rate', {
                'rate': float(self._rate),
            })
        # Producer owns the PV exclusively; route the rate change
        # through it so the next-block render uses the new rate AND
        # the stamp the callback sees carries the right value.
        if rate_changed and self._producer is not None:
            self._producer.set_rate(rate)

    def seek_to_chart_time(self, chart_t: float) -> None:
        """Jump to chart-time `chart_t`. Negative values become a lead-in
        offset; the PV is parked at source-frame 0 and the engine emits
        silent frames until the source-frame counter passes the lead-in
        mark. Positive values seek the PV directly.
        """
        if not self.ready:
            return
        chart_t = float(chart_t)
        if self._proc_client is not None:
            self._proc_client.seek_to_chart_time(chart_t)
            # Mirror lead-in / chart-time so engine-side getters that
            # don't go through the proc client (e.g. tests) read
            # something sensible.
            if chart_t < 0.0:
                self._lead_in_seconds = -chart_t
                self._chart_time = 0.0
            else:
                self._lead_in_seconds = 0.0
                self._chart_time = chart_t
            self._ended = False
            return
        if chart_t < 0.0:
            lead_in_seconds = -chart_t
            audio_t = 0.0
        else:
            lead_in_seconds = 0.0
            audio_t = chart_t
        with self._lock:
            self._chart_time = audio_t
            self._scheduled_chart_pos = audio_t
            self._lead_in_seconds = lead_in_seconds
            self._dac_anchor_valid = False
            self._ended = False
            self._debug_log_locked('audio_seek_to_chart_time', {
                'chart_t': chart_t,
                'audio_t': audio_t,
                'lead_in_seconds': lead_in_seconds,
            })
        # Producer owns the PV; queue the seek there. The producer
        # resets the ring (so the consumer sees silence until refill)
        # and calls `pv.seek(audio_t)` on its next tick.
        if self._producer is not None:
            self._producer.request_seek(audio_t, lead_in_seconds)

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
        if self._proc_client is not None:
            self._proc_client.stop()
            self._proc_client = None
            return
        self._stream_worker.close(self._stream)
        self._stream = None
        if self._producer is not None:
            self._producer.stop()
            self._producer = None
        self._ring = None

    def prewarm_rates(self, rates) -> None:
        """No-op ; the streaming PV has no per-rate precompute."""
        return


def _monotonic() -> float:
    return time.monotonic()
