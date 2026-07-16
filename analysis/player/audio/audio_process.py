"""Out-of-process audio backend.

The in-process producer-thread design (`audio_producer.py`) reduces
underflows but can't fully eliminate them: when the Qt main thread
holds the GIL through a long native paint, the producer thread cannot
run, period. `sys.setswitchinterval` does not help because it only
fires between Python bytecode instructions.

This module runs the entire audio path -- PortAudio, the phase vocoder,
the producer ring -- inside a child process. The child has its own
GIL, so Qt paints in the parent cannot starve it. The parent process
is reduced to:

    GUI thread  ──control msg──▶  child process
                ◀──status snapshot via shared memory

The child re-uses ``AudioProducer`` + ``AudioRing`` from the in-process
version unchanged; from the child's point of view it's just a normal
Python process running the audio engine.

# Cross-platform notes

We use ``multiprocessing.get_context('spawn')`` explicitly. Spawn:
  - works on Linux, macOS, Windows
  - does NOT inherit Qt state from the parent (fork would, and Qt is
    not fork-safe)
  - costs ~0.5-1.0 s of startup time, paid once at engine init.

The child entry point is a module-level function so it pickles cleanly
under spawn. We pass the audio path + initial pitch_correct flag as
plain strings/ints; the child does its own decode + PortAudio open.

# Status surface

The parent reads chart-time via a small lock-free shared double array.
The GUI polls this every frame, so reads must never wait on the audio
child. A snapshot can see fields from adjacent callback writes; that is
acceptable for HUD/playhead display because the next frame corrects it.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
import traceback
from dataclasses import dataclass


# ── shared-state field layout ──────────────────────────────────────

# Indices into the shared Array('d', _NUM_STATUS_FIELDS). Keep these
# in sync between parent (`AudioProcessClient`) and child (`_run_audio_child`).
_F_HW_POS               = 0   # chart-time at the END of the most recent block
_F_HW_WALL              = 1   # PortAudio DAC time matching _F_HW_POS
_F_HW_RATE              = 2   # rate the most recent block was rendered at
_F_DAC_ANCHOR_VALID     = 3   # 0.0 or 1.0
_F_LEAD_IN_SECONDS      = 4   # producer-side lead-in, mirrors engine state
_F_BASE_DURATION        = 5   # source duration in seconds (filled at startup)
_F_READY                = 6   # 0.0 / 1.0; set after stream is open
_F_ENDED                = 7   # 0.0 / 1.0; set when source runs out
_F_CB_STATUS_COUNT      = 8   # PortAudio callback status flag count
_F_CB_RING_UNDERFLOW    = 9   # ring-empty events seen by callback
_F_RING_FILL_FRAMES     = 10  # readable frames in the producer ring
_F_RING_CAPACITY_FRAMES = 11  # ring capacity (constant after init)
_F_HW_MONO              = 12  # time.monotonic() at which _F_HW_POS plays
_F_SEEK_GEN             = 13  # count of seek commands the child processed
_NUM_STATUS_FIELDS      = 14


# ── command protocol ───────────────────────────────────────────────

# Commands flow parent -> child via a Queue. Each command is a
# (op, payload) tuple of plain Python types; see `_run_audio_child`
# for the dispatch.
_OP_SET_VOLUME        = 'volume'
_OP_SET_SILENT        = 'silent'
_OP_SET_RATE          = 'rate'
_OP_SET_PITCH_CORRECT = 'pitch_correct'
_OP_SEEK              = 'seek'
_OP_PREWARM           = 'prewarm'
_OP_STOP              = 'stop'


@dataclass
class AudioProcessConfig:
    """Everything the child needs to bootstrap. Picklable."""
    audio_path: str
    pitch_correct: bool = True
    volume: float = 0.5
    block_size: int = 512
    # 512 frames minimal block size and overflows in the single 
    # digits on stress tests. So use 1024 for some headroom.
    ring_capacity: int = 1024


# ── child entry point ──────────────────────────────────────────────

def _run_audio_child(config: AudioProcessConfig,
                     cmd_queue: 'multiprocessing.Queue',
                     status: 'multiprocessing.Array',
                     last_status_str: 'multiprocessing.Array') -> None:
    """Child-process main loop.

    Owns: audio decode, PortAudio stream, phase vocoder, producer ring.
    Communication:
      - reads commands from `cmd_queue`
      - writes status into `status` (shared doubles)
      - writes the most recent PortAudio status string into
        `last_status_str` (a `Array('c', N)` of bytes; truncated to
        fit). The parent decodes it for the HUD.

    Errors during decode / open are non-fatal: we set _F_READY=0 and
    keep the queue poll loop alive so the parent can still issue
    `stop`. The parent treats _F_READY=0 as "no audio".
    """
    # Imports are inside the child so the parent doesn't pay for
    # decoder libraries it doesn't need; also keeps the child
    # self-contained when launched under `spawn`.
    import numpy as np
    from analysis.player.audio.audio_producer import AudioProducer, AudioRing
    from analysis.player.audio.phase_vocoder import StreamingPhaseVocoder
    from analysis.player.audio.source import WaveSource

    # Decode the audio file. The child does this on its own so the
    # parent never has to ship a multi-megabyte numpy array through
    # the pipe.
    try:
        import sounddevice as sd
    except Exception as e:
        _publish_error(status, last_status_str, f'no sounddevice: {e}')
        _drain_until_stop(cmd_queue)
        return

    samples, sr = _decode_audio(config.audio_path, last_status_str)
    if samples is None:
        _publish_error(status, last_status_str, 'decode failed')
        _drain_until_stop(cmd_queue)
        return

    source = WaveSource(samples, sr)
    pv = StreamingPhaseVocoder(source, rate=1.0,
                                pitch_correct=config.pitch_correct)
    ring = AudioRing(config.ring_capacity, source.src_channels,
                     block_size=config.block_size)
    producer = AudioProducer(pv, ring, sr=sr)
    producer.set_silent(True)
    producer.start()

    state = _ChildState(
        status=status,
        last_status_str=last_status_str,
        ring=ring,
        sr=sr,
        block_size=config.block_size,
        volume=float(config.volume),
        rate=1.0,
        chart_time=0.0,
        scheduled_chart_pos=0.0,
        lead_in_seconds=0.0,
    )
    status[_F_BASE_DURATION] = source.duration
    status[_F_RING_CAPACITY_FRAMES] = float(config.ring_capacity)

    scratch = np.zeros((config.block_size, source.src_channels),
                       dtype=np.float32)

    def _callback(outdata, frames, time_info, pa_status):
        if pa_status:
            state.cb_status_count += 1
            text = str(pa_status)
            _write_status_str(last_status_str, text)
            status[_F_CB_STATUS_COUNT] = float(state.cb_status_count)
        stamp = ring.read_block(scratch)
        if stamp is None:
            outdata.fill(0.0)
            state.cb_ring_underflow += 1
            status[_F_CB_RING_UNDERFLOW] = float(state.cb_ring_underflow)
            rate_est = state.rate
            anchor_chart_end = state.scheduled_chart_pos \
                + (frames / state.sr) * rate_est
        else:
            np.multiply(scratch, state.volume, out=outdata)
            rate_est = stamp.rate
            anchor_chart_end = stamp.chart_end
        producer.signal_drain()
        # DAC anchor.
        block_period = frames / state.sr
        dac_end = float(time_info.outputBufferDacTime) + block_period
        # The parent can't read this stream's PortAudio clock, so publish
        # the anchor's deadline in time.monotonic()'s timebase instead
        # (CLOCK_MONOTONIC / QPC tick system-wide, shared across
        # processes). Some host APIs report zero time_info fields; fall
        # back to "one block from now".
        dac_now = float(getattr(time_info, 'currentTime', 0.0) or 0.0)
        if dac_now > 0.0 and dac_end > dac_now:
            mono_end = time.monotonic() + (dac_end - dac_now)
        else:
            mono_end = time.monotonic() + block_period
        block_dt = block_period * rate_est
        block_chart_start = anchor_chart_end - block_dt
        if block_chart_start > state.chart_time:
            state.chart_time = block_chart_start
        if stamp is not None and stamp.ended:
            status[_F_ENDED] = 1.0
            if anchor_chart_end > state.chart_time:
                state.chart_time = anchor_chart_end
        state.scheduled_chart_pos = anchor_chart_end
        # Publish: write all DAC-anchor fields together. There's no
        # producer-side reader of these (the parent is read-only), so
        # tear-free isn't strictly required, but writing in field order
        # keeps any debugger snapshots coherent.
        status[_F_HW_POS] = anchor_chart_end
        status[_F_HW_WALL] = dac_end
        status[_F_HW_MONO] = mono_end
        status[_F_HW_RATE] = rate_est
        status[_F_DAC_ANCHOR_VALID] = 1.0
        status[_F_RING_FILL_FRAMES] = float(ring.readable_frames())

    try:
        stream = sd.OutputStream(
            samplerate=sr,
            channels=source.src_channels,
            dtype='float32',
            blocksize=config.block_size,
            callback=_callback,
        )
        stream.start()
    except Exception as e:
        _publish_error(status, last_status_str, f'stream open failed: {e}')
        producer.stop()
        _drain_until_stop(cmd_queue)
        return

    status[_F_READY] = 1.0
    state.stream = stream
    state.producer = producer
    state.source = source

    _command_loop(cmd_queue, state)

    # Shutdown
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
    producer.stop()


@dataclass
class _ChildState:
    """Mutable bag for the child-process callback + command loop."""
    status: 'multiprocessing.Array'
    last_status_str: 'multiprocessing.Array'
    ring: object
    sr: int
    block_size: int
    volume: float
    rate: float
    chart_time: float
    scheduled_chart_pos: float
    lead_in_seconds: float
    cb_status_count: int = 0
    cb_ring_underflow: int = 0
    seek_gen: int = 0
    stream: object = None
    producer: object = None
    source: object = None


def _command_loop(cmd_queue, state: _ChildState) -> None:
    """Block on the command queue, dispatching commands until 'stop'.

    Runs on the child's main thread (PortAudio callback runs on a
    different OS thread inside the child). All command handlers
    forward into the producer (set_silent / set_rate / etc.); none
    of them touch the audio callback directly.
    """
    while True:
        try:
            op, payload = cmd_queue.get(timeout=1.0)
        except Exception:
            # Timeout: check parent still alive. multiprocessing
            # daemonizes the child so it exits when the parent does,
            # but if the parent is hung we still exit voluntarily
            # after a long idle.
            if os.getppid() == 1:  # reparented to init -> parent died
                return
            continue
        try:
            if op == _OP_STOP:
                return
            elif op == _OP_SET_VOLUME:
                state.volume = float(payload)
            elif op == _OP_SET_SILENT:
                state.producer.set_silent(bool(payload))
            elif op == _OP_SET_RATE:
                rate = max(0.05, float(payload))
                state.rate = rate
                state.producer.set_rate(rate)
            elif op == _OP_SET_PITCH_CORRECT:
                state.producer.set_pitch_correct(bool(payload))
            elif op == _OP_SEEK:
                chart_t = float(payload)
                if chart_t < 0.0:
                    lead_in = -chart_t
                    audio_t = 0.0
                else:
                    lead_in = 0.0
                    audio_t = chart_t
                state.lead_in_seconds = lead_in
                state.status[_F_LEAD_IN_SECONDS] = lead_in
                state.status[_F_DAC_ANCHOR_VALID] = 0.0
                state.status[_F_ENDED] = 0.0
                state.scheduled_chart_pos = audio_t
                state.chart_time = audio_t
                state.producer.request_seek(audio_t, lead_in)
                # Acknowledge last: once the parent sees the new gen,
                # every anchor field it reads is post-seek (invalid
                # until the next callback re-anchors).
                state.seek_gen += 1
                state.status[_F_SEEK_GEN] = float(state.seek_gen)
            elif op == _OP_PREWARM:
                # No-op for now; the in-process engine had a prewarm
                # path that's not strictly necessary.
                pass
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            _write_status_str(state.last_status_str,
                              f'cmd {op} failed: {e}')


# ── decoder helpers (child side) ───────────────────────────────────


def _decode_audio(path: str, last_status_str) -> tuple:
    """Decode `path` into (samples, sr) using the same fallback chain
    the in-process engine uses. Returns (None, 0) on failure."""
    import numpy as np
    try:
        import soundfile as sf
        arr, sr = sf.read(path, dtype='float32', always_2d=True)
        return arr, int(sr)
    except Exception as sf_err:
        _write_status_str(last_status_str, f'soundfile: {sf_err}')
    # Audioread fallback for mp3/m4a.
    try:
        import audioread
        with audioread.audio_open(path) as f:
            sr = int(f.samplerate)
            channels = int(f.channels)
            chunks = []
            for buf in f:
                chunks.append(np.frombuffer(buf, dtype='<i2'))
            if not chunks:
                return None, 0
            pcm = np.concatenate(chunks)
            arr = pcm.astype(np.float32) / 32768.0
            arr = arr.reshape(-1, channels) if channels > 1 else arr.reshape(-1, 1)
            return arr, sr
    except Exception as ar_err:
        _write_status_str(last_status_str, f'audioread: {ar_err}')
    return None, 0


# ── shared-status helpers ──────────────────────────────────────────


_STATUS_STR_LEN = 64


def _write_status_str(arr, text: str) -> None:
    """Pack a status string into a fixed-length shared bytes buffer.
    Truncates to ``_STATUS_STR_LEN`` and zero-fills the rest so the
    parent can decode by stripping NULs."""
    encoded = text.encode('utf-8')[:_STATUS_STR_LEN - 1]
    n = len(encoded)
    arr[:n] = encoded
    if n < _STATUS_STR_LEN:
        arr[n] = 0


def _read_status_str(arr) -> str:
    raw = bytes(arr[:])
    end = raw.find(b'\x00')
    if end < 0:
        end = len(raw)
    try:
        return raw[:end].decode('utf-8', errors='replace')
    except Exception:
        return ''


def _publish_error(status, last_status_str, msg: str) -> None:
    status[_F_READY] = 0.0
    _write_status_str(last_status_str, msg)


def _drain_until_stop(cmd_queue) -> None:
    """When the child can't initialize audio, keep the command queue
    alive so the parent's `stop()` lands cleanly instead of leaving
    the child running."""
    while True:
        try:
            op, _ = cmd_queue.get(timeout=5.0)
            if op == _OP_STOP:
                return
        except Exception:
            if os.getppid() == 1:
                return


# ── parent-side client ─────────────────────────────────────────────


class AudioProcessClient:
    """Parent-side handle to the audio child process.

    Mirrors the operational subset of `AudioEngine`: set_rate,
    set_silent, seek_to_chart_time, set_pitch_correct, set_volume,
    stop. Reads chart-time and status from shared memory.

    `ready` becomes True once the child publishes its READY flag in
    the shared status array. Until then, methods are no-ops.
    """

    def __init__(self, audio_path: str, *, pitch_correct: bool = True,
                 volume: float = 0.5) -> None:
        self._ctx = multiprocessing.get_context('spawn')
        self._cmd_queue = self._ctx.Queue(maxsize=64)
        self._status = self._ctx.Array('d', _NUM_STATUS_FIELDS, lock=False)
        self._last_status_str = self._ctx.Array('b', _STATUS_STR_LEN,
                                                 lock=False)
        config = AudioProcessConfig(
            audio_path=audio_path,
            pitch_correct=pitch_correct,
            volume=volume,
        )
        # Monotone floor for current_chart_time(); reset on seek, which
        # is the only legitimate backward move of the playhead. The gen
        # counter gates reads after a seek: until the child acknowledges
        # via _F_SEEK_GEN, the anchor fields are pre-seek and must not
        # be extrapolated from (they'd re-poison the floor).
        self._chart_time_floor = -float('inf')
        self._seek_gen_sent = 0
        self._last_seek_target = 0.0
        self._proc = self._ctx.Process(
            target=_run_audio_child,
            args=(config, self._cmd_queue, self._status,
                  self._last_status_str),
            daemon=True,
            name='vsrg-audio-child',
        )
        self._proc.start()
        # Wait for the child to publish READY (or fail fast). 5 s is
        # generous: spawn + decode + open stream rarely exceeds 2 s.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._status[_F_READY]:
                break
            if not self._proc.is_alive():
                break
            time.sleep(0.02)

    # ── status surface (parent reads shared memory) ───────────────

    @property
    def ready(self) -> bool:
        return bool(self._status[_F_READY])

    @property
    def base_duration(self) -> float:
        return float(self._status[_F_BASE_DURATION])

    def current_chart_time(self) -> float:
        """Chart-time of the audible playhead. Mirrors the in-process
        engine's `_current_source_position_locked` formula:
        anchor + (now - anchor_deadline) * hw_rate, minus lead-in.

        The child publishes the anchor's play deadline translated into
        `time.monotonic()`'s timebase (`_F_HW_MONO`), so we can
        extrapolate between audio callbacks here. Without that, reads
        would quantize to one block period (~12 ms), which renders as a
        staircase playhead.

        Reads are lock-free and can tear across a callback's field
        writes, mispairing pos/mono by up to one block. The monotone
        clamp turns any resulting backward step into a one-frame flat
        hold instead of a visible snap.

        After a seek, reads return the seek target until the child has
        both acknowledged the seek (gen counter) and re-anchored on its
        next callback; the anchor fields are stale or invalid until
        then.
        """
        anchor_valid = self._status[_F_DAC_ANCHOR_VALID]
        seek_pending = self._status[_F_SEEK_GEN] < self._seek_gen_sent
        if seek_pending or not anchor_valid:
            if self._seek_gen_sent:
                return self._last_seek_target
            lead_in = self._status[_F_LEAD_IN_SECONDS]
            return -lead_in if lead_in else 0.0

        hw_pos = self._status[_F_HW_POS]
        hw_mono = self._status[_F_HW_MONO]
        hw_rate = self._status[_F_HW_RATE]
        lead_in = self._status[_F_LEAD_IN_SECONDS]
        t = hw_pos + (time.monotonic() - hw_mono) * hw_rate - lead_in
        if t > self._chart_time_floor:
            self._chart_time_floor = t
        return self._chart_time_floor

    def callback_status_snapshot(self) -> tuple[int, str]:
        pa_n = int(self._status[_F_CB_STATUS_COUNT])
        ring_n = int(self._status[_F_CB_RING_UNDERFLOW])
        fill = int(self._status[_F_RING_FILL_FRAMES])
        cap = int(self._status[_F_RING_CAPACITY_FRAMES])
        text = _read_status_str(self._last_status_str)
        gauge = ''
        if cap:
            gauge = f' fill={int(100 * fill / cap)}%'
        if ring_n and not pa_n:
            return ring_n, f'ring underflow{gauge}'
        if pa_n and not ring_n:
            return pa_n, f'{text}{gauge}'
        if pa_n or ring_n:
            return pa_n + ring_n, (
                f'{"ring underflow" if ring_n else text}{gauge}'
            )
        return 0, gauge.lstrip()

    # ── command surface (parent enqueues messages) ────────────────

    def set_volume(self, v: float) -> None:
        self._send(_OP_SET_VOLUME, float(v))

    def set_silent(self, silent: bool) -> None:
        self._send(_OP_SET_SILENT, bool(silent))

    def set_rate(self, rate: float) -> None:
        self._send(_OP_SET_RATE, float(rate))

    def set_pitch_correct(self, on: bool) -> None:
        self._send(_OP_SET_PITCH_CORRECT, bool(on))

    def seek_to_chart_time(self, chart_t: float) -> None:
        chart_t = float(chart_t)
        self._chart_time_floor = -float('inf')
        self._last_seek_target = chart_t
        self._seek_gen_sent += 1
        self._send(_OP_SEEK, chart_t)

    def stop(self) -> None:
        self._send(_OP_STOP, None)
        if self._proc.is_alive():
            self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)

    def _send(self, op: str, payload) -> None:
        try:
            self._cmd_queue.put_nowait((op, payload))
        except Exception:
            # Queue full or pipe closed. Drop. We never expect this
            # in practice (the queue caps at 64 messages and the
            # child's command loop drains it continuously).
            pass
