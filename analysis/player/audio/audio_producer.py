"""Decoupled producer for the audio engine.

The PortAudio callback has a hard real-time deadline (~11.6 ms at 512
frames @ 44.1 kHz). On Python the GIL is the bottleneck: when the GUI
thread holds it for a long paint, the callback thread waits, misses the
deadline, and PortAudio reports an output-underflow. The user hears
that as choppy audio.

# GIL contention is the real bottleneck

The producer thread is pure Python + numpy; it shares the GIL with the
Qt main thread. When Qt is in a long native paint call (a single
`QPainter.drawPixmap` etc.) it holds the GIL for the entire call --
~47 ms on SV-heavy charts. During that window the producer thread
cannot run, period. `sys.setswitchinterval` only affects checks
between Python bytecode instructions, so it does nothing for a long
native call holding the GIL.

What actually mitigates this in-process:

  - Numpy ops INSIDE `pv.generate()` (FFTs, convolutions) release the
    GIL during their C-level work, letting the producer interleave
    with shorter Python operations on the GUI thread.
  - A deep ring -- 700+ ms here -- lets the consumer ride out a
    multi-frame GUI hitch without underrunning.

What actually fixes it (out of scope for this module, but the
direction we head if `ring underflow` events stay non-zero):

  - CFFI callback path: replace the Python PortAudio callback with a
    C function that drains a `pa_ringbuffer`. The callback never takes
    the GIL, so producer starvation just slows refill rather than
    starving the consumer. This is what pyo and Carla do.
  - Multiprocess producer: move `pv.generate()` into a separate
    Process with a shared-memory ring. Qt paints can't starve another
    process at all.

This module decouples PV cost from the callback. A worker thread runs
``pv.generate()`` ahead of playback and stamps each emitted block into
two parallel SPSC rings:

    audio ring    -- raw float32 samples, ``(capacity_frames, channels)``
    stamp ring    -- one ``BlockStamp`` per block (chart-time end + flags)

The PortAudio callback then becomes one numpy ``copy_to`` of the next
block's slice plus one stamp pop -- microseconds, no Python compute, no
PV lock.

# Why SPSC works without a lock

There is exactly one writer (the producer thread) and exactly one
reader (the callback). Indices are monotone integers; CPython makes
plain-int reads and writes atomic under the GIL. We rely on the
ordering "write data, then bump the write index" so that the moment
the consumer sees an advanced write index, the data backing it is
already in place. The capacity is power-of-two so wrap-around is a
mask, not a branch.

# What the producer owns

* The phase vocoder. Only the producer calls ``pv.generate()``,
  ``pv.seek()``, ``pv.set_rate()``, ``pv.set_pitch_correct()``. The
  callback never touches the PV.
* The source-domain "next block" cursor (chart-time of the next sample
  the producer will write).
* Lead-in zero-fill. While the source position is below the lead-in
  threshold the producer writes silent blocks -- consistent with the
  pre-ring engine's policy -- so the source-frame counter advances and
  chart-time crosses zero on an exact frame boundary.

# What the callback owns

* The DAC anchor (``hw_pos``, ``hw_wall``). These are derived from
  PortAudio's ``time_info`` and the chart-time stamp the producer left
  for the block being played; the callback is the only place we have
  PortAudio's clock, so this still has to live there.
* The output-volume multiply. One numpy op.

# Underflow handling

If the producer falls behind (worker preempted, PV burst), the audio
ring may be empty when the callback fires. The callback emits silence
in that case and counts the event so we can correlate with PortAudio's
own status flag. We never block the callback waiting for samples --
that would defeat the whole point of decoupling.

# Seek semantics

Seeks happen on the GUI thread but must take effect coherently with
PV state. The GUI enqueues a ``Seek`` command; the producer applies it
on its next tick by:
  1. Resetting the audio ring (write_idx = read_idx -- consumer sees
     empty, so subsequent callbacks emit silence until the producer
     refills).
  2. Calling ``pv.seek(audio_t)``.
  3. Updating the next-block chart-time cursor.

This is the same pattern the existing engine uses today, just shifted
behind the ring boundary.

# Rate changes

The producer applies rate changes immediately by setting ``pv.rate`` and
its own ``rate`` cursor. Frames already in the ring continue to play at
the rate they were rendered with -- there's no way to retroactively
change them. Audible chart-time keeps advancing correctly because we
stamp each block's chart-end at write time using the rate that was
active when that block was generated.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BlockStamp:
    """Per-block metadata the producer leaves for the callback.

    Fields:
      chart_end:    chart-time of the last sample in this block, in
                    source-time seconds.
      rate:         playback rate active while this block was rendered.
                    The DAC anchor in the callback uses this for its
                    extrapolation math.
      ended:        True if this block was the last one before the
                    source ran out (consumer can latch ``_ended``).
      silent:       True if this block is the lead-in / GUI-paused
                    silence pad. Useful for diagnostic logs.
    """
    chart_end: float
    rate: float
    ended: bool
    silent: bool


class AudioRing:
    """SPSC ring of float32 audio frames + parallel ring of BlockStamps.

    Capacity is in frames and must be a power of two. The two rings
    advance in lockstep: every block written / read updates both audio
    and stamp cursors by the same amount.

    Thread safety:
      Producer thread is the sole writer of ``write_idx``.
      Consumer (callback) thread is the sole writer of ``read_idx``.
      Both are plain ints; CPython int reads/writes are atomic under
      the GIL. We rely on the producer publishing data before the
      cursor (data first, then index).
    """

    def __init__(self, capacity_frames: int, channels: int,
                 *, block_size: int) -> None:
        if capacity_frames & (capacity_frames - 1):
            raise ValueError('capacity_frames must be a power of two')
        if capacity_frames % block_size:
            raise ValueError('capacity_frames must be a multiple of block_size')
        self._cap = int(capacity_frames)
        self._mask = self._cap - 1
        self._channels = int(channels)
        self._block_size = int(block_size)
        self._buf = np.zeros((self._cap, self._channels), dtype=np.float32)
        # One stamp per block. We size the stamp ring to match the
        # block-count of the audio ring so the two cursors are
        # interchangeable when divided by block_size.
        self._n_blocks = self._cap // self._block_size
        self._stamp_mask = self._n_blocks - 1
        self._stamps: list[BlockStamp | None] = [None] * self._n_blocks
        # Cursors in *frames*. Monotone; never wrap. Index into the
        # buffer with `& mask`.
        self._write_idx = 0
        self._read_idx = 0

    # ── producer-only ─────────────────────────────────────────────────

    def writable_frames(self) -> int:
        """Number of frames the producer can write without overrunning."""
        return self._cap - (self._write_idx - self._read_idx)

    def write_block(self, samples: np.ndarray, stamp: BlockStamp) -> None:
        """Write `block_size` frames + stamp. Caller must check
        `writable_frames() >= block_size` first."""
        n = self._block_size
        start = self._write_idx & self._mask
        end = start + n
        if end <= self._cap:
            self._buf[start:end] = samples
        else:
            tail = self._cap - start
            self._buf[start:] = samples[:tail]
            self._buf[:end - self._cap] = samples[tail:]
        # Stamp index follows the same ring; one stamp per block.
        block_idx = (self._write_idx // n) & self._stamp_mask
        self._stamps[block_idx] = stamp
        # Publish data + stamp before advancing the cursor; the
        # consumer reads the cursor first and then the data, so this
        # ordering keeps it consistent.
        self._write_idx += n

    # ── consumer-only ─────────────────────────────────────────────────

    def readable_frames(self) -> int:
        return self._write_idx - self._read_idx

    def read_block(self, out: np.ndarray) -> BlockStamp | None:
        """Pop one `block_size` block of frames into `out`. Returns the
        stamp for that block, or None if the ring was empty (consumer
        should emit silence). `out` must be shape `(block_size,
        channels)`."""
        n = self._block_size
        if self.readable_frames() < n:
            return None
        start = self._read_idx & self._mask
        end = start + n
        if end <= self._cap:
            np.copyto(out, self._buf[start:end])
        else:
            tail = self._cap - start
            np.copyto(out[:tail], self._buf[start:])
            np.copyto(out[tail:], self._buf[:end - self._cap])
        block_idx = (self._read_idx // n) & self._stamp_mask
        stamp = self._stamps[block_idx]
        self._read_idx += n
        return stamp

    # ── shared (any thread, but caller must serialize seeks) ──────────

    def reset_to_empty(self) -> None:
        """Drop everything in the ring. Called by the producer when it
        applies a seek -- frames already queued correspond to the old
        position. Setting ``read_idx = write_idx`` makes the consumer
        see an empty ring on its next pop."""
        # Producer is the only thread that calls this AND the only
        # writer of write_idx, so this is consistent. The consumer
        # may be mid-read at this exact moment but it can only
        # *advance* read_idx; if it advances past write_idx (which
        # can't happen given the readable_frames check), it'd be a
        # bug. After this call, readable_frames() returns <= 0.
        self._read_idx = self._write_idx


# ── Producer thread ───────────────────────────────────────────────────


class _StopThread(Exception):
    pass


class AudioProducer:
    """Background thread that fills an `AudioRing` from a phase vocoder.

    Owns the PV exclusively. The audio engine forwards seek / set_rate
    / set_silent / set_pitch_correct calls into the producer via thread-
    safe setters; the producer applies them on its next tick alongside
    the next block of generation.

    The producer doesn't need a lock around `pv.generate()` because no
    other thread touches the PV. The setters use a small command lock
    so multiple GUI calls don't race against each other or against the
    producer's read of those fields.
    """

    # The producer fills the ring as much as the ring will hold and only
    # stops when there isn't room for one more block. This is the
    # opposite of the textbook "high-water mark + sleep" pattern, but
    # it matters here because the producer thread shares the GIL with
    # the GUI thread: a 47 ms GUI paint can completely starve the
    # producer for that window, so the only defense against a
    # consumer-side ring underflow is to keep the ring as deep as
    # possible at all times.

    def __init__(self, pv, ring: AudioRing, sr: int) -> None:
        self._pv = pv
        self._ring = ring
        self._sr = int(sr)
        self._block_size = ring._block_size
        # Producer-thread-local cursor: chart-time of the next sample
        # the producer is about to render. Mirrors the legacy engine's
        # `_scheduled_chart_pos` but lives on the producer side.
        self._next_chart_t = 0.0
        # Lead-in: while next_chart_t < lead_in_seconds we emit silent
        # blocks but keep advancing the chart cursor.
        self._lead_in_seconds = 0.0
        # Command state. The producer reads these on every tick; the
        # GUI writes them under `_cmd_lock`. We use a separate "seek
        # generation" counter so the producer can detect a fresh seek
        # even if the requested chart-time happens to equal the
        # current next_chart_t.
        self._cmd_lock = threading.Lock()
        self._silent = True
        self._rate = 1.0
        self._pending_seek_t: float | None = None
        self._pending_seek_lead_in: float = 0.0
        self._seek_gen = 0
        self._applied_seek_gen = 0
        # Wake/stop signaling. Wake gets set when the consumer drains
        # below the low-water mark or a command is enqueued; stop gets
        # set on engine shutdown.
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Diagnostic counters
        self._ended_latched = False

    # ── public API (called from GUI / engine threads) ─────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name='vsrg-audio-producer',
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None

    def set_silent(self, silent: bool) -> None:
        with self._cmd_lock:
            self._silent = bool(silent)
        self._wake.set()

    def set_rate(self, rate: float) -> None:
        rate = max(0.05, float(rate))
        with self._cmd_lock:
            self._rate = rate
        # Propagate to the PV so the next `pv.generate()` call resamples
        # at the new rate. `pv.set_rate` is a single-float write; safe
        # to call from the GUI thread because the producer is the only
        # other PV reader and `_step` re-reads `self.rate` each
        # iteration -- a torn-mid-block rate just transitions over the
        # next block, which is fine.
        self._pv.set_rate(rate)
        self._wake.set()

    def set_pitch_correct(self, on: bool) -> None:
        # The PV's pitch_correct field is a single bool write; safe
        # without a lock for the same reason `set_rate` is. We poke
        # it directly so the change takes effect on the producer's
        # very next `pv.generate()` call without waiting for a tick.
        self._pv.pitch_correct = bool(on)

    def request_seek(self, chart_t: float, lead_in_seconds: float) -> None:
        """Queue a seek to `chart_t` (audio-domain seconds, >= 0) with
        the given lead-in. The producer applies it on its next tick;
        the audio ring is reset so the callback sees silence between
        the request and the first refilled block."""
        with self._cmd_lock:
            self._pending_seek_t = float(chart_t)
            self._pending_seek_lead_in = float(lead_in_seconds)
            self._seek_gen += 1
        self._wake.set()

    def signal_drain(self) -> None:
        """Called by the consumer (callback) after each pop. Wakes the
        producer if it was asleep on the high-water mark."""
        self._wake.set()

    # ── producer loop ─────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.is_set():
            self._apply_pending_seek()
            silent, rate = self._read_cmd_state()
            # Fill the ring as much as it will hold. Each pass through
            # the inner loop generates one block; we exit when there's
            # no room for one more. No high-water gate -- the deeper
            # the ring stays, the more GIL hitch we can absorb before
            # the consumer underruns.
            while self._ring.writable_frames() >= self._block_size:
                self._produce_block(silent=silent, rate=rate)
                # Re-check pending commands inside the fill loop so a
                # seek arriving mid-fill doesn't have to wait for the
                # producer to fill the entire ring before it applies.
                if self._seek_gen != self._applied_seek_gen:
                    break
            # Ring is full (or seek pending). Sleep until the consumer
            # drains a block (signal_drain wakes us) or a command lands.
            # Bounded sleep so a missed wake doesn't stall us; one block
            # period is ~12 ms, so 5 ms is generous slack.
            self._wake.wait(timeout=0.005)
            self._wake.clear()

    def _apply_pending_seek(self) -> None:
        with self._cmd_lock:
            if self._seek_gen == self._applied_seek_gen:
                return
            target_t = self._pending_seek_t
            lead_in = self._pending_seek_lead_in
            self._applied_seek_gen = self._seek_gen
            self._pending_seek_t = None
        # Reset the ring first so the consumer sees silence rather
        # than stale-position frames; THEN reseat the PV.
        self._ring.reset_to_empty()
        self._pv.seek(float(target_t or 0.0))
        self._next_chart_t = float(target_t or 0.0)
        self._lead_in_seconds = float(lead_in)
        self._ended_latched = False

    def _read_cmd_state(self) -> tuple[bool, float]:
        with self._cmd_lock:
            return self._silent, self._rate

    def _produce_block(self, *, silent: bool, rate: float) -> None:
        n = self._block_size
        in_lead_in = self._next_chart_t < self._lead_in_seconds
        emit_silent = silent or in_lead_in or self._ended_latched
        if emit_silent:
            samples = np.zeros((n, self._ring._channels), dtype=np.float32)
            cont = True
        else:
            try:
                samples, cont = self._pv.generate(n)
            except Exception as e:
                # Match the legacy engine's behavior: emit silence
                # rather than crashing the producer thread.
                print(f'audio producer: {e}')
                samples = np.zeros((n, self._ring._channels),
                                   dtype=np.float32)
                cont = False
            if samples.shape[0] < n:
                pad = np.zeros((n - samples.shape[0], self._ring._channels),
                               dtype=np.float32)
                samples = np.concatenate([samples, pad], axis=0)
        block_dt = (n / self._sr) * rate
        chart_end = self._next_chart_t + block_dt
        ended = not cont
        stamp = BlockStamp(
            chart_end=chart_end,
            rate=rate,
            ended=ended,
            silent=emit_silent,
        )
        self._ring.write_block(samples, stamp)
        self._next_chart_t = chart_end
        if ended:
            self._ended_latched = True
