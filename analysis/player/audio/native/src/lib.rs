//! PyO3 extension: a native SPSC audio ring + a PortAudio callback registered
//! directly with ``Pa_OpenStream``.
//!
//! The Python producer (phase vocoder, chart-time
//! stamping, seek/rate) writes blocks into the ring via `write_block`. PortAudio
//! calls `ring_callback` on its own real-time thread; that function is pure
//! Rust, holds no GIL, allocates nothing, and just drains one block from the
//! ring into the output buffer (or emits silence on underrun).

use std::os::raw::{c_ulong, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Borrow a C-contiguous f32 PyBuffer as a `&[f32]` of exactly `want` elements,
/// without copying. Errors if the buffer is the wrong length or not contiguous.
fn buffer_as_slice<'a>(
    _py: Python<'_>,
    buf: &'a PyBuffer<f32>,
    want: usize,
) -> PyResult<&'a [f32]> {
    if buf.item_count() != want {
        return Err(PyRuntimeError::new_err(format!(
            "write_block expects {want} samples, got {}",
            buf.item_count()
        )));
    }
    if !buf.is_c_contiguous() {
        // c_contiguous for zero copy borrow
        return Err(PyRuntimeError::new_err("write_block buffer must be C-contiguous"));
    }
    // SAFETY: PyBuffer guarantees the pointer is valid for item_count elements
    // for the buffer's lifetime; we hold `buf` (and the GIL on entry) so the
    // exporting object can't be freed. The slice is only read.
    let ptr = buf.buf_ptr() as *const f32;
    Ok(unsafe { std::slice::from_raw_parts(ptr, want) })
}

mod pa;
mod ring;

use ring::{Ring, Stamp};

/// If u r reading this u r cool
/// Shared context handed to the C callback as `user_data`. Lives behind an
/// `Arc`; the callback holds a raw `*const StreamCtx` into the same allocation.
struct StreamCtx {
    ring: Ring,
    sr: f64,
    /// DAC anchor + chart-time, published by the callback, read by Python.
    /// Stored as bit-punned u64 so a single atomic covers each f64 field.
    hw_chart_end: AtomicU64, // f64 bits: chart-time of last sample in last block
    hw_dac_time: AtomicU64,  // f64 bits: PortAudio DAC time for that sample
    hw_rate: AtomicU64,      // f64 bits: rate the last block was rendered at
    anchor_valid: AtomicBool,
    ended: AtomicBool,
}

#[inline]
fn store_f64(a: &AtomicU64, v: f64) {
    a.store(v.to_bits(), Ordering::Relaxed);
}
#[inline]
fn load_f64(a: &AtomicU64) -> f64 {
    f64::from_bits(a.load(Ordering::Relaxed))
}

/// Real time callback. PortAudio calls this on its own thread. 
/// Removes need for a GIL for the producer (actual villian causing underruns)
unsafe extern "C" fn ring_callback(
    _input: *const c_void,
    output: *mut c_void,
    frame_count: c_ulong,
    time_info: *const pa::PaStreamCallbackTimeInfo,
    _status_flags: pa::PaStreamCallbackFlags,
    user_data: *mut c_void,
) -> i32 {
    let ctx = &*(user_data as *const StreamCtx);
    let frames = frame_count as usize;
    let ch = ctx.ring.channels();
    let out = std::slice::from_raw_parts_mut(output as *mut f32, frames * ch);

    ctx.ring.stats.callbacks.fetch_add(1, Ordering::Relaxed);

    // The stream is opened with frames_per_buffer == ring block size, so a
    // single read_block fills the whole callback. If PortAudio ever hands us a
    // different frame_count we fall back to silence for the remainder rather
    // than risk an out-of-bounds copy.
    let block = ctx.ring.block();
    let stamp = if frames == block {
        match ctx.ring.read_block(out) {
            Some(s) => Some(s),
            None => {
                ctx.ring.stats.underruns.fetch_add(1, Ordering::Relaxed);
                out.fill(0.0);
                None
            }
        }
    } else {
        ctx.ring.stats.underruns.fetch_add(1, Ordering::Relaxed);
        out.fill(0.0);
        None
    };

    // Publish the DAC anchor for the parent's playhead extrapolation.
    if let Some(s) = stamp {
        let block_period = frames as f64 / ctx.sr;
        let dac_end = if time_info.is_null() {
            0.0
        } else {
            (*time_info).output_buffer_dac_time + block_period
        };
        store_f64(&ctx.hw_chart_end, s.chart_end);
        store_f64(&ctx.hw_dac_time, dac_end);
        store_f64(&ctx.hw_rate, s.rate);
        ctx.anchor_valid.store(true, Ordering::Release);
        if s.ended {
            ctx.ended.store(true, Ordering::Relaxed);
        }
    }
    pa::PA_CONTINUE
}

/// Python-facing handle. Owns the ring (via the Arc'd ctx) and the PortAudio
/// stream.
///
/// This class is sent across threads on purpose: the Python producer thread
/// calls the ring methods (`writable_frames` / `write_block` / `reset_to_empty`)
/// while the child's main thread opens/closes the stream and reads the anchor.
/// That split is sound because:
///   - the ring is `Send + Sync` and its SPSC discipline is the whole point;
///   - the raw `*mut PaStream` is only ever touched by open/close/stream_time,
///     which the producer never calls.
/// PyO3's default `#[pyclass]` requires `Send`; we assert it below with the
/// reasoning that the only non-`Send` field (the stream pointer) is confined to
/// the owning thread by convention.
#[pyclass]
struct NativeAudioStream {
    ctx: Arc<StreamCtx>,
    stream: *mut pa::PaStream,
    block_samples: usize, // block * channels; the exact write_block length
    started: bool,
}

// SAFETY: see the type doc. The `*mut PaStream` is only dereferenced by the
// owning (main) thread's open/close/stream_time; the ring (the part the
// producer thread touches concurrently) is `Send + Sync`. PyO3 requires both
// Send and Sync on a #[pyclass] because Python may call methods from multiple
// threads; the stream pointer stays confined to the main thread by convention
unsafe impl Send for NativeAudioStream {}
unsafe impl Sync for NativeAudioStream {}

#[pymethods]
impl NativeAudioStream {
    /// Open + start a PortAudio output stream draining a fresh ring.
    ///
    /// device < 0 -> default output device. `block` is both the ring block size
    /// and PortAudio's frames_per_buffer, so one callback == one ring block.
    #[new]
    #[pyo3(signature = (samplerate, channels, block, capacity_frames, device=-1, suggested_latency=0.0))]
    fn new(
        samplerate: f64,
        channels: usize,
        block: usize,
        capacity_frames: usize,
        device: i32,
        suggested_latency: f64,
    ) -> PyResult<Self> {
        let ring = Ring::new(capacity_frames, channels, block)
            .map_err(PyRuntimeError::new_err)?;
        let ctx = Arc::new(StreamCtx {
            ring,
            sr: samplerate,
            hw_chart_end: AtomicU64::new(0),
            hw_dac_time: AtomicU64::new(0),
            hw_rate: AtomicU64::new(0f64.to_bits()),
            anchor_valid: AtomicBool::new(false),
            ended: AtomicBool::new(false),
        });

        unsafe {
            let err = pa::Pa_Initialize();
            if err != pa::PA_NO_ERROR {
                return Err(PyRuntimeError::new_err(format!("Pa_Initialize failed: {err}")));
            }
            let dev = if device < 0 { pa::Pa_GetDefaultOutputDevice() } else { device };
            let params = pa::PaStreamParameters {
                device: dev,
                channel_count: channels as i32,
                sample_format: pa::PA_FLOAT32,
                suggested_latency: if suggested_latency > 0.0 {
                    suggested_latency
                } else {
                    block as f64 / samplerate
                },
                host_api_specific_stream_info: std::ptr::null_mut(),
            };
            let mut stream: *mut pa::PaStream = std::ptr::null_mut();
            // Hand the callback a raw pointer INTO the Arc allocation. We keep
            // the Arc alive in `self.ctx` for the stream's whole lifetime, and
            // stop+close the stream in Drop before the Arc drops
            let ctx_ptr = Arc::as_ptr(&ctx) as *mut c_void;
            let err = pa::Pa_OpenStream(
                &mut stream,
                std::ptr::null(),
                &params,
                samplerate,
                block as c_ulong,
                0,
                Some(ring_callback),
                ctx_ptr,
            );
            if err != pa::PA_NO_ERROR {
                pa::Pa_Terminate();
                return Err(PyRuntimeError::new_err(format!("Pa_OpenStream failed: {err}")));
            }
            let err = pa::Pa_StartStream(stream);
            if err != pa::PA_NO_ERROR {
                pa::Pa_CloseStream(stream);
                pa::Pa_Terminate();
                return Err(PyRuntimeError::new_err(format!("Pa_StartStream failed: {err}")));
            }
            Ok(NativeAudioStream { ctx, stream, block_samples: block * channels, started: true })
        }
    }

    /// Frames the producer may write right now without overrunning.
    fn writable_frames(&self) -> usize {
        self.ctx.ring.writable_frames()
    }

    /// Write one block. `samples` is a flat interleaved, C-contiguous f32 numpy
    /// buffer of exactly `block * channels`. The producer must check
    /// `writable_frames() >= block` first (matches the Python AudioRing
    /// contract). We borrow the numpy buffer ZERO-COPY (no per-block Vec
    /// allocation/conversion) and release the GIL (consumer ) so the
    /// producer can't hold it against the callback.
    #[pyo3(signature = (samples, chart_end, rate, ended, silent))]
    fn write_block(
        &self,
        py: Python<'_>,
        samples: PyBuffer<f32>,
        chart_end: f64,
        rate: f64,
        ended: bool,
        silent: bool,
    ) -> PyResult<()> {
        let src = buffer_as_slice(py, &samples, self.block_samples)?;
        let stamp = Stamp { chart_end, rate, ended, silent };
        let ring = &self.ctx.ring;
        py.detach(|| ring.write_block(src, stamp));
        Ok(())
    }

    fn reset_to_empty(&self) {
        self.ctx.ring.reset_to_empty();
    }

    fn readable_frames(&self) -> usize {
        self.ctx.ring.readable_frames()
    }

    fn capacity_frames(&self) -> usize {
        self.ctx.ring.capacity_frames()
    }

    // ---- status surface (parent reads; all lock-free) ----

    /// (chart_end, dac_time, rate, anchor_valid, ended)
    fn anchor(&self) -> (f64, f64, f64, bool, bool) {
        (
            load_f64(&self.ctx.hw_chart_end),
            load_f64(&self.ctx.hw_dac_time),
            load_f64(&self.ctx.hw_rate),
            self.ctx.anchor_valid.load(Ordering::Acquire),
            self.ctx.ended.load(Ordering::Relaxed),
        )
    }

    /// Invalidate the anchor across a seek (producer-side), matching the Python
    /// engine's `_F_DAC_ANCHOR_VALID = 0` on seek.
    fn invalidate_anchor(&self) {
        self.ctx.anchor_valid.store(false, Ordering::Release);
        self.ctx.ended.store(false, Ordering::Relaxed);
    }

    /// PortAudio's current stream DAC clock (seconds). Pairs with `anchor()`'s
    /// dac_time so the parent can translate the anchor into monotonic time.
    fn stream_time(&self) -> f64 {
        unsafe { pa::Pa_GetStreamTime(self.stream) }
    }

    /// (underruns, callbacks, last_fill_frames, capacity_frames)
    fn stats(&self) -> (u64, u64, usize, usize) {
        (
            self.ctx.ring.stats.underruns.load(Ordering::Relaxed),
            self.ctx.ring.stats.callbacks.load(Ordering::Relaxed),
            self.ctx.ring.stats.last_fill_frames.load(Ordering::Relaxed),
            self.ctx.ring.capacity_frames(),
        )
    }

    fn close(&mut self) {
        self.shutdown();
    }
}

impl NativeAudioStream {
    fn shutdown(&mut self) {
        if !self.started {
            return;
        }
        self.started = false;
        // Stop + close the stream BEFORE the Arc can drop, so the callback (which
        // reads through the raw ctx pointer) is guaranteed not to fire again.
        unsafe {
            if !self.stream.is_null() {
                pa::Pa_StopStream(self.stream);
                pa::Pa_CloseStream(self.stream);
                self.stream = std::ptr::null_mut();
            }
            pa::Pa_Terminate();
        }
    }
}

impl Drop for NativeAudioStream {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Headless ring for benchmarking / testing: the same `Ring` the stream uses,
/// with no PortAudio stream attached, so `write_block` and `read_block` cost can
/// be measured in isolation (and against the Python `AudioRing`). Not used by
/// the audio engine.
#[pyclass]
struct RingBench {
    ring: Ring,
    out: Vec<f32>,
}

#[pymethods]
impl RingBench {
    #[new]
    fn new(channels: usize, block: usize, capacity_frames: usize) -> PyResult<Self> {
        let ring = Ring::new(capacity_frames, channels, block)
            .map_err(PyRuntimeError::new_err)?;
        Ok(RingBench { ring, out: vec![0.0; block * channels] })
    }

    fn writable_frames(&self) -> usize { self.ring.writable_frames() }
    fn readable_frames(&self) -> usize { self.ring.readable_frames() }

    fn write_block(&self, py: Python<'_>, samples: PyBuffer<f32>) -> PyResult<()> {
        let src = buffer_as_slice(py, &samples, self.out.len())?;
        let ring = &self.ring;
        let stamp = Stamp { chart_end: 0.0, rate: 1.0, ended: false, silent: false };
        py.detach(|| ring.write_block(src, stamp));
        Ok(())
    }

    /// Pop one block (discarding the samples); returns True if a block was read.
    fn read_block(&mut self) -> bool {
        self.ring.read_block(&mut self.out).is_some()
    }

    fn reset_to_empty(&self) { self.ring.reset_to_empty(); }
}

#[pymodule]
fn audio_ring_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeAudioStream>()?;
    m.add_class::<RingBench>()?;
    Ok(())
}
