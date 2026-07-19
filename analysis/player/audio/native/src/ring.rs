//! Lock-free SPSC ring of interleaved f32 frames + a parallel ring of block
//! stamps. One writer (the Python producer, across FFI) and one reader (the
//! PortAudio callback, native).
//!
//! # Memory ordering
//!
//! The producer publishes data (the sample copy + the stamp) BEFORE bumping
//! ``write_idx`` with a `Release` store. The consumer loads ``write_idx`` with
//! `Acquire` before touching the backing memory. That pair is the
//! happens-before edge: the moment the consumer observes an advanced write
//! index, the frames and stamp behind it are already visible. Symmetrically the
//! consumer publishes its progress via a `Release` store of ``read_idx`` and the
//! producer reads it `Acquire` to know how much room exists.
//!
//! Capacity is in frames and must be a power of two, and a multiple of the
//! block size, so one stamp maps to one block and wrap-around is a mask.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::atomic::AtomicUsize;

/// Per-block metadata the producer leaves for the callback, POD
#[derive(Clone, Copy)]
pub struct Stamp {
    pub chart_end: f64,
    pub rate: f64,
    pub ended: bool,
    pub silent: bool,
}

impl Default for Stamp {
    fn default() -> Self {
        Stamp { chart_end: 0.0, rate: 0.0, ended: false, silent: false }
    }
}

/// Diagnostic counters
#[derive(Default)]
pub struct RingStats {
    /// Callback invocations that found fewer than a block readable (underrun).
    pub underruns: AtomicU64,
    /// Total callback invocations.
    pub callbacks: AtomicU64,
    /// Frames currently readable
    pub last_fill_frames: AtomicUsize,
}

pub struct Ring {
    cap: usize,            // frames, power of two
    mask: usize,           // cap - 1
    channels: usize,
    block: usize,          // frames per block
    n_blocks: usize,       // cap / block
    stamp_mask: usize,     // n_blocks - 1, maps a block counter onto an idx in the stamps arr
    buf: Box<[f32]>,       // cap * channels, interleaved
    stamps: Box<[Stamp]>,  // n_blocks
    write_idx: AtomicU64,  // frames written (monotone, never wraps)
    read_idx: AtomicU64,   // frames read (monotone, never wraps)
    pub stats: RingStats,
}

// The whole point: the callback thread (consumer) and the producer thread touch
// this concurrently
// Raw buffer writes/reads are disjoint by construction to enable sharing `&Ring`
// across threads
unsafe impl Sync for Ring {}
unsafe impl Send for Ring {}

impl Ring {
    pub fn new(capacity_frames: usize, channels: usize, block: usize) -> Result<Ring, String> {
        if capacity_frames == 0 || (capacity_frames & (capacity_frames - 1)) != 0 {
            return Err("capacity_frames must be a power of two".into());
        }
        if block == 0 || capacity_frames % block != 0 {
            return Err("capacity_frames must be a multiple of block".into());
        }
        let n_blocks = capacity_frames / block;
        if (n_blocks & (n_blocks - 1)) != 0 {
            return Err("capacity/block (block count) must be a power of two".into());
        }
        Ok(Ring {
            cap: capacity_frames,
            mask: capacity_frames - 1,
            channels,
            block,
            n_blocks,
            stamp_mask: n_blocks - 1,
            buf: vec![0.0f32; capacity_frames * channels].into_boxed_slice(),
            stamps: vec![Stamp::default(); n_blocks].into_boxed_slice(),
            write_idx: AtomicU64::new(0),
            read_idx: AtomicU64::new(0),
            stats: RingStats::default(),
        })
    }

    #[inline]
    pub fn block(&self) -> usize { self.block }
    #[inline]
    pub fn channels(&self) -> usize { self.channels }
    #[inline]
    pub fn capacity_frames(&self) -> usize { self.cap }

    // ---- producer side (called from Python across FFI, GIL released) ----

    /// Frames the producer may write without overrunning the consumer.
    #[inline]
    pub fn writable_frames(&self) -> usize {
        let w = self.write_idx.load(Ordering::Relaxed);
        let r = self.read_idx.load(Ordering::Acquire);
        self.cap - (w - r) as usize
    }

    /// Write exactly one block of interleaved f32 (`block * channels` samples)
    /// plus its stamp. Caller must have checked `writable_frames() >= block`.
    /// `samples` must be `block * channels` long; enforced by the caller
    pub fn write_block(&self, samples: &[f32], stamp: Stamp) {
        let n = self.block;
        let w = self.write_idx.load(Ordering::Relaxed);
        let start_frame = (w as usize) & self.mask;
        let start = start_frame * self.channels;
        let count = n * self.channels;
        // SAFETY: the consumer never touches cells in [read_idx, write_idx);
        // we only write the region ahead of write_idx that the writable check
        // guaranteed is free. Split into head/tail on wrap.
        let end_frame = start_frame + n;
        if end_frame <= self.cap {
            self.copy_in(start, &samples[..count]);
        } else {
            let head_frames = self.cap - start_frame;
            let head = head_frames * self.channels;
            self.copy_in(start, &samples[..head]);
            self.copy_in(0, &samples[head..count]);
        }
        let block_idx = ((w as usize) / n) & self.stamp_mask;
        // SAFETY: same disjointness argument for the stamp cell.
        unsafe {
            let sp = self.stamps.as_ptr().add(block_idx) as *mut Stamp;
            std::ptr::write(sp, stamp);
        }
        // Publish: Release so the consumer's Acquire load sees data+stamp.
        self.write_idx.store(w + n as u64, Ordering::Release);
    }

    #[inline]
    fn copy_in(&self, at: usize, src: &[f32]) {
        // SAFETY: `at + src.len()` stays within `buf` by the wrap arithmetic in
        // the caller; the region is exclusively the producer's.
        unsafe {
            let dst = self.buf.as_ptr().add(at) as *mut f32;
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
    }

    /// Drop resets queued state e.g on a seek
    pub fn reset_to_empty(&self) {
        let w = self.write_idx.load(Ordering::Relaxed);
        self.read_idx.store(w, Ordering::Release);
    }

    // ---- consumer side (called from the PortAudio callback, no GIL) ----

    /// Pop one block into `out` (`block * channels` samples). Returns the block's
    /// stamp, or None if the ring had less than a block readable (caller emits
    /// silence). ONLY function the real-time thread runs on the ring.
    pub fn read_block(&self, out: &mut [f32]) -> Option<Stamp> {
        let n = self.block;
        let w = self.write_idx.load(Ordering::Acquire);
        let r = self.read_idx.load(Ordering::Relaxed);
        let readable = (w - r) as usize;
        self.stats.last_fill_frames.store(readable, Ordering::Relaxed);
        if readable < n {
            return None;
        }
        let start_frame = (r as usize) & self.mask;
        let start = start_frame * self.channels;
        let count = n * self.channels;
        let end_frame = start_frame + n;
        if end_frame <= self.cap {
            self.copy_out(&mut out[..count], start);
        } else {
            let head_frames = self.cap - start_frame;
            let head = head_frames * self.channels;
            self.copy_out(&mut out[..head], start);
            self.copy_out(&mut out[head..count], 0);
        }
        let block_idx = ((r as usize) / n) & self.stamp_mask;
        // SAFETY: the producer published this stamp before advancing write_idx,
        // which our Acquire load observed; the cell is stable until we advance
        // read_idx past it.
        let stamp = unsafe { *self.stamps.as_ptr().add(block_idx) };
        self.read_idx.store(r + n as u64, Ordering::Release);
        Some(stamp)
    }

    #[inline]
    fn copy_out(&self, dst: &mut [f32], at: usize) {
        // SAFETY: `at + dst.len()` stays within `buf`; the region was published
        // by the producer and is not reused until we advance read_idx.
        unsafe {
            let src = self.buf.as_ptr().add(at);
            std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
        }
    }

    /// Returns number of frames ready
    #[inline]
    pub fn readable_frames(&self) -> usize {
        let w = self.write_idx.load(Ordering::Acquire);
        let r = self.read_idx.load(Ordering::Relaxed);
        (w - r) as usize
    }
}
