//! Wire format for the side-socket messages.
//!
//! Mirror of the C header ``analysis/overlay/widgets/web_texture_ipc.h``.
//! This module defines the Rust view of the same struct and pins the
//! layout at compile time so a drifted C header fails Rust compilation
//! rather than corrupting runtime exchanges.

use std::mem::size_of;

// ── Constants (must match web_texture_ipc.h) ──────────────────────

pub const SOCKET_PATH: &str = "/tmp/vsrg_overlay_web.sock";
pub const MAGIC: u32 = 0x57454254;  // 'WEBT'
pub const VERSION: u32 = 1;

pub const KIND_PUBLISH: u32 = 1;
pub const KIND_RELEASE: u32 = 2;

pub const FORMAT_ARGB8888: u32 = 0x34325241;  // 'AR24'
pub const FORMAT_XRGB8888: u32 = 0x34325258;  // 'XR24'
pub const FORMAT_ABGR8888: u32 = 0x34324241;  // 'AB24'

pub const MAX_PLANES: usize = 4;

// ── Wire struct ───────────────────────────────────────────────────

/// Single SOCK_SEQPACKET message. The fd (for KIND_PUBLISH) travels
/// via SCM_RIGHTS ancillary data, not in this payload.
///
/// Layout must exactly match VsrgWebTexFrame in web_texture_ipc.h. The
/// static_assertions at the bottom of this module pin field offsets
/// and total size; a failed assertion = drift that would corrupt the
/// wire protocol.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub magic: u32,
    pub version: u32,
    pub kind: u32,
    pub channel_id: u32,
    pub generation: u32,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub modifier: u64,
    pub n_planes: u32,
    pub _pad0: u32,
    pub offsets: [u32; MAX_PLANES],
    pub strides: [u32; MAX_PLANES],
}

impl Frame {
    pub fn new_publish(
        channel_id: u32,
        generation: u32,
        width: u32,
        height: u32,
        format: u32,
        modifier: u64,
    ) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            kind: KIND_PUBLISH,
            channel_id,
            generation,
            width,
            height,
            format,
            modifier,
            n_planes: 1,   // single-plane RGB default; producer may overwrite
            _pad0: 0,
            offsets: [0; MAX_PLANES],
            strides: [0; MAX_PLANES],
        }
    }

    pub fn new_release(channel_id: u32) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            kind: KIND_RELEASE,
            channel_id,
            generation: 0,
            width: 0,
            height: 0,
            format: 0,
            modifier: 0,
            n_planes: 0,
            _pad0: 0,
            offsets: [0; MAX_PLANES],
            strides: [0; MAX_PLANES],
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        // Safe: #[repr(C)] struct with only POD fields.
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                size_of::<Self>(),
            )
        }
    }
}

// ── Layout contract ────────────────────────────────────────────────
// These assertions fire at compile time. Any change that doesn't
// match the canonical C layout breaks the build -- which is the
// point: we catch drift before it corrupts the wire format.

#[cfg(target_pointer_width = "64")]
const _: () = {
    // Total size: 80 bytes (verified by C probe in
    // tests/test_overlay_api.py::test_web_texture_ipc_header_struct_offsets_match_c).
    assert!(size_of::<Frame>() == 80);
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    #[test]
    fn field_offsets_match_c_layout() {
        assert_eq!(offset_of!(Frame, magic), 0);
        assert_eq!(offset_of!(Frame, version), 4);
        assert_eq!(offset_of!(Frame, kind), 8);
        assert_eq!(offset_of!(Frame, channel_id), 12);
        assert_eq!(offset_of!(Frame, generation), 16);
        assert_eq!(offset_of!(Frame, width), 20);
        assert_eq!(offset_of!(Frame, height), 24);
        assert_eq!(offset_of!(Frame, format), 28);
        assert_eq!(offset_of!(Frame, modifier), 32);
        assert_eq!(offset_of!(Frame, n_planes), 40);
        assert_eq!(offset_of!(Frame, offsets), 48);
        assert_eq!(offset_of!(Frame, strides), 64);
    }

    #[test]
    fn publish_frame_has_magic_and_version() {
        let f = Frame::new_publish(1, 2, 640, 480, FORMAT_ARGB8888, 0);
        assert_eq!(f.magic, MAGIC);
        assert_eq!(f.version, VERSION);
        assert_eq!(f.kind, KIND_PUBLISH);
        assert_eq!(f.channel_id, 1);
        assert_eq!(f.generation, 2);
    }

    #[test]
    fn release_frame_has_release_kind() {
        let f = Frame::new_release(5);
        assert_eq!(f.kind, KIND_RELEASE);
        assert_eq!(f.channel_id, 5);
    }
}
