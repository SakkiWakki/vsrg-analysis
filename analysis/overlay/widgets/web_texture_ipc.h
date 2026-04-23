// Side-socket protocol for shipping dmabuf fds to the overlay.
//
// The main ``overlay_shm`` IPC is shared-memory with fixed-size POD
// widgets. It works for rect/text but cannot carry file descriptors --
// dmabufs are process-local OS handles that travel via SCM_RIGHTS on a
// Unix socket. This header defines that side channel.
//
// Flow:
//
//   producer (our process)                  overlay (gl_layer in osu!)
//   -----------------------------           ---------------------------
//   connect AF_UNIX SOCK_SEQPACKET
//   to /tmp/vsrg_overlay_web.sock    ───►   accept()
//
//   sendmsg(VsrgWebTexFrame, fd)     ───►   recvmsg() -> import as
//                                            EGLImage + GL texture,
//                                            cache keyed by
//                                            (channel_id, generation)
//
//   shm publish widget with
//   kind=WEB_TEXTURE, same
//   (channel_id, generation)         ───►   draw_widgets() picks up the
//                                            slot, looks up the cached
//                                            texture, blits at rect.
//
// The socket type is SOCK_SEQPACKET so each sendmsg is one message
// (no framing concerns) and we never partial-receive. Path convention
// matches /dev/shm/vsrg_overlay; one socket per session.
//
// Message shape is a single fixed-size struct. Future message types
// (e.g. "release channel N") extend via the ``kind`` tag; unknown
// kinds are dropped by the overlay so the producer can add new ones
// without breaking older overlays.

#ifndef VSRG_OVERLAY_WEB_TEXTURE_IPC_H
#define VSRG_OVERLAY_WEB_TEXTURE_IPC_H

#include <stdint.h>

#define VSRG_WEB_TEX_SOCKET_PATH  "/tmp/vsrg_overlay_web.sock"
#define VSRG_WEB_TEX_MAGIC        0x57454254u   // 'WEBT'
#define VSRG_WEB_TEX_VERSION      1

// Message kinds.
//   PUBLISH: a fresh dmabuf fd (passed via SCM_RIGHTS). The overlay
//     imports it and caches under (channel_id, generation).
//   RELEASE: the producer is done with this channel; the overlay may
//     drop any cached textures for it. Sent on producer shutdown or
//     when the overlay widget is removed. Best-effort; if the fd is
//     already gone the overlay cleans up lazily.
#define VSRG_WEB_TEX_KIND_PUBLISH  1u
#define VSRG_WEB_TEX_KIND_RELEASE  2u

// Pixel formats this protocol understands. DRM FourCC values so the
// overlay can feed them straight to EGL_LINUX_DMA_BUF_EXT attribute
// arrays.
//   ARGB8888 / XRGB8888 are little-endian so memory layout is B,G,R,A
//   which matches Chromium's BGRA8 compositor output on Linux.
#define VSRG_WEB_TEX_FORMAT_ARGB8888  0x34325241u   // 'AR24'
#define VSRG_WEB_TEX_FORMAT_XRGB8888  0x34325258u   // 'XR24'
#define VSRG_WEB_TEX_FORMAT_ABGR8888  0x34324241u   // 'AB24'

// Maximum plane count. Single-planar RGB dmabufs use 1; NV12 etc. up
// to 3. We only advertise single-plane formats above, but the
// protocol carries multi-plane metadata so a future YUV producer
// doesn't need a version bump.
#define VSRG_WEB_TEX_MAX_PLANES  4

typedef struct {
    // Header: stamp every message so a corrupted socket read fails
    // fast rather than the overlay importing garbage.
    uint32_t magic;         // VSRG_WEB_TEX_MAGIC
    uint32_t version;       // VSRG_WEB_TEX_VERSION
    uint32_t kind;          // one of VSRG_WEB_TEX_KIND_*

    // Channel + generation. Must match the values the producer later
    // stamps on its shm widget. The overlay keys its EGLImage cache
    // on (channel_id, generation).
    uint32_t channel_id;
    uint32_t generation;

    // Image shape. width/height are pixels; format is a DRM FourCC
    // from VSRG_WEB_TEX_FORMAT_*.
    uint32_t width;
    uint32_t height;
    uint32_t format;

    // DRM modifier. 0 on older drivers (linear layout); nonzero for
    // tiled/compressed layouts. The producer gets this from
    // EGL_EXT_image_dma_buf_export_modifiers; the overlay feeds it
    // back into EGL_LINUX_DMA_BUF_EXT attribute list.
    uint64_t modifier;

    // Per-plane layout. ``n_planes`` is how many entries below are
    // live; single-plane RGB dmabufs set n_planes=1.
    uint32_t n_planes;
    uint32_t _pad0;

    uint32_t offsets[VSRG_WEB_TEX_MAX_PLANES];
    uint32_t strides[VSRG_WEB_TEX_MAX_PLANES];
} VsrgWebTexFrame;

#endif
