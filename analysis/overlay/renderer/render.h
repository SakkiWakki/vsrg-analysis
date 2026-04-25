// Thin drawing shim for the overlay binary.
//
// This is the ONLY file outside render.c that gets to see the
// underlying graphics library. Today that library is NanoVG; if we
// ever swap it out (Skia, direct-GL, Vulkan), only render.c needs to
// change ; osu_overlay.c and everything else calls through this API.
//
// Contract:
//  - render_init() must be called once after the GL context is
//    current. Returns 1 on success, 0 on failure.
//  - Each frame is bracketed by render_begin_frame / render_end_frame.
//    The caller passes the drawable size in pixels (not DPR-scaled;
//    our overlay runs at the compositor's native resolution).
//  - Colours are the publisher's RGBA32 format (byte 0 = R,
//    byte 3 = A ; see analysis.overlay.api::rgba).
//  - Text is drawn from a top-left anchor at the caller's specified
//    pixel height, consistent with the publisher's bitmap-era layout.
//    render_text_measure returns the exact pixel advance + ascent
//    used by render_text so hit-testing stays accurate.

#ifndef VSRG_OVERLAY_RENDER_H
#define VSRG_OVERLAY_RENDER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int  render_init(void);
void render_shutdown(void);

void render_begin_frame(int width, int height);
void render_end_frame(void);

void render_rect(float x, float y, float w, float h, uint32_t rgba);
void render_rect_outline(float x, float y, float w, float h,
                         uint32_t rgba, float stroke_width);

// px_height is the requested visible ink height of the line. The
// shim handles the font-metric conversion (NanoVG asks for a
// "font size", which roughly matches em height; we derive a size
// that yields the requested cap height so numbers line up with
// the bitmap-era layouts).
void render_text(const char *s, float x, float y,
                 float px_height, uint32_t rgba);

// Measured advance of ``s`` at ``px_height``. Returns 0 if the
// font isn't loaded. Matches render_text's advancing pen exactly.
float render_text_width(const char *s, float px_height);

// Height of a text box at ``px_height`` ; used so the resolve-box
// layout math produces a bounding rect the hit-test can trust.
float render_text_height(float px_height);

// Draw an externally-owned GL texture at (x, y) with size (w, h).
// ``gl_texture_id`` must refer to a GL_TEXTURE_2D valid in the
// caller's current GL context -- typically a dmabuf imported via
// EGLImage + glEGLImageTargetTexture2DOES, living in the context
// the game is rendering into.
//
// ``tex_w`` / ``tex_h`` describe the source texture in pixels and
// drive NanoVG's UV mapping. ``flip_y=1`` flips vertically, needed
// for dmabuf-imported textures whose GL memory order ends up
// upside-down relative to Chromium's top-left origin.
//
// Backed by ``nvglCreateImageFromHandleGL``. NanoVG caches the
// image handle for ~60 frames internally; calling this repeatedly
// with the same ``gl_texture_id`` is cheap (no per-frame alloc).
void render_gl_texture(uint32_t gl_texture_id,
                       int tex_w, int tex_h,
                       float x, float y, float w, float h,
                       int flip_y);

#ifdef __cplusplus
}
#endif

#endif
