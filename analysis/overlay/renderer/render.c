// NanoVG-backed implementation of render.h.
//
// Everything NanoVG-specific lives here. osu_overlay.c sees only the
// render_* API. If NanoVG is ever replaced (Skia, direct-GL, Vulkan,
// etc.) only this translation unit changes.
//
// GL backend: GL2. We keep the overlay's existing compatibility-profile
// GL context from glXCreateNewContext so the context-creation code in
// osu_overlay.c doesn't have to change. NanoVG's GL2 path works fine
// with fixed-function contexts.
//
// Font: DejaVu Sans Mono, loaded at init from the system path. If the
// font is missing, render_init still returns 1 (drawing still works)
// and text calls become no-ops ; matching the behaviour of the old
// font_ttf fallback so the rest of the overlay degrades gracefully
// rather than failing to start.

// GL_GLEXT_PROTOTYPES exposes glGenBuffers / glUseProgram / etc. from
// the system libGL, which is what NanoVG's GL2 backend needs. We go
// this route instead of pulling in GLEW so the overlay binary has no
// extra runtime deps.
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>

#include <stdio.h>
#include <stdlib.h>

#include "render.h"

// Pull in NanoVG and its GL2 implementation in this single TU.
#define NANOVG_GL2_IMPLEMENTATION
#include "nanovg.h"
#include "nanovg_gl.h"

#define DEFAULT_FONT_PATH "/usr/share/fonts/TTF/DejaVuSansMono.ttf"
#define FONT_NAME         "mono"

// DejaVu Sans Mono cap-height is roughly this fraction of its em.
// We divide the caller's requested visible-ink height by this factor
// to get a NanoVG font size that yields the requested cap height.
// If text looks the wrong size after a font swap, this is the knob
// to tune. NanoVG's nvgTextMetrics would let us derive this exactly,
// but the constant is stable enough for our uses and saves per-frame
// work.
#define CAP_HEIGHT_RATIO  0.72f

static NVGcontext *g_vg      = NULL;
static int         g_font_id = -1;

static void rgba_to_nvg(uint32_t c, NVGcolor *out) {
    // byte 0 = R, byte 1 = G, byte 2 = B, byte 3 = A
    out->r = ((c >>  0) & 0xff) / 255.0f;
    out->g = ((c >>  8) & 0xff) / 255.0f;
    out->b = ((c >> 16) & 0xff) / 255.0f;
    out->a = ((c >> 24) & 0xff) / 255.0f;
}

static float size_for_visible(float px_height) {
    return px_height / CAP_HEIGHT_RATIO;
}

int render_init(void) {
    if (g_vg) return 1;
    // NVG_ANTIALIAS = smoother edges without MSAA; NVG_STENCIL_STROKES
    // would be nicer but needs a stencil buffer we didn't request.
    g_vg = nvgCreateGL2(NVG_ANTIALIAS);
    if (!g_vg) {
        fprintf(stderr, "[render] nvgCreateGL2 failed\n");
        return 0;
    }

    const char *font_path = getenv("VSRG_OVERLAY_FONT");
    if (!font_path || !*font_path) font_path = DEFAULT_FONT_PATH;
    g_font_id = nvgCreateFont(g_vg, FONT_NAME, font_path);
    if (g_font_id == -1) {
        fprintf(stderr, "[render] nvgCreateFont('%s') failed; "
                        "text will not render\n", font_path);
        // Still a successful init ; rects keep working.
    } else {
        printf("[render] NanoVG GL2 up; font='%s'\n", font_path);
    }
    return 1;
}

void render_shutdown(void) {
    if (g_vg) {
        nvgDeleteGL2(g_vg);
        g_vg = NULL;
        g_font_id = -1;
    }
}

void render_begin_frame(int width, int height) {
    if (!g_vg) return;
    // Pixel ratio 1.0 ; gamescope reports the true compositor size and
    // we don't do HiDPI scaling inside the overlay.
    nvgBeginFrame(g_vg, (float)width, (float)height, 1.0f);
}

void render_end_frame(void) {
    if (!g_vg) return;
    nvgEndFrame(g_vg);
}

void render_rect(float x, float y, float w, float h, uint32_t rgba) {
    if (!g_vg) return;
    NVGcolor c;
    rgba_to_nvg(rgba, &c);
    nvgBeginPath(g_vg);
    nvgRect(g_vg, x, y, w, h);
    nvgFillColor(g_vg, c);
    nvgFill(g_vg);
}

void render_rect_outline(float x, float y, float w, float h,
                         uint32_t rgba, float stroke_width) {
    if (!g_vg) return;
    NVGcolor c;
    rgba_to_nvg(rgba, &c);
    nvgBeginPath(g_vg);
    nvgRect(g_vg, x, y, w, h);
    nvgStrokeColor(g_vg, c);
    nvgStrokeWidth(g_vg, stroke_width);
    nvgStroke(g_vg);
}

static int ensure_text_ready(float px_height) {
    if (!g_vg || g_font_id < 0) return 0;
    nvgFontFaceId(g_vg, g_font_id);
    nvgFontSize(g_vg, size_for_visible(px_height));
    return 1;
}

void render_text(const char *s, float x, float y,
                 float px_height, uint32_t rgba) {
    if (!s || !*s) return;
    if (!ensure_text_ready(px_height)) return;
    NVGcolor c;
    rgba_to_nvg(rgba, &c);
    // Top-left alignment so (x, y) is the top-left of the text box ;
    // same contract the publisher's layout math assumes.
    nvgTextAlign(g_vg, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
    nvgFillColor(g_vg, c);
    nvgText(g_vg, x, y, s, NULL);
}

float render_text_width(const char *s, float px_height) {
    if (!s || !*s) return 0.0f;
    if (!ensure_text_ready(px_height)) return 0.0f;
    float bounds[4] = {0, 0, 0, 0};
    nvgTextAlign(g_vg, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
    return nvgTextBounds(g_vg, 0.0f, 0.0f, s, NULL, bounds);
}

float render_text_height(float px_height) {
    // The caller's px_height is the visible-ink target, and our
    // hit-testing / layout assumes that's the bounding-box height.
    // Return it unchanged so resolve_box stays consistent.
    return px_height;
}

// ── External GL texture draw ──────────────────────────────────────
//
// The web-texture IPC path delivers a dmabuf fd; the caller imports
// it into a GL texture (via EGL) and passes the GL name here each
// frame. We wrap the raw GL id in a NanoVG image handle so NanoVG's
// rasterizer can draw a textured quad with the same transform stack
// as our other primitives.
//
// Per-frame overhead we pay: one nvglCreateImageFromHandleGL2 +
// nvgDeleteImage. ``NVG_IMAGE_NODELETE`` tells NanoVG not to glDelete
// our texture when we drop its image -- the caller owns that
// lifecycle (they created the texture via EGLImage and will delete
// it when the producer releases the channel).
//
// Caching by gl_texture_id would cut the create/delete churn, but
// NanoVG treats image handles as short-lived and doesn't expose a
// "same underlying texture, new frame" path. The per-call cost is a
// handful of struct mallocs; negligible at ~30 Hz widget refresh.

void render_gl_texture(uint32_t gl_texture_id,
                       int tex_w, int tex_h,
                       float x, float y, float w, float h,
                       int flip_y) {
    if (!g_vg || gl_texture_id == 0) return;
    if (tex_w <= 0 || tex_h <= 0 || w <= 0.0f || h <= 0.0f) return;

    int image = nvglCreateImageFromHandleGL2(
        g_vg, (GLuint)gl_texture_id, tex_w, tex_h, NVG_IMAGE_NODELETE);
    if (image <= 0) return;

    // nvgImagePattern takes (ox, oy, ex, ey, angle, image, alpha).
    //   (ox, oy): top-left corner of the image's rect in world coords.
    //   (ex, ey): pattern repeat extent (we set to draw rect size so
    //             it tiles exactly once).
    // For flip_y, shift the origin down by h and negate the extent.
    NVGpaint paint;
    if (flip_y) {
        paint = nvgImagePattern(g_vg, x, y + h, w, -h, 0.0f, image, 1.0f);
    } else {
        paint = nvgImagePattern(g_vg, x, y,     w,  h, 0.0f, image, 1.0f);
    }
    nvgBeginPath(g_vg);
    nvgRect(g_vg, x, y, w, h);
    nvgFillPaint(g_vg, paint);
    nvgFill(g_vg);

    // Drop the NVG image handle; NVG_IMAGE_NODELETE means the
    // underlying GL texture survives (the caller owns it).
    nvgDeleteImage(g_vg, image);
}
