// Implementation for widgets.h. Lifted verbatim from the original
// osu_overlay.c — the code was never osu-specific, only its
// location was.

#include "widgets.h"

#include <math.h>
#include <stddef.h>

#include "../renderer/render.h"

// Translucent colours used by edit-mode decorations. Packed in the
// publisher's RGBA32 byte order (byte 0=R, 3=A) so they flow through
// render_rect unchanged.
#define RGBA_PACK(r, g, b, a) \
    ((uint32_t)((r) | ((g) << 8) | ((b) << 16) | ((a) << 24)))
#define EDIT_DIM            RGBA_PACK(0,   0,   0,   115)
#define EDIT_HELP_BG        RGBA_PACK(0,   0,   0,   204)
#define EDIT_HELP_TEXT      RGBA_PACK(255, 255, 255, 255)
#define EDIT_OUTLINE_HOVER  RGBA_PACK(255, 217, 51,  255)
#define EDIT_OUTLINE_IDLE   RGBA_PACK(255, 255, 255, 166)

float vsrg_px_height_for_scale(float px_scale) {
    return px_scale * VSRG_TEXT_HEIGHT_PER_PX_SCALE;
}

// ── Web-texture resolver ─────────────────────────────────────────
//
// The resolver pointer is written by the host (once at startup) and
// read by the render thread (once per frame per web-texture slot).
// A plain pointer plus __atomic_load/store is sufficient -- we never
// call through a half-written pointer because the atomic store is a
// release and the load is an acquire, so everything the host did to
// prepare its texture cache before installing the resolver happens-
// before any render-thread call.
static VsrgWebTextureResolver g_web_tex_resolver = NULL;

void vsrg_set_web_texture_resolver(VsrgWebTextureResolver fn) {
    __atomic_store_n(&g_web_tex_resolver, fn, __ATOMIC_RELEASE);
}

static VsrgWebTextureResolver web_tex_resolver(void) {
    return __atomic_load_n(&g_web_tex_resolver, __ATOMIC_ACQUIRE);
}

float vsrg_measure_text(const char *s, float px_scale) {
    return render_text_width(s, vsrg_px_height_for_scale(px_scale));
}

float vsrg_text_height(float px_scale) {
    return render_text_height(vsrg_px_height_for_scale(px_scale));
}

VsrgResolvedBox vsrg_resolve_box(const VsrgOverlayWidget *w,
                                 int canvas_w, int canvas_h) {
    float pw, ph;
    if (w->kind == VSRG_OVERLAY_KIND_TEXT) {
        pw = vsrg_measure_text(w->text, w->px_scale);
        ph = vsrg_text_height(w->px_scale);
    } else {
        pw = w->w * canvas_w;
        ph = w->h * canvas_h;
    }
    float px = w->x * canvas_w;
    float py = w->y * canvas_h;
    switch (w->anchor) {
        case VSRG_OVERLAY_ANCHOR_TR: px = canvas_w - px - pw; break;
        case VSRG_OVERLAY_ANCHOR_BL: py = canvas_h - py - ph; break;
        case VSRG_OVERLAY_ANCHOR_BR: px = canvas_w - px - pw;
                                     py = canvas_h - py - ph; break;
        case VSRG_OVERLAY_ANCHOR_C:  px += canvas_w * 0.5f - pw * 0.5f;
                                     py += canvas_h * 0.5f - ph * 0.5f; break;
        default: break;  // TL: already correct.
    }
    VsrgResolvedBox rb = { px, py, pw, ph };
    return rb;
}

void vsrg_reverse_anchor(const VsrgOverlayWidget *w,
                         int canvas_w, int canvas_h,
                         float pw, float ph,
                         float target_px, float target_py,
                         float *out_nx, float *out_ny) {
    float ux = target_px, uy = target_py;
    switch (w->anchor) {
        case VSRG_OVERLAY_ANCHOR_TR: ux = canvas_w - target_px - pw; break;
        case VSRG_OVERLAY_ANCHOR_BL: uy = canvas_h - target_py - ph; break;
        case VSRG_OVERLAY_ANCHOR_BR: ux = canvas_w - target_px - pw;
                                     uy = canvas_h - target_py - ph; break;
        case VSRG_OVERLAY_ANCHOR_C:  ux = target_px - (canvas_w * 0.5f - pw * 0.5f);
                                     uy = target_py - (canvas_h * 0.5f - ph * 0.5f); break;
        default: break;
    }
    *out_nx = ux / (float)canvas_w;
    *out_ny = uy / (float)canvas_h;
}

int vsrg_hit_test(const VsrgOverlayShm *s,
                  int canvas_w, int canvas_h,
                  int mx, int my) {
    int n = (int)s->n_widgets;
    if (n > VSRG_OVERLAY_MAX_WIDGETS) n = VSRG_OVERLAY_MAX_WIDGETS;
    for (int i = n - 1; i >= 0; i--) {
        const VsrgOverlayWidget *w = &s->widgets[i];
        if (w->kind == VSRG_OVERLAY_KIND_UNUSED) continue;
        VsrgResolvedBox rb = vsrg_resolve_box(w, canvas_w, canvas_h);
        // Small margin so thin text is easier to grab.
        float x0 = rb.px - 2, y0 = rb.py - 2;
        float x1 = rb.px + rb.pw + 2, y1 = rb.py + rb.ph + 2;
        if ((float)mx >= x0 && (float)mx <= x1
            && (float)my >= y0 && (float)my <= y1) {
            return i;
        }
    }
    return -1;
}

void vsrg_draw_widgets(const VsrgOverlayShm *s,
                       int canvas_w, int canvas_h) {
    uint32_t n = s->n_widgets;
    if (n > VSRG_OVERLAY_MAX_WIDGETS) n = VSRG_OVERLAY_MAX_WIDGETS;
    for (uint32_t i = 0; i < n; i++) {
        const VsrgOverlayWidget *w = &s->widgets[i];
        if (w->kind == VSRG_OVERLAY_KIND_UNUSED) continue;
        VsrgResolvedBox rb = vsrg_resolve_box(w, canvas_w, canvas_h);
        if (w->kind == VSRG_OVERLAY_KIND_RECT) {
            render_rect(rb.px, rb.py, rb.pw, rb.ph, w->color);
        } else if (w->kind == VSRG_OVERLAY_KIND_TEXT) {
            render_text(w->text, rb.px, rb.py,
                        vsrg_px_height_for_scale(w->px_scale), w->color);
        } else if (w->kind == VSRG_OVERLAY_KIND_WEB_TEXTURE) {
            // Lookup the live GL texture for this (channel, gen). The
            // resolver returns 0 when the dmabuf fd hasn't landed yet
            // (publisher pushed the shm widget before the fd went
            // over the socket, or the fd hasn't been imported on the
            // render thread). In that case skip -- next frame likely
            // has it.
            VsrgWebTextureResolver resolve = web_tex_resolver();
            if (resolve == NULL) continue;
            uint32_t gl_tex = 0;
            int tex_w = 0, tex_h = 0;
            if (!resolve(w->channel_id, w->generation,
                         &gl_tex, &tex_w, &tex_h)) {
                continue;
            }
            // dmabuf-imported textures land y-flipped relative to the
            // producer's Chromium top-left origin. flip_y=1 corrects
            // that in the NanoVG UV mapping.
            render_gl_texture(gl_tex, tex_w, tex_h,
                              rb.px, rb.py, rb.pw, rb.ph,
                              /*flip_y=*/1);
        }
    }
}

void vsrg_draw_edit_decorations(const VsrgOverlayShm *s,
                                int canvas_w, int canvas_h,
                                int hover_idx) {
    render_rect(0, 0, (float)canvas_w, (float)canvas_h, EDIT_DIM);

    uint32_t n = s->n_widgets;
    if (n > VSRG_OVERLAY_MAX_WIDGETS) n = VSRG_OVERLAY_MAX_WIDGETS;
    for (uint32_t i = 0; i < n; i++) {
        const VsrgOverlayWidget *w = &s->widgets[i];
        if (w->kind == VSRG_OVERLAY_KIND_UNUSED) continue;
        VsrgResolvedBox rb = vsrg_resolve_box(w, canvas_w, canvas_h);
        uint32_t col = ((int)i == hover_idx)
                       ? EDIT_OUTLINE_HOVER : EDIT_OUTLINE_IDLE;
        float stroke = ((int)i == hover_idx) ? 2.5f : 1.0f;
        render_rect_outline(rb.px - 2.0f, rb.py - 2.0f,
                            rb.pw + 4.0f, rb.ph + 4.0f,
                            col, stroke);
    }

    render_rect(0, 0, (float)canvas_w, 28.0f, EDIT_HELP_BG);
    render_text("EDIT MODE  DRAG WIDGETS  SHIFT+TAB TO EXIT",
                12.0f, 6.0f, vsrg_px_height_for_scale(1.8f),
                EDIT_HELP_TEXT);
}

int vsrg_drag_tick(VsrgDragState *ds,
                   const VsrgInputState *in,
                   const VsrgOverlayShm *s,
                   VsrgOverlayShm *shm_mut,
                   int canvas_w, int canvas_h) {
    if (!in->valid) return -1;

    if (in->edit_toggle_pressed) {
        ds->edit_mode = !ds->edit_mode;
        if (shm_mut) {
            __atomic_store_n(&shm_mut->edit_mode,
                             ds->edit_mode,
                             __ATOMIC_RELEASE);
        }
        if (!ds->edit_mode && ds->drag_idx >= 0) {
            ds->drag_idx = -1;
            if (shm_mut) {
                __atomic_store_n(&shm_mut->drag_active, 0u,
                                 __ATOMIC_RELEASE);
            }
        }
    }

    if (!ds->edit_mode) {
        ds->drag_idx = -1;
        return -1;
    }

    if (in->primary_button_pressed && ds->drag_idx < 0 && s) {
        int idx = vsrg_hit_test(s, canvas_w, canvas_h,
                                in->mouse_x, in->mouse_y);
        if (idx >= 0) {
            ds->drag_idx     = idx;
            ds->drag_grab_mx = in->mouse_x;
            ds->drag_grab_my = in->mouse_y;
            if (shm_mut) {
                __atomic_store_n(&shm_mut->drag_active, 1u,
                                 __ATOMIC_RELEASE);
                __atomic_store_n(&shm_mut->dragged_widget_id,
                                 s->widgets[idx].widget_id,
                                 __ATOMIC_RELEASE);
            }
        }
    }

    if (ds->drag_idx >= 0 && s && shm_mut
            && (uint32_t)ds->drag_idx < s->n_widgets) {
        VsrgOverlayWidget w_cur = s->widgets[ds->drag_idx];
        float dx_px = (float)(in->mouse_x - ds->drag_grab_mx);
        float dy_px = (float)(in->mouse_y - ds->drag_grab_my);
        uint32_t group_id = w_cur.group_id;

        uint32_t n = s->n_widgets;
        if (n > VSRG_OVERLAY_MAX_WIDGETS) n = VSRG_OVERLAY_MAX_WIDGETS;
        for (uint32_t i = 0; i < n; i++) {
            VsrgOverlayWidget wi = s->widgets[i];
            if (wi.kind == VSRG_OVERLAY_KIND_UNUSED) continue;
            int move_this = (group_id == 0)
                          ? ((int)i == ds->drag_idx)
                          : (wi.group_id == group_id);
            if (!move_this) continue;

            VsrgResolvedBox rb_i = vsrg_resolve_box(&wi, canvas_w, canvas_h);
            float new_resolved_x = rb_i.px + dx_px;
            float new_resolved_y = rb_i.py + dy_px;
            float nx, ny;
            vsrg_reverse_anchor(&wi, canvas_w, canvas_h, rb_i.pw, rb_i.ph,
                                new_resolved_x, new_resolved_y, &nx, &ny);
            if (nx < 0.0f) nx = 0.0f;
            if (nx > 1.0f) nx = 1.0f;
            if (ny < 0.0f) ny = 0.0f;
            if (ny > 1.0f) ny = 1.0f;
            VsrgOverlayWidget *wi_shm = &shm_mut->widgets[i];
            wi_shm->x = nx;
            wi_shm->y = ny;
        }

        ds->drag_grab_mx = in->mouse_x;
        ds->drag_grab_my = in->mouse_y;
    }

    if (in->primary_button_released && ds->drag_idx >= 0) {
        ds->drag_idx = -1;
        if (shm_mut) {
            __atomic_store_n(&shm_mut->drag_active, 0u,
                             __ATOMIC_RELEASE);
            __atomic_add_fetch(&shm_mut->dragged_seq, 1,
                               __ATOMIC_RELEASE);
        }
    }

    if (!s) return -1;
    return vsrg_hit_test(s, canvas_w, canvas_h,
                         in->mouse_x, in->mouse_y);
}
