// Generic gamescope external-overlay renderer.
//
// This binary is intentionally dumb: it knows how to draw rects
// and bitmap text from a widget array kept in shared memory. It
// does NOT know about osu!, mania, combos, accuracy, or any
// specific game. That semantics lives entirely in the Python
// publisher (see analysis/overlay/publisher.py + any plugin-specific
// consumer, e.g. plugins/unsafe/osu_live/shm_publisher.py).
//
// Why: a plugin ecosystem. Any plugin can ship its own publisher
// and get a HUD in-game without touching C or rebuilding.
//
// Shm contract: analysis/games/osu/gamescope_overlay/overlay_shm.h
// Launch: osu_overlay --feed /dev/shm/vsrg_overlay_<plugin_key>
//                     --width W --height H
//
// Controls:
//   Shift+Tab  toggle edit mode. In edit mode, widgets get a dashed
//              outline, the screen dims, and the user can drag a
//              widget with the left mouse button. On release we
//              write the new (x, y) back into the shm slot; the
//              publisher picks it up on its next frame and persists
//              via the app's ConfigStore.
//
// Rendering: immediate-mode GL, fixed-function pipeline, no shaders
// or VBOs. The 8x8 bitmap font is drawn as GL_QUADS per lit pixel.
// Vsync is pinned to the compositor via GLX_EXT_swap_control so we
// don't beat gamescope's cadence.

#define _GNU_SOURCE
#include <GL/gl.h>
#include <GL/glx.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "font8x8.h"
#include "overlay_shm.h"

// ─── Globals (small, C-program scale) ───────────────────────────────────

static VsrgOverlayShm *g_shm;          // mmap'd shared region
static const char     *g_feed_path;    // where we're attached

// ─── Shared memory consumer ─────────────────────────────────────────────

static void shm_attach(void) {
    int fd = open(g_feed_path, O_CREAT | O_RDWR, 0600);
    if (fd < 0) return;
    // Ensure at least sizeof(VsrgOverlayShm) — we're happy to grow
    // (or create) the file so the publisher's mmap succeeds too.
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return; }
    if (st.st_size < (off_t)sizeof(VsrgOverlayShm)) {
        if (ftruncate(fd, sizeof(VsrgOverlayShm)) < 0) {
            close(fd); return;
        }
    }
    void *p = mmap(NULL, sizeof(VsrgOverlayShm),
                   PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return;
    g_shm = (VsrgOverlayShm *)p;
}

// Seqlock read of the full shm into `out`. Returns 1 on success.
// Re-reads up to 16 times if the writer is mid-update (odd seq).
static int shm_read(VsrgOverlayShm *out) {
    if (!g_shm) return 0;
    for (int tries = 0; tries < 16; tries++) {
        uint32_t s0 = __atomic_load_n(&g_shm->seq, __ATOMIC_ACQUIRE);
        if (s0 & 1u) continue;
        memcpy(out, (const void *)g_shm, sizeof(*out));
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        uint32_t s1 = __atomic_load_n(&g_shm->seq, __ATOMIC_ACQUIRE);
        if (s0 == s1) {
            return out->magic == VSRG_OVERLAY_MAGIC
                && out->version == VSRG_OVERLAY_VERSION;
        }
    }
    return 0;
}

// ─── X11 setup helpers ──────────────────────────────────────────────────

static void set_cardinal_prop(Display *dpy, Window w,
                              const char *name, unsigned long value) {
    Atom a = XInternAtom(dpy, name, False);
    XChangeProperty(dpy, w, a, XA_CARDINAL, 32, PropModeReplace,
                    (unsigned char *)&value, 1);
}

// ─── Drawing primitives ─────────────────────────────────────────────────

static void rect_verts(float x, float y, float w, float h) {
    glVertex2f(x,     y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x,     y + h);
}

static void draw_filled_rect(float x, float y, float w, float h) {
    glBegin(GL_QUADS);
    rect_verts(x, y, w, h);
    glEnd();
}

static void draw_outline_rect(float x, float y, float w, float h) {
    glBegin(GL_LINE_LOOP);
    glVertex2f(x,     y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x,     y + h);
    glEnd();
}

static void draw_glyph(char c, float x, float y, float px) {
    const Glyph8x8 *g = font_glyph(c);
    for (int row = 0; row < 8; row++) {
        uint8_t bits = (*g)[row];
        for (int col = 0; col < 8; col++) {
            if (bits & (0x80 >> col)) {
                rect_verts(x + col * px, y + row * px, px, px);
            }
        }
    }
}

static void draw_text(const char *s, float x, float y, float px) {
    glBegin(GL_QUADS);
    float cx = x;
    for (; *s; s++) {
        if (*s == ' ') { cx += 4.0f * px; continue; }
        draw_glyph(*s, cx, y, px);
        cx += 9.0f * px;
    }
    glEnd();
}

// Width in pixels the renderer will use for a text string at scale
// ``px``. Matches draw_text's per-glyph advance so drag hit-testing
// can compute the same bounding box.
static float measure_text(const char *s, float px) {
    float w = 0.0f;
    for (; *s; s++) {
        if (*s == ' ') w += 4.0f * px;
        else           w += 9.0f * px;
    }
    return w;
}

static void set_color_rgba32(uint32_t c) {
    // Byte 0 = R, byte 3 = A (see analysis.overlay.api::rgba).
    glColor4f(((c >>  0) & 0xff) / 255.0f,
              ((c >>  8) & 0xff) / 255.0f,
              ((c >> 16) & 0xff) / 255.0f,
              ((c >> 24) & 0xff) / 255.0f);
}

// ─── Widget layout (normalized → pixels) ────────────────────────────────

// Resolve a widget's normalized (x, y) + anchor into a pixel
// top-left corner. Size is resolved to an on-screen bounding box
// (for rect: w*canvas_w, h*canvas_h; for text: measured string).
typedef struct {
    float px, py;          // top-left in pixels
    float pw, ph;          // size in pixels (for hit-testing)
} ResolvedBox;

static ResolvedBox resolve_box(const VsrgOverlayWidget *w,
                               int canvas_w, int canvas_h) {
    float pw, ph;
    if (w->kind == VSRG_OVERLAY_KIND_TEXT) {
        pw = measure_text(w->text, w->px_scale);
        ph = 8.0f * w->px_scale;
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
    ResolvedBox rb = { px, py, pw, ph };
    return rb;
}

// Inverse of resolve_box for the (x, y) field: given the desired
// on-screen pixel top-left ``(target_px, target_py)``, what
// *normalized, pre-anchor* (x, y) should we write back so the next
// resolve yields that pixel position? Used during drag.
static void reverse_anchor(const VsrgOverlayWidget *w,
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

// ─── Rendering ──────────────────────────────────────────────────────────

static void render_widgets(const VsrgOverlayShm *s, int width, int height) {
    uint32_t n = s->n_widgets;
    if (n > VSRG_OVERLAY_MAX_WIDGETS) n = VSRG_OVERLAY_MAX_WIDGETS;
    for (uint32_t i = 0; i < n; i++) {
        const VsrgOverlayWidget *w = &s->widgets[i];
        if (w->kind == VSRG_OVERLAY_KIND_UNUSED) continue;
        ResolvedBox rb = resolve_box(w, width, height);
        set_color_rgba32(w->color);
        if (w->kind == VSRG_OVERLAY_KIND_RECT) {
            draw_filled_rect(rb.px, rb.py, rb.pw, rb.ph);
        } else if (w->kind == VSRG_OVERLAY_KIND_TEXT) {
            draw_text(w->text, rb.px, rb.py, w->px_scale);
        }
    }
}

// Edit-mode overlay: dim the whole canvas, then outline every
// widget. If ``hover_idx`` >= 0, brighten that outline.
static void render_edit_decorations(const VsrgOverlayShm *s,
                                    int width, int height,
                                    int hover_idx) {
    // Full-screen dim quad so the player notices they're in edit
    // mode and the widgets stand out.
    glColor4f(0.0f, 0.0f, 0.0f, 0.45f);
    draw_filled_rect(0, 0, (float)width, (float)height);

    uint32_t n = s->n_widgets;
    if (n > VSRG_OVERLAY_MAX_WIDGETS) n = VSRG_OVERLAY_MAX_WIDGETS;
    for (uint32_t i = 0; i < n; i++) {
        const VsrgOverlayWidget *w = &s->widgets[i];
        if (w->kind == VSRG_OVERLAY_KIND_UNUSED) continue;
        ResolvedBox rb = resolve_box(w, width, height);
        if ((int)i == hover_idx) {
            glColor4f(1.0f, 0.85f, 0.2f, 1.0f);
            glLineWidth(2.5f);
        } else {
            glColor4f(1.0f, 1.0f, 1.0f, 0.65f);
            glLineWidth(1.0f);
        }
        // Pad the outline a couple pixels so text widgets don't
        // have the outline clipped against the glyph pixels.
        draw_outline_rect(rb.px - 2.0f, rb.py - 2.0f,
                          rb.pw + 4.0f, rb.ph + 4.0f);
    }
    glLineWidth(1.0f);

    // Help strip at top of screen.
    glColor4f(0.0f, 0.0f, 0.0f, 0.8f);
    draw_filled_rect(0, 0, (float)width, 28.0f);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    draw_text("EDIT MODE  DRAG WIDGETS  SHIFT+TAB TO EXIT",
              12.0f, 6.0f, 1.8f);
}

// Pick the topmost (last-rendered) widget under a pixel.
static int hit_test(const VsrgOverlayShm *s, int width, int height,
                    int mx, int my) {
    int n = (int)s->n_widgets;
    if (n > VSRG_OVERLAY_MAX_WIDGETS) n = VSRG_OVERLAY_MAX_WIDGETS;
    for (int i = n - 1; i >= 0; i--) {
        const VsrgOverlayWidget *w = &s->widgets[i];
        if (w->kind == VSRG_OVERLAY_KIND_UNUSED) continue;
        ResolvedBox rb = resolve_box(w, width, height);
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

// ─── Entry point ────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    int width = 1920, height = 1080;
    const char *feed = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--width") && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--height") && i + 1 < argc) {
            height = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--feed") && i + 1 < argc) {
            feed = argv[++i];
        }
    }
    if (!feed) {
        // Back-compat with the old osu_live path so existing
        // runner scripts keep working during the transition.
        feed = "/dev/shm/vsrg_overlay_osu_live";
    }
    g_feed_path = feed;

    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) {
        fprintf(stderr, "[overlay] XOpenDisplay(NULL) failed. "
                "Run this inside gamescope (DISPLAY=%s)\n",
                getenv("DISPLAY"));
        return 1;
    }
    int screen = DefaultScreen(dpy);
    Window root = RootWindow(dpy, screen);

    int fb_attribs[] = {
        GLX_X_RENDERABLE,  True,
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_RENDER_TYPE,   GLX_RGBA_BIT,
        GLX_DOUBLEBUFFER,  True,
        GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
        GLX_ALPHA_SIZE, 8, GLX_DEPTH_SIZE, 0, None
    };
    int nfb = 0;
    GLXFBConfig *fbs = glXChooseFBConfig(dpy, screen, fb_attribs, &nfb);
    if (!fbs || nfb == 0) {
        fprintf(stderr, "[overlay] glXChooseFBConfig returned no configs\n");
        return 1;
    }
    GLXFBConfig fb = 0;
    XVisualInfo *vi = NULL;
    for (int i = 0; i < nfb; i++) {
        XVisualInfo *cand = glXGetVisualFromFBConfig(dpy, fbs[i]);
        if (!cand) continue;
        int a = 0;
        glXGetFBConfigAttrib(dpy, fbs[i], GLX_ALPHA_SIZE, &a);
        if (cand->depth == 32 && a == 8) { fb = fbs[i]; vi = cand; break; }
        XFree(cand);
    }
    XFree(fbs);
    if (!vi) {
        fprintf(stderr, "[overlay] no 32-bit (RGBA8888) FBConfig found\n");
        return 1;
    }

    Colormap cmap = XCreateColormap(dpy, root, vi->visual, AllocNone);
    XSetWindowAttributes swa = {0};
    swa.colormap = cmap;
    swa.border_pixel = 0;
    swa.background_pixel = 0;
    // Always select key/pointer events — we need them in edit mode,
    // and selecting them in normal mode is harmless (we only grab
    // the pointer when edit mode turns on).
    swa.event_mask = StructureNotifyMask | KeyPressMask
                   | ButtonPressMask | ButtonReleaseMask
                   | PointerMotionMask;

    Window win = XCreateWindow(
        dpy, root, 0, 0, width, height, 0,
        vi->depth, InputOutput, vi->visual,
        CWColormap | CWBorderPixel | CWBackPixel | CWEventMask, &swa);
    XStoreName(dpy, win, "vsrg-analysis overlay");

    set_cardinal_prop(dpy, win, "GAMESCOPE_EXTERNAL_OVERLAY", 1);
    set_cardinal_prop(dpy, win, "GAMESCOPE_NO_FOCUS",         1);

    // Passive global grab on Shift+Tab so we receive the key chord
    // regardless of focus. GrabModeAsync for both keyboard and
    // pointer so other apps keep running normally; we only get the
    // chord routed to us.
    KeyCode tab_kc = XKeysymToKeycode(dpy, XK_Tab);
    if (tab_kc != 0) {
        XGrabKey(dpy, tab_kc, ShiftMask, root, True,
                 GrabModeAsync, GrabModeAsync);
    } else {
        fprintf(stderr, "[overlay] XKeysymToKeycode(XK_Tab) failed; "
                "Shift+Tab edit toggle will not work\n");
    }

    GLXContext ctx = glXCreateNewContext(dpy, fb, GLX_RGBA_TYPE, NULL, True);
    if (!ctx) { fprintf(stderr, "[overlay] glXCreateNewContext failed\n"); return 1; }
    XMapWindow(dpy, win);
    glXMakeCurrent(dpy, win, ctx);

    // Vsync.
    typedef void (*PFNGLXSWAPINTERVALEXTPROC)(Display *, GLXDrawable, int);
    const char *glx_exts = glXQueryExtensionsString(dpy, screen);
    if (glx_exts && strstr(glx_exts, "GLX_EXT_swap_control")) {
        PFNGLXSWAPINTERVALEXTPROC set_swap =
            (PFNGLXSWAPINTERVALEXTPROC)glXGetProcAddressARB(
                (const GLubyte *)"glXSwapIntervalEXT");
        if (set_swap) set_swap(dpy, win, 1);
    }

    printf("[overlay] mapped %dx%d  DISPLAY=%s  feed=%s  GL=%s\n",
           width, height, getenv("DISPLAY"), feed,
           (const char *)glGetString(GL_VERSION));
    fflush(stdout);

    shm_attach();
    if (!g_shm) {
        printf("[overlay] %s not present yet; will retry on each frame\n",
               feed);
        fflush(stdout);
    }

    // ── Drag state ────────────────────────────────────────────
    int edit_mode      = 0;
    int drag_idx       = -1;    // widget slot being dragged, -1 = none
    int drag_grab_mx   = 0;     // pointer x at mouse-down (in px)
    int drag_grab_my   = 0;
    int hover_idx      = -1;
    int mouse_x        = 0;
    int mouse_y        = 0;

    uint64_t last_hash = 0;
    int      have_drawn = 0;
    unsigned long frame = 0;

    for (;;) {
        // ── X events ────────────────────────────────────────
        while (XPending(dpy)) {
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type == ConfigureNotify) {
                width  = ev.xconfigure.width;
                height = ev.xconfigure.height;
            } else if (ev.type == KeyPress) {
                KeySym ks = XLookupKeysym(&ev.xkey, 0);
                if (ks == XK_Tab && (ev.xkey.state & ShiftMask)) {
                    edit_mode = !edit_mode;
                    if (g_shm) {
                        // Expose the flag so Python can gate
                        // stateful publisher logic on it (e.g.
                        // keep positions stable while editing).
                        __atomic_store_n(&g_shm->edit_mode,
                                         (uint8_t)edit_mode,
                                         __ATOMIC_RELEASE);
                    }
                    if (edit_mode) {
                        XGrabPointer(dpy, win, True,
                            ButtonPressMask | ButtonReleaseMask
                                | PointerMotionMask,
                            GrabModeAsync, GrabModeAsync,
                            None, None, CurrentTime);
                    } else {
                        XUngrabPointer(dpy, CurrentTime);
                        drag_idx = -1;
                        if (g_shm) {
                            __atomic_store_n(&g_shm->drag_active, 0u,
                                             __ATOMIC_RELEASE);
                        }
                    }
                }
            } else if (ev.type == ButtonPress && edit_mode) {
                mouse_x = ev.xbutton.x;
                mouse_y = ev.xbutton.y;
                if (ev.xbutton.button == Button1 && g_shm) {
                    VsrgOverlayShm snap;
                    if (shm_read(&snap)) {
                        int idx = hit_test(&snap, width, height,
                                           mouse_x, mouse_y);
                        if (idx >= 0) {
                            drag_idx     = idx;
                            drag_grab_mx = mouse_x;
                            drag_grab_my = mouse_y;
                            // Tell the publisher: hands off the
                            // dragged widget's (x, y) until release.
                            __atomic_store_n(&g_shm->drag_active, 1u,
                                             __ATOMIC_RELEASE);
                            __atomic_store_n(&g_shm->dragged_widget_id,
                                             snap.widgets[idx].widget_id,
                                             __ATOMIC_RELEASE);
                        }
                    }
                }
            } else if (ev.type == ButtonRelease && edit_mode) {
                if (ev.xbutton.button == Button1 && drag_idx >= 0) {
                    drag_idx = -1;
                    if (g_shm) {
                        // Release hand-off: publisher now owns the
                        // position again and should capture the
                        // final delta. dragged_seq bumps exactly
                        // once per drag so the publisher persists
                        // the result at the end, not every frame.
                        __atomic_store_n(&g_shm->drag_active, 0u,
                                         __ATOMIC_RELEASE);
                        __atomic_add_fetch(&g_shm->dragged_seq, 1,
                                           __ATOMIC_RELEASE);
                    }
                }
            } else if (ev.type == MotionNotify) {
                mouse_x = ev.xmotion.x;
                mouse_y = ev.xmotion.y;
            }
        }

        if (!g_shm) shm_attach();

        VsrgOverlayShm snap;
        int have = shm_read(&snap);

        // ── Live drag: move the dragged widget's group in shm ───
        // Group semantics: if the widget under the cursor has
        // group_id != 0, every widget sharing that group_id moves
        // by the same *pixel* delta — the user perceives the
        // composite HUD as one piece. group_id == 0 means drag
        // only this widget (singleton group keyed by widget_id on
        // the Python side).
        if (edit_mode && drag_idx >= 0 && have && g_shm
                && (uint32_t)drag_idx < snap.n_widgets) {
            VsrgOverlayWidget w_cur = snap.widgets[drag_idx];
            float dx_px = (float)(mouse_x - drag_grab_mx);
            float dy_px = (float)(mouse_y - drag_grab_my);
            uint32_t group_id = w_cur.group_id;

            uint32_t n = snap.n_widgets;
            if (n > VSRG_OVERLAY_MAX_WIDGETS) n = VSRG_OVERLAY_MAX_WIDGETS;
            for (uint32_t i = 0; i < n; i++) {
                VsrgOverlayWidget wi = snap.widgets[i];
                if (wi.kind == VSRG_OVERLAY_KIND_UNUSED) continue;
                int move_this = (group_id == 0)
                              ? ((int)i == drag_idx)
                              : (wi.group_id == group_id);
                if (!move_this) continue;

                ResolvedBox rb_i = resolve_box(&wi, width, height);
                float new_resolved_x = rb_i.px + dx_px;
                float new_resolved_y = rb_i.py + dy_px;
                float nx, ny;
                reverse_anchor(&wi, width, height, rb_i.pw, rb_i.ph,
                               new_resolved_x, new_resolved_y, &nx, &ny);
                if (nx < 0.0f) nx = 0.0f;
                if (nx > 1.0f) nx = 1.0f;
                if (ny < 0.0f) ny = 0.0f;
                if (ny > 1.0f) ny = 1.0f;
                VsrgOverlayWidget *wi_shm = &g_shm->widgets[i];
                wi_shm->x = nx;
                wi_shm->y = ny;
            }

            // Advance grab point so next frame's delta is
            // incremental relative to where the cursor is now.
            drag_grab_mx = mouse_x;
            drag_grab_my = mouse_y;
            // Don't bump dragged_seq here — the publisher is told
            // to ignore us mid-drag via drag_active. We bump once
            // on ButtonRelease so persistence happens exactly once
            // per drag (not 60× per second).
            shm_read(&snap);
        }

        // ── Hover highlight in edit mode ────────────────────
        hover_idx = -1;
        if (edit_mode && have) {
            hover_idx = hit_test(&snap, width, height, mouse_x, mouse_y);
        }

        // ── Hash to skip no-op frames ───────────────────────
        // Hash the visible widgets + edit mode + hover so we
        // only redraw + swap when the display actually changes.
        // Skipping both the render and the swap is what fixes
        // gamescope's "same buffer committed twice" spam.
        uint64_t hash = 0xcbf29ce484222325ull;
        #define FNV_MIX(b) do { hash ^= (uint8_t)(b); \
                                hash *= 0x100000001b3ull; } while (0)
        #define FNV_BYTES(ptr, n) do { \
            const uint8_t *_p = (const uint8_t *)(ptr); \
            for (size_t _i = 0; _i < (n); _i++) FNV_MIX(_p[_i]); \
        } while (0)
        FNV_MIX((uint8_t)edit_mode);
        FNV_MIX((uint8_t)(hover_idx + 1));
        FNV_MIX((uint8_t)(width  & 0xff)); FNV_MIX((uint8_t)(width  >> 8));
        FNV_MIX((uint8_t)(height & 0xff)); FNV_MIX((uint8_t)(height >> 8));
        if (have) {
            uint32_t n = snap.n_widgets;
            if (n > VSRG_OVERLAY_MAX_WIDGETS) n = VSRG_OVERLAY_MAX_WIDGETS;
            for (uint32_t i = 0; i < n; i++) {
                const VsrgOverlayWidget *w = &snap.widgets[i];
                FNV_MIX(w->kind);
                FNV_MIX(w->anchor);
                FNV_BYTES(&w->widget_id, sizeof(w->widget_id));
                // Quantize floats so 1ulp jitter doesn't flip the
                // hash. Positions → 1/10 px, sizes → 1/10 px,
                // px_scale → 1/10 px.
                int qx = (int)lroundf(w->x * 10000.0f);
                int qy = (int)lroundf(w->y * 10000.0f);
                int qw = (int)lroundf(w->w * 10000.0f);
                int qh = (int)lroundf(w->h * 10000.0f);
                int qs = (int)lroundf(w->px_scale * 10.0f);
                FNV_BYTES(&qx, sizeof(qx)); FNV_BYTES(&qy, sizeof(qy));
                FNV_BYTES(&qw, sizeof(qw)); FNV_BYTES(&qh, sizeof(qh));
                FNV_BYTES(&qs, sizeof(qs));
                FNV_BYTES(&w->color, sizeof(w->color));
                if (w->kind == VSRG_OVERLAY_KIND_TEXT) {
                    FNV_BYTES(w->text, strnlen(w->text, VSRG_OVERLAY_TEXT_LEN));
                }
            }
        }
        #undef FNV_MIX
        #undef FNV_BYTES

        if (have_drawn && hash == last_hash) {
            usleep(16 * 1000);
            continue;
        }

        // ── Draw ────────────────────────────────────────────
        glViewport(0, 0, width, height);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width, height, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (have) {
            render_widgets(&snap, width, height);
            if (edit_mode) {
                render_edit_decorations(&snap, width, height, hover_idx);
            }
        }

        glXSwapBuffers(dpy, win);
        last_hash  = hash;
        have_drawn = 1;
        frame++;

        if (frame == 1 || frame % 300 == 0) {
            printf("[overlay] frame %lu  feed=%s  shm=%s  n=%u  edit=%d\n",
                   frame, feed, g_shm ? "ok" : "none",
                   have ? snap.n_widgets : 0, edit_mode);
            fflush(stdout);
        }
    }
}
