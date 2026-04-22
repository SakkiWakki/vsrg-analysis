// Generic gamescope external-overlay host.
//
// This binary is intentionally dumb: it stands up a transparent
// RGBA8888 GLX window, reads a widget array from shared memory, and
// replays the widgets through the game-agnostic renderer in
// analysis/overlay/renderer + widgets/. It does NOT know about osu!,
// mania, combos, accuracy, or any specific game. That semantics
// lives entirely in the Python publisher (see
// analysis/overlay/publisher.py + any plugin-specific consumer,
// e.g. plugins/unsafe/osu_live/shm_publisher.py).
//
// Why the osu!-shaped path: this is the first host we shipped. The
// same drawing code is also called from the GL preload layer
// (analysis/games/osu/gl_layer) and can trivially host etterna or
// any other VSRG the publisher supports.
//
// Shm contract:  analysis/overlay/widgets/overlay_shm.h
// Launch:        osu_overlay --feed /dev/shm/vsrg_overlay
//                            --width W --height H
//
// Controls:
//   Shift+Tab  toggle edit mode. Widgets get outlined, the screen
//              dims, and the user can drag a widget with the left
//              mouse button. On release we write the new (x, y)
//              back into shm; the publisher picks it up on its
//              next frame and persists via ConfigStore.

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

#include "../../../overlay/widgets/overlay_shm.h"
#include "../../../overlay/widgets/widgets.h"
#include "../../../overlay/renderer/render.h"

// ─── Globals (small, C-program scale) ───────────────────────────────────

static VsrgOverlayShm *g_shm;          // mmap'd shared region
static const char     *g_feed_path;    // where we're attached

// ─── Shared memory consumer ─────────────────────────────────────────────

static void shm_attach(void) {
    int fd = open(g_feed_path, O_CREAT | O_RDWR, 0600);
    if (fd < 0) return;
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
        feed = "/dev/shm/vsrg_overlay";
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

    if (!render_init()) {
        fprintf(stderr, "[overlay] render_init failed; aborting\n");
        return 1;
    }

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
    int drag_idx       = -1;
    int drag_grab_mx   = 0;
    int drag_grab_my   = 0;
    int hover_idx      = -1;
    int mouse_x        = 0;
    int mouse_y        = 0;

    uint64_t last_hash = 0;
    int      have_drawn = 0;
    unsigned long frame = 0;

    for (;;) {
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
                        int idx = vsrg_hit_test(&snap, width, height,
                                                mouse_x, mouse_y);
                        if (idx >= 0) {
                            drag_idx     = idx;
                            drag_grab_mx = mouse_x;
                            drag_grab_my = mouse_y;
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

                VsrgResolvedBox rb_i = vsrg_resolve_box(&wi, width, height);
                float new_resolved_x = rb_i.px + dx_px;
                float new_resolved_y = rb_i.py + dy_px;
                float nx, ny;
                vsrg_reverse_anchor(&wi, width, height, rb_i.pw, rb_i.ph,
                                    new_resolved_x, new_resolved_y, &nx, &ny);
                if (nx < 0.0f) nx = 0.0f;
                if (nx > 1.0f) nx = 1.0f;
                if (ny < 0.0f) ny = 0.0f;
                if (ny > 1.0f) ny = 1.0f;
                VsrgOverlayWidget *wi_shm = &g_shm->widgets[i];
                wi_shm->x = nx;
                wi_shm->y = ny;
            }

            drag_grab_mx = mouse_x;
            drag_grab_my = mouse_y;
            shm_read(&snap);
        }

        // ── Hover highlight in edit mode ────────────────────
        hover_idx = -1;
        if (edit_mode && have) {
            hover_idx = vsrg_hit_test(&snap, width, height, mouse_x, mouse_y);
        }

        // ── Hash to skip no-op frames ───────────────────────
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

        render_begin_frame(width, height);
        if (have) {
            vsrg_draw_widgets(&snap, width, height);
            if (edit_mode) {
                vsrg_draw_edit_decorations(&snap, width, height, hover_idx);
            }
        }
        render_end_frame();

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
