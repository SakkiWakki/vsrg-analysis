// Native gamescope external-overlay client.
//
// Runs as a sibling client inside the gamescope nested X server
// (same one osu!-under-wine is rendering into). We do *not* touch
// osu!'s render path — gamescope composites our surface on top
// via the GAMESCOPE_EXTERNAL_OVERLAY window property.
//
// Data is pulled from /dev/shm/osu_live_overlay, a seqlock-guarded
// POD struct written by plugins/unsafe/osu_live/shm_publisher.py.
// The overlay polls the region at 60 Hz; if the publisher hasn't
// touched it yet, we fall back to a "waiting for feed" banner so
// the user can tell whether the data plumbing or the overlay
// plumbing is broken.
//
// Rendering: immediate-mode GL, fixed-function pipeline, no
// shaders / VBOs / textures. Text is an 8x8 bitmap font
// (font8x8.h) drawn as GL_QUADS per pixel. Fine for tens of
// glyphs at 60 Hz.

#define _GNU_SOURCE
#include <GL/gl.h>
#include <GL/glx.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/extensions/shape.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "font8x8.h"
#include "shm_layout.h"

// ─── Shared memory consumer ─────────────────────────────────────────────

static OsuLiveShm *g_shm;

static void shm_attach(void) {
    int fd = open("/dev/shm/osu_live_overlay", O_RDONLY);
    if (fd < 0) {
        // Publisher not up yet — we'll draw a waiting banner.
        return;
    }
    struct stat st;
    if (fstat(fd, &st) < 0 || st.st_size < (off_t)sizeof(OsuLiveShm)) {
        close(fd);
        return;
    }
    void *p = mmap(NULL, sizeof(OsuLiveShm), PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return;
    g_shm = (OsuLiveShm *)p;
}

// Seqlock read: loops until two bracketing reads of `seq` agree
// and are even. Returns 1 on success (state filled), 0 if shm not
// attached or the header is wrong.
static int shm_read(OsuLiveShm *out) {
    if (!g_shm) return 0;
    for (int tries = 0; tries < 16; tries++) {
        uint32_t s0 = __atomic_load_n(&g_shm->seq, __ATOMIC_ACQUIRE);
        if (s0 & 1u) continue;   // writer mid-update; spin
        memcpy(out, (const void *)g_shm, sizeof(*out));
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        uint32_t s1 = __atomic_load_n(&g_shm->seq, __ATOMIC_ACQUIRE);
        if (s0 == s1) {
            return out->magic == OSU_LIVE_SHM_MAGIC
                && out->version == OSU_LIVE_SHM_VERSION;
        }
    }
    return 0;
}

// ─── X11 + GL setup ─────────────────────────────────────────────────────

static void set_cardinal_prop(Display *dpy, Window w,
                              const char *name, unsigned long value) {
    Atom a = XInternAtom(dpy, name, False);
    XChangeProperty(dpy, w, a, XA_CARDINAL, 32, PropModeReplace,
                    (unsigned char *)&value, 1);
}

// ─── Drawing helpers ────────────────────────────────────────────────────

// Draw a filled 2D rect in current color. Assumes GL_QUADS already
// begun, or the caller wraps.
static void rect(float x, float y, float w, float h) {
    glVertex2f(x,     y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x,     y + h);
}

// Render a single 8x8 glyph at (x, y) with pixel size `px`.
// Uses GL_QUADS — caller wraps glBegin/glEnd for throughput.
static void draw_glyph(char c, float x, float y, float px) {
    const Glyph8x8 *g = font_glyph(c);
    for (int row = 0; row < 8; row++) {
        uint8_t bits = (*g)[row];
        for (int col = 0; col < 8; col++) {
            if (bits & (0x80 >> col)) {
                float gx = x + col * px;
                float gy = y + row * px;
                rect(gx, gy, px, px);
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
        cx += 9.0f * px;   // 8 px glyph + 1 px kerning
    }
    glEnd();
}

static void draw_filled_rect(float x, float y, float w, float h) {
    glBegin(GL_QUADS);
    rect(x, y, w, h);
    glEnd();
}

// ─── Main overlay layout ────────────────────────────────────────────────

static void format_accuracy(float acc, char *buf, size_t n) {
    snprintf(buf, n, "%.2f%%", acc);
}

static void render_hud(int width, int height, const OsuLiveShm *s) {
    (void)width;
    // Origin is top-left after the ortho call in main().
    const float pad = 24.0f;

    // Background panel — subtly dark so text stays legible over
    // bright maps. Height sized for 3 lines of text + histogram.
    glColor4f(0.04f, 0.04f, 0.06f, 0.55f);
    float panel_x = pad;
    float panel_y = pad;
    float panel_w = 520.0f;
    float panel_h = 220.0f;
    draw_filled_rect(panel_x, panel_y, panel_w, panel_h);

    // Accent bar at top of panel.
    glColor4f(0.29f, 0.64f, 1.0f, 0.95f);
    draw_filled_rect(panel_x, panel_y, panel_w, 3.0f);

    // Text lines.
    glColor4f(0.98f, 0.98f, 0.98f, 1.0f);

    char line[128];
    float tx = panel_x + 18.0f;
    float ty = panel_y + 18.0f;
    float px = 2.5f;   // pixel scale for the 8x8 font

    // Line 1: combo.
    snprintf(line, sizeof(line), "%dX", s->combo);
    draw_text(line, tx, ty, px);

    // Line 2: accuracy, UR.
    ty += 9.0f * px + 6.0f;
    char acc_buf[32];
    format_accuracy(s->accuracy, acc_buf, sizeof(acc_buf));
    snprintf(line, sizeof(line), "%s  UR %.1f", acc_buf, s->unstable_rate);
    draw_text(line, tx, ty, 2.0f);

    // Line 3: hit counts.
    ty += 9.0f * 2.0f + 6.0f;
    snprintf(line, sizeof(line), "%d:%d:%d:%d",
             s->hits_300, s->hits_100, s->hits_50, s->hits_miss);
    draw_text(line, tx, ty, 1.6f);

    // Histogram along the bottom of the panel.
    float hist_x = panel_x + 18.0f;
    float hist_y = panel_y + panel_h - 60.0f;
    float hist_w = panel_w - 36.0f;
    float hist_h = 48.0f;

    // Baseline.
    glColor4f(0.25f, 0.25f, 0.3f, 0.9f);
    draw_filled_rect(hist_x, hist_y + hist_h - 1.0f, hist_w, 1.0f);

    // Find the tallest bin to normalize heights.
    uint32_t peak = 1;
    for (int i = 0; i < OSU_LIVE_SHM_HIST_BINS; i++) {
        if (s->hist[i] > peak) peak = s->hist[i];
    }
    float bin_w = hist_w / (float)OSU_LIVE_SHM_HIST_BINS;

    glColor4f(0.29f, 0.64f, 1.0f, 0.9f);
    glBegin(GL_QUADS);
    for (int i = 0; i < OSU_LIVE_SHM_HIST_BINS; i++) {
        float h = ((float)s->hist[i] / (float)peak) * hist_h;
        rect(hist_x + i * bin_w, hist_y + (hist_h - h),
             bin_w - 1.0f, h);
    }
    glEnd();

    // Zero-offset marker (center of ±100 ms histogram).
    glColor4f(1.0f, 1.0f, 1.0f, 0.4f);
    draw_filled_rect(hist_x + hist_w * 0.5f - 0.5f,
                     hist_y, 1.0f, hist_h);
    (void)height;
}

static void render_banner(const char *msg, int width, int height) {
    (void)width; (void)height;
    glColor4f(0.04f, 0.04f, 0.06f, 0.55f);
    draw_filled_rect(24.0f, 24.0f, 520.0f, 48.0f);
    glColor4f(1.0f, 0.7f, 0.2f, 1.0f);
    draw_text(msg, 36.0f, 36.0f, 2.0f);
}

// ─── Entry point ────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    int width = 1920, height = 1080;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--width") && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--height") && i + 1 < argc) {
            height = atoi(argv[++i]);
        }
    }

    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) {
        fprintf(stderr, "[osu_overlay] XOpenDisplay(NULL) failed. "
                "Make sure this process is launched inside gamescope "
                "(DISPLAY=%s)\n", getenv("DISPLAY"));
        return 1;
    }
    int screen = DefaultScreen(dpy);
    Window root = RootWindow(dpy, screen);

    // FBConfig with a real 8-bit alpha channel — gamescope blends
    // us on top of osu!'s frame using this.
    int fb_attribs[] = {
        GLX_X_RENDERABLE,  True,
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_RENDER_TYPE,   GLX_RGBA_BIT,
        GLX_DOUBLEBUFFER,  True,
        GLX_RED_SIZE,      8,
        GLX_GREEN_SIZE,    8,
        GLX_BLUE_SIZE,     8,
        GLX_ALPHA_SIZE,    8,
        GLX_DEPTH_SIZE,    0,
        None
    };
    int nfb = 0;
    GLXFBConfig *fbs = glXChooseFBConfig(dpy, screen, fb_attribs, &nfb);
    if (!fbs || nfb == 0) {
        fprintf(stderr, "[osu_overlay] glXChooseFBConfig returned no configs\n");
        return 1;
    }
    GLXFBConfig fb = 0;
    XVisualInfo *vi = NULL;
    for (int i = 0; i < nfb; i++) {
        XVisualInfo *cand = glXGetVisualFromFBConfig(dpy, fbs[i]);
        if (!cand) continue;
        int a = 0;
        glXGetFBConfigAttrib(dpy, fbs[i], GLX_ALPHA_SIZE, &a);
        if (cand->depth == 32 && a == 8) {
            fb = fbs[i];
            vi = cand;
            break;
        }
        XFree(cand);
    }
    XFree(fbs);
    if (!vi) {
        fprintf(stderr, "[osu_overlay] no 32-bit (RGBA8888) FBConfig found\n");
        return 1;
    }
    printf("[osu_overlay] picked visual 0x%lx depth=%d\n",
           vi->visualid, vi->depth);

    Colormap cmap = XCreateColormap(dpy, root, vi->visual, AllocNone);
    XSetWindowAttributes swa;
    memset(&swa, 0, sizeof(swa));
    swa.colormap = cmap;
    swa.border_pixel = 0;
    swa.background_pixel = 0;
    swa.event_mask = StructureNotifyMask;

    Window win = XCreateWindow(
        dpy, root,
        0, 0, width, height, 0,
        vi->depth, InputOutput, vi->visual,
        CWColormap | CWBorderPixel | CWBackPixel | CWEventMask,
        &swa);

    XStoreName(dpy, win, "etterna-analysis osu overlay");

    // GAMESCOPE_EXTERNAL_OVERLAY = 1 tags this window for the
    // overlay layer. GAMESCOPE_NO_FOCUS is harmless on current
    // gamescope (not read) but is the mangoapp convention — keep
    // it for forward-compat.
    set_cardinal_prop(dpy, win, "GAMESCOPE_EXTERNAL_OVERLAY", 1);
    set_cardinal_prop(dpy, win, "GAMESCOPE_NO_FOCUS",         1);

    // Make the overlay input-transparent: empty ShapeInput region
    // means no pointer or keyboard events ever land here. Without
    // this, gamescope can route focus (and Alt+F4) to the overlay
    // window, so hitting Alt+F4 in osu! closes our window — which
    // terminates the gamescope session and kills osu! with it.
    // With an empty input shape, gamescope sees osu! as the only
    // viable keyboard target and Alt+F4 goes to osu!.
    int shape_evt = 0, shape_err = 0;
    if (XShapeQueryExtension(dpy, &shape_evt, &shape_err)) {
        XRectangle empty = {0, 0, 0, 0};
        XShapeCombineRectangles(dpy, win, ShapeInput, 0, 0,
                                &empty, 0, ShapeSet, Unsorted);
    } else {
        fprintf(stderr, "[osu_overlay] XShape not available — overlay "
                "may steal keyboard focus from osu!\n");
    }

    GLXContext ctx = glXCreateNewContext(dpy, fb, GLX_RGBA_TYPE, NULL, True);
    if (!ctx) {
        fprintf(stderr, "[osu_overlay] glXCreateNewContext failed\n");
        return 1;
    }

    XMapWindow(dpy, win);
    glXMakeCurrent(dpy, win, ctx);

    printf("[osu_overlay] mapped %dx%d on DISPLAY=%s (GL %s)\n",
           width, height, getenv("DISPLAY"),
           (const char *)glGetString(GL_VERSION));
    fflush(stdout);

    shm_attach();
    if (!g_shm) {
        printf("[osu_overlay] /dev/shm/osu_live_overlay not present yet; "
               "will retry on each frame\n");
        fflush(stdout);
    }

    unsigned long frame = 0;
    for (;;) {
        while (XPending(dpy)) {
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type == ConfigureNotify) {
                width  = ev.xconfigure.width;
                height = ev.xconfigure.height;
            }
        }

        // Reattach lazily so starting the publisher after the
        // overlay is fine.
        if (!g_shm) shm_attach();

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

        OsuLiveShm snap;
        int have = shm_read(&snap);
        if (!have) {
            render_banner("WAITING FOR FEED", width, height);
        } else if (!snap.connected) {
            render_banner("OSU NOT CONNECTED", width, height);
        } else if (!snap.in_gameplay) {
            // Outside of gameplay the overlay is useless clutter;
            // show nothing (osu!'s own UI is already on-screen).
        } else {
            render_hud(width, height, &snap);
        }

        glXSwapBuffers(dpy, win);
        frame++;

        if (frame == 1 || frame % 300 == 0) {
            printf("[osu_overlay] frame %lu  shm=%s  connected=%d  in_gameplay=%d\n",
                   frame,
                   g_shm ? "ok" : "none",
                   have ? snap.connected : -1,
                   have ? snap.in_gameplay : -1);
            fflush(stdout);
        }

        usleep(16 * 1000);
    }
}
