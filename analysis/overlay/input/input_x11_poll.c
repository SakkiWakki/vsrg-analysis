// X11 poll backend — opens an independent Display connection and asks
// the server for keyboard + pointer state each frame. Same idea as
// MangoHud's X11 keybind path.
//
// Why a second Display: when we're loaded into a Wine process, Wine
// owns its own Display connection for event pumping; we must not
// poke at it. X11 is multi-client, so dialing $DISPLAY from our side
// gets us a parallel connection to the same server. Both clients
// receive consistent answers about the global keyboard + pointer.
//
// The tradeoff is that this is a polling backend, not an event one:
// input between frames is not observed. At osu!'s typical frame
// rates (>300 Hz) this is fine for edit-mode toggling and drag.
//
// TODO: windowed osu!. For the first cut we use root-relative pointer
// coords directly, which only match the render surface when the game
// is fullscreen (exclusive or borderless fullscreen covering 0,0→WxH).
// To support windowed mode we need to translate via the game window's
// origin — ``set_surface`` records the Window XID and we can call
// XTranslateCoordinates on each poll to map root→client. Left for
// later since osu!stable-with-overlay is a fullscreen-first flow.

#include "input_backend.h"

#include <X11/Xlib.h>
#include <X11/keysym.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Display *g_dpy          = NULL;
static Window   g_surface      = 0;    // informational — see note below
static KeyCode  g_kc_shift_l   = 0;
static KeyCode  g_kc_shift_r   = 0;
static KeyCode  g_kc_tab       = 0;

// Override Xlib's default handler, which calls exit() on BadWindow etc.
static int x11_poll_error_handler(Display *dpy, XErrorEvent *ev) {
    (void)dpy;
    (void)ev;
    return 0;
}

static int x11_poll_init(void) {
    if (g_dpy) return 1;
    g_dpy = XOpenDisplay(NULL);
    if (!g_dpy) {
        fprintf(stderr, "[input/x11_poll] XOpenDisplay(NULL) failed "
                        "- DISPLAY not set or unreachable\n");
        return 0;
    }
    // Install once per process. This is a global handler — if Wine
    // itself installs one later we'll get replaced, which is fine
    // (Wine's handler is also non-fatal for its own errors).
    XSetErrorHandler(x11_poll_error_handler);
    g_kc_shift_l = XKeysymToKeycode(g_dpy, XK_Shift_L);
    g_kc_shift_r = XKeysymToKeycode(g_dpy, XK_Shift_R);
    g_kc_tab     = XKeysymToKeycode(g_dpy, XK_Tab);
    return 1;
}

static void x11_poll_set_surface(uintptr_t handle) {
    g_surface = (Window)handle;
}

static int keycode_pressed(const char keymap[32], KeyCode kc) {
    if (kc == 0) return 0;
    return !!(keymap[kc >> 3] & (1 << (kc & 7)));
}

static void x11_poll_poll(int w, int h, VsrgInputRaw *out) {
    (void)w;
    (void)h;
    if (!g_dpy) return;

    char keymap[32];
    XQueryKeymap(g_dpy, keymap);

    out->shift_down = keycode_pressed(keymap, g_kc_shift_l)
                   || keycode_pressed(keymap, g_kc_shift_r);
    out->tab_down   = keycode_pressed(keymap, g_kc_tab);

    // Always query against *our* Display's root window. The surface
    // handle the host passes is a Window XID from a different Xlib
    // connection (Wine's); XIDs are not shared across Display
    // connections, so querying it here yields BadWindow.
    //
    // Root-relative coords are fine for fullscreen games — osu!stable
    // covers 0,0 → WxH. TODO: windowed mode needs a root→client
    // translation step; we still record the handle in g_surface so a
    // future version can use it.
    Window root = DefaultRootWindow(g_dpy);
    Window root_ret, child_ret;
    int root_x = 0, root_y = 0;
    int win_x  = 0, win_y  = 0;
    unsigned int mask = 0;

    if (XQueryPointer(g_dpy, root,
                      &root_ret, &child_ret,
                      &root_x, &root_y,
                      &win_x,  &win_y,
                      &mask)) {
        out->mouse_x = root_x;
        out->mouse_y = root_y;
        out->primary_button_down = !!(mask & Button1Mask);
        out->valid = 1;
    }
}

static void x11_poll_shutdown(void) {
    if (g_dpy) {
        XCloseDisplay(g_dpy);
        g_dpy = NULL;
    }
    g_surface    = 0;
    g_kc_shift_l = 0;
    g_kc_shift_r = 0;
    g_kc_tab     = 0;
}

const VsrgInputBackend vsrg_input_backend_x11_poll = {
    .name        = "x11_poll",
    .init        = x11_poll_init,
    .set_surface = x11_poll_set_surface,
    .poll        = x11_poll_poll,
    .shutdown    = x11_poll_shutdown,
};
