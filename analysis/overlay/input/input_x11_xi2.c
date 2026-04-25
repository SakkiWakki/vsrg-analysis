// X11 XInput2 raw-event backend.
//
// Why this exists: the polling backend samples key state at the host's
// swap rate. osu!stable drops to ~30 Hz on menus, and a shift+tab tap
// can complete entirely between two samples ; both keys appear to flip
// simultaneously, which misses the rising-edge semantics the overlay's
// edit-mode toggle relies on.
//
// XI2 raw events are delivered by the X server as they happen, queued
// per-client on our own Display connection. We drain the queue each
// poll(), fold the events into level state, and return that ; the
// dispatcher in input.c still turns the level state into edges, but
// now the level already reflects every transition that happened since
// the last poll, not just the state at sample time.
//
// "Raw" vs. regular XI2 events matters because osu! often grabs the
// pointer (and sometimes keyboard) on menus; regular events would stop
// flowing to us during the grab. Raw events are delivered to any client
// that asked for them regardless of grabs ; same mechanism MangoHud
// uses for its F12 keybind.
//
// The backend still has to know "is shift down right now?" so we keep a
// level map keyed by keycode. At startup we sync it by running an
// XQueryKeymap once; after that every raw key event updates it, and we
// only return the shift_down / tab_down bits (the public interface
// doesn't need arbitrary keys).

#include "input_backend.h"

#include <X11/Xlib.h>
#include <X11/extensions/XInput2.h>
#include <X11/keysym.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Display *g_dpy          = NULL;
static int      g_xi_opcode    = 0;
static KeyCode  g_kc_shift_l   = 0;
static KeyCode  g_kc_shift_r   = 0;
static KeyCode  g_kc_tab       = 0;

// Level mirror updated from raw key events. Covers the full X11 keycode
// range (8..255). 1 = down.
static uint8_t  g_keys_down[256];
static uint8_t  g_button1_down = 0;

static int xi2_error_handler(Display *dpy, XErrorEvent *ev) {
    (void)dpy; (void)ev;
    return 0;  // never kill the host process
}

static void resync_keys_from_server(void) {
    // Prime the level map once from the server. Without this, a key
    // that was already held when we initialised would look "up" until
    // its next transition.
    char keymap[32];
    XQueryKeymap(g_dpy, keymap);
    for (int kc = 0; kc < 256; kc++) {
        g_keys_down[kc] = (keymap[kc >> 3] >> (kc & 7)) & 1;
    }
}

static int xi2_init(void) {
    if (g_dpy) return 1;
    g_dpy = XOpenDisplay(NULL);
    if (!g_dpy) {
        fprintf(stderr, "[input/xi2] XOpenDisplay(NULL) failed\n");
        return 0;
    }
    XSetErrorHandler(xi2_error_handler);

    int event, error;
    if (!XQueryExtension(g_dpy, "XInputExtension",
                         &g_xi_opcode, &event, &error)) {
        fprintf(stderr, "[input/xi2] XInput extension not present\n");
        XCloseDisplay(g_dpy);
        g_dpy = NULL;
        return 0;
    }
    int major = 2, minor = 0;
    if (XIQueryVersion(g_dpy, &major, &minor) != Success) {
        fprintf(stderr, "[input/xi2] XInput2 not supported\n");
        XCloseDisplay(g_dpy);
        g_dpy = NULL;
        return 0;
    }

    // Select raw events on the root. Raw = delivered regardless of
    // focus/grabs, and regardless of whether anyone else has selected
    // events on the same window. Every X client that asks for raw
    // events receives its own copy.
    XIEventMask mask;
    unsigned char bits[(XI_LASTEVENT + 7) / 8];
    memset(bits, 0, sizeof(bits));
    XISetMask(bits, XI_RawKeyPress);
    XISetMask(bits, XI_RawKeyRelease);
    XISetMask(bits, XI_RawButtonPress);
    XISetMask(bits, XI_RawButtonRelease);
    mask.deviceid = XIAllMasterDevices;
    mask.mask_len = sizeof(bits);
    mask.mask     = bits;
    if (XISelectEvents(g_dpy, DefaultRootWindow(g_dpy), &mask, 1)
            != Success) {
        fprintf(stderr, "[input/xi2] XISelectEvents failed\n");
        XCloseDisplay(g_dpy);
        g_dpy = NULL;
        return 0;
    }
    XFlush(g_dpy);

    g_kc_shift_l = XKeysymToKeycode(g_dpy, XK_Shift_L);
    g_kc_shift_r = XKeysymToKeycode(g_dpy, XK_Shift_R);
    g_kc_tab     = XKeysymToKeycode(g_dpy, XK_Tab);

    memset(g_keys_down, 0, sizeof(g_keys_down));
    g_button1_down = 0;
    resync_keys_from_server();
    // Also prime mouse button1 from the current pointer state.
    Window rr, cr; int rx, ry, wx, wy; unsigned int mask_ret = 0;
    if (XQueryPointer(g_dpy, DefaultRootWindow(g_dpy),
                      &rr, &cr, &rx, &ry, &wx, &wy, &mask_ret)) {
        g_button1_down = !!(mask_ret & Button1Mask);
    }

    fprintf(stderr, "[input/xi2] attached (XI %d.%d)\n", major, minor);
    return 1;
}

static void xi2_set_surface(uintptr_t handle) {
    (void)handle;  // raw events don't care which window has focus
}

static void drain_events(void) {
    // Non-blocking drain. XPending only returns events already in our
    // client's queue, so this never waits on the server.
    while (XPending(g_dpy) > 0) {
        XEvent ev;
        XNextEvent(g_dpy, &ev);
        if (ev.xcookie.type != GenericEvent
                || ev.xcookie.extension != g_xi_opcode) {
            continue;
        }
        if (!XGetEventData(g_dpy, &ev.xcookie)) continue;

        XIRawEvent *re = (XIRawEvent *)ev.xcookie.data;
        // Drop auto-repeat synthetic events. When a user holds a key,
        // X generates release/press pairs at the repeat rate; without
        // this filter the edge detector sees each pair as a new tap
        // and fires the shift+tab toggle repeatedly (observed flicker).
        int is_repeat = (re->flags & XIKeyRepeat) != 0;
        switch (ev.xcookie.evtype) {
            case XI_RawKeyPress:
                if (!is_repeat && re->detail >= 0 && re->detail < 256) {
                    g_keys_down[re->detail] = 1;
                }
                break;
            case XI_RawKeyRelease:
                if (!is_repeat && re->detail >= 0 && re->detail < 256) {
                    g_keys_down[re->detail] = 0;
                }
                break;
            case XI_RawButtonPress:
                if (re->detail == 1) g_button1_down = 1;
                break;
            case XI_RawButtonRelease:
                if (re->detail == 1) g_button1_down = 0;
                break;
            default: break;
        }
        XFreeEventData(g_dpy, &ev.xcookie);
    }
}

static void xi2_poll(int w, int h, VsrgInputRaw *out) {
    (void)w; (void)h;
    if (!g_dpy) return;

    drain_events();

    out->shift_down = g_keys_down[g_kc_shift_l]
                   || g_keys_down[g_kc_shift_r];
    out->tab_down   = g_keys_down[g_kc_tab];
    out->primary_button_down = g_button1_down;

    // Pointer position still needs a query ; raw motion events are
    // relative deltas and reconstructing absolute screen coords from
    // them is more fragile than just asking the server.
    Window root = DefaultRootWindow(g_dpy);
    Window rr, cr;
    int rx = 0, ry = 0, wx = 0, wy = 0;
    unsigned int mask_ret = 0;
    if (XQueryPointer(g_dpy, root, &rr, &cr,
                      &rx, &ry, &wx, &wy, &mask_ret)) {
        out->mouse_x = rx;
        out->mouse_y = ry;
        // Belt-and-braces: if we somehow missed a button event (e.g.
        // clicked before init synced), trust the current mask.
        out->primary_button_down =
            g_button1_down || !!(mask_ret & Button1Mask);
        out->valid = 1;
    }
}

static void xi2_shutdown(void) {
    if (g_dpy) {
        XCloseDisplay(g_dpy);
        g_dpy = NULL;
    }
    g_xi_opcode    = 0;
    g_kc_shift_l   = 0;
    g_kc_shift_r   = 0;
    g_kc_tab       = 0;
    g_button1_down = 0;
    memset(g_keys_down, 0, sizeof(g_keys_down));
}

const VsrgInputBackend vsrg_input_backend_x11_xi2 = {
    .name        = "x11_xi2",
    .init        = xi2_init,
    .set_surface = xi2_set_surface,
    .poll        = xi2_poll,
    .shutdown    = xi2_shutdown,
};
