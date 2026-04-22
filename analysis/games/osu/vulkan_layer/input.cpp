// Global hotkey polling for the Vulkan layer.
//
// Pattern borrowed from vkBasalt's keyboard_input_x11.cpp: open a
// private Display* to whichever X server is on $DISPLAY (the Wine
// one in our case), call XQueryKeymap every present to read the
// current keyboard state, detect edges to flip the HUD's edit mode.
//
// Why not hook vkCreate*SurfaceKHR and install an X event listener?
// Would work but adds threading complexity and needs us to share the
// game's Display* without stepping on its event queue. XQueryKeymap
// on a separate Display is focus-independent and stateless.

#include "overlay.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <X11/Xlib.h>
#include <X11/keysym.h>

namespace vsrg {

namespace {

Display* g_dpy = nullptr;
bool     g_dpy_tried = false;
KeyCode  g_tab_kc    = 0;
bool     g_prev_chord_down = false;

void ensure_display(void) {
    if (g_dpy || g_dpy_tried) return;
    g_dpy_tried = true;
    // getenv("DISPLAY") may be null in odd environments; XOpenDisplay
    // accepts null and uses the default.
    g_dpy = XOpenDisplay(nullptr);
    if (!g_dpy) {
        std::fprintf(stderr,
                     "[vsrg-layer] XOpenDisplay failed; "
                     "Shift+Tab edit toggle disabled\n");
        return;
    }
    g_tab_kc = XKeysymToKeycode(g_dpy, XK_Tab);
    if (g_tab_kc == 0) {
        std::fprintf(stderr,
                     "[vsrg-layer] XKeysymToKeycode(XK_Tab) failed\n");
    }
}

// Is the given keycode pressed in the XQueryKeymap result?
bool keymap_has(const char keys[32], KeyCode kc) {
    if (kc == 0) return false;
    return (keys[kc / 8] & (1 << (kc % 8))) != 0;
}

// Any of the Shift keycodes down?
bool any_shift_down(Display* dpy, const char keys[32]) {
    KeyCode l = XKeysymToKeycode(dpy, XK_Shift_L);
    KeyCode r = XKeysymToKeycode(dpy, XK_Shift_R);
    return keymap_has(keys, l) || keymap_has(keys, r);
}

}  // namespace

bool input_poll_edit_toggle(void) {
    ensure_display();
    if (!g_dpy || g_tab_kc == 0) return false;
    char keys[32];
    XQueryKeymap(g_dpy, keys);
    bool chord_down = keymap_has(keys, g_tab_kc) && any_shift_down(g_dpy, keys);
    bool edge = chord_down && !g_prev_chord_down;
    g_prev_chord_down = chord_down;
    return edge;
}

}  // namespace vsrg
