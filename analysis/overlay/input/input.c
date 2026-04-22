// Backend dispatch + edge detection.
//
// The public API (input.h) promises semantic fields like
// ``edit_toggle_pressed``. Backends produce raw level state (is shift
// down? is tab down? is the mouse button down?) and this file turns
// that into rising/falling edges by remembering the previous poll.

#define _POSIX_C_SOURCE 200809L

#include "input.h"
#include "input_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const VsrgInputBackend *g_backend = NULL;
static VsrgInputRaw            g_prev;
static int                     g_have_prev = 0;
static int                     g_debug     = 0;  // VSRG_INPUT_DEBUG=1
static unsigned long           g_frame     = 0;

// Log only transitions, not every frame — per-frame log would flood
// the file at 1000 Hz gameplay. Each line shows the fields that
// actually changed plus the running frame index so you can tell how
// long a key was held.
static void log_raw_transitions(const VsrgInputRaw *prev,
                                const VsrgInputRaw *cur,
                                unsigned long frame) {
    if (!g_debug) return;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double t = ts.tv_sec + ts.tv_nsec * 1e-9;

    int shift_changed = prev->shift_down != cur->shift_down;
    int tab_changed   = prev->tab_down   != cur->tab_down;
    int btn_changed   =
        prev->primary_button_down != cur->primary_button_down;

    if (!shift_changed && !tab_changed && !btn_changed) return;

    fprintf(stderr,
            "[input/dbg] t=%.3f f=%lu shift=%d%s tab=%d%s btn=%d%s"
            " xy=(%d,%d)\n",
            t, frame,
            cur->shift_down, shift_changed ? "*" : "",
            cur->tab_down,   tab_changed   ? "*" : "",
            cur->primary_button_down, btn_changed ? "*" : "",
            cur->mouse_x, cur->mouse_y);
}

// Backend selection: xi2 (default) → poll (fallback). XI2 needs the
// XInput extension at the server end; every modern X server has it,
// but we keep poll reachable both as an escape hatch
// (VSRG_INPUT_BACKEND=poll) and as an automatic fallback if xi2_init
// fails for any reason.
static const VsrgInputBackend *pick_backend(void) {
    const char *name = getenv("VSRG_INPUT_BACKEND");
    if (!name || !*name) name = "xi2";
    if (strcmp(name, "xi2")  == 0) return &vsrg_input_backend_x11_xi2;
    if (strcmp(name, "poll") == 0) return &vsrg_input_backend_x11_poll;
    fprintf(stderr, "[input] unknown VSRG_INPUT_BACKEND='%s', "
                    "falling back to xi2\n", name);
    return &vsrg_input_backend_x11_xi2;
}

int vsrg_input_init(void) {
    if (g_backend) return 1;
    const VsrgInputBackend *b = pick_backend();
    if (!b->init()) {
        // If the requested backend didn't come up and it wasn't the
        // poll backend, try poll before giving up. A server without
        // XInput2 is unlikely in 2026 but not impossible.
        if (b != &vsrg_input_backend_x11_poll) {
            fprintf(stderr,
                    "[input] '%s' init failed, trying 'poll'\n", b->name);
            b = &vsrg_input_backend_x11_poll;
            if (!b->init()) return 0;
        } else {
            return 0;
        }
    }
    g_backend = b;
    memset(&g_prev, 0, sizeof(g_prev));
    g_have_prev = 0;
    g_frame     = 0;
    const char *dbg = getenv("VSRG_INPUT_DEBUG");
    g_debug = (dbg && *dbg && strcmp(dbg, "0") != 0);
    fprintf(stderr, "[input] backend='%s' attached%s\n",
            b->name, g_debug ? " (debug on)" : "");
    return 1;
}

void vsrg_input_set_surface(uintptr_t handle) {
    if (!g_backend || !g_backend->set_surface) return;
    g_backend->set_surface(handle);
}

void vsrg_input_shutdown(void) {
    if (!g_backend) return;
    g_backend->shutdown();
    g_backend = NULL;
    g_have_prev = 0;
}

void vsrg_input_poll(int surface_w, int surface_h, VsrgInputState *out) {
    memset(out, 0, sizeof(*out));
    if (!g_backend) return;

    VsrgInputRaw raw;
    memset(&raw, 0, sizeof(raw));
    g_backend->poll(surface_w, surface_h, &raw);
    if (!raw.valid) {
        // Don't cross an invalid poll with the previous valid one —
        // that would produce a spurious release edge. Drop the prev
        // history so the next valid poll starts clean.
        g_have_prev = 0;
        return;
    }

    out->valid                    = 1;
    out->mouse_x                  = raw.mouse_x;
    out->mouse_y                  = raw.mouse_y;
    out->primary_button_down      = raw.primary_button_down;

    if (g_have_prev) {
        out->primary_button_pressed =
            raw.primary_button_down && !g_prev.primary_button_down;
        out->primary_button_released =
            !raw.primary_button_down && g_prev.primary_button_down;

        // Edit-mode toggle: shift+tab rising edge. We trigger on the
        // tab transition (0→1) while shift is held, not on the shift
        // edge, so tapping tab again while still holding shift
        // re-toggles.
        out->edit_toggle_pressed =
            raw.shift_down && raw.tab_down && !g_prev.tab_down;

        log_raw_transitions(&g_prev, &raw, g_frame);
        if (g_debug && out->edit_toggle_pressed) {
            fprintf(stderr, "[input/dbg] EDIT_TOGGLE fired at f=%lu\n",
                    g_frame);
        }
    }

    g_prev      = raw;
    g_have_prev = 1;
    g_frame++;
}
