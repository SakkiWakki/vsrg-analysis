#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "input.h"
#include "input_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static HWND  g_surface    = NULL;
static int   g_have_prev  = 0;
static int   g_debug      = 0;
static unsigned long g_frame = 0;

static VsrgInputRaw g_prev;

static int win32_init(void) {
    const char *dbg = getenv("VSRG_INPUT_DEBUG");
    g_debug = (dbg && *dbg && strcmp(dbg, "0") != 0);
    fprintf(stderr, "[input/win32] attached%s\n", g_debug ? " (debug on)" : "");
    return 1;
}

static void win32_set_surface(uintptr_t handle) {
    g_surface = (HWND)handle;
}

static void win32_poll(int w, int h, VsrgInputRaw *out) {
    (void)w; (void)h;

    SHORT shift = GetAsyncKeyState(VK_SHIFT);
    SHORT tab   = GetAsyncKeyState(VK_TAB);
    SHORT lbtn  = GetAsyncKeyState(VK_LBUTTON);

    POINT pt;
    if (!GetCursorPos(&pt)) return;

    if (g_surface) {
        ScreenToClient(g_surface, &pt);
    }

    out->valid               = 1;
    out->mouse_x             = pt.x;
    out->mouse_y             = pt.y;
    out->primary_button_down = (lbtn & 0x8000) ? 1 : 0;
    out->shift_down          = (shift & 0x8000) ? 1 : 0;
    out->tab_down            = (tab   & 0x8000) ? 1 : 0;
}

static void win32_shutdown(void) {
    g_surface   = NULL;
    g_have_prev = 0;
}

static const VsrgInputBackend win32_backend = {
    "win32",
    win32_init,
    win32_set_surface,
    win32_poll,
    win32_shutdown,
};

// Implements input.h directly — no platform-agnostic input.c on Windows.

static const VsrgInputBackend *g_backend = NULL;

int vsrg_input_init(void) {
    if (g_backend) return 1;
    if (!win32_backend.init()) return 0;
    g_backend = &win32_backend;
    memset(&g_prev, 0, sizeof(g_prev));
    g_have_prev = 0;
    g_frame     = 0;
    return 1;
}

void vsrg_input_set_surface(uintptr_t handle) {
    if (g_backend && g_backend->set_surface)
        g_backend->set_surface(handle);
}

void vsrg_input_shutdown(void) {
    if (!g_backend) return;
    g_backend->shutdown();
    g_backend   = NULL;
    g_have_prev = 0;
}

void vsrg_input_poll(int surface_w, int surface_h, VsrgInputState *out) {
    memset(out, 0, sizeof(*out));
    if (!g_backend) return;

    VsrgInputRaw raw;
    memset(&raw, 0, sizeof(raw));
    g_backend->poll(surface_w, surface_h, &raw);
    if (!raw.valid) {
        g_have_prev = 0;
        return;
    }

    out->valid               = 1;
    out->mouse_x             = raw.mouse_x;
    out->mouse_y             = raw.mouse_y;
    out->primary_button_down = raw.primary_button_down;

    if (g_have_prev) {
        out->primary_button_pressed  =  raw.primary_button_down && !g_prev.primary_button_down;
        out->primary_button_released = !raw.primary_button_down &&  g_prev.primary_button_down;
        out->edit_toggle_pressed     =  raw.shift_down && raw.tab_down && !g_prev.tab_down;
    }

    g_prev      = raw;
    g_have_prev = 1;
    g_frame++;
}
