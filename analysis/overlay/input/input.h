// Game-agnostic input snapshot for the overlay.
//
// The overlay has two jobs that need input: toggling edit mode, and
// dragging a widget. Both are driven from a per-frame snapshot of the
// user's keyboard + mouse state ; ``vsrg_input_poll`` fills one in.
//
// We intentionally do NOT expose X11 (or evdev, or Wayland) types
// through this header. Hosts speak in semantic fields only:
//
//   edit_toggle_pressed   ; one-frame rising edge of the bind
//                           (shift+tab today)
//   mouse_x / mouse_y     ; client-area pixels, same coord space as
//                           the renderer's begin_frame(w, h)
//   primary_button_down   ; current level of left mouse
//   primary_button_pressed / _released ; one-frame edges
//
// This lets us swap the backend (poll vs. event-hook vs. evdev) without
// touching the host. Pick one at init via VSRG_INPUT_BACKEND; default
// is "poll" which uses XQueryKeymap/XQueryPointer on our own X11
// connection (see input_x11_poll.c for why ; MangoHud does the same).

#ifndef VSRG_OVERLAY_INPUT_H
#define VSRG_OVERLAY_INPUT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int valid;                        // 0 if backend couldn't read state

    int mouse_x;                      // client-area pixels
    int mouse_y;

    uint8_t primary_button_down;      // level
    uint8_t primary_button_pressed;   // edge: down this frame
    uint8_t primary_button_released;  // edge: up this frame

    uint8_t edit_toggle_pressed;      // edge: shift+tab this frame
} VsrgInputState;

// Initialise the configured backend. Idempotent; safe to call every
// frame (the first call attaches, subsequent calls are no-ops until
// shutdown). Returns 1 on success, 0 if no backend could attach.
int  vsrg_input_init(void);

// Tell the backend which X11 window (or equivalent surface handle)
// the renderer is drawing into, so cursor coordinates can be mapped
// into client-area pixels. ``handle`` is a Window XID cast to
// uintptr_t; passing 0 resets to "no target" (cursor coords will be
// root-relative). Hosts that render into a window they created
// (gamescope binary) call this once; the gl_layer calls it each
// frame with the drawable glXSwapBuffers received.
//
// We take uintptr_t instead of Window so callers that don't include
// Xlib.h (e.g. C++ hosts) don't have to.
void vsrg_input_set_surface(uintptr_t handle);

// Fill ``out`` with the current semantic state. Edges are computed
// here against the previous poll, so the caller only needs to react
// to the *_pressed/_released fields (they're true for exactly one
// frame after the transition).
//
// ``surface_w`` / ``surface_h`` is the drawable size the host is
// rendering into ; some backends need it to clip root-relative
// cursor coords into the surface.
void vsrg_input_poll(int surface_w, int surface_h, VsrgInputState *out);

void vsrg_input_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
