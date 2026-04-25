// Private backend contract. Each backend exposes three functions; the
// dispatcher in input.c picks one at init time and calls it per frame.
//
// Raw state (no edge detection) ; the dispatcher computes the edges
// against the previous poll so backends only have to answer "what is
// true right now?"

#ifndef VSRG_OVERLAY_INPUT_BACKEND_H
#define VSRG_OVERLAY_INPUT_BACKEND_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int     valid;
    int     mouse_x;
    int     mouse_y;
    uint8_t primary_button_down;
    uint8_t shift_down;
    uint8_t tab_down;
} VsrgInputRaw;

typedef struct {
    const char *name;                      // for logging
    int  (*init)(void);                    // 1 on success
    void (*set_surface)(uintptr_t handle); // may be NULL
    void (*poll)(int w, int h, VsrgInputRaw *out);
    void (*shutdown)(void);
} VsrgInputBackend;

// Backends implemented elsewhere.
extern const VsrgInputBackend vsrg_input_backend_x11_poll;
extern const VsrgInputBackend vsrg_input_backend_x11_xi2;

#endif
