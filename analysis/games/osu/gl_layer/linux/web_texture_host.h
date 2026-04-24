// Web-texture host for the LD_PRELOAD gl_layer.
//
// Owns the socket listener thread, the fd queue, the per-render-thread
// EGLImage cache, and the resolver hook that ``widgets.c`` calls each
// frame. The listener thread never touches GL; it only receives fds
// and stashes metadata. All EGL work happens on the render thread
// inside the swap-buffers hook, where the game's GL context is
// guaranteed to be current.

#ifndef VSRG_WEB_TEXTURE_HOST_H
#define VSRG_WEB_TEXTURE_HOST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Start the listener + register the resolver on ``widgets.c``.
// Creates the Unix socket at ``/tmp/vsrg_overlay_web.sock`` (bind +
// listen + SOCK_SEQPACKET) and spins a background thread. Returns 1
// on success, 0 on any failure (socket busy, bind error, thread
// creation failed). Failure is non-fatal: the rest of the overlay
// keeps working, just without web textures.
int  web_texture_host_start(void);

// Drain every queued fd and import pending ones into GL textures in
// the current context. MUST be called from the render thread with a
// live GL context, once per frame, BEFORE ``vsrg_draw_widgets``. If
// no fds are pending, this is a cheap mutex check + early return.
void web_texture_host_tick(void);

// Stop the listener, close the socket, and release every cached
// texture. Idempotent; safe to call from ``atexit``.
void web_texture_host_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
