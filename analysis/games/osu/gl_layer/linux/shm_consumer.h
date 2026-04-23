// Thread-safe shm consumer for the gl_layer's render path.
//
// Called per-frame from the preload hook, which may run on any GL
// thread the game chooses. We keep one mmap'd pointer and a local
// snapshot; ``shm_consumer_read`` does a seqlock-protected copy so
// the caller can iterate the widget array without locking.

#ifndef VSRG_GL_LAYER_SHM_CONSUMER_H
#define VSRG_GL_LAYER_SHM_CONSUMER_H

#include "../../../overlay/widgets/overlay_shm.h"

#ifdef __cplusplus
extern "C" {
#endif

// Attach to /dev/shm/vsrg_overlay lazily. Safe to call repeatedly;
// once attached, it's a no-op. Returns 1 if attached, 0 if not.
int shm_consumer_ensure(void);

// Seqlock read of the current shm state into ``out``. Returns 1 on
// a consistent read with a valid magic/version, 0 otherwise (either
// the publisher hasn't started yet, the shm file is missing, or
// the writer was racing us for all 16 retry attempts).
int shm_consumer_read(VsrgOverlayShm *out);

// Writable pointer into the same shm region, or NULL if drag writes
// aren't available. Used by vsrg_drag_tick to publish widget moves +
// drag state back so the Python publisher can persist them.
//
// We try RDWR on first attach; if the file was created read-only we
// silently fall back to read-only and this returns NULL. Callers
// must treat it as "nice to have" — drag simply becomes a no-op
// when it's missing.
VsrgOverlayShm *shm_consumer_writable(void);

#ifdef __cplusplus
}
#endif

#endif
