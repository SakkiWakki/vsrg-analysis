// Seqlock consumer for /dev/shm/vsrg_overlay.
//
// Mirrors the attach/read logic in osu_overlay.c but read-only from
// the game's process: we do not write drag state back through the
// layer yet (that belongs to step 3 once we have input wired up).

#include "shm_consumer.h"

#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define SHM_PATH "/dev/shm/vsrg_overlay"

static VsrgOverlayShm *g_shm     = NULL;
static VsrgOverlayShm *g_shm_mut = NULL;  // == g_shm if RDWR succeeded

int shm_consumer_ensure(void) {
    if (g_shm) return 1;
    // Try RDWR first so drag can publish widget moves. If the file
    // is read-only (or we raced its creation), fall back to RDONLY
    // so the HUD still renders ; drag just becomes a no-op.
    int fd       = open(SHM_PATH, O_RDWR);
    int writable = 1;
    if (fd < 0) {
        fd       = open(SHM_PATH, O_RDONLY);
        writable = 0;
    }
    if (fd < 0) return 0;
    struct stat st;
    if (fstat(fd, &st) < 0 || st.st_size < (off_t)sizeof(VsrgOverlayShm)) {
        close(fd);
        return 0;
    }
    int prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
    void *p = mmap(NULL, sizeof(VsrgOverlayShm),
                   prot, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return 0;
    g_shm     = (VsrgOverlayShm *)p;
    g_shm_mut = writable ? g_shm : NULL;
    return 1;
}

VsrgOverlayShm *shm_consumer_writable(void) {
    return g_shm_mut;
}

int shm_consumer_read(VsrgOverlayShm *out) {
    if (!g_shm) return 0;
    for (int tries = 0; tries < 16; tries++) {
        uint32_t s0 = __atomic_load_n(&g_shm->seq, __ATOMIC_ACQUIRE);
        if (s0 & 1u) continue;
        memcpy(out, (const void *)g_shm, sizeof(*out));
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        uint32_t s1 = __atomic_load_n(&g_shm->seq, __ATOMIC_ACQUIRE);
        if (s0 == s1) {
            return out->magic == VSRG_OVERLAY_MAGIC
                && out->version == VSRG_OVERLAY_VERSION;
        }
    }
    return 0;
}
