#include "win_shm_consumer.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string.h>
#include <stdio.h>

// No namespace prefix ; Windows puts the mapping in the session-default
// namespace, which both the publisher (our GUI) and the consumer (osu!.exe
// as our child process) share. Avoids Global\'s SeCreateGlobalPrivilege
// requirement and Local\'s odd interaction with CPython's mmap tagname.
#define SHM_NAME "vsrg_overlay"

static HANDLE           g_map     = NULL;
static VsrgOverlayShm  *g_shm     = NULL;
static VsrgOverlayShm  *g_shm_mut = NULL;

int shm_consumer_ensure(void) {
    if (g_shm) return 1;

    g_map = OpenFileMappingA(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, SHM_NAME);
    int writable = 1;
    if (!g_map) {
        g_map = OpenFileMappingA(FILE_MAP_READ, FALSE, SHM_NAME);
        writable = 0;
    }
    if (!g_map) return 0;

    DWORD access = writable ? (FILE_MAP_READ | FILE_MAP_WRITE) : FILE_MAP_READ;
    void *p = MapViewOfFile(g_map, access, 0, 0, sizeof(VsrgOverlayShm));
    if (!p) {
        CloseHandle(g_map);
        g_map = NULL;
        return 0;
    }

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
        uint32_t s0 = InterlockedCompareExchange(
            (volatile LONG *)&g_shm->seq, 0, 0);
        if (s0 & 1u) continue;
        memcpy(out, (const void *)g_shm, sizeof(*out));
        MemoryBarrier();
        uint32_t s1 = InterlockedCompareExchange(
            (volatile LONG *)&g_shm->seq, 0, 0);
        if (s0 == s1) {
            return out->magic   == VSRG_OVERLAY_MAGIC
                && out->version == VSRG_OVERLAY_VERSION;
        }
    }
    return 0;
}
