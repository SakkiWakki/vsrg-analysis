// Shared-memory contract between the Python live poller (producer)
// and the C gamescope overlay (consumer).
//
// Path: /dev/shm/osu_live_overlay (POSIX shm_open("/osu_live_overlay"))
// Size: sizeof(OsuLiveShm), rounded up to page. mmap'd shared RW.
//
// Synchronization is a seqlock: the writer bumps `seq` to an odd
// value before mutating fields, mutates them, then bumps to the
// next even value. Readers loop: read seq0; memory-barrier; read
// fields; memory-barrier; read seq1. If seq0 == seq1 and both are
// even, the read is consistent. No locks, no syscalls — good for
// a 60 Hz redraw loop.
//
// We deliberately use fixed-size arrays (no pointers, no lengths
// that could exceed the array) so the consumer can treat the
// mapping as a plain POD struct and never chase indirection into
// an address space it doesn't own.
#ifndef OSU_LIVE_SHM_LAYOUT_H
#define OSU_LIVE_SHM_LAYOUT_H

#include <stdint.h>

#define OSU_LIVE_SHM_PATH  "/osu_live_overlay"
#define OSU_LIVE_SHM_MAGIC 0x4F53554Cu   // 'OSUL'
#define OSU_LIVE_SHM_VERSION 1
#define OSU_LIVE_SHM_MAP_TITLE_LEN 128
#define OSU_LIVE_SHM_HIST_BINS 41   // ±100 ms in 5 ms bins

typedef struct {
    uint32_t magic;            // must equal OSU_LIVE_SHM_MAGIC
    uint32_t version;          // bumped on schema-breaking changes
    volatile uint32_t seq;     // seqlock counter (see header note)
    uint32_t _pad0;

    // Gameplay state. `in_gameplay` gates the overlay's visibility.
    uint8_t  connected;        // poller has live data (vs. tosu down)
    uint8_t  in_gameplay;      // GameState.play (vs. menu / results)
    uint8_t  keycount;         // mania key count; 4K / 7K
    uint8_t  _pad1;

    int32_t  combo;
    int32_t  max_combo;
    int32_t  hits_300, hits_100, hits_50, hits_miss;

    float    accuracy;         // 0..100
    float    unstable_rate;    // ms (10 * stdev of hit offsets)

    // Hit-offset histogram: bin i covers [-100 + 5*i, -95 + 5*i] ms.
    // Producer rebuilds from snapshot.offsets on each tick — cheap.
    uint32_t hist[OSU_LIVE_SHM_HIST_BINS];

    // Null-terminated UTF-8 title. Fixed buffer so the consumer
    // never has to malloc.
    char     map_title[OSU_LIVE_SHM_MAP_TITLE_LEN];
} OsuLiveShm;

#endif
