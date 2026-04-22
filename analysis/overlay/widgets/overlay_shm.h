// Generic widget-slot shared-memory contract between any Python
// publisher and the gamescope overlay binary.
//
// Path convention: /dev/shm/vsrg_overlay
// One shm segment per session, shared by every Python publisher
// (legacy register_overlay plugins + unified components). The C
// overlay reads this single segment; new plugins just add widgets
// to the next frame instead of standing up their own shm + binary.
//
// Layout:
//   - Fixed-size header (magic/version/seq + edit-mode state).
//   - Fixed array of 32 widget slots. Each slot has a kind byte
//     (0 = unused), stable widget_id for identity, normalized
//     position/size, color, and either a text payload or nothing
//     (for rect widgets).
//
// Concurrency: seqlock on `seq` (odd = writer mid-update). The
// Python publisher is the primary writer. In edit mode the C
// overlay *also* writes, but only into each widget's (x, y)
// fields — the publisher treats those fields as read-from-shm
// rather than authoritative on its side. That's a one-way drag
// handoff; no cross-seqlock contention because the publisher
// only *reads* (x, y) then writes the full slot once per tick.

#ifndef VSRG_OVERLAY_SHM_H
#define VSRG_OVERLAY_SHM_H

#include <stdint.h>

#define VSRG_OVERLAY_MAGIC        0x56524F56u   // 'VROV'
// v2: added group_id so widgets in the same group drag together.
#define VSRG_OVERLAY_VERSION      2
// Bumped from 32 → 128 when the overlay collapsed to one shared
// segment. Multiple plugins now share each frame, so the cap is the
// session-wide widget budget, not a per-plugin one.
#define VSRG_OVERLAY_MAX_WIDGETS  128
#define VSRG_OVERLAY_TEXT_LEN     48

// Widget.kind
#define VSRG_OVERLAY_KIND_UNUSED  0u
#define VSRG_OVERLAY_KIND_RECT    1u
#define VSRG_OVERLAY_KIND_TEXT    2u

// Widget.anchor — the corner of the canvas (x, y) is measured
// from. Lets publishers pin widgets to an edge without knowing
// the resolution. 0 = top-left, 1 = top-right, 2 = bottom-left,
// 3 = bottom-right, 4 = center.
#define VSRG_OVERLAY_ANCHOR_TL    0u
#define VSRG_OVERLAY_ANCHOR_TR    1u
#define VSRG_OVERLAY_ANCHOR_BL    2u
#define VSRG_OVERLAY_ANCHOR_BR    3u
#define VSRG_OVERLAY_ANCHOR_C     4u

typedef struct {
    uint8_t  kind;        // 0 = unused slot
    uint8_t  anchor;      // one of VSRG_OVERLAY_ANCHOR_*
    uint8_t  _pad0[2];
    uint32_t widget_id;   // stable across frames; FNV of id string

    // Grouping for drag. 0 = standalone (drag moves only this
    // widget). Non-zero = drag moves every widget that shares the
    // same group_id by the same pixel delta. Publishers use this
    // to compose a HUD out of many drawing primitives while the
    // user still perceives and drags it as one unit.
    uint32_t group_id;

    // All coords normalized to [0, 1] of canvas width/height.
    // Width/height for rects; for text, w/h are unused (size is
    // implied by text length + px_scale).
    float x, y, w, h;

    // RGBA8 (R in byte 0, A in byte 3).
    uint32_t color;

    // Text payload (null-terminated, truncated at TEXT_LEN-1).
    // Unused for rect widgets.
    char text[VSRG_OVERLAY_TEXT_LEN];

    // Font scale in pixels per font-unit for text widgets.
    // 1.0 = 8px tall; 2.0 = 16px tall.
    float px_scale;
} VsrgOverlayWidget;

typedef struct {
    uint32_t magic;         // VSRG_OVERLAY_MAGIC
    uint32_t version;       // VSRG_OVERLAY_VERSION
    volatile uint32_t seq;  // seqlock counter
    uint32_t n_widgets;     // count of active slots (rest are unused)

    // Edit-mode bridge. The C overlay writes these; the Python
    // publisher reads them to persist drag deltas.
    //   edit_mode: 0 normal, 1 edit (shift+tab toggles).
    //   drag_active: 1 while the user is holding the mouse button
    //       down on a widget. While set, the publisher must NOT
    //       re-stamp the dragged widget's (x, y) — the C side is
    //       the source of truth for position until release, so that
    //       the user sees the widget follow the cursor instead of
    //       snapping back to baseline+delta each frame.
    //   dragged_widget_id: last widget whose (x, y) C moved. 0 if none.
    //   dragged_seq: bumps on ButtonRelease so the publisher captures
    //       the final (x, y) exactly once and writes it to config.
    uint8_t  edit_mode;
    uint8_t  drag_active;
    uint8_t  _pad0[2];
    uint32_t dragged_widget_id;
    uint32_t dragged_seq;
    uint32_t _pad1;

    VsrgOverlayWidget widgets[VSRG_OVERLAY_MAX_WIDGETS];
} VsrgOverlayShm;

#endif
