// Game-agnostic widget replay for the vsrg overlay.
//
// Given a seqlock-protected ``VsrgOverlayShm`` snapshot and a canvas
// size, this module resolves every widget's normalized (x, y) +
// anchor into pixel coordinates, then emits ``render_*`` calls from
// render.h. It is the one place where the on-screen layout math
// lives — anyone drawing the HUD (the gamescope external overlay,
// the stable GL preload layer, an imaginary future SDL backend)
// calls through here.
//
// Dependencies:
//   - analysis/overlay/widgets/overlay_shm.h (wire format)
//   - analysis/overlay/renderer/render.h      (draw primitives)

#ifndef VSRG_OVERLAY_WIDGETS_H
#define VSRG_OVERLAY_WIDGETS_H

#include <stdint.h>
#include "overlay_shm.h"
#include "../input/input.h"

#ifdef __cplusplus
extern "C" {
#endif

// px_scale (bitmap-era: 1.0 == 8 px of visible ink) → pixel height
// for the TTF renderer. The publisher addresses text in these units
// so its layout constants don't have to know which renderer is
// active. If a font swap makes text visibly wrong-sized, retune
// this scalar (not CAP_HEIGHT_RATIO inside render.c).
#define VSRG_TEXT_HEIGHT_PER_PX_SCALE  8.0f

typedef struct {
    float px, py;      // top-left in pixels
    float pw, ph;      // size in pixels (used for hit-testing)
} VsrgResolvedBox;

// Convert a widget's px_scale to a pixel line height.
float vsrg_px_height_for_scale(float px_scale);

// Measured pixel advance of ``s`` at the publisher's px_scale.
// Returns 0 if the font isn't loaded. Exact match for render_text's
// pen advance so hit-testing stays accurate.
float vsrg_measure_text(const char *s, float px_scale);

// Measured pixel height of a text line at ``px_scale``.
float vsrg_text_height(float px_scale);

// Resolve a widget's normalized (x, y, w, h) + anchor into a pixel
// bounding box for the current canvas size.
VsrgResolvedBox vsrg_resolve_box(const VsrgOverlayWidget *w,
                                 int canvas_w, int canvas_h);

// Given a desired resolved pixel top-left, compute the
// normalized, pre-anchor (x, y) that would yield it. Used during
// drag — caller typically clamps the result into [0, 1] before
// writing it back to shm.
void vsrg_reverse_anchor(const VsrgOverlayWidget *w,
                         int canvas_w, int canvas_h,
                         float pw, float ph,
                         float target_px, float target_py,
                         float *out_nx, float *out_ny);

// Pick the topmost (last-rendered) widget under a pixel, or -1.
int  vsrg_hit_test(const VsrgOverlayShm *s,
                   int canvas_w, int canvas_h,
                   int mx, int my);

// Draw every visible widget from the snapshot.
void vsrg_draw_widgets(const VsrgOverlayShm *s,
                       int canvas_w, int canvas_h);

// Overlay edit-mode decorations (screen dim + per-widget outlines +
// help strip). ``hover_idx`` < 0 means no widget is hovered.
void vsrg_draw_edit_decorations(const VsrgOverlayShm *s,
                                int canvas_w, int canvas_h,
                                int hover_idx);

// Edit-mode + drag state machine, shared by every host. The host
// owns the struct (zero-initialised is "clean: no edit, no drag");
// each frame it fills in input (from vsrg_input_poll) and the
// current shm snapshot and calls vsrg_drag_tick. The function:
//
//   * toggles ``edit_mode`` on input.edit_toggle_pressed;
//   * on primary_button_pressed, hit-tests the snapshot and starts
//     a drag (recording the grabbed widget's group);
//   * while dragged, rewrites (x, y) in ``shm_mut`` for the whole
//     group (anchor-reversed + clamped to [0, 1]);
//   * on primary_button_released, ends the drag and bumps
//     ``dragged_seq`` so the publisher knows it should persist.
//
// ``shm_mut`` is the publisher-writable mapping (same as the snapshot
// read target in the gamescope binary; NULL is tolerated and makes
// drag a no-op — useful if the host hasn't attached yet). The
// snapshot ``s`` is read-only — drag math reads positions from it.
//
// Returns the hover index (-1 if no widget hovered, only valid in
// edit mode) so the host can pass it to vsrg_draw_edit_decorations.
typedef struct {
    uint8_t edit_mode;         // persisted across frames
    int     drag_idx;          // -1 when not dragging
    int     drag_grab_mx;      // last pointer x we applied
    int     drag_grab_my;      // last pointer y we applied
} VsrgDragState;

int vsrg_drag_tick(VsrgDragState *ds,
                   const VsrgInputState *in,
                   const VsrgOverlayShm *s,
                   VsrgOverlayShm *shm_mut,
                   int canvas_w, int canvas_h);

#ifdef __cplusplus
}
#endif

#endif
