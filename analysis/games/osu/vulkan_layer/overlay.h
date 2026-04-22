// Shared state between the Vulkan-layer glue (layer.cpp) and the
// HUD/input modules (overlay.cpp, input.cpp).
//
// The goal of this split is that layer.cpp is the *only* file that
// speaks Vulkan. overlay.cpp and input.cpp see:
//   - the overlay_shm.h widget records (same shm the NanoVG path
//     reads; no publisher changes)
//   - an ImDrawList pointer to emit primitives into
// They never touch VkInstance / VkDevice / VkQueue.
//
// Threading: vkQueuePresentKHR can be called from multiple DXVK
// threads. The layer serialises overlay_tick() under its own mutex
// before handing off to this module, so the HUD module itself is
// single-threaded.

#ifndef VSRG_VULKAN_LAYER_OVERLAY_H
#define VSRG_VULKAN_LAYER_OVERLAY_H

#include <stdint.h>

struct ImDrawList;

namespace vsrg {

// Lazily opens the shm segment (same path as the gamescope overlay:
// /dev/shm/vsrg_overlay). Returns true once attached. Safe to call
// every frame — no-op once attached.
bool shm_ensure(void);

// Draw the current widget snapshot into the ImGui background draw
// list for this frame. canvas_{w,h} are the swapchain image size in
// pixels. Returns the number of widgets emitted (for diagnostics).
int overlay_tick(ImDrawList* draw_list, int canvas_w, int canvas_h);

// Font handle used by overlay_tick. layer.cpp is responsible for
// calling font_load() once during ImGui setup and for triggering the
// font-atlas rebuild. If the TTF load fails, the HUD falls back to
// ImGui's default pixel font.
void font_load(void);

// Hotkey polling. Safe to call every frame; opens its own X
// connection on first use. Returns true if edit-mode should toggle
// this tick (one-shot edge detection for Shift+Tab).
bool input_poll_edit_toggle(void);

// Whether overlay_tick should render edit-mode decorations.
// layer.cpp flips this in response to input_poll_edit_toggle().
bool overlay_edit_mode(void);
void overlay_set_edit_mode(bool on);

}  // namespace vsrg

#endif
