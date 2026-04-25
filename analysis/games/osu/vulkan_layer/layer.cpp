// Vulkan layer glue ; the only file in this module that speaks Vulkan.
//
// Scope (stage 1): prove the implicit-layer discovery + chainloading
// path end-to-end. Override vkCreateInstance, vkCreateDevice,
// vkCreateSwapchainKHR, vkQueuePresentKHR. Track the swapchain image
// size and call overlay_tick() once per present to populate the HUD,
// but do not yet render ImGui into the game's swapchain image.
//
// Stage 2 (separate commit): record ImGui draw commands into a
// command buffer targeting the image about to be presented, submit
// before forwarding the present. Deferred because it needs per-
// swapchain framebuffers / descriptor pools / fences and is
// materially more code than the discovery path.
//
// Gating: the manifest lists VSRG_OVERLAY_LAYER in enable_environment,
// so the loader only instantiates us when that env var is set. No
// runtime gate needed here.
//
// Threading note for stage 2: vkQueuePresentKHR can be called from
// multiple DXVK threads. All shared state (imgui context, shm
// snapshot) will go under a single std::mutex there. Stage 1's
// overlay_tick() already takes an ImDrawList pointer and runs
// serially per-present, so we just need to wrap the ImGui frame in
// a mutex when we add it.

#include "vkroots.h"
#include "overlay.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace vsrg {

namespace {

// Swapchain image size, keyed by VkSwapchainKHR handle. Populated in
// CreateSwapchainKHR and read in QueuePresentKHR so overlay_tick
// knows the canvas size without re-querying Vulkan.
struct SwapchainInfo {
    uint32_t width;
    uint32_t height;
};

std::mutex                                        g_swapchain_mu;
std::unordered_map<VkSwapchainKHR, SwapchainInfo> g_swapchains;

// Diagnostic latches so we don't flood stderr from a 240 Hz present
// loop. Log only on state transitions (first present, size change).
bool     g_logged_first_present = false;

void log_once(const char* msg) {
    std::fprintf(stderr, "[vsrg-layer] %s\n", msg);
}

}  // namespace

class VkInstanceOverrides {
public:
    static VkResult CreateInstance(
        PFN_vkCreateInstance          create,
        const VkInstanceCreateInfo*   pCreateInfo,
        const VkAllocationCallbacks*  pAllocator,
        VkInstance*                   pInstance) {
        VkResult r = create(pCreateInfo, pAllocator, pInstance);
        if (r == VK_SUCCESS) {
            log_once("instance created ; layer live");
        }
        return r;
    }

    static VkResult CreateDevice(
        const vkroots::VkPhysicalDeviceDispatch& d,
        VkPhysicalDevice                         physicalDevice,
        const VkDeviceCreateInfo*                pCreateInfo,
        const VkAllocationCallbacks*             pAllocator,
        VkDevice*                                pDevice) {
        VkResult r = d.CreateDevice(physicalDevice, pCreateInfo, pAllocator,
                                    pDevice);
        if (r == VK_SUCCESS) {
            log_once("device created ; device dispatch live");
        }
        return r;
    }
};

class VkDeviceOverrides {
public:
    static VkResult CreateSwapchainKHR(
        const vkroots::VkDeviceDispatch&   d,
        VkDevice                           device,
        const VkSwapchainCreateInfoKHR*    pCreateInfo,
        const VkAllocationCallbacks*       pAllocator,
        VkSwapchainKHR*                    pSwapchain) {
        VkResult r = d.CreateSwapchainKHR(device, pCreateInfo, pAllocator,
                                          pSwapchain);
        if (r == VK_SUCCESS && pSwapchain && *pSwapchain) {
            std::lock_guard<std::mutex> lock(g_swapchain_mu);
            g_swapchains[*pSwapchain] = {
                pCreateInfo->imageExtent.width,
                pCreateInfo->imageExtent.height,
            };
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "swapchain %p created %ux%u format=%d",
                          (void*)*pSwapchain,
                          pCreateInfo->imageExtent.width,
                          pCreateInfo->imageExtent.height,
                          (int)pCreateInfo->imageFormat);
            log_once(buf);
        }
        return r;
    }

    static void DestroySwapchainKHR(
        const vkroots::VkDeviceDispatch&   d,
        VkDevice                           device,
        VkSwapchainKHR                     swapchain,
        const VkAllocationCallbacks*       pAllocator) {
        {
            std::lock_guard<std::mutex> lock(g_swapchain_mu);
            g_swapchains.erase(swapchain);
        }
        d.DestroySwapchainKHR(device, swapchain, pAllocator);
    }

    static VkResult QueuePresentKHR(
        const vkroots::VkQueueDispatch&    d,
        VkQueue                            queue,
        const VkPresentInfoKHR*            pPresentInfo) {
        // One-shot edit-mode hotkey poll + shm sanity check. Cheap
        // (XQueryKeymap + memcpy) so doing it per-present is fine.
        if (input_poll_edit_toggle()) {
            overlay_set_edit_mode(!overlay_edit_mode());
        }
        shm_ensure();

        if (!g_logged_first_present) {
            g_logged_first_present = true;
            log_once("first vkQueuePresentKHR intercepted");
        }

        // Stage 2 will insert a cmd buffer here that renders ImGui
        // into the image being presented. Stage 1 just forwards.
        return d.QueuePresentKHR(queue, pPresentInfo);
    }
};

static_assert(std::is_same_v<
              decltype(&VkDeviceOverrides::CreateSwapchainKHR),
              VkResult (*)(const vkroots::VkDeviceDispatch&, VkDevice,
                           const VkSwapchainCreateInfoKHR*,
                           const VkAllocationCallbacks*, VkSwapchainKHR*)>);
static_assert(std::is_same_v<
              decltype(&VkDeviceOverrides::DestroySwapchainKHR),
              void (*)(const vkroots::VkDeviceDispatch&, VkDevice,
                       VkSwapchainKHR, const VkAllocationCallbacks*)>);
static_assert(std::is_same_v<
              decltype(&VkDeviceOverrides::QueuePresentKHR),
              VkResult (*)(const vkroots::VkQueueDispatch&, VkQueue,
                           const VkPresentInfoKHR*)>);
static_assert(std::is_same_v<
              decltype(&VkInstanceOverrides::CreateDevice),
              VkResult (*)(const vkroots::VkPhysicalDeviceDispatch&,
                           VkPhysicalDevice, const VkDeviceCreateInfo*,
                           const VkAllocationCallbacks*, VkDevice*)>);

}  // namespace vsrg

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char* pName);
VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char* pName);
VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_layerGetPhysicalDeviceProcAddr(VkInstance instance, const char* pName);

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (!pName) {
        return nullptr;
    }

    if (!std::strcmp(pName, "vkGetInstanceProcAddr")) {
        return (PFN_vkVoidFunction)&vkGetInstanceProcAddr;
    }
    if (!std::strcmp(pName, "vkGetDeviceProcAddr")) {
        return (PFN_vkVoidFunction)&vkGetDeviceProcAddr;
    }
    if (!std::strcmp(pName, "vk_layerGetPhysicalDeviceProcAddr")) {
        return (PFN_vkVoidFunction)&vk_layerGetPhysicalDeviceProcAddr;
    }

    if (!std::strcmp(pName, "vkCreateDevice")) {
        std::fprintf(stderr,
                     "[vsrg-layer] vkGetInstanceProcAddr(vkCreateDevice)\n");
    }
    return vkroots::GetInstanceProcAddr<vsrg::VkInstanceOverrides,
                                        vsrg::VkDeviceOverrides>(instance,
                                                                 pName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    if (!pName) {
        return nullptr;
    }

    if (!std::strcmp(pName, "vkGetDeviceProcAddr")) {
        return (PFN_vkVoidFunction)&vkGetDeviceProcAddr;
    }

    // The loader's layer rules require vkGetDeviceProcAddr to expose
    // vkCreateDevice for layers that build a device chain. vkroots only
    // exposes it through GetInstanceProcAddr/GetPhysicalDeviceProcAddr.
    if (!std::strcmp(pName, "vkCreateDevice")) {
        std::fprintf(stderr,
                     "[vsrg-layer] vkGetDeviceProcAddr(vkCreateDevice)\n");
        return (PFN_vkVoidFunction)
            &vkroots::wrap_CreateDevice<vsrg::VkInstanceOverrides,
                                        vsrg::VkDeviceOverrides>;
    }

    if (!std::strcmp(pName, "vkCreateSwapchainKHR") ||
        !std::strcmp(pName, "vkQueuePresentKHR")) {
        std::fprintf(stderr, "[vsrg-layer] vkGetDeviceProcAddr(%s)\n", pName);
    }
    return vkroots::GetDeviceProcAddr<vsrg::VkInstanceOverrides,
                                      vsrg::VkDeviceOverrides>(device, pName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_layerGetPhysicalDeviceProcAddr(VkInstance instance, const char* pName) {
    if (pName && !std::strcmp(pName, "vkCreateDevice")) {
        std::fprintf(
            stderr,
            "[vsrg-layer] vk_layerGetPhysicalDeviceProcAddr(vkCreateDevice)\n");
    }
    return vkroots::GetPhysicalDeviceProcAddr<vsrg::VkInstanceOverrides,
                                              vsrg::VkDeviceOverrides>(
        instance, pName);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct) {
    std::fprintf(stderr, "[vsrg-layer] negotiate loader/layer interface\n");

    VkResult r =
        vkroots::NegotiateLoaderLayerInterfaceVersion<vsrg::VkInstanceOverrides,
                                                      vsrg::VkDeviceOverrides>(
            pVersionStruct);
    if (r == VK_SUCCESS) {
        pVersionStruct->pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr = &vkGetDeviceProcAddr;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr =
            &vk_layerGetPhysicalDeviceProcAddr;
    }
    return r;
}
