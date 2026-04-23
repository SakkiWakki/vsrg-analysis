cmake_minimum_required(VERSION 3.20)
project(vsrg_win_overlay C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD   11)

set(ROOT ${CMAKE_SOURCE_DIR})

# ── vsrg_gl_overlay.dll ─────────────────────────────────────────────

set(MINHOOK_SRC
    ${ROOT}/third_party/minhook/src/buffer.c
    ${ROOT}/third_party/minhook/src/hook.c
    ${ROOT}/third_party/minhook/src/trampoline.c
    ${ROOT}/third_party/minhook/src/hde/hde32.c
    ${ROOT}/third_party/minhook/src/hde/hde64.c
)

set(OVERLAY_SRC
    ${ROOT}/analysis/overlay/renderer/nanovg.c
    ${ROOT}/analysis/overlay/widgets/widgets.c
)

add_library(vsrg_gl_overlay SHARED
    win_gl_layer.cpp
    win_gl_loader.c
    win_render.c
    win_shm_consumer.c
    input_win32.c
    ${OVERLAY_SRC}
    ${MINHOOK_SRC}
)

target_include_directories(vsrg_gl_overlay PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/stubs
    ${ROOT}/third_party/minhook/include
    ${ROOT}/analysis/overlay/renderer
    ${ROOT}/analysis/overlay/widgets
    ${ROOT}/analysis/overlay/input
)

target_compile_definitions(vsrg_gl_overlay PRIVATE NOMINMAX)

if(MSVC)
    target_compile_options(vsrg_gl_overlay PRIVATE
        /FI${CMAKE_CURRENT_SOURCE_DIR}/gcc_atomic_shim.h
    )
endif()

target_link_libraries(vsrg_gl_overlay PRIVATE opengl32)

set_target_properties(vsrg_gl_overlay PROPERTIES PREFIX "")

# ── inject.exe ──────────────────────────────────────────────────────

add_executable(inject inject/inject.c)

if(MSVC)
    set_target_properties(inject PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
endif()
