# vsrg-analysis ; top-level build orchestration.
#
# Targets group by subsystem:
#   make            → (default) build everything and launch the GUI
#   make all        → build everything (venv + native + overlay), no tests
#   make run        → alias for the default: build then launch
#   make venv       → virtualenv + pip install
#   make native     → Rust osu_memory_native PyO3 extension (maturin develop)
#   make overlay    → C gamescope external-overlay binary
#   make gl-layer   → LD_PRELOAD OpenGL/EGL/GLX hooks for osu!stable
#   make build      → native + overlay
#   make test       → pytest
#   make gui        → launch the Qt GUI (assumes build already done)
#   make clean      → wipe build artifacts (keeps venv)
#   make distclean  → also wipe the venv
#
# All targets are idempotent. Each one echoes what it's doing; set
# Q= to see the underlying commands.
Q ?= @

PY         ?= python3
VENV       ?= .venv
VENV_PY    := $(VENV)/bin/python
VENV_PIP   := $(VENV)/bin/pip
VENV_MATURIN := $(VENV)/bin/maturin
VENV_PYTEST  := $(VENV)/bin/pytest

NATIVE_DIR    := analysis/games/osu/native
FRAME_NATIVE_DIR := analysis/games/notitg/native
WEBTEX_DIR    := analysis/overlay/web_texture_ipc
OVERLAY_DIR   := analysis/games/osu/gamescope_overlay
OVERLAY_BIN   := $(OVERLAY_DIR)/osu_overlay

# Game-agnostic rendering + widget replay. Shared by every host
# (the gamescope external overlay binary here, the stable GL
# preload layer below, and anything else that wants to draw HUD
# widgets from the /dev/shm/vsrg_overlay feed).
RENDERER_DIR := analysis/overlay/renderer
WIDGETS_DIR  := analysis/overlay/widgets
INPUT_DIR    := analysis/overlay/input

# Default target: one command that installs, builds, and launches.
# This is the "I just cloned the repo, what do I type?" path.
.DEFAULT_GOAL := run

.PHONY: help
help:
	@echo "vsrg-analysis build targets:"
	@echo "  make            - (default) build everything and launch the GUI"
	@echo "  make all        - build everything, no tests, no launch"
	@echo "  make run        - alias for the default"
	@echo "  make venv       - create $(VENV) and install requirements"
	@echo "  make native     - build osu_memory_native Rust extension into the venv"
	@echo "  make overlay    - build the gamescope in-game overlay binary"
	@echo "  make gl-layer   - build OpenGL/EGL/GLX preload hooks"
	@echo "  make build      - native + overlay"
	@echo "  make test       - run pytest"
	@echo "  make gui        - launch the Qt GUI (assumes build already done)"
	@echo "  make clean      - remove build artifacts (venv kept)"
	@echo "  make distclean  - also remove the venv"

# ─── venv ──────────────────────────────────────────────────────────────

$(VENV_PY):
	$(Q)echo "[venv] creating $(VENV)"
	$(Q)$(PY) -m venv $(VENV)
	$(Q)$(VENV_PIP) install --upgrade pip wheel >/dev/null

.PHONY: venv
venv: $(VENV_PY)
	$(Q)echo "[venv] pip install -r requirements.txt"
	$(Q)$(VENV_PIP) install -r requirements.txt
	$(Q)$(VENV_PIP) install maturin pytest

# ─── native PyO3 extension ─────────────────────────────────────────────

# maturin develop builds the cdylib and drops it straight into the
# venv's site-packages, so ``import osu_memory_native`` works from
# any script run through $(VENV_PY). We depend on the Cargo sources
# so an unchanged tree short-circuits the rebuild.
NATIVE_SRCS := $(shell find $(NATIVE_DIR)/src -type f -name '*.rs' 2>/dev/null) \
               $(NATIVE_DIR)/Cargo.toml

NATIVE_STAMP := $(NATIVE_DIR)/.maturin-stamp

$(NATIVE_STAMP): $(NATIVE_SRCS) | venv
	$(Q)echo "[native] maturin develop --release"
	$(Q)cd $(NATIVE_DIR) && ../../../../$(VENV_MATURIN) develop --release
	$(Q)$(VENV_PY) -c "import osu_memory_native" \
	    || { echo "[native] post-build import failed ; venv mismatch?"; exit 1; }
	$(Q)touch $@

.PHONY: native
native: $(NATIVE_STAMP)

# ─── NotITG frame-interpreter core (script->timeline compiler) ─────────

# The native residue tick-loop interpreter (notitg_frame_native). Same
# maturin develop pattern; the sim imports it opt-in via
# `use_native_body` and falls back to the Python interpreter when the
# wheel is absent, so this target is not required to run the app.
FRAME_NATIVE_SRCS := $(shell find $(FRAME_NATIVE_DIR)/src -type f -name '*.rs' 2>/dev/null) \
                     $(FRAME_NATIVE_DIR)/Cargo.toml
FRAME_NATIVE_STAMP := $(FRAME_NATIVE_DIR)/.maturin-stamp

$(FRAME_NATIVE_STAMP): $(FRAME_NATIVE_SRCS) | venv
	$(Q)echo "[frame-native] maturin develop --release"
	$(Q)cd $(FRAME_NATIVE_DIR) && ../../../../$(VENV_MATURIN) develop --release
	$(Q)$(VENV_PY) -c "import notitg_frame_native" \
	    || { echo "[frame-native] post-build import failed ; venv mismatch?"; exit 1; }
	$(Q)touch $@

.PHONY: frame-native
frame-native: $(FRAME_NATIVE_STAMP)

# ─── web-texture IPC Rust extension (Linux dmabuf side channel) ───────

WEBTEX_SRCS := $(shell find $(WEBTEX_DIR)/src -type f -name '*.rs' 2>/dev/null) \
               $(WEBTEX_DIR)/Cargo.toml
WEBTEX_STAMP := $(WEBTEX_DIR)/.maturin-stamp

$(WEBTEX_STAMP): $(WEBTEX_SRCS) | venv
	$(Q)echo "[webtex] maturin develop --release"
	$(Q)cd $(WEBTEX_DIR) && ../../../$(VENV_MATURIN) develop --release
	$(Q)$(VENV_PY) -c "import web_texture_ipc" \
	    || { echo "[webtex] post-build import failed ; venv mismatch?"; exit 1; }
	$(Q)touch $@

.PHONY: webtex
webtex: $(WEBTEX_STAMP)

# ─── gamescope overlay ─────────────────────────────────────────────────

OVERLAY_SRCS := $(OVERLAY_DIR)/osu_overlay.c \
                $(RENDERER_DIR)/render.c \
                $(RENDERER_DIR)/render.h \
                $(RENDERER_DIR)/nanovg.c \
                $(RENDERER_DIR)/nanovg.h \
                $(RENDERER_DIR)/nanovg_gl.h \
                $(RENDERER_DIR)/fontstash.h \
                $(RENDERER_DIR)/stb_truetype.h \
                $(RENDERER_DIR)/stb_image.h \
                $(WIDGETS_DIR)/widgets.c \
                $(WIDGETS_DIR)/widgets.h \
                $(WIDGETS_DIR)/overlay_shm.h \
                $(INPUT_DIR)/input.h

OVERLAY_CFLAGS := -O2 -Wall -Wextra \
                  -I$(RENDERER_DIR) -I$(WIDGETS_DIR) -I$(INPUT_DIR) \
                  $(shell pkg-config --cflags x11 gl 2>/dev/null)
OVERLAY_LIBS   := $(shell pkg-config --libs x11 gl 2>/dev/null) -lm

# nanovg.c + the bundled stb headers throw a swarm of -Wextra warnings
# (unused functions/parameters, sign comparisons) that we don't want
# to patch upstream. Suppress them only for the overlay binary; nothing
# else in the repo uses these flags.
$(OVERLAY_BIN): $(OVERLAY_SRCS)
	$(Q)echo "[overlay] gcc $(notdir $@)"
	$(Q)gcc $(OVERLAY_CFLAGS) \
	    -Wno-unused-function -Wno-unused-parameter \
	    -Wno-sign-compare -Wno-unused-but-set-variable \
	    -Wno-misleading-indentation \
	    -Wno-shift-negative-value -Wno-implicit-fallthrough \
	    -o $@ \
	    $(OVERLAY_DIR)/osu_overlay.c \
	    $(RENDERER_DIR)/render.c \
	    $(RENDERER_DIR)/nanovg.c \
	    $(WIDGETS_DIR)/widgets.c \
	    $(OVERLAY_LIBS)

.PHONY: overlay
overlay: $(OVERLAY_BIN)

# ─── OpenGL/EGL/GLX preload hook (osu!stable) ─────────────────────────
#
# osu!stable under Wine may present through OpenGL/EGL instead of
# Vulkan/DXVK. These shared objects are injected with LD_PRELOAD and log
# the first buffer-present call they intercept. Build both 64-bit and
# 32-bit variants because old stable/Wine setups can use either Unix
# process bitness.

GL_LAYER_DIR := analysis/games/osu/gl_layer/linux
GL_LAYER_SO  := $(GL_LAYER_DIR)/lib/libvsrg_gl_overlay.so
GL_LAYER_SO64_ALT := $(GL_LAYER_DIR)/lib64/libvsrg_gl_overlay.so
GL_LAYER_SO32 := $(GL_LAYER_DIR)/lib32/libvsrg_gl_overlay.so

GL_LAYER_SRCS  := $(GL_LAYER_DIR)/gl_layer.cpp

# 64-bit build pulls in the same NanoVG-backed shim the gamescope
# overlay uses (from analysis/overlay/renderer + widgets). 32-bit
# stays logging-only until we have multilib fontstash/GL headers;
# stable under modern Wine runs 64-bit so the renderer only needs
# to live in the 64-bit .so.
GL_LAYER_RENDER_SRCS_C := $(RENDERER_DIR)/render.c \
                          $(RENDERER_DIR)/nanovg.c \
                          $(WIDGETS_DIR)/widgets.c \
                          $(INPUT_DIR)/input.c \
                          $(INPUT_DIR)/input_x11_poll.c \
                          $(INPUT_DIR)/input_x11_xi2.c \
                          $(GL_LAYER_DIR)/shm_consumer.c \
                          $(GL_LAYER_DIR)/web_texture_host.c

GL_LAYER_CXXFLAGS := -std=c++20 -O2 -fPIC -fvisibility=hidden \
                     -Wall -Wextra -Wno-unused-parameter \
                     -I$(RENDERER_DIR) -I$(WIDGETS_DIR) -I$(INPUT_DIR) \
                     $(shell pkg-config --cflags x11 egl gl 2>/dev/null)
GL_LAYER_CFLAGS_C := -std=c11 -O2 -fPIC -fvisibility=hidden \
                     -Wno-unused-function -Wno-unused-parameter \
                     -Wno-sign-compare -Wno-unused-but-set-variable \
                     -Wno-misleading-indentation \
                     -Wno-shift-negative-value -Wno-implicit-fallthrough \
                     -I$(RENDERER_DIR) -I$(WIDGETS_DIR) -I$(INPUT_DIR) \
                     $(shell pkg-config --cflags x11 gl 2>/dev/null)
GL_LAYER_LDFLAGS  := -shared -fvisibility=hidden
GL_LAYER_LIBS     := -ldl -lm -lpthread \
                     $(shell pkg-config --libs x11 xi egl gl 2>/dev/null)

GL_LAYER_BUILD_DIR := $(GL_LAYER_DIR)/.build
GL_LAYER_C_OBJS := $(patsubst %.c,$(GL_LAYER_BUILD_DIR)/%.o, \
                       $(notdir $(GL_LAYER_RENDER_SRCS_C)))
GL_LAYER_CPP_OBJS := $(patsubst %.cpp,$(GL_LAYER_BUILD_DIR)/%.o, \
                         $(notdir $(GL_LAYER_SRCS)))

# Per-file compile rules. g++'s ``-x c`` is single-shot (it applies
# only to the next input file), so the previous single-invocation
# ``g++ -x c a.c b.c`` pattern silently compiled b.c as C++ and
# mangled render_text_width's reference from widgets.c into a C++
# symbol that didn't match render.c's unmangled definition. Building
# each TU separately with the right compiler avoids that.

# C files: renderer + widgets + shm consumer.
$(GL_LAYER_BUILD_DIR)/%.o: $(RENDERER_DIR)/%.c \
                           $(RENDERER_DIR)/render.h \
                           $(WIDGETS_DIR)/widgets.h
	$(Q)mkdir -p $(@D)
	$(Q)gcc $(GL_LAYER_CFLAGS_C) -DVSRG_GL_LAYER_HAS_RENDERER -c -o $@ $<

$(GL_LAYER_BUILD_DIR)/%.o: $(WIDGETS_DIR)/%.c \
                           $(RENDERER_DIR)/render.h \
                           $(WIDGETS_DIR)/widgets.h \
                           $(INPUT_DIR)/input.h
	$(Q)mkdir -p $(@D)
	$(Q)gcc $(GL_LAYER_CFLAGS_C) -DVSRG_GL_LAYER_HAS_RENDERER -c -o $@ $<

$(GL_LAYER_BUILD_DIR)/%.o: $(INPUT_DIR)/%.c \
                           $(INPUT_DIR)/input.h \
                           $(INPUT_DIR)/input_backend.h
	$(Q)mkdir -p $(@D)
	$(Q)gcc $(GL_LAYER_CFLAGS_C) -DVSRG_GL_LAYER_HAS_RENDERER -c -o $@ $<

$(GL_LAYER_BUILD_DIR)/%.o: $(GL_LAYER_DIR)/%.c \
                           $(GL_LAYER_DIR)/shm_consumer.h \
                           $(GL_LAYER_DIR)/web_texture_host.h \
                           $(WIDGETS_DIR)/web_texture_ipc.h
	$(Q)mkdir -p $(@D)
	$(Q)gcc $(GL_LAYER_CFLAGS_C) -DVSRG_GL_LAYER_HAS_RENDERER -c -o $@ $<

# C++ file: gl_layer.cpp.
$(GL_LAYER_BUILD_DIR)/%.o: $(GL_LAYER_DIR)/%.cpp \
                           $(RENDERER_DIR)/render.h \
                           $(WIDGETS_DIR)/widgets.h \
                           $(INPUT_DIR)/input.h \
                           $(GL_LAYER_DIR)/shm_consumer.h \
                           $(GL_LAYER_DIR)/web_texture_host.h
	$(Q)mkdir -p $(@D)
	$(Q)g++ $(GL_LAYER_CXXFLAGS) -DVSRG_GL_LAYER_HAS_RENDERER -c -o $@ $<

$(GL_LAYER_SO): $(GL_LAYER_C_OBJS) $(GL_LAYER_CPP_OBJS)
	$(Q)echo "[gl-layer] g++ $(notdir $@) (with NanoVG renderer)"
	$(Q)mkdir -p $(@D)
	$(Q)g++ $(GL_LAYER_CXXFLAGS) $(GL_LAYER_LDFLAGS) \
	    -o $@ $(GL_LAYER_C_OBJS) $(GL_LAYER_CPP_OBJS) $(GL_LAYER_LIBS)

$(GL_LAYER_SO64_ALT): $(GL_LAYER_SO)
	$(Q)echo "[gl-layer] cp $(notdir $@)"
	$(Q)mkdir -p $(@D)
	$(Q)cp $< $@

$(GL_LAYER_SO32): $(GL_LAYER_SRCS)
	$(Q)echo "[gl-layer] g++ -m32 $(notdir $@) (logging only)"
	$(Q)mkdir -p $(@D)
	$(Q)g++ -m32 $(GL_LAYER_CXXFLAGS) $(GL_LAYER_LDFLAGS) \
	    -o $@ $(GL_LAYER_SRCS) $(GL_LAYER_LIBS)

.PHONY: gl-layer
gl-layer: $(GL_LAYER_SO) $(GL_LAYER_SO64_ALT) $(GL_LAYER_SO32)

# ─── vulkan layer (in-process HUD) ─────────────────────────────────────
#
# Preferred overlay path: a Vulkan layer loaded by DXVK inside osu!'s
# own process. Zero compositor, zero input latency. Falls back to the
# gamescope overlay above if the layer isn't installed or fails to
# load (see the launcher in plugins/unsafe/osu_live/viz/live_drift.py).

LAYER_DIR  := analysis/games/osu/vulkan_layer
LAYER_SO   := $(LAYER_DIR)/libVkLayer_vsrg_overlay.so
LAYER_JSON := $(LAYER_DIR)/VkLayer_vsrg_overlay.json

# Install location: XDG_DATA_HOME per-user. Loader discovers layers
# here before system paths, which is exactly what we want during dev.
XDG_DATA_HOME_OR_DEFAULT := $(if $(XDG_DATA_HOME),$(XDG_DATA_HOME),$(HOME)/.local/share)
LAYER_INSTALL_DIR        := $(XDG_DATA_HOME_OR_DEFAULT)/vulkan/implicit_layer.d

LAYER_SRCS := $(LAYER_DIR)/layer.cpp \
              $(LAYER_DIR)/overlay.cpp \
              $(LAYER_DIR)/input.cpp \
              $(LAYER_DIR)/imgui/imgui.cpp \
              $(LAYER_DIR)/imgui/imgui_draw.cpp \
              $(LAYER_DIR)/imgui/imgui_tables.cpp \
              $(LAYER_DIR)/imgui/imgui_widgets.cpp

LAYER_CXXFLAGS := -std=c++20 -O2 -fPIC -fvisibility=hidden \
                  -Wall -Wextra -Wno-unused-parameter \
                  -I$(LAYER_DIR) -I$(LAYER_DIR)/imgui \
                  $(shell pkg-config --cflags vulkan x11 2>/dev/null)
LAYER_LDFLAGS  := -shared -fvisibility=hidden
LAYER_LIBS     := $(shell pkg-config --libs vulkan x11 2>/dev/null)

$(LAYER_SO): $(LAYER_SRCS) $(LAYER_DIR)/vkroots.h $(LAYER_DIR)/overlay.h
	$(Q)echo "[layer] g++ $(notdir $@)"
	$(Q)g++ $(LAYER_CXXFLAGS) $(LAYER_LDFLAGS) \
	    -o $@ $(LAYER_SRCS) $(LAYER_LIBS)

# Manifest generated from template with the absolute .so path so the
# loader doesn't depend on LD_LIBRARY_PATH.
$(LAYER_JSON): $(LAYER_DIR)/VkLayer_vsrg_overlay.json.in $(LAYER_SO)
	$(Q)echo "[layer] generating $(notdir $@)"
	$(Q)sed 's|@LIBRARY_PATH@|$(abspath $(LAYER_SO))|' \
	    $(LAYER_DIR)/VkLayer_vsrg_overlay.json.in > $@

.PHONY: vulkan-layer
vulkan-layer: $(LAYER_SO) $(LAYER_JSON)

.PHONY: vulkan-layer-install
vulkan-layer-install: vulkan-layer
	$(Q)echo "[layer] install → $(LAYER_INSTALL_DIR)"
	$(Q)mkdir -p $(LAYER_INSTALL_DIR)
	$(Q)cp $(LAYER_JSON) $(LAYER_INSTALL_DIR)/VkLayer_vsrg_overlay.json
	$(Q)echo "[layer] installed. Enable with: VSRG_OVERLAY_LAYER=1"

.PHONY: vulkan-layer-uninstall
vulkan-layer-uninstall:
	$(Q)echo "[layer] uninstall from $(LAYER_INSTALL_DIR)"
	$(Q)rm -f $(LAYER_INSTALL_DIR)/VkLayer_vsrg_overlay.json

# ─── aggregate ─────────────────────────────────────────────────────────

.PHONY: build
build: native frame-native overlay

# "Build everything" without running tests or launching. For CI or
# packaging. The venv is implied via the native dep chain.
.PHONY: all
all: build

# Default path for end users: build everything, then launch the GUI.
# Depending on `build` ensures a first-run user gets a working app
# from one command; subsequent runs short-circuit the build targets.
.PHONY: run
run: build gui

# ─── test ──────────────────────────────────────────────────────────────

.PHONY: test
test: venv
	$(Q)echo "[test] pytest"
	$(Q)$(VENV_PYTEST) tests

# ─── run ───────────────────────────────────────────────────────────────

.PHONY: gui
gui: venv
	$(Q)echo "[gui] launching"
	$(Q)$(VENV_PY) -m analysis.gui.app

# ─── clean ─────────────────────────────────────────────────────────────

# ─── release ───────────────────────────────────────────────────────────
# Build the Linux equivalent of `make.bat release`: a zip containing
# the Python package + plugins, prebuilt .so overlay, the native wheel,
# and run.sh. No source-only files (Rust, C, CMake, Makefile, tests).
# Users extract and run ./run.sh.

DIST_DIR    ?= dist
REV         := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
PYVER       := $(shell $(VENV_PY) -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
RELEASE_NAME := vsrg-analysis-linux-$(REV)-py$(PYVER)
RELEASE_DIR  := $(DIST_DIR)/$(RELEASE_NAME)

.PHONY: release
release: build gl-layer
	$(Q)echo "[release] staging $(RELEASE_DIR)"
	$(Q)rm -rf $(RELEASE_DIR)
	$(Q)mkdir -p $(RELEASE_DIR)/overlay $(RELEASE_DIR)/native
	$(Q)rsync -a --exclude='__pycache__' --exclude='target' \
	    --exclude='.maturin-stamp' analysis $(RELEASE_DIR)/
	$(Q)rsync -a --exclude='__pycache__' plugins $(RELEASE_DIR)/
	$(Q)cp run.sh requirements.txt $(RELEASE_DIR)/
	$(Q)chmod +x $(RELEASE_DIR)/run.sh
	$(Q)test -f analyze && cp analyze $(RELEASE_DIR)/ || true
	$(Q)cp $(GL_LAYER_SO) $(RELEASE_DIR)/overlay/
	$(Q)echo "[release] maturin build --release"
	$(Q)cd $(NATIVE_DIR) && ../../../../$(VENV_MATURIN) build --release >/dev/null
	$(Q)cp $(NATIVE_DIR)/target/wheels/*.whl $(RELEASE_DIR)/native/
	$(Q)printf 'vsrg-analysis %s (Linux, Python 3.%s)\n\nHOW TO RUN:\n  ./run.sh\n\nFirst run installs Python dependencies into a local .venv folder.\nOnly Python 3.10+ is required; no Rust, C compiler, or CMake needed.\n' \
	    "$(REV)" "$(patsubst 3%,%,$(PYVER))" > $(RELEASE_DIR)/README.txt
	$(Q)cd $(DIST_DIR) && zip -qr $(RELEASE_NAME).zip $(RELEASE_NAME)
	$(Q)echo "[release] done: $(DIST_DIR)/$(RELEASE_NAME).zip"

# ─── clean ─────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	$(Q)echo "[clean] overlay binary + maturin stamp + target/"
	$(Q)rm -f $(OVERLAY_BIN) $(GL_LAYER_SO) $(GL_LAYER_SO64_ALT) \
	    $(GL_LAYER_SO32) $(GL_LAYER_DIR)/libvsrg_gl_overlay.so \
	    $(GL_LAYER_DIR)/libvsrg_gl_overlay32.so $(NATIVE_STAMP)
	$(Q)rm -rf $(NATIVE_DIR)/target
	$(Q)find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

.PHONY: distclean
distclean: clean
	$(Q)echo "[distclean] removing $(VENV)"
	$(Q)rm -rf $(VENV)
