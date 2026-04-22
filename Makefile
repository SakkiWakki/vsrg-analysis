# vsrg-analysis — top-level build orchestration.
#
# Targets group by subsystem:
#   make            → (default) build everything and launch the GUI
#   make all        → build everything (venv + native + overlay), no tests
#   make run        → alias for the default: build then launch
#   make venv       → virtualenv + pip install
#   make native     → Rust osu_memory_native PyO3 extension (maturin develop)
#   make overlay    → C gamescope external-overlay binary
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

NATIVE_DIR  := analysis/games/osu/native
OVERLAY_DIR := analysis/games/osu/gamescope_overlay
OVERLAY_BIN := $(OVERLAY_DIR)/osu_overlay

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
	    || { echo "[native] post-build import failed — venv mismatch?"; exit 1; }
	$(Q)touch $@

.PHONY: native
native: $(NATIVE_STAMP)

# ─── gamescope overlay ─────────────────────────────────────────────────

OVERLAY_SRCS := $(OVERLAY_DIR)/osu_overlay.c \
                $(OVERLAY_DIR)/render.c \
                $(OVERLAY_DIR)/render.h \
                $(OVERLAY_DIR)/nanovg.c \
                $(OVERLAY_DIR)/nanovg.h \
                $(OVERLAY_DIR)/nanovg_gl.h \
                $(OVERLAY_DIR)/fontstash.h \
                $(OVERLAY_DIR)/stb_truetype.h \
                $(OVERLAY_DIR)/stb_image.h \
                $(OVERLAY_DIR)/overlay_shm.h

OVERLAY_CFLAGS := -O2 -Wall -Wextra $(shell pkg-config --cflags x11 gl 2>/dev/null)
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
	    -o $@ \
	    $(OVERLAY_DIR)/osu_overlay.c \
	    $(OVERLAY_DIR)/render.c \
	    $(OVERLAY_DIR)/nanovg.c \
	    $(OVERLAY_LIBS)

.PHONY: overlay
overlay: $(OVERLAY_BIN)

# ─── aggregate ─────────────────────────────────────────────────────────

.PHONY: build
build: native overlay

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

.PHONY: clean
clean:
	$(Q)echo "[clean] overlay binary + maturin stamp + target/"
	$(Q)rm -f $(OVERLAY_BIN) $(NATIVE_STAMP)
	$(Q)rm -rf $(NATIVE_DIR)/target
	$(Q)find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

.PHONY: distclean
distclean: clean
	$(Q)echo "[distclean] removing $(VENV)"
	$(Q)rm -rf $(VENV)
