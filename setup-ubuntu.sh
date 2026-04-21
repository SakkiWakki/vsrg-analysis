#!/usr/bin/env bash
# One-shot setup for vsrg-analysis on Ubuntu (and Debian-based
# distros that use apt).
#
# Installs:
#   - build toolchain (build-essential, pkg-config)
#   - Python 3 + venv
#   - Rust toolchain (prefers apt's rustc; falls back to rustup
#     if apt's version is too old for pyo3)
#   - X11/GL/Xext development headers (for the gamescope overlay)
#   - gamescope (the live in-game overlay compositor) — only on
#     Ubuntu 24.04+/Debian 13+; older releases don't ship it and
#     the user needs to build from source
# Then runs `make all` to build everything.
#
# osu-winello is *not* installed automatically — it has its own
# installer with interactive prompts. The script points you at it
# at the end if osu-wine isn't already on PATH.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

say() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }

if ! command -v apt-get >/dev/null 2>&1; then
    echo "This script is for Debian/Ubuntu-based distros (apt-get not found)."
    echo "On Arch, run ./setup-arch.sh instead."
    exit 1
fi

SUDO=""
if [[ $EUID -ne 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
        echo "sudo is required (or run this script as root)."
        exit 1
    fi
    SUDO="sudo"
fi

say "updating apt metadata"
$SUDO apt-get update

say "installing build + python + overlay dependencies"
# libxext-dev → X Shape extension (input-transparent overlay).
# libgl1-mesa-dev → GL headers; the runtime GL lib comes from the
# user's GPU driver (nvidia / mesa) and is already installed.
$SUDO apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    curl \
    python3 \
    python3-venv \
    python3-pip \
    libx11-dev \
    libxext-dev \
    libgl1-mesa-dev

# ── Rust ────────────────────────────────────────────────────────────
# pyo3 (used by osu_memory_native) needs rustc ≥ 1.75. Ubuntu 24.04
# ships 1.75, 22.04 ships 1.70 — too old. We probe apt's version
# and fall back to rustup if necessary.
need_rustup=1
if apt-cache show rustc >/dev/null 2>&1; then
    apt_rust_ver="$(apt-cache policy rustc | awk '/Candidate:/ {print $2}' | head -1)"
    apt_rust_major="$(printf '%s\n' "$apt_rust_ver" | awk -F. '{print $1}')"
    apt_rust_minor="$(printf '%s\n' "$apt_rust_ver" | awk -F. '{print $2}')"
    if [[ -n "$apt_rust_major" && -n "$apt_rust_minor" ]]; then
        if (( apt_rust_major > 1 )) || (( apt_rust_major == 1 && apt_rust_minor >= 75 )); then
            say "installing rustc/cargo from apt ($apt_rust_ver)"
            $SUDO apt-get install -y --no-install-recommends rustc cargo
            need_rustup=0
        fi
    fi
fi
if (( need_rustup )); then
    if command -v rustc >/dev/null 2>&1; then
        say "rustc already installed ($(rustc --version))"
    else
        say "apt's rustc is too old for pyo3; installing via rustup"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --default-toolchain stable --profile minimal
        # rustup puts cargo in ~/.cargo/bin — make sure the rest of
        # this script picks it up.
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
fi

# ── gamescope ───────────────────────────────────────────────────────
# Ubuntu 24.04 (noble) and Debian 13 (trixie) onwards ship gamescope
# in the main repos. Older releases need to build it themselves —
# the GUI still works, just not the in-game overlay.
if command -v gamescope >/dev/null 2>&1; then
    say "gamescope already installed"
elif apt-cache show gamescope >/dev/null 2>&1; then
    say "installing gamescope from apt"
    $SUDO apt-get install -y --no-install-recommends gamescope
else
    warn "gamescope is not in your apt repos (older Ubuntu/Debian)."
    warn "The GUI and offline analysis will still work."
    warn "To get the in-game overlay, build gamescope from source:"
    warn "  https://github.com/ValveSoftware/gamescope"
fi

say "building everything (venv + native + overlay) via make all"
make all

if ! command -v osu-wine >/dev/null 2>&1; then
    warn "osu-wine not found on PATH."
    warn "To play osu! with our overlay you also need osu-winello:"
    warn "  https://github.com/NelloKudo/osu-winello"
    warn "(Install it, then the 'Start osu (with overlay)' action"
    warn " in the GUI will Just Work.)"
fi

say "done. Launch the GUI with:   make gui"
say "or the full build+launch path: make"
