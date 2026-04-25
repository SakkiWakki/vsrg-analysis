#!/usr/bin/env bash
# One-shot setup for vsrg-analysis on Arch Linux (and derivatives
# that use pacman: Manjaro, EndeavourOS, etc.).
#
# Installs:
#   - build toolchain (base-devel, pkgconf, gcc)
#   - Python 3 + venv module
#   - Rust toolchain (for the osu_memory_native PyO3 extension)
#   - X11/GL/Xext development headers (for the gamescope overlay)
#   - gamescope (the live in-game overlay compositor)
# Then runs `make all` to build everything.
#
# osu-winello is *not* installed automatically ; it has its own
# installer with interactive prompts. The script points you at it
# at the end if osu-wine isn't already on PATH.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

say() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }

if ! command -v pacman >/dev/null 2>&1; then
    echo "This script is for Arch-based distros (pacman not found)."
    echo "On Ubuntu/Debian, run ./setup-ubuntu.sh instead."
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

say "installing system packages via pacman"
# --needed skips packages that are already current, so re-running
# this script is cheap and idempotent.
$SUDO pacman -S --needed --noconfirm \
    base-devel \
    pkgconf \
    git \
    python \
    python-pip \
    rust \
    libx11 \
    libxext \
    mesa \
    gamescope

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
