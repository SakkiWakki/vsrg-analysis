#!/usr/bin/env bash
# vsrg-analysis ; end-user launcher for a prebuilt release zip.
# First run creates a venv, installs dependencies + bundled native
# wheel, and stages the overlay .so. Subsequent runs jump to the GUI.
#
# A prebuilt release only needs Python 3.10+. No Rust, C compilers,
# or CMake required ; all native pieces are pre-compiled.
#
# Source contributors should run `make` instead.

set -eu
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV="$HERE/.venv"
VENV_PY="$VENV/bin/python"
VENV_PIP="$VENV/bin/pip"

if [[ ! -x "$VENV_PY" ]]; then
    echo "[setup] first-run setup (this takes a minute)"

    PY=""
    if command -v python3 >/dev/null 2>&1; then
        PY=python3
    elif command -v python >/dev/null 2>&1; then
        PY=python
    fi
    if [[ -z "$PY" ]]; then
        echo
        echo "[setup] Python is not installed or not on PATH."
        echo
        if [[ "$(uname -s)" == "Darwin" ]]; then
            echo "  Install (Homebrew, recommended):"
            echo "    brew install python@3.12"
            echo
            echo "  Or download from https://www.python.org/downloads/"
        else
            echo "  Install via your package manager, e.g.:"
            echo "    Debian/Ubuntu:  sudo apt install python3 python3-venv python3-pip"
            echo "    Arch:           sudo pacman -S python python-pip"
            echo "    Fedora:         sudo dnf install python3 python3-pip"
        fi
        echo
        echo "After installing, run ./run.sh again."
        exit 1
    fi

    # Some distros (notably Debian/Ubuntu) ship python3 without the
    # venv module; catch that specifically because the error message
    # is cryptic otherwise.
    if ! "$PY" -c 'import venv' 2>/dev/null; then
        echo
        echo "[setup] Python's venv module is missing."
        echo "        Debian/Ubuntu: sudo apt install python3-venv"
        echo "        Other distros: install your python venv package."
        exit 1
    fi

    "$PY" -m venv "$VENV"
    "$VENV_PY" -m pip install --upgrade pip wheel >/dev/null
    echo "[setup] installing requirements..."
    "$VENV_PIP" install -r requirements.txt

    if compgen -G "native/*.whl" > /dev/null; then
        echo "[setup] installing bundled osu_memory_native wheel..."
        "$VENV_PIP" install native/*.whl
    else
        echo
        echo "[setup] native/*.whl not found."
        echo "        This looks like a source checkout rather than a release"
        echo "        zip. For source builds, run 'make' instead."
        echo "        osu_live will fall back to the tosu HTTP bridge at runtime."
    fi

    # Stage a bundled overlay .so where the plugin looks for it.
    if [[ -d overlay ]] && compgen -G "overlay/*.so" > /dev/null; then
        dest="analysis/games/osu/gl_layer/linux/lib64"
        mkdir -p "$dest"
        cp -f overlay/*.so "$dest/"
        echo "[setup] overlay staged."
    fi

    echo "[setup] done."
fi

exec "$VENV_PY" -m analysis.gui.app "$@"
