#!/usr/bin/env bash
# Launch the VSRG analysis GUI.
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
exec python3 -m analysis.gui.app "$@"
