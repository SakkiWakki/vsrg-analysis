#!/usr/bin/env bash
# Build the native gamescope external-overlay client. Native x86_64
# ; this is NOT the 32-bit LD_PRELOAD from the earlier attempt,
# it's a standalone overlay that lives outside wine.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

out="$here/osu_overlay"
echo "[build] compiling $out"
gcc -O2 -Wall -Wextra -o "$out" osu_overlay.c \
    $(pkg-config --cflags --libs x11 gl)

echo "[build] done: $(file "$out" | sed 's/.*: //')"
