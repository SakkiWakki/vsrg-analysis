#!/usr/bin/env bash
# Run osu! + our custom overlay as two clients inside one gamescope
# nested session. Invoke via:
#
#   gamescope -f -w 2560 -h 1440 -W 2560 -H 1440 -- \
#       analysis/games/osu/gamescope_overlay/run-osu-gamescope-overlay.sh
#
# Adjust -w/-h/-W/-H to match your desired gamescope virtual
# resolution; the overlay picks up its size from the same values
# via --width/--height.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

width="${GAMESCOPE_WIDTH:-2560}"
height="${GAMESCOPE_HEIGHT:-1440}"

echo "[runner] gamescope DISPLAY=$DISPLAY  ${width}x${height}"

# Start osu! FIRST so gamescope promotes it as the primary game
# surface. If the overlay maps before osu! does, gamescope picks
# the overlay as the game window and osu! ends up hidden beneath
# gamescope's blank background.
osu-wine &
osu_pid=$!

# Give osu! a few seconds to create its X window before we spawn
# the overlay. The overlay self-tags with GAMESCOPE_EXTERNAL_OVERLAY
# and should land on the overlay layer regardless of map order,
# but empirically gamescope's surface-promotion pass only
# considers windows that exist when it first sees a MapRequest.
sleep 4

"$here/osu_overlay" --width "$width" --height "$height" \
    --feed /dev/shm/vsrg_overlay \
    > /tmp/osu_overlay.log 2>&1 &
overlay_pid=$!

trap 'kill "$overlay_pid" 2>/dev/null || true' EXIT
wait "$osu_pid" || true
