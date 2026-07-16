"""Per-note-type hooks for the player + analysis pipeline.

Today these modules mostly just expose the Etterna TapNoteType enum
values each type maps to. They exist as deliberately-thin placeholders
so that per-type behavior (mine damage, fake-skip, lift-on-release,
roll-retap) has a natural home when it lands ; the Player shouldn't
grow a `if notetype == ...` ladder for every new feature.

Source of truth for the enum values is still
analysis.games.etterna.sm_chart (which has to match Etterna's
Steps::GenerateChartKey expectations). We re-export the names here so
the rest of the player can `from analysis.player.notetypes import taps`
without reaching into the adapter."""
from __future__ import annotations

from analysis.games.etterna.sm_chart import (  # noqa: F401
    NT_TAP,
    NT_HOLD_HEAD,
    NT_ROLL_HEAD,
    NT_MINE,
    NT_LIFT,
    NT_FAKE,
    NT_AUTO_KEYSOUND,
)

# Not an Etterna type: fluXis tick notes (chunithm-style "activates
# while the key is down"). Value chosen clear of the sm_chart enum.
NT_TICK = 8
