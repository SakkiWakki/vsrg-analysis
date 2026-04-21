"""Regular taps (SM character '1', Etterna TapNoteType_Tap).

The most common note type. Judged against the tap windows the game's
adapter provides (osu OD, Etterna J1..J9). The Player's hit logic
treats every note as a tap by default; this module exists so future
tap-only behavior (CB-ignore for 'blurs' mods, per-column offsets,
etc.) has somewhere to go that isn't `if notetype == NT_TAP:` in
Player."""
from __future__ import annotations

from analysis.player.notetypes import NT_TAP  # noqa: F401
