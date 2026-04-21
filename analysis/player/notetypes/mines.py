"""Mines (SM character 'M', Etterna TapNoteType_Mine).

Pressing a column while a mine is inside the mine window deducts life
in Etterna. We don't currently model life drain, so mines round-trip
through the parser but don't affect judging. The mine window is
supplied by `analysis.games.etterna.judgment.extra_windows_for(...)`
(75 ms at J4, scales down with harder judges)."""
from __future__ import annotations

from analysis.player.notetypes import NT_MINE  # noqa: F401
