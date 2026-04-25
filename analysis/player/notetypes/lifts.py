"""Lifts (SM character 'L', Etterna TapNoteType_Lift).

Judged on key *release* instead of press. Same window as a regular tap
but from the release edge of the keystroke. Replays record lifts with
the Tap path, so the judge logic currently treats them as taps ; when
we start honoring release timing this module is where the logic will
live."""
from __future__ import annotations

from analysis.player.notetypes import NT_LIFT  # noqa: F401
