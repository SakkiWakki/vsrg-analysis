"""Long notes — holds ('2'→'3') and rolls ('4'→'3').

In Etterna these are separate TapNoteType values but share the same
start+end layout: a head row, then a span of empty rows, then a tail
row. Rolls also require continuous re-tap during the span; holds only
require holding the key. The render path treats both identically (LN
body + head + tail), and the adapter's `prepare_replay_times` emits
`(head_row, col) -> tail_time_sec` into `hold_tails` with no
distinction.

Per-type split lives here to give roll-specific logic (retap-rate
tracking, roll-drop judgment) a place to land without polluting the
Player."""
from __future__ import annotations

from analysis.player.notetypes import NT_HOLD_HEAD, NT_ROLL_HEAD  # noqa: F401


def is_hold_type(notetype) -> bool:
    """Is this note a long-note head? Covers both hold and roll heads."""
    return notetype in (NT_HOLD_HEAD, NT_ROLL_HEAD)
