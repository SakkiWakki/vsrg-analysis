"""fluXis lane-switch visibility tables + mask timeline builder.

Transcribed mechanically from `LaneSwitchEvent.cs` (TeamFluXis/fluXis):
`_VISIBILITY_V2` is `SWITCH_VISIBILITY_V2` ("considers finger positions
properly"), `_VISIBILITY_V1` the legacy `SWITCH_VISIBILITY`; a map's
`ls-v2` flag picks which applies. Indexing: `table[keymode - 2][active
- 1][lane]` -> 1 when the lane stays active. Events with `count ==
keymode` (or count outside 1..keymode-1) show every lane.

Masks are lane tuples of 0/1, lane 0 = leftmost.
"""
from __future__ import annotations

_VISIBILITY_V2 = (
    # 2k: rows are active-lane counts 1..1
    (
        (1, 0),
    ),
    # 3k: rows are active-lane counts 1..2
    (
        (0, 1, 0),
        (1, 0, 1),
    ),
    # 4k: rows are active-lane counts 1..3
    (
        (0, 1, 0, 0),
        (0, 1, 1, 0),
        (1, 1, 0, 1),
    ),
    # 5k: rows are active-lane counts 1..4
    (
        (0, 0, 1, 0, 0),
        (0, 1, 0, 1, 0),
        (0, 1, 1, 1, 0),
        (1, 1, 0, 1, 1),
    ),
    # 6k: rows are active-lane counts 1..5
    (
        (0, 0, 1, 0, 0, 0),
        (0, 0, 1, 1, 0, 0),
        (0, 1, 1, 0, 1, 0),
        (0, 1, 1, 1, 1, 0),
        (1, 1, 1, 0, 1, 1),
    ),
    # 7k: rows are active-lane counts 1..6
    (
        (0, 0, 0, 1, 0, 0, 0),
        (0, 0, 1, 0, 1, 0, 0),
        (0, 0, 1, 1, 1, 0, 0),
        (0, 1, 1, 0, 1, 1, 0),
        (0, 1, 1, 1, 1, 1, 0),
        (1, 1, 1, 0, 1, 1, 1),
    ),
    # 8k: rows are active-lane counts 1..7
    (
        (0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 1, 0, 0),
        (0, 0, 1, 1, 0, 1, 0, 0),
        (0, 1, 1, 0, 0, 1, 1, 0),
        (0, 1, 1, 1, 0, 1, 1, 0),
        (1, 1, 1, 0, 0, 1, 1, 1),
        (1, 1, 1, 1, 0, 1, 1, 1),
    ),
    # 9k: rows are active-lane counts 1..8
    (
        (0, 0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 1, 1, 0, 0, 0),
        (0, 0, 1, 1, 0, 1, 1, 0, 0),
        (0, 0, 1, 1, 1, 1, 1, 0, 0),
        (0, 1, 1, 1, 0, 1, 1, 1, 0),
        (0, 1, 1, 1, 1, 1, 1, 1, 0),
        (1, 1, 1, 1, 0, 1, 1, 1, 1),
    ),
    # 10k: rows are active-lane counts 1..9
    (
        (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 1, 0, 1, 0, 0, 0),
        (0, 0, 1, 1, 0, 0, 1, 1, 0, 0),
        (0, 0, 1, 1, 1, 0, 1, 1, 0, 0),
        (0, 1, 1, 1, 0, 0, 1, 1, 1, 0),
        (0, 1, 1, 1, 1, 0, 1, 1, 1, 0),
        (1, 1, 1, 1, 0, 0, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 0, 1, 1, 1, 1),
    ),
)

_VISIBILITY_V1 = (
    # 2k: rows are active-lane counts 1..1
    (
        (1, 0),
    ),
    # 3k: rows are active-lane counts 1..2
    (
        (0, 1, 0),
        (1, 0, 1),
    ),
    # 4k: rows are active-lane counts 1..3
    (
        (0, 1, 0, 0),
        (0, 1, 1, 0),
        (1, 1, 0, 1),
    ),
    # 5k: rows are active-lane counts 1..4
    (
        (0, 0, 1, 0, 0),
        (0, 1, 0, 1, 0),
        (0, 1, 1, 1, 0),
        (1, 1, 0, 1, 1),
    ),
    # 6k: rows are active-lane counts 1..5
    (
        (0, 0, 1, 0, 0, 0),
        (0, 0, 1, 1, 0, 0),
        (0, 1, 1, 0, 1, 0),
        (0, 1, 1, 1, 1, 0),
        (1, 1, 1, 0, 1, 1),
    ),
    # 7k: rows are active-lane counts 1..6
    (
        (0, 0, 0, 1, 0, 0, 0),
        (0, 0, 1, 0, 1, 0, 0),
        (0, 0, 1, 1, 1, 0, 0),
        (0, 1, 1, 0, 1, 1, 0),
        (0, 1, 1, 1, 1, 1, 0),
        (1, 1, 1, 0, 1, 1, 1),
    ),
    # 8k: rows are active-lane counts 1..7
    (
        (0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 1, 0, 0),
        (0, 0, 1, 1, 0, 1, 0, 0),
        (0, 1, 1, 0, 0, 1, 1, 0),
        (0, 1, 1, 1, 0, 1, 1, 0),
        (1, 1, 1, 0, 0, 1, 1, 1),
        (1, 1, 1, 1, 0, 1, 1, 1),
    ),
    # 9k: rows are active-lane counts 1..8
    (
        (0, 0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 1, 1, 0, 0, 0),
        (0, 0, 1, 1, 0, 1, 1, 0, 0),
        (0, 0, 1, 1, 1, 1, 1, 0, 0),
        (0, 1, 1, 1, 0, 1, 1, 1, 0),
        (0, 1, 1, 1, 1, 1, 1, 1, 0),
        (1, 1, 1, 1, 0, 1, 1, 1, 1),
    ),
    # 10k: rows are active-lane counts 1..9
    (
        (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 1, 0, 1, 0, 0, 0),
        (0, 0, 1, 1, 0, 0, 1, 1, 0, 0),
        (0, 0, 1, 1, 1, 0, 1, 1, 0, 0),
        (0, 1, 1, 1, 0, 0, 1, 1, 1, 0),
        (0, 1, 1, 1, 1, 0, 1, 1, 1, 0),
        (1, 1, 1, 1, 0, 0, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 0, 1, 1, 1, 1),
    ),
)


def _generated_mask(keymode: int, active_count: int):
    """Centered-block fallback for keymodes outside the transcribed
    tables. The real tables are hand-authored per finger position and
    not rule-derivable (e.g. 6k/3 is the asymmetric 011010, chosen over
    its equally-plausible mirror), so this generator only covers shapes
    fluXis itself doesn't define."""
    lead = (keymode - active_count) // 2
    return tuple(1 if lead <= i < lead + active_count else 0
                 for i in range(keymode))


def lane_mask_for(keymode: int, active_count: int, *, v2: bool = True):
    """Active-lane mask for `active_count` visible lanes in a
    `keymode`-lane chart. Full mask when the count covers every lane;
    fluXis's tables for keymodes 2..10, generated centered block
    beyond them."""
    if active_count >= keymode or active_count < 1:
        return (1,) * keymode
    table = _VISIBILITY_V2 if v2 else _VISIBILITY_V1
    if not 2 <= keymode <= len(table) + 1:
        return _generated_mask(keymode, active_count)
    return tuple(int(x) for x in table[keymode - 2][active_count - 1])


def build_lane_mask_timeline(lane_switches, keymode: int, *, v2: bool):
    """`[(t_start_s, mask)]` from raw lane-switch events (ms), sorted,
    deduplicated. Empty when the chart never changes lane count."""
    if not lane_switches:
        return []
    events = sorted(lane_switches, key=lambda e: float(e.get('time', 0.0)))
    timeline = []
    for e in events:
        mask = lane_mask_for(keymode, int(e.get('count', keymode)), v2=v2)
        t = float(e.get('time', 0.0)) / 1000.0
        if timeline and timeline[-1][1] == mask:
            continue
        timeline.append((t, mask))
    full = (1,) * keymode
    if len(timeline) == 1 and timeline[0][1] == full:
        return []
    return timeline
