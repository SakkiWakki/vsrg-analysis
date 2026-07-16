"""fluXis `.frp` replay parser.

A `.frp` is JSON: `{PlayerID, Frames: [{Time, Type, Actions}]}` where
`Type` 0 = Input (Actions = the complete pressed-action set at that
instant) and 1 = Sync (no actions). Times are milliseconds.

Actions are `FluXisGameplayKeybind` enum values, laid out sequentially
by keymode: Key1k1=0, Key2k1..2=1..2, Key3k*=3..5, Key4k*=6..9, ... so
a keymode-`k` chart's lanes map to `[offset, offset + k)` with
`offset = k*(k-1)/2`. (Verified against a real 6K play: actions 15..20.)
"""
from __future__ import annotations

import json

FRAME_INPUT = 0


def keybind_offset(keycount: int) -> int:
    return keycount * (keycount - 1) // 2


def parse_frp(path):
    """Returns `(frames, player_id)`; frames are `(t_ms, actions_set)`
    for input frames only, time-sorted."""
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)

    frames = []
    for fr in raw.get('Frames') or []:
        if int(fr.get('Type', FRAME_INPUT)) != FRAME_INPUT:
            continue
        actions = frozenset(int(a) for a in fr.get('Actions') or [])
        frames.append((float(fr.get('Time', 0.0)), actions))
    frames.sort(key=lambda x: x[0])
    return frames, int(raw.get('PlayerID', -1))


def extract_key_events(frames, keycount: int):
    """Per-column chronological `(t_ms, is_press)` lists, mirroring the
    Quaver/osu extractors so the judgement sim stays column-symmetric.
    Actions outside this keymode's bind range are ignored (other
    keymodes' binds can't fire in a normal play)."""
    offset = keybind_offset(keycount)
    out = [[] for _ in range(keycount)]
    prev = frozenset()
    for t, actions in frames:
        for a in actions - prev:
            lane = a - offset
            if 0 <= lane < keycount:
                out[lane].append((t, True))
        for a in prev - actions:
            lane = a - offset
            if 0 <= lane < keycount:
                out[lane].append((t, False))
        prev = actions
    return out
