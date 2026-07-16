"""fluXis `.fsc` chart + `.ffx` effect-file parser.

Both are plain JSON (Newtonsoft-serialized `MapInfo` / `MapEvents`).
Times are milliseconds throughout; lanes are 1-based in the file and
0-based columns in the returned dict, matching the other game parsers.

`HitObjectType`: 0 = Normal (tap / hold via `holdtime`), 1 = Tick,
2 = Landmine.

The full effect-stream dict is stashed under `effect_streams` for the
future modchart renderer (playfield transforms, shaders, flashes);
only `laneswitch` is interpreted today.
"""
from __future__ import annotations

import json
from pathlib import Path

HO_NORMAL = 0
HO_TICK = 1
HO_LANDMINE = 2

DEFAULT_ACCURACY_DIFFICULTY = 8.0


def _coerce_hitobject(h):
    lane = int(h.get('lane', 0))
    if lane < 1:
        return None
    hold_ms = float(h.get('holdtime', 0.0) or 0.0)
    kind = int(h.get('type', HO_NORMAL) or HO_NORMAL)
    time_ms = float(h.get('time', 0.0))
    return {
        'time': time_ms,
        'column': lane - 1,
        'is_hold': kind == HO_NORMAL and hold_ms > 0,
        'end_time': time_ms + hold_ms if hold_ms > 0 else None,
        'type': kind,
    }


_LOOPS_KEY = 'loops'


def _group_key(event) -> str:
    group = event.get('group') if isinstance(event, dict) else None
    return str(group).lower().strip() if group else ''


def _grouped_events(streams):
    """Map lowercased group name -> [(stream_name, event), ...] over every
    stream. Mirrors MapEvents.GetGroups: events without a group are
    skipped, and loop events (which cannot carry a JSON group) never join
    a group so a loop can't target itself."""
    groups = {}
    for stream_name, events in streams.items():
        if stream_name == _LOOPS_KEY or not isinstance(events, list):
            continue
        for event in events:
            key = _group_key(event)
            if key:
                groups.setdefault(key, []).append((stream_name, event))
    return groups


def _apply_loop(loop, groups, streams):
    """Clone one loop's target group per MapEvents.Compile: for i in
    0..count-1 the clones land at `loop.time + distance * i`, each keeping
    its event's delta above the group's earliest time. Clones append to
    the stream they came from; originals are untouched."""
    key = str(loop.get('target') or '').lower().strip()
    members = groups.get(key)
    if not members:
        return
    lowest = min(float(e.get('time', 0.0)) for _s, e in members)
    loop_time = float(loop.get('time', 0.0))
    distance = float(loop.get('distance', 0.0))
    for i in range(int(loop.get('count', 0))):
        start = loop_time + distance * i
        for stream_name, event in members:
            clone = dict(event)
            clone['time'] = start + (float(event.get('time', 0.0)) - lowest)
            streams[stream_name].append(clone)


def _expand_loops(streams):
    """Port MapEvents.Compile: expand every `loops` event into its target
    group, then drop the `loops` stream so effects never see groups or
    loops. Loop iteration starts at i = 0, so the first clone lands at
    `loop.time + delta`, not at the original event's time; originals stay
    wherever the author placed them."""
    loops = streams.get(_LOOPS_KEY)
    if isinstance(loops, list) and loops:
        groups = _grouped_events(streams)
        for loop in loops:
            if isinstance(loop, dict):
                _apply_loop(loop, groups, streams)
    streams.pop(_LOOPS_KEY, None)
    return streams


def parse_ffx(ffx_path):
    """Raw effect-stream dict with `loops` expanded; {} when the file is
    absent/unreadable."""
    try:
        with open(ffx_path, encoding='utf-8') as f:
            return _expand_loops(json.load(f))
    except (OSError, json.JSONDecodeError):
        return {}


def parse_fsc(fsc_path):
    """Parse a `.fsc` (and its sibling `.ffx` when referenced).

    Returns: title, artist, mapper, difficulty, audio,
    accuracy_difficulty, keycount, hitobjects, timing_points (ms, bpm),
    scroll_velocities (ms, multiplier), lane_switches, ls_v2,
    effect_streams.
    """
    fsc_path = Path(fsc_path)
    with open(fsc_path, encoding='utf-8') as f:
        raw = json.load(f)

    meta = raw.get('metadata') or {}
    hitobjects = [ho for ho in (_coerce_hitobject(h)
                                for h in raw.get('HitObjects') or []
                                if isinstance(h, dict))
                  if ho is not None]
    hitobjects.sort(key=lambda h: (h['time'], h['column']))

    timing_points = sorted(
        ((float(tp.get('time', 0.0)), float(tp.get('bpm', 120.0)) or 120.0)
         for tp in raw.get('TimingPoints') or [] if isinstance(tp, dict)),
        key=lambda x: x[0])

    # Per-lane SV masks (`mask` on each velocity) are ignored for now:
    # the time-density engine integrates one stream. Rare, modchart-only.
    scroll_velocities = sorted(
        ((float(sv.get('time', 0.0)), float(sv.get('multiplier', 1.0)))
         for sv in raw.get('ScrollVelocities') or [] if isinstance(sv, dict)),
        key=lambda x: x[0])

    effect_streams = {}
    effect_file = (raw.get('EffectFile') or '').strip()
    if effect_file:
        effect_streams = parse_ffx(fsc_path.parent / effect_file)
    lane_switches = [e for e in effect_streams.get('laneswitch') or []
                     if isinstance(e, dict)]

    max_lane = max((h['column'] + 1 for h in hitobjects), default=4)
    max_switch = max((int(e.get('count', 0)) for e in lane_switches),
                     default=0)
    keycount = max(max_lane, max_switch)

    return {
        'title': str(meta.get('title', '')),
        'artist': str(meta.get('artist', '')),
        'mapper': str(meta.get('mapper', '')),
        'difficulty': str(meta.get('difficulty', '')),
        'audio': str(raw.get('AudioFile', '')),
        'background': str(raw.get('BackgroundFile', '')),
        'storyboard': str(raw.get('StoryboardFile', '') or ''),
        'accuracy_difficulty': float(
            raw.get('AccuracyDifficulty') or DEFAULT_ACCURACY_DIFFICULTY),
        'keycount': keycount,
        'hitobjects': hitobjects,
        'timing_points': timing_points or [(0.0, 120.0)],
        'scroll_velocities': scroll_velocities,
        'lane_switches': lane_switches,
        'ls_v2': bool(raw.get('ls-v2', False)),
        'effect_streams': effect_streams,
    }
