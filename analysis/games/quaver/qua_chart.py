"""`.qua` chart parsing.

`.qua` is YAML, but the dialect Quaver writes is regular: 2-space indent,
no anchors/flow style, scalars are unquoted strings or numbers, lists are
sequences of `- key: value` blocks. We hand-roll a parser for that subset
so we don't take a hard dep on PyYAML for one game.

Output shape mirrors `analysis/games/osu/replay/chart.parse_osu_file`:
metadata fields, `hitobjects` list, `timing_points` list, `sv_sections`
already converted to seconds, plus the Quaver-specific
`initial_velocity` and `bpm_does_not_affect_sv` fields the SV-engine
factory in `analysis/player/sv/render.py` consumes.
"""
from __future__ import annotations

import numpy as np


def _f32(x):
    """Round through single-precision so equality checks reproduce
    Quaver's `float` arithmetic. The normalization pass dedupes
    consecutive multipliers; in `float64` we keep ~1e-8 drift that fires
    spurious emits across BPM changes."""
    return float(np.float32(x))


# Map Quaver GameMode enum (Quaver.API/Enums/GameMode.cs) -> keycount.
_GAME_MODE_KEYS = {
    'Keys1': 1, 'Keys2': 2, 'Keys3': 3, 'Keys4': 4, 'Keys5': 5,
    'Keys6': 6, 'Keys7': 7, 'Keys8': 8, 'Keys9': 9, 'Keys10': 10,
}

# Default scroll group ID -- Qua.DefaultScrollGroupId. Notes that don't
# carry an explicit `TimingGroup:` field belong to this group, and the
# top-level `SliderVelocities` / `InitialScrollVelocity` fields populate
# its stream.
DEFAULT_GROUP_ID = '$Default'


def parse_qua_file(qua_path):
    """Parse a `.qua` file and return a metadata dict.

    Keys returned:
        title, artist, creator, version (DifficultyName), audio,
        keycount, mode, hitobjects, timing_points, sv_sections,
        initial_velocity, bpm_does_not_affect_sv, base_bpm, length_ms.

    `sv_sections` is `[(time_sec, multiplier), ...]` already normalized
    so `BPMDoesNotAffectScrollVelocity == True`. Charts written in the
    denormalized format are normalized on the fly using the same
    common-BPM weighting Quaver's `Qua.NormalizeSVs` uses.
    """
    with open(qua_path, encoding='utf-8', errors='replace') as f:
        text = f.read()

    raw = _parse_qua_yaml(text)
    return _post_process(raw)


# ----------------------------------------------------------------------
# YAML parser (Quaver-flavoured subset)
# ----------------------------------------------------------------------


def _parse_qua_yaml(text):
    """Return a dict mirroring `Qua` field shape. Lists become list[dict].
    Scalar coercion is done in `_post_process`; here we keep raw strings
    so unknown keys round-trip without surprise truncation.

    Walk lines once, tracking a stack of (target, indent, kind) frames
    where `kind` is 'map' (target is a dict) or 'list' (target is a list
    of dicts whose newest item receives nested keys). Pop frames whose
    indent is greater-or-equal than the current line's indent before
    deciding where to land the line."""
    root = {}
    # Stack frames are dicts (mutated in place) so a 'pending' frame can
    # resolve into a 'map' or 'list' frame without rebuilding tuples:
    #   kind == 'map'      target is a dict, new keys land here.
    #   kind == 'list'     target is a list, new `- ` items land here.
    #   kind == 'pending'  parent opened a block-value with empty `key:`
    #                      but the body shape is still unknown. The first
    #                      child line resolves it.
    stack = [{'target': root, 'indent': -1, 'kind': 'map'}]
    is_list_item = lambda line: line.startswith('- ') or line == '-'

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith('#'):
            continue
        if raw.strip() == '---':
            continue

        indent = len(raw) - len(raw.lstrip(' '))
        line = raw[indent:]
        starts_item = is_list_item(line)

        # Pop deeper frames. List frames at the same indent stick around
        # so consecutive `- ` items at that indent keep landing in the
        # same list ; map and pending frames at the same indent get popped
        # because a new key/value replaces the open slot.
        while len(stack) > 1:
            top = stack[-1]
            if indent < top['indent']:
                stack.pop()
                continue
            if indent != top['indent']:
                break
            # Equal indent. Behavior depends on what frame is at the top:
            #   - 'list' / 'pending': stays alive iff the new line is a
            #     `- ` item (consecutive list items, or pending->list
            #     resolution). A non-item line means the list has ended
            #     and the next sibling key belongs to its parent map.
            #   - 'map' under a list at the same indent ('list-item
            #     map'): always close, since both another `- ` sibling
            #     and a key at the list's indent terminate it.
            #   - other 'map': stays alive (sibling keys reuse the
            #     same dict at this indent level).
            if top['kind'] in ('list', 'pending'):
                if starts_item:
                    break
                stack.pop()
                continue
            is_list_item_map = (top['kind'] == 'map'
                                and len(stack) >= 2
                                and stack[-2]['kind'] == 'list'
                                and stack[-2]['indent'] == indent)
            if is_list_item_map:
                stack.pop()
            else:
                break

        # Resolve a pending block frame now that we know what its body looks
        # like: `- ` => list at the opener's indent, `key:` => dict whose
        # keys live at the current line's deeper indent.
        top = stack[-1]
        if top['kind'] == 'pending':
            container = [] if starts_item else {}
            top['parent'][top['pending_key']] = container
            top['target'] = container
            top['kind'] = 'list' if starts_item else 'map'
            if not starts_item:
                top['indent'] = indent

        target = top['target']
        kind = top['kind']

        if starts_item:
            list_target = _enclosing_list(stack)
            if list_target is None:
                continue
            item = {}
            list_target.append(item)
            body = line[2:] if line.startswith('- ') else ''
            _set_inline_kv(item, body)
            stack.append({'target': item, 'indent': indent, 'kind': 'map'})
            continue

        key, sep, value = line.partition(':')
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if kind != 'map':
            # Defensive: a scalar line under a list frame should re-
            # target the active list item (top of stack after pop).
            target = stack[-1]['target'] if isinstance(stack[-1]['target'], dict) else root

        # Strip a leading YAML local tag like `!ScrollGroup`. We only need
        # the data shape, not the type annotation, and Quaver uses tags only
        # to disambiguate ScrollGroup vs. plain mapping in its TimingGroups
        # block.
        if value.startswith('!'):
            value = value.split(None, 1)[1].strip() if ' ' in value else ''

        if value == '':
            # Defer choosing list vs. dict until the first child line lands.
            stack.append({'parent': target, 'pending_key': key,
                          'target': None, 'indent': indent, 'kind': 'pending'})
        else:
            target[key] = value

    return root


def _enclosing_list(stack):
    """Return the topmost list frame on the stack. Lists are the only
    frames whose target is a `list` instance; everything else is a dict."""
    for frame in reversed(stack):
        if frame['kind'] == 'list':
            return frame['target']
    return None


def _set_inline_kv(item, body):
    """Populate the first key/value pair of a list item from the same
    line as the `- `. Subsequent keys arrive as indented lines whose
    container is this item."""
    key, sep, value = body.partition(':')
    if not sep:
        return
    item[key.strip()] = value.strip()


# ----------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------


def _post_process(raw):
    mode_str = raw.get('Mode', 'Keys4')
    keycount = _GAME_MODE_KEYS.get(mode_str, 4)

    timing_points = [_coerce_timing_point(tp)
                     for tp in raw.get('TimingPoints') or []
                     if isinstance(tp, dict)]
    timing_points.sort(key=lambda x: x[0])

    slider_velocities = [_coerce_slider_velocity(sv)
                         for sv in raw.get('SliderVelocities') or []
                         if isinstance(sv, dict)]
    slider_velocities.sort(key=lambda x: x[0])

    ssf_default = [_coerce_scroll_speed_factor(s)
                   for s in raw.get('ScrollSpeedFactors') or []
                   if isinstance(s, dict)]
    ssf_default.sort(key=lambda x: x[0])
    # Quaver stores SSF times in ms; convert to seconds once here so the
    # engine never needs to think about units.
    ssf_default = [(t / 1000.0, m) for t, m in ssf_default]

    hitobjects = [ho for ho in (_coerce_hitobject(h, keycount)
                                for h in raw.get('HitObjects') or []
                                if isinstance(h, dict))
                  if ho is not None]
    hitobjects.sort(key=lambda h: (h['time'], h['column']))

    bpm_does_not_affect_sv = _coerce_bool(
        raw.get('BPMDoesNotAffectScrollVelocity', 'false'))
    initial_velocity = _coerce_float(raw.get('InitialScrollVelocity', '1.0'),
                                      default=1.0)
    legacy_ln_rendering = _coerce_bool(
        raw.get('LegacyLNRendering', 'false'))

    base_bpm = _common_bpm(timing_points, hitobjects)
    sv_sections, normalized_initial = _normalize_svs(
        slider_velocities, timing_points, bpm_does_not_affect_sv, base_bpm)
    if not bpm_does_not_affect_sv and normalized_initial is not None:
        # When a denormalized chart is normalized on the fly, Quaver
        # rewrites InitialScrollVelocity to the first adjusted multiplier
        # (`initialSvMultiplier ?? 1`); the engine reads it for the
        # pre-first-section pad, so use the same seed value here.
        initial_velocity = normalized_initial

    # Per-group SV streams. The `$Default` group carries the chart's
    # top-level SV; explicit `TimingGroups: { SG_N: !ScrollGroup ... }`
    # entries each contribute their own (sections, initial_velocity).
    # A note's selector is its `TimingGroup` field; lookup is straight
    # dict access in the engine.
    groups = {DEFAULT_GROUP_ID: {
        'initial_velocity': initial_velocity,
        'sections': sv_sections,
        'ssf': ssf_default,
    }}
    raw_groups = raw.get('TimingGroups')
    if isinstance(raw_groups, dict):
        for group_id, group_raw in raw_groups.items():
            if not isinstance(group_raw, dict):
                continue
            group_svs = [_coerce_slider_velocity(sv)
                         for sv in group_raw.get('ScrollVelocities') or []
                         if isinstance(sv, dict)]
            group_svs.sort(key=lambda x: x[0])
            group_init = _coerce_float(
                group_raw.get('InitialScrollVelocity', '1.0'),
                default=1.0)
            group_sections, group_norm_init = _normalize_svs(
                group_svs, timing_points, bpm_does_not_affect_sv, base_bpm)
            if not bpm_does_not_affect_sv and group_norm_init is not None:
                group_init = group_norm_init
            group_ssf = [_coerce_scroll_speed_factor(s)
                         for s in group_raw.get('ScrollSpeedFactors') or []
                         if isinstance(s, dict)]
            group_ssf.sort(key=lambda x: x[0])
            group_ssf = [(t / 1000.0, m) for t, m in group_ssf]
            groups[group_id] = {
                'initial_velocity': group_init,
                'sections': group_sections,
                'ssf': group_ssf,
            }

    length_ms = 0
    if hitobjects:
        last = hitobjects[-1]
        length_ms = max(last['time'], last.get('end_time') or 0)

    return {
        'title': raw.get('Title', ''),
        'artist': raw.get('Artist', ''),
        'creator': raw.get('Creator', ''),
        'version': raw.get('DifficultyName', ''),
        'audio': raw.get('AudioFile', ''),
        'mode': mode_str,
        'keycount': keycount,
        'hitobjects': hitobjects,
        'timing_points': timing_points,
        'sv_sections': sv_sections,
        'initial_velocity': initial_velocity,
        'bpm_does_not_affect_sv': bpm_does_not_affect_sv,
        'legacy_ln_rendering': legacy_ln_rendering,
        'base_bpm': base_bpm,
        'groups': groups,
        'length_ms': length_ms,
    }


def _coerce_float(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _coerce_int(s, default=0):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return default


def _coerce_bool(s):
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ('true', 'yes', '1')


def _coerce_timing_point(tp):
    return (_coerce_float(tp.get('StartTime', '0')),
            _coerce_float(tp.get('Bpm', '120')))


def _coerce_slider_velocity(sv):
    # Quaver deserializes SliderVelocityInfo.Multiplier into a plain float;
    # when old/minimized YAML omits it, the game sees the language zero value.
    return (_coerce_float(sv.get('StartTime', '0')),
            _coerce_float(sv.get('Multiplier', '0')))


def _coerce_scroll_speed_factor(sf):
    """Same shape as SV but consumed as a position-dependent multiplier
    on the scroll speed (Quaver's `ScrollSpeedFactorInfo`). Lerps
    linearly between consecutive entries."""
    return (_coerce_float(sf.get('StartTime', '0')),
            _coerce_float(sf.get('Multiplier', '1')))


def _coerce_hitobject(h, keycount):
    lane = _coerce_int(h.get('Lane', '1'))
    if lane < 1 or lane > keycount:
        return None
    end_time = _coerce_int(h.get('EndTime', '0'))
    raw_group = (h.get('TimingGroup') or '').strip()
    return {
        'time': _coerce_int(h.get('StartTime', '0')),
        'column': lane - 1,
        'is_hold': end_time > 0,
        'end_time': end_time if end_time > 0 else None,
        'group': raw_group or DEFAULT_GROUP_ID,
    }


# ----------------------------------------------------------------------
# Common BPM (port of Qua.GetCommonBpm)
# ----------------------------------------------------------------------


def _common_bpm(timing_points, hitobjects):
    """Quaver's `Qua.GetCommonBpm`: BPM that covers the most time between
    chart start and the last note. Used as the base for SV normalization."""
    if not timing_points:
        return 120.0
    if not hitobjects:
        return timing_points[0][1]
    last_obj = max(hitobjects,
                   key=lambda h: h['end_time'] if h.get('end_time') else h['time'])
    last_time = last_obj['end_time'] if last_obj.get('end_time') else last_obj['time']
    durations = {}
    cursor = last_time
    for i in range(len(timing_points) - 1, -1, -1):
        start, bpm = timing_points[i]
        if start > cursor:
            continue
        seg_start = 0 if i == 0 else start
        durations[bpm] = durations.get(bpm, 0) + max(0, cursor - seg_start)
        cursor = start
    if not durations:
        return timing_points[0][1]
    return max(durations.items(), key=lambda kv: kv[1])[0]


# ----------------------------------------------------------------------
# SV normalization (port of Qua.NormalizeSVs)
# ----------------------------------------------------------------------


def _normalize_svs(slider_velocities, timing_points, bpm_does_not_affect_sv,
                   base_bpm):
    """Convert raw `(time_ms, multiplier)` pairs into the time-space
    `(time_sec, multiplier)` shape `QuaverSVEngine` consumes.

    When the chart is already normalized (`BPMDoesNotAffectScrollVelocity`
    is true) we just unit-convert; otherwise we replay the same merge
    Quaver does between the SV list and the timing-point list, scaling
    each SV by `currentBpm / baseBpm` and emitting a synthetic SV at
    each timing point.

    Multipliers go through `_f32` so the equality check that prunes
    redundant entries matches Quaver's `float`-precision arithmetic --
    a Python `float64` keeps tiny rounding drift that would emit
    near-1.0 spurious sections.
    """
    if bpm_does_not_affect_sv or not timing_points or base_bpm <= 0:
        return [(t / 1000.0, m) for t, m in slider_velocities], None

    # Port of Qua.NormalizeSVs. The first SV/timing-point we visit
    # *seeds* `cur_adjusted` (and the new InitialScrollVelocity) without
    # emitting; subsequent points emit only when their adjusted multiplier
    # changes. That seed-without-emit pattern is why a chart whose first
    # adjusted SV is 1.0 produces zero leading entries instead of one.
    out = []
    sv_idx = 0
    cur_bpm = timing_points[0][1]
    cur_sv_mult = 1.0
    cur_sv_start = None
    cur_adjusted = None
    initial_seed = None

    for i, (tp_time, tp_bpm) in enumerate(timing_points):
        next_same = (i + 1 < len(timing_points)
                     and timing_points[i + 1][0] == tp_time)

        while sv_idx < len(slider_velocities):
            sv_time, sv_mult = slider_velocities[sv_idx]
            if sv_time > tp_time:
                break
            if next_same and sv_time == tp_time:
                break

            if sv_time < tp_time:
                m = _f32(sv_mult * (cur_bpm / base_bpm))
                if cur_adjusted is None:
                    cur_adjusted = m
                    initial_seed = m
                elif m != cur_adjusted:
                    out.append((sv_time, m))
                    cur_adjusted = m

            cur_sv_start = sv_time
            cur_sv_mult = sv_mult
            sv_idx += 1

        # Timing points reset the previous SV multiplier (Quaver behaviour).
        if cur_sv_start is None or cur_sv_start < tp_time:
            cur_sv_mult = 1.0

        cur_bpm = tp_bpm
        m = _f32(cur_sv_mult * (cur_bpm / base_bpm))
        if cur_adjusted is None:
            cur_adjusted = m
            initial_seed = m
        elif m != cur_adjusted:
            out.append((tp_time, m))
            cur_adjusted = m

    while sv_idx < len(slider_velocities):
        sv_time, sv_mult = slider_velocities[sv_idx]
        m = _f32(sv_mult * (cur_bpm / base_bpm))
        if cur_adjusted is None:
            cur_adjusted = m
            initial_seed = m
        elif m != cur_adjusted:
            out.append((sv_time, m))
            cur_adjusted = m
        sv_idx += 1

    out.sort(key=lambda x: x[0])
    return [(t / 1000.0, m) for t, m in out], (initial_seed if initial_seed is not None else 1.0)


# ----------------------------------------------------------------------
# Hash + lookup helpers (parallel to chart.find_osu_by_hash)
# ----------------------------------------------------------------------


def find_qua_by_hash(md5_hash, songs_dir):
    """Look up the `.qua` whose MD5 matches `md5_hash` against the
    persistent chart-hash index that the library scan maintains.

    Read-only on the hot path: a miss returns None instead of triggering
    a full Songs-folder rebuild, since rebuilding here would block a
    single play-action on hashing thousands of charts. The library scan
    is the one place that grows the index ; `songs_dir` is unused but
    kept for signature stability with callers that don't know that yet."""
    del songs_dir
    if not md5_hash:
        return None
    try:
        from analysis.games.quaver.adapter import _CHART_INDEX_CACHE
    except ImportError:
        return None
    cached = _CHART_INDEX_CACHE.load() or {}
    for path_str, (_m, _s, md5, _meta) in cached.items():
        if md5 == md5_hash:
            return path_str
    return None
