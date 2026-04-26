"""`.osu` file parsing: [General]/[Metadata]/[Difficulty]/[TimingPoints]/
[HitObjects] sections, plus the derived SV-section timeline."""


def _set_int(meta, key, v):
    try:
        meta[key] = int(float(v))
    except ValueError:
        pass


def _set_float(meta, key, v):
    try:
        meta[key] = float(v)
    except ValueError:
        pass


_META_KEYS = {
    'Title': ('title', str),
    'Artist': ('artist', str),
    'Creator': ('creator', str),
    'Version': ('version', str),
    'AudioFilename': ('audio', str),
    'CircleSize': ('keycount', _set_int),
    'OverallDifficulty': ('od', _set_float),
}


def _parse_hitobject(line, keycount):
    parts = line.split(',')
    if len(parts) < 5:
        return None
    x = int(parts[0])
    time = int(parts[2])
    obj_type = int(parts[3])
    column = min(int(x * keycount / 512), keycount - 1)
    is_hold = bool(obj_type & 128)
    end_time = None
    if is_hold:
        # mania hold extras field: endTime:hitSample
        head = (parts[5] if len(parts) > 5 else '').split(':', 1)[0]
        try:
            end_time = int(head)
        except ValueError:
            end_time = None
    return {'time': time, 'column': column,
            'is_hold': is_hold, 'end_time': end_time}


def parse_osu_file(osu_path):
    """Parse a `.osu` file. Return metadata + hitobjects + sv_sections
    (list of `(time_sec, sv_multiplier)` from [TimingPoints])."""
    meta = {'title': '', 'artist': '', 'creator': '', 'version': '',
            'audio': '', 'keycount': None, 'hitobjects': [],
            'timing_points': [], 'sv_sections': [], 'od': 8.0}
    section = None
    with open(osu_path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('//'):
                continue
            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1]
                continue
            if section == 'TimingPoints':
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        meta['timing_points'].append(
                            (float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
                continue
            if section in ('General', 'Metadata', 'Difficulty'):
                if ':' not in line:
                    continue
                k, v = (s.strip() for s in line.split(':', 1))
                entry = _META_KEYS.get(k)
                if entry is None:
                    continue
                key, coerce = entry
                if coerce is str:
                    meta[key] = v
                else:
                    coerce(meta, key, v)
            elif section == 'HitObjects':
                # Keycount may appear after HitObjects in weird charts, but
                # osu!'s own loader assumes 4 when missing ; match that.
                obj = _parse_hitobject(line, meta.get('keycount') or 4)
                if obj is not None:
                    meta['hitobjects'].append(obj)
    meta['hitobjects'].sort(key=lambda h: (h['time'], h['column']))
    meta['sv_sections'] = _compute_sv_sections(meta['timing_points'])
    return meta


def _compute_sv_sections(timing_points):
    """Port of pset6's SV parser.
    Uninherited TP (ms_per_beat > 0): SV = bpm / base_bpm.
    Inherited   TP (ms_per_beat < 0): SV = -100 / ms_per_beat.
    """
    if not timing_points:
        return []
    uninherited = [(t, mpb) for t, mpb in timing_points if mpb > 0]
    inherited = [(t, mpb) for t, mpb in timing_points if mpb < 0]
    base_bpm = _base_bpm(uninherited)
    sections = [(t_ms * 0.001, (60000.0 / mpb) / base_bpm)
                for t_ms, mpb in uninherited]
    sections += [(t_ms * 0.001, -100.0 / mpb) for t_ms, mpb in inherited]
    sections.sort(key=lambda x: x[0])
    return sections


def _base_bpm(uninherited):
    """Pick the dominant-duration uninherited BPM so a brief intro
    timing point doesn't set the base rate for the rest of the chart."""
    if not uninherited:
        return 120.0
    if len(uninherited) == 1:
        return 60000.0 / uninherited[0][1]
    ordered = sorted(uninherited, key=lambda x: x[0])
    last_t = ordered[-1][0]
    durations = {}
    for i, (t_ms, mpb) in enumerate(ordered):
        end = ordered[i + 1][0] if i + 1 < len(ordered) else last_t
        durations[mpb] = durations.get(mpb, 0) + max(0, end - t_ms)
    dominant_ms = max(durations, key=durations.get)
    return 60000.0 / dominant_ms


def find_osu_by_hash(md5_hash, songs_dir):
    """Look up the `.osu` whose MD5 matches `md5_hash` against the
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
        from analysis.games.osu.adapter import _CHART_INDEX_CACHE
    except ImportError:
        return None
    cached = _CHART_INDEX_CACHE.load() or {}
    for path_str, (_m, _s, md5, _meta) in cached.items():
        if md5 == md5_hash:
            return path_str
    return None
