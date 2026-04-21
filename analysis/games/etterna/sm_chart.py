"""Minimal .sm/.ssc chart parser. Gives us BPM map + noterow->time conversion,
plus per-chart note lists (times, columns, holds). Keyed by chart difficulty."""
import os
import re
import bisect
import hashlib
from pathlib import Path


ROWS_PER_BEAT = 48  # SM stores 192 subdivisions per measure (4 beats) = 48 rows per beat
# NOTE: Etterna uses noterow = row_in_measure * 4 scale. In SM files, each measure
# is N rows and each measure = 4 beats. So rows_per_beat depends on note density.
# But for Etterna replays specifically, noterow = 48 * beat_position (standard).
# We'll go with 48 rows/beat which matches Etterna's internal NoteData.


def _strip_comments(text):
    return re.sub(r'//[^\n]*', '', text)


def parse_sm(path):
    """Parse .sm file. Returns dict with bpms (list of (beat, bpm)), offset, charts (list)."""
    with open(path, encoding='utf-8', errors='replace') as f:
        text = _strip_comments(f.read())
    tags = {}
    for m in re.finditer(r'#([A-Z]+):([^;]*);', text, flags=re.DOTALL):
        tags[m.group(1)] = m.group(2).strip()

    offset = float(tags.get('OFFSET', '0') or 0)
    bpms = []
    for pair in (tags.get('BPMS', '') or '').split(','):
        pair = pair.strip()
        if not pair:
            continue
        try:
            b, v = pair.split('=')
            bpms.append((float(b), float(v)))
        except ValueError:
            pass
    if not bpms:
        bpms = [(0.0, 120.0)]

    charts = []
    # SM block: #NOTES: type: desc: diff: meter: radar: notedata;
    for m in re.finditer(r'#NOTES:(.*?);', text, flags=re.DOTALL):
        block = m.group(1)
        parts = block.split(':', 5)
        if len(parts) < 6:
            continue
        notedata = parts[5]
        charts.append({
            'stepstype': parts[0].strip(),
            'description': parts[1].strip(),
            'difficulty': parts[2].strip(),
            'meter': parts[3].strip(),
            'notedata': notedata,
        })
    return {
        'offset': offset,
        'bpms': bpms,
        'charts': charts,
        'title': tags.get('TITLE', ''),
        'artist': tags.get('ARTIST', ''),
        'music': tags.get('MUSIC', ''),
        'path': str(path),
    }


def parse_ssc(path):
    """Parse .ssc (per-chart BPMs)."""
    with open(path, encoding='utf-8', errors='replace') as f:
        text = _strip_comments(f.read())
    # split into song header + NOTEDATA blocks
    sections = re.split(r'#NOTEDATA:\s*;', text)
    header = sections[0]

    def parse_tags(sec):
        tags = {}
        for m in re.finditer(r'#([A-Z]+):([^;]*);', sec, flags=re.DOTALL):
            tags[m.group(1)] = m.group(2).strip()
        return tags

    h = parse_tags(header)
    offset = float(h.get('OFFSET', '0') or 0)
    bpms_h = _parse_bpms(h.get('BPMS', ''))

    charts = []
    for sec in sections[1:]:
        t = parse_tags(sec)
        notes = t.get('NOTES', '')
        bpms = _parse_bpms(t.get('BPMS', '')) or bpms_h
        charts.append({
            'stepstype': t.get('STEPSTYPE', ''),
            'description': t.get('DESCRIPTION', ''),
            'difficulty': t.get('DIFFICULTY', ''),
            'meter': t.get('METER', ''),
            'chartname': t.get('CHARTNAME', ''),
            'bpms': bpms,
            'offset': float(t.get('OFFSET', offset) or offset),
            'notedata': notes,
        })
    return {
        'offset': offset,
        'bpms': bpms_h,
        'charts': charts,
        'title': h.get('TITLE', ''),
        'artist': h.get('ARTIST', ''),
        'music': h.get('MUSIC', ''),
        'path': str(path),
    }


def _parse_bpms(s):
    out = []
    for pair in (s or '').split(','):
        pair = pair.strip()
        if not pair:
            continue
        try:
            b, v = pair.split('=')
            out.append((float(b), float(v)))
        except ValueError:
            pass
    return out


def beat_to_time(beat, bpms, offset):
    """Convert beat position -> time in seconds using BPM change points."""
    if not bpms:
        return offset + beat * (60.0 / 120.0)
    t = -offset
    prev_beat, prev_bpm = bpms[0]
    t = -offset  # OFFSET is positive=audio starts later, so t0 of beat0 = -offset
    # Walk through segments
    bpms_sorted = sorted(bpms)
    if beat <= bpms_sorted[0][0]:
        bpm = bpms_sorted[0][1]
        return -offset + (beat - bpms_sorted[0][0]) * (60.0 / bpm)

    for i in range(len(bpms_sorted)):
        b0, bpm0 = bpms_sorted[i]
        b1 = bpms_sorted[i + 1][0] if i + 1 < len(bpms_sorted) else float('inf')
        if beat <= b1:
            # time at b0:
            t0 = -offset
            for j in range(i):
                jb, jbpm = bpms_sorted[j]
                next_b = bpms_sorted[j + 1][0]
                t0 += (next_b - jb) * (60.0 / jbpm)
            return t0 + (beat - b0) * (60.0 / bpm0)
    return 0.0


def row_to_time(noterow, bpms, offset):
    """noterow / 48 = beat. Returns seconds since audio start of first beat."""
    return beat_to_time(noterow / 48.0, bpms, offset)


def parse_notes_block(notedata, keycount_hint=None):
    """Parse SM notedata block into list of (noterow, column, notetype, end_row_if_hold)."""
    measures = notedata.strip().split(',')
    notes = []  # (noterow, column, notetype)
    holds_open = {}  # (col) -> start_row
    holds = []  # (start_row, col, end_row)
    for mi, measure in enumerate(measures):
        lines = [ln.strip() for ln in measure.strip().split('\n')
                 if ln.strip() and not ln.strip().startswith('//')]
        if not lines:
            continue
        subdiv = len(lines)
        rows_per_line = (ROWS_PER_BEAT * 4) // subdiv
        for li, line in enumerate(lines):
            row = mi * (ROWS_PER_BEAT * 4) + li * rows_per_line
            for col, ch in enumerate(line):
                if ch in '0':
                    continue
                if ch == '1':
                    notes.append((row, col, 1))
                elif ch == '2':  # hold head
                    notes.append((row, col, 2))
                    holds_open[col] = row
                elif ch == '4':  # roll head
                    notes.append((row, col, 4))
                    holds_open[col] = row
                elif ch == '3':  # hold/roll tail
                    if col in holds_open:
                        holds.append((holds_open[col], col, row))
                        del holds_open[col]
                elif ch == 'M':  # mine
                    notes.append((row, col, -1))
                elif ch == 'L':  # lift
                    notes.append((row, col, 5))
                elif ch == 'F':  # fake
                    notes.append((row, col, 6))
                # else unknown — skip
    return notes, holds


def stepstype_keycount(stepstype):
    mapping = {
        'dance-single': 4, 'dance-solo': 6, 'dance-double': 8,
        'dance-couple': 8, 'dance-routine': 8,
        'pump-single': 5, 'pump-halfdouble': 6, 'pump-double': 10,
        'kb7-single': 7,
        'beat-single5': 6, 'beat-versus5': 6, 'beat-single7': 8,
        'beat-versus7': 8,
    }
    return mapping.get(stepstype, 4)


def chart_hash(stepstype, notedata):
    """Rough reproducible hash to match against chartkeys. Not exact Etterna chartkey."""
    return hashlib.md5((stepstype + notedata).encode()).hexdigest()[:16]


CHARTKEY_INDEX_PATH = Path.home() / '.cache' / 'etterna-analysis' / 'chartkey_index.pkl'


def _scan_one_ssc(p):
    """Extract (path_str, [(chartkey, chart_index), ...]) from one .ssc file.
    Returns None on I/O error. Pure CPU/IO — safe under ThreadPoolExecutor."""
    try:
        with open(p, encoding='utf-8', errors='replace') as f:
            text = _strip_comments(f.read())
    except OSError:
        return None
    sections = re.split(r'#NOTEDATA:\s*;', text)
    out = []
    for ci, sec in enumerate(sections[1:]):
        m = re.search(r'#CHARTKEY:([^;]+);', sec)
        if m:
            out.append((m.group(1).strip(), ci))
    return str(p), out


def _build_chartkey_index(songs_dir, progress=None):
    """Walk Songs once, extract every chart's (chartkey, file, chart_index)
    from .ssc files' #CHARTKEY lines. Returns dict[chartkey] = (file, idx).
    Only .ssc files carry chartkeys; .sm charts can't be indexed this way.
    Parses .ssc files in parallel — the scan is IO-bound but each file is
    big enough that threading still helps meaningfully."""
    from concurrent.futures import ThreadPoolExecutor
    index = {}
    root = Path(songs_dir)
    paths = list(root.rglob('*.ssc'))
    max_workers = min(32, (os.cpu_count() or 4) * 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, res in enumerate(ex.map(_scan_one_ssc, paths, chunksize=16)):
            if progress and i % 200 == 0:
                progress(f'indexing {i}/{len(paths)} charts…')
            if res is None:
                continue
            path_str, entries = res
            for chartkey, ci in entries:
                index[chartkey] = (path_str, ci)
    return index


def _load_chartkey_index():
    import pickle
    if not CHARTKEY_INDEX_PATH.exists():
        return None
    try:
        with open(CHARTKEY_INDEX_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_chartkey_index(data):
    import pickle
    CHARTKEY_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(CHARTKEY_INDEX_PATH, 'wb') as f:
            pickle.dump(data, f)
    except OSError:
        pass


def get_chartkey_index(songs_dir, refresh=False, progress=None):
    """Return {chartkey: (ssc_file, chart_index)} for all charts under
    songs_dir. Cached on disk; refreshes when Songs dir mtime changes."""
    songs_dir = str(songs_dir)
    try:
        songs_mtime = os.stat(songs_dir).st_mtime
    except OSError:
        songs_mtime = 0.0
    cached = _load_chartkey_index() if not refresh else None
    if cached and cached.get('songs_dir') == songs_dir and \
            abs(cached.get('mtime', 0) - songs_mtime) < 1.0:
        return cached['index']
    if progress:
        progress('scanning Songs for chartkeys…')
    index = _build_chartkey_index(songs_dir, progress=progress)
    _save_chartkey_index({
        'songs_dir': songs_dir,
        'mtime': songs_mtime,
        'index': index,
    })
    return index


def find_chart_by_key(chartkey, songs_dir):
    """Fast lookup via cached chartkey index. Returns dict like
    find_chart_for_replay or None."""
    if not chartkey:
        return None
    idx = get_chartkey_index(songs_dir)
    hit = idx.get(chartkey)
    if hit is None:
        return None
    ssc_file, chart_idx = hit
    try:
        data = parse_ssc(ssc_file)
    except Exception:
        return None
    if chart_idx >= len(data['charts']):
        return None
    return {'file': ssc_file, 'data': data, 'chart': data['charts'][chart_idx]}


FINGERPRINT_N = 50
FINGERPRINT_INDEX_PATH = Path.home() / '.cache' / 'etterna-analysis' / 'fingerprint_index_v4.pkl'


def _normalize_fingerprint(rows_cols, n=None):
    """Sort columns within each noterow and return the longest prefix made
    up of complete chord groups whose length is <= n. Charts list chord
    columns ascending; replays record them in press order — without sorting
    the two disagree on any chord. And without stopping on a group boundary,
    a chord straddling n would reintroduce order ambiguity at the tail."""
    out = []
    i = 0
    while i < len(rows_cols):
        j = i
        while j < len(rows_cols) and rows_cols[j][0] == rows_cols[i][0]:
            j += 1
        if n is not None and len(out) + (j - i) > n:
            break
        out.extend(sorted(rows_cols[i:j], key=lambda p: p[1]))
        i = j
    return tuple(out)


def _chart_fingerprint(notedata, n=FINGERPRINT_N):
    """Return the first `n` (noterow, column) tuples for a chart, with
    chord columns sorted. None if the chart has fewer notes than n."""
    notes, _ = parse_notes_block(notedata)
    if len(notes) < n:
        return None
    return _normalize_fingerprint([(nr, c) for (nr, c, _) in notes], n)


def _scan_one_chartfile_fp(p):
    """Return (path_str, [(chart_index, fingerprint), ...]) for one .ssc/.sm
    file. Pure CPU/IO, safe under ThreadPoolExecutor."""
    try:
        p_str = str(p)
        if p_str.endswith('.ssc'):
            data = parse_ssc(p)
        else:
            data = parse_sm(p)
    except Exception:
        return None
    out = []
    for ci, ch in enumerate(data['charts']):
        fp = _chart_fingerprint(ch['notedata'])
        if fp is not None:
            out.append((ci, fp))
    return str(p), out


def _build_fingerprint_index(songs_dir, progress=None):
    """Walk Songs, fingerprint every chart. Returns {fingerprint: (file, idx)}.
    Parsed in parallel since this is the slow path used when chartkey lookup
    misses (e.g. .sm files, or .ssc charts with missing #CHARTKEY)."""
    from concurrent.futures import ThreadPoolExecutor
    index = {}
    root = Path(songs_dir)
    paths = list(root.rglob('*.ssc')) + list(root.rglob('*.sm'))
    max_workers = min(32, (os.cpu_count() or 4) * 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, res in enumerate(ex.map(_scan_one_chartfile_fp, paths, chunksize=8)):
            if progress and i % 200 == 0:
                progress(f'fingerprinting {i}/{len(paths)} charts…')
            if res is None:
                continue
            path_str, entries = res
            for chart_idx, fp in entries:
                # First match wins; duplicate fingerprints are harmless since
                # the chart content is identical.
                index.setdefault(fp, (path_str, chart_idx))
    return index


def _load_fingerprint_index():
    import pickle
    if not FINGERPRINT_INDEX_PATH.exists():
        return None
    try:
        with open(FINGERPRINT_INDEX_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_fingerprint_index(data):
    import pickle
    FINGERPRINT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(FINGERPRINT_INDEX_PATH, 'wb') as f:
            pickle.dump(data, f)
    except OSError:
        pass


def get_fingerprint_index(songs_dir, refresh=False, progress=None):
    """Return {fingerprint: (path, chart_idx)}. Cached on disk, invalidated
    when Songs dir mtime changes."""
    songs_dir = str(songs_dir)
    try:
        songs_mtime = os.stat(songs_dir).st_mtime
    except OSError:
        songs_mtime = 0.0
    cached = _load_fingerprint_index() if not refresh else None
    if cached and cached.get('songs_dir') == songs_dir and \
            abs(cached.get('mtime', 0) - songs_mtime) < 1.0:
        return cached['index']
    if progress:
        progress('scanning Songs for fingerprints…')
    index = _build_fingerprint_index(songs_dir, progress=progress)
    _save_fingerprint_index({
        'songs_dir': songs_dir,
        'mtime': songs_mtime,
        'index': index,
    })
    return index


def find_chart_for_replay(replay_noterows, replay_columns, songs_dir,
                          chartkey_hint=None, progress=None):
    """Fast fingerprint match using the cached fingerprint index. Builds the
    index on first use (parallelized .ssc/.sm scan); subsequent calls are
    O(1) dict lookups."""
    if len(replay_noterows) < FINGERPRINT_N:
        return None
    # Pass the full list so the normalizer can walk full chord groups and
    # stop when the next group would exceed FINGERPRINT_N. Truncating the
    # input first would chop a chord straddling n into incomplete halves.
    fp = _normalize_fingerprint(
        list(zip(replay_noterows.tolist(), replay_columns.tolist())),
        FINGERPRINT_N)
    idx = get_fingerprint_index(songs_dir, progress=progress)
    hit = idx.get(fp)
    if hit is None:
        return None
    path_str, chart_idx = hit
    try:
        data = parse_ssc(path_str) if path_str.endswith('.ssc') else parse_sm(path_str)
    except Exception:
        return None
    if chart_idx >= len(data['charts']):
        return None
    return {'file': path_str, 'data': data, 'chart': data['charts'][chart_idx]}
