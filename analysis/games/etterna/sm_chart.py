"""Minimal .sm/.ssc chart parser. Gives us BPM map + noterow->time conversion,
plus per-chart note lists (times, columns, holds). Keyed by chart difficulty."""
import os
import re
import bisect
import hashlib
from pathlib import Path

from analysis.core.cache import Cache

# Note: This is the worst implementation for anything I've ever seen. Why does Etterna do this???

ROWS_PER_BEAT = 48  # SM stores 192 subdivisions per measure (4 beats) = 48 rows per beat
# NOTE: Etterna uses noterow = row_in_measure * 4 scale. In SM files, each measure
# is N rows and each measure = 4 beats. So rows_per_beat depends on note density.
# But for Etterna replays specifically, noterow = 48 * beat_position (standard).
# We'll go with 48 rows/beat which matches Etterna's internal NoteData.
_SSC_VERSION_SPLIT_TIMING = 0.7


def _beat_to_row(beat):
    """Match Etterna's BeatToNoteRow for timing segment placement."""
    import math
    x = float(beat) * ROWS_PER_BEAT
    return int(math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5))


def _row_to_beat(row):
    return int(row) / float(ROWS_PER_BEAT)


def _quantize_beat(beat):
    return _row_to_beat(_beat_to_row(beat))


def _quantize_pairs(pairs):
    return [(_quantize_beat(b), v) for b, v in (pairs or [])]


def _quantize_warps(warps):
    out = []
    for b, length in (warps or []):
        length_rows = _beat_to_row(length)
        if length_rows > 0:
            out.append((_quantize_beat(b), _row_to_beat(length_rows)))
    return out


def _strip_comments(text):
    return re.sub(r'//[^\n]*', '', text)


# SMLoader::ProcessBPMsAndStops (NotesLoaderSM.cpp:511) preprocesses raw
# BPMS+STOPS so that BPM<0, BPM>FAST_BPM_WARP, and negative stops become
# explicit WarpSegments. The note loader runs this BEFORE TidyUpData
# (NotesLoaderSM.cpp:1219-1221, comment: "Turn negative time changes into
# warps"). All downstream timing/render code (IsWarpAtRow,
# GetBeatAndBPSFromElapsedTime) only knows about clean BPMs + warps - it
# never sees the raw negatives. We mirror that contract here.
_FAST_BPM_WARP = 9999999.0


def process_bpms_and_stops(bpms, stops):
    """Port of SMLoader::ProcessBPMsAndStops. Returns
    (out_bpms, out_stops, out_warps)."""
    bpms = sorted(bpms or [])
    stops = sorted(stops or [])
    out_bpms: list[tuple[float, float]] = []
    out_stops: list[tuple[float, float]] = []
    out_warps: list[tuple[float, float]] = []

    bpm = 0.0
    prev_beat = 0.0
    warp_start = -1.0
    prewarp_bpm = 0.0
    timeofs = 0.0

    i_bpm = 0
    while i_bpm < len(bpms) and bpms[i_bpm][0] <= 0:
        bpm = bpms[i_bpm][1]
        i_bpm += 1
    i_stop = 0
    while i_stop < len(stops) and stops[i_stop][0] < 0:
        i_stop += 1

    if bpm == 0:
        bpm = bpms[i_bpm][1] if i_bpm < len(bpms) else 60.0
        if i_bpm < len(bpms):
            i_bpm += 1
    if 0 < bpm <= _FAST_BPM_WARP:
        out_bpms.append((0.0, bpm))

    while i_bpm < len(bpms) or i_stop < len(stops):
        change_is_bpm = (
            i_stop >= len(stops)
            or (i_bpm < len(bpms) and bpms[i_bpm][0] <= stops[i_stop][0])
        )
        change = bpms[i_bpm] if change_is_bpm else stops[i_stop]

        if bpm <= _FAST_BPM_WARP:
            timeofs += (change[0] - prev_beat) * 60.0 / bpm
            if warp_start >= 0 and bpm > 0 and timeofs > 0:
                warp_end = change[0] - (timeofs * bpm / 60.0)
                out_warps.append((warp_start, warp_end - warp_start))
                if bpm != prewarp_bpm:
                    out_bpms.append((warp_start, bpm))
                warp_start = -1.0
        prev_beat = change[0]

        if change_is_bpm:
            if warp_start < 0 and (change[1] < 0 or change[1] > _FAST_BPM_WARP):
                warp_start = change[0]
                prewarp_bpm = bpm
                timeofs = 0.0
            elif warp_start < 0:
                out_bpms.append((change[0], change[1]))
            bpm = change[1]
            i_bpm += 1
        else:
            if warp_start < 0 and change[1] < 0:
                warp_start = change[0]
                prewarp_bpm = bpm
                timeofs = change[1]
            elif warp_start < 0:
                out_stops.append((change[0], change[1]))
            else:
                timeofs += change[1]
                if change[1] > 0 and timeofs > 0:
                    warp_end = change[0]
                    out_warps.append((warp_start, warp_end - warp_start))
                    out_stops.append((change[0], timeofs))
                    if bpm < 0 or bpm > _FAST_BPM_WARP:
                        warp_start = change[0]
                        timeofs = 0.0
                    else:
                        if bpm != prewarp_bpm:
                            out_bpms.append((warp_start, bpm))
                        warp_start = -1.0
            i_stop += 1

    if warp_start >= 0:
        if bpm < 0 or bpm > _FAST_BPM_WARP:
            warp_end = 99999999.0
        else:
            warp_end = prev_beat - (timeofs * bpm / 60.0)
        out_warps.append((warp_start, warp_end - warp_start))
        if bpm != prewarp_bpm:
            out_bpms.append((warp_start, bpm))

    if not out_bpms:
        out_bpms.append((0.0, 60.0))
    return _quantize_pairs(out_bpms), _quantize_pairs(out_stops), _quantize_warps(out_warps)


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

    scrolls = _quantize_pairs(_parse_scrolls(tags.get('SCROLLS', '')))
    speeds = _quantize_speeds(_parse_speeds(tags.get('SPEEDS', '')))
    stops = _parse_bpms(tags.get('STOPS', ''))
    delays = _quantize_pairs(_parse_bpms(tags.get('DELAYS', '')))
    warps = _quantize_warps(_parse_bpms(tags.get('WARPS', '')))
    bpms, stops, derived_warps = process_bpms_and_stops(bpms, stops)
    warps = sorted(warps + derived_warps)

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
            'scrolls': scrolls,
            'speeds': speeds,
            'stops': stops,
            'delays': delays,
            'warps': warps,
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
    try:
        version = float(h.get('VERSION', '0') or 0)
    except ValueError:
        version = 0.0
    bpms_h = _parse_bpms(h.get('BPMS', ''))
    scrolls_h = _quantize_pairs(_parse_scrolls(h.get('SCROLLS', '')))
    speeds_h = _quantize_speeds(_parse_speeds(h.get('SPEEDS', '')))
    stops_h = _parse_bpms(h.get('STOPS', ''))
    delays_h = _quantize_pairs(_parse_bpms(h.get('DELAYS', '')))
    warps_h = _parse_warps(h.get('WARPS', ''), version)

    charts = []
    for sec in sections[1:]:
        t = parse_tags(sec)
        notes = t.get('NOTES', '')
        bpms = _parse_bpms(t.get('BPMS', '')) or bpms_h
        scrolls = _quantize_pairs(_parse_scrolls(t.get('SCROLLS', ''))) or scrolls_h
        speeds = _quantize_speeds(_parse_speeds(t.get('SPEEDS', ''))) or speeds_h
        stops = _parse_bpms(t.get('STOPS', '')) or stops_h
        delays = _quantize_pairs(_parse_bpms(t.get('DELAYS', ''))) or delays_h
        warps = _parse_warps(t.get('WARPS', ''), version) or warps_h
        bpms, stops, derived_warps = process_bpms_and_stops(bpms, stops)
        warps = sorted(warps + derived_warps)
        charts.append({
            'stepstype': t.get('STEPSTYPE', ''),
            'description': t.get('DESCRIPTION', ''),
            'difficulty': t.get('DIFFICULTY', ''),
            'meter': t.get('METER', ''),
            'chartname': t.get('CHARTNAME', ''),
            'chartkey': t.get('CHARTKEY', ''),
            'bpms': bpms,
            'offset': float(t.get('OFFSET', offset) or offset),
            'notedata': notes,
            'scrolls': scrolls,
            'speeds': speeds,
            'stops': stops,
            'delays': delays,
            'warps': warps,
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


def _parse_warps(s, version=_SSC_VERSION_SPLIT_TIMING):
    out = []
    for beat, value in _parse_bpms(s):
        if version < _SSC_VERSION_SPLIT_TIMING and value > beat:
            value = value - beat
        if value > 0:
            out.append((_quantize_beat(beat), _quantize_beat(value)))
    return out


def _parse_scrolls(s):
    """Parse #SCROLLS: beat=factor, ... into [(beat, factor)]."""
    return _parse_bpms(s)


def _quantize_speeds(speeds):
    out = []
    for seg in speeds or []:
        if len(seg) >= 4:
            beat, ratio, delay, unit = seg[:4]
            out.append((_quantize_beat(beat), ratio, delay, unit))
        else:
            beat, ratio = seg[:2]
            out.append((_quantize_beat(beat), ratio, *seg[2:]))
    return out


def _parse_speeds(s):
    """Parse #SPEEDS entries.

    Returns either:
    - `(beat, ratio)` when only those fields are present, or
    - `(beat, ratio, delay, unit)` when transition metadata is present.

    `unit` matches Etterna's `SpeedSegment::BaseUnit` encoding:
    - `0` = beats
    - `1` = seconds
    """
    out = []
    for entry in (s or '').split(','):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split('=')
        if len(parts) < 2:
            continue
        try:
            beat = float(parts[0])
            ratio = float(parts[1])
        except ValueError:
            pass
            continue
        if len(parts) < 3:
            out.append((beat, ratio))
            continue

        delay_txt = parts[2].strip()
        unit = 0
        if delay_txt.lower().endswith('s'):
            unit = 1
            delay_txt = delay_txt[:-1]
        try:
            delay = float(delay_txt or '0')
        except ValueError:
            out.append((beat, ratio))
            continue

        if len(parts) >= 4 and parts[3].strip():
            try:
                unit = 0 if int(float(parts[3])) == 0 else 1
            except ValueError:
                pass

        out.append((beat, ratio, delay, unit))
    return out


def sv_sections_from_chart(chart, bpms, offset):
    """Build sv_sections list for the Player from a parsed chart dict.

    Combines #SCROLLS (per-beat scroll factor) and #SPEEDS (per-beat speed
    multiplier) into a single [(time_sec, combined_multiplier)] list sorted
    by time. Either tag can be absent; the other still contributes.

    The combined multiplier at any beat is scrolls(beat) * speeds(beat), where
    both step-functions are piecewise-constant and sampled at every change point
    from either source.
    """
    scrolls = chart.get('scrolls') or []
    speeds = chart.get('speeds') or []
    if not scrolls and not speeds:
        return []

    # Collect all change-point beats from both sources
    beats = sorted({s[0] for s in scrolls} | {s[0] for s in speeds})

    def last_value_at(pairs, beat):
        val = 1.0
        for item in pairs:
            b, v = item[0], item[1]
            if b <= beat:
                val = v
            else:
                break
        return val

    # pairs are already sorted (parsed in order from the file)
    sections = []
    for beat in beats:
        t = beat_to_time(beat, bpms, offset)
        s_val = last_value_at(scrolls, beat)
        sp_val = last_value_at(speeds, beat)
        sections.append((t, s_val * sp_val))

    sections.sort(key=lambda x: x[0])
    return sections


def beat_to_time(beat, bpms, offset, stops=None, delays=None, warps=None):
    """Convert beat position -> time in seconds.

    Handles the four Etterna timing events (ports TimingData.cpp::
    GetElapsedTimeInternal):

      BPMS:   change bps going forward
      STOPS:  beat=duration_sec; time advances while beat stays put, AFTER
              the beat's events have triggered
      DELAYS: beat=duration_sec; time advances while beat stays put, BEFORE
              the beat's events trigger
      WARPS:  beat=length_beats; at the given beat, jump forward in
              beat-space by `length_beats` with no time advance. Notes and
              events inside the warp range are skipped.

    `offset` is OFFSET from the simfile: positive means audio starts LATER
    than beat 0, so beat 0 maps to `-offset` seconds.
    """
    bpms = sorted(bpms or [(0.0, 120.0)])
    stops = sorted(stops or [])
    delays = sorted(delays or [])
    warps = sorted(warps or [])

    # Port TimingData::GetElapsedTimeInternal closely. The target beat is a
    # marker in the event search, so same-row precedence matters:
    # BPM -> DELAY -> marker -> STOP -> WARP. This is why a note exactly on a
    # STOP row maps to the pre-stop time, while a note exactly on a DELAY row
    # maps to the post-delay time.
    t = -offset
    last_beat = 0.0
    bps = bpms[0][1] / 60.0
    i_bpm = 0
    i_delay = 0
    i_stop = 0
    i_warp = 0
    is_warping = False
    warp_destination = -float('inf')

    while True:
        event_beat = float('inf')
        event_type = None

        if is_warping and warp_destination < event_beat:
            event_beat = warp_destination
            event_type = 'warp_destination'
        if i_bpm < len(bpms) and bpms[i_bpm][0] < event_beat:
            event_beat = bpms[i_bpm][0]
            event_type = 'bpm'
        if i_delay < len(delays) and delays[i_delay][0] < event_beat:
            event_beat = delays[i_delay][0]
            event_type = 'delay'
        if beat < event_beat:
            event_beat = beat
            event_type = 'marker'
        if i_stop < len(stops) and stops[i_stop][0] < event_beat:
            event_beat = stops[i_stop][0]
            event_type = 'stop'
        if i_warp < len(warps) and warps[i_warp][0] < event_beat:
            event_beat = warps[i_warp][0]
            event_type = 'warp'

        if event_type is None:
            return t

        if event_beat >= 0:
            if not is_warping:
                t += (event_beat - last_beat) / bps
            last_beat = event_beat

        if event_type == 'marker':
            return t
        if event_type == 'warp_destination':
            is_warping = False
        elif event_type == 'bpm':
            bps = float(bpms[i_bpm][1]) / 60.0
            i_bpm += 1
        elif event_type == 'delay':
            t += float(delays[i_delay][1])
            i_delay += 1
        elif event_type == 'stop':
            t += float(stops[i_stop][1])
            i_stop += 1
        elif event_type == 'warp':
            is_warping = True
            warp_destination = max(
                warp_destination,
                float(warps[i_warp][0]) + float(warps[i_warp][1]),
            )
            i_warp += 1


def row_to_time(noterow, bpms, offset, stops=None, delays=None, warps=None):
    """noterow / 48 = beat. Returns seconds since audio start of first beat."""
    return beat_to_time(noterow / 48.0, bpms, offset, stops, delays, warps)


def is_beat_in_warp(beat, warps, stops=None, delays=None):
    """Port of TimingData::IsWarpAtRow. Beats strictly inside any
    [warp_start, warp_start + warp_length) are unjudgable / unrendered,
    EXCEPT when a stop or delay also lands exactly at that beat (Etterna
    allows stop-inside-warp gimmicks)."""
    if not warps:
        return False
    for wb, wl in warps:
        if wb <= beat < wb + wl:
            if stops or delays:
                for sb, _ in (stops or []):
                    if sb == beat:
                        return False
                for db, _ in (delays or []):
                    if db == beat:
                        return False
            return True
    return False


# parse_notes_block notetype codes. Values match Etterna's TapNoteType enum
# (NoteTypes.h) where practical so downstream code can reason about them
# without a translation table. '-1' for mine is legacy from before this file
# learned about the enum; kept for back-compat with the fingerprint path.
NT_TAP = 1
NT_HOLD_HEAD = 2      # '2'
NT_ROLL_HEAD = 4      # '4' ; distinct from HOLD for scoring; stored as its
                      # own value so the fingerprint can tell them apart even
                      # though Etterna's chartkey coalesces them to enum 2.
NT_MINE = -1          # 'M'
NT_LIFT = 5           # 'L'
NT_FAKE = 6           # 'F'
NT_AUTO_KEYSOUND = 7  # 'K'


def parse_notes_block(notedata, keycount_hint=None):
    """Parse SM notedata block into (notes, holds).
    notes: list of (noterow, column, notetype).
    holds: list of (start_row, column, end_row) for '2'→'3' and '4'→'3' spans.

    Recognizes every note character Etterna's NoteDataUtil currently parses:
    0 (empty), 1 (tap), 2 (hold head), 3 (tail), 4 (roll head), M (mine),
    K (auto-keysound), L (lift), F (fake). Commented-out cases in upstream
    (A/I/N) are skipped the same way Etterna does. Unknown chars are
    ignored rather than asserted, matching the upstream 'be lenient with
    broken .sm files' behavior."""
    measures = notedata.strip().split(',')
    notes = []
    holds_open = {}  # col -> start_row
    holds = []
    for mi, measure in enumerate(measures):
        lines = [ln.strip() for ln in measure.strip().split('\n')
                 if ln.strip() and not ln.strip().startswith('//')]
        if not lines:
            continue
        subdiv = len(lines)
        # Round-to-nearest, not floor: matches BeatToNoteRow's lrintf so rows
        # line up with Etterna's for non-power-of-2 subdivisions.
        span = ROWS_PER_BEAT * 4
        for li, line in enumerate(lines):
            row = mi * span + int(span * li / subdiv + 0.5)
            for col, ch in enumerate(line):
                if ch == '0':
                    continue
                if ch == '1':
                    notes.append((row, col, NT_TAP))
                elif ch == '2':
                    notes.append((row, col, NT_HOLD_HEAD))
                    holds_open[col] = (row, NT_HOLD_HEAD)
                elif ch == '4':
                    notes.append((row, col, NT_ROLL_HEAD))
                    holds_open[col] = (row, NT_ROLL_HEAD)
                elif ch == '3':
                    head = holds_open.pop(col, None)
                    if head is not None:
                        holds.append((head[0], col, row))
                elif ch == 'M':
                    notes.append((row, col, NT_MINE))
                elif ch == 'L':
                    notes.append((row, col, NT_LIFT))
                elif ch == 'F':
                    notes.append((row, col, NT_FAKE))
                elif ch == 'K':
                    notes.append((row, col, NT_AUTO_KEYSOUND))
                # else: unknown/commented-out (A/I/N/etc.) ; skip silently
    return notes, holds


# --- Etterna-compatible chartkey generation ---------------------------------
# Mirrors Steps::GenerateChartKey in upstream Etterna (Steps.cpp:549). The key
# is a SHA-1 over a per-row string: for each non-empty row, walk every track
# in column order and append TapNoteType.type as a decimal int, then append
# int(BPM_at_row + 0.374643). Prepend "X" and hex-encode.
#
# Reproducing this for .sm files lets us skip the fingerprint fallback ; the
# XML's chartkey resolves deterministically to the right chart file even
# when multiple charts share an identical first-N-notes prefix.
#
# TapNoteType enum (Etterna NoteTypes.h):
#   0 Empty, 1 Tap, 2 HoldHead, 3 HoldTail, 4 Mine, 5 Lift,
#   6 AutoKeysound, 7 Fake
# Our .sm parser uses different ad-hoc codes in parse_notes_block; the map
# below converts the file character directly to Etterna's enum value. Hold
# interior rows (between '2'/'4' and '3') are Empty in Etterna's NoteData,
# matching AddHoldNote behavior.
_ETT_TAPTYPE = {
    '1': 1,  # Tap
    '2': 2,  # HoldHead (hold)
    '4': 2,  # HoldHead (roll) ; same enum value in Etterna
    'M': 4,  # Mine
    'L': 5,  # Lift
    'F': 7,  # Fake
    'K': 6,  # AutoKeysound (rare; still non-empty)
}


def _iter_chart_rows(notedata, num_tracks):
    """Yield (row, [taptype_0..taptype_{num_tracks-1}]) for every row that
    has at least one non-Empty tap. Follows Etterna's row semantics: a hold
    head goes at the start row; cells between head and tail are Empty; the
    tail row itself is Empty for chartkey purposes.

    Row computation mirrors Etterna's `BeatToNoteRow((m + l/size) * 4)`:
    a round-to-nearest based on float position into the measure, not plain
    integer division. This matters for measures whose line count doesn't
    divide 192 evenly (quintuplets, 7-lets, etc.) ; otherwise the rows
    drift and the BPM-at-row / per-row types end up at different indices
    than Etterna's, changing the chartkey."""
    measures = _strip_comments(notedata).strip().split(',')
    rows = {}
    for mi, measure in enumerate(measures):
        lines = [ln.strip() for ln in measure.strip().split('\n')
                 if ln.strip()]
        if not lines:
            continue
        subdiv = len(lines)
        base = mi * (ROWS_PER_BEAT * 4)  # rows in one measure = 48*4 = 192
        span = ROWS_PER_BEAT * 4
        for li, line in enumerate(lines):
            # BeatToNoteRow's lrintf rounds half away-from-zero; since values
            # are non-negative, round-half-up matches.
            row = base + int(span * li / subdiv + 0.5)
            tracks = rows.setdefault(row, [0] * num_tracks)
            for col, ch in enumerate(line[:num_tracks]):
                t = _ETT_TAPTYPE.get(ch, 0)
                if t:
                    tracks[col] = t
    for row in sorted(rows):
        tracks = rows[row]
        if any(tracks):
            yield row, tracks


def _bpm_at_row(row, bpm_segments):
    """Return the BPM active at `row`. `bpm_segments` is a list of
    (beat, bpm); Etterna stores BPM segment positions as note rows, so a
    segment becomes active when BeatToNoteRow(segment_beat) <= query row."""
    if not bpm_segments:
        return 120.0
    segs = sorted(bpm_segments, key=lambda seg: seg[0])
    cur = segs[0][1]
    for b, v in segs:
        bpm_row = int(b * ROWS_PER_BEAT + 0.5)
        if bpm_row <= row:
            cur = v
        else:
            break
    return cur


def generate_chartkey(notedata: str,
                      bpms: list[tuple[float, float]],
                      stepstype: str) -> str | None:
    """Compute Etterna's chartkey for a single chart. Matches the C++
    Steps::GenerateChartKey byte-for-byte so the resulting "X…" hash equals
    the one Etterna writes to Etterna.xml and to .ssc files' #CHARTKEY tag.
    Returns 'X' + 40 hex chars, or None if the notedata is empty."""
    num_tracks = stepstype_keycount(stepstype)
    buf = []
    any_row = False
    for row, tracks in _iter_chart_rows(notedata, num_tracks):
        any_row = True
        for t in tracks:
            buf.append(str(t))
        # The +0.374643 nudge biases near-integer BPMs to round up, matching
        # the C++ cast int(bpm + 0.374643F). Without this, BPMs like 119.999
        # stored in some packs would hash differently from 120.000.
        bpm = _bpm_at_row(row, bpms)
        buf.append(str(int(bpm + 0.374643)))
    if not any_row:
        return None
    return 'X' + hashlib.sha1(''.join(buf).encode('utf-8')).hexdigest()


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


_CHARTKEY_INDEX_CACHE = Cache('chartkey_index.pkl')


def _scan_one_chartfile(p):
    """Extract (path_str, [(chartkey, chart_index), ...]) for one .ssc or
    .sm file. For .ssc, index the file's stored #CHARTKEY tag and also the
    generated key when it differs; real files can carry stale tags after note
    edits. .sm files and .ssc blocks missing the tag use the generated key.
    Pure CPU/IO ; safe under ThreadPoolExecutor."""
    p_str = str(p)
    try:
        if p_str.endswith('.ssc'):
            data = parse_ssc(p)
        else:
            data = parse_sm(p)
    except Exception:
        return None
    out = []
    for ci, ch in enumerate(data['charts']):
        keys = []
        stored_key = ch.get('chartkey') if p_str.endswith('.ssc') else None
        if stored_key:
            keys.append(stored_key)
        bpms = ch.get('bpms') or data['bpms']
        try:
            generated_key = generate_chartkey(
                ch['notedata'], bpms, ch.get('stepstype', 'dance-single'))
        except Exception:
            generated_key = None
        if generated_key and generated_key not in keys:
            keys.append(generated_key)
        for key in keys:
            out.append((key, ci))
    return p_str, out


def _build_chartkey_index(songs_dir, progress=None):
    """Walk Songs once, extract or synthesize every chart's chartkey, and
    build `{chartkey: (file, chart_index)}`. Covers both .ssc (stored or
    computed) and .sm (computed).

    Uses processes, not threads: SHA-1 + notedata walks are CPU-bound and
    Python's GIL caps thread parallelism to ~1 core. Across 22k files this
    is the difference between ~30s and ~90s on the first run. Cached
    afterwards so subsequent launches are instant."""
    from concurrent.futures import ProcessPoolExecutor
    import time
    index = {}
    root = Path(songs_dir)
    if progress:
        progress('scanning Songs directory…')
    paths = list(root.rglob('*.ssc')) + list(root.rglob('*.sm'))
    total = len(paths)
    # TODO: Stop this from happening every new replay
    if progress:
        progress(f'indexing charts 0/{total} (0%)')
    max_workers = max(2, (os.cpu_count() or 4))
    # Emit ~every 200 files or 0.25s, whichever is sparser, so the status
    # line tracks the scan without flooding the Qt event loop.
    step = max(50, total // 200) if total else 1
    last_emit = time.monotonic()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for i, res in enumerate(ex.map(_scan_one_chartfile, paths,
                                        chunksize=32), start=1):
            if res is not None:
                path_str, entries = res
                for chartkey, ci in entries:
                    # First match wins. Duplicate chartkeys across files
                    # are rare and mean identical notes+BPMs.
                    index.setdefault(chartkey, (path_str, ci))
            now = time.monotonic()
            if progress and (i % step == 0 or i == total) and \
                    (now - last_emit) > 0.25:
                pct = int(100 * i / total) if total else 100
                progress(f'indexing charts {i}/{total} ({pct}%)')
                last_emit = now
    if progress:
        progress(f'indexed {total}/{total} (100%)')
    return index


def get_chartkey_index(songs_dir, refresh=False, progress=None):
    """Return {chartkey: (ssc_file, chart_index)} for all charts under
    songs_dir. Cached on disk; refreshes when Songs dir mtime changes."""
    songs_dir = str(songs_dir)
    try:
        songs_mtime = os.stat(songs_dir).st_mtime
    except OSError:
        songs_mtime = 0.0
    fp = (songs_dir, songs_mtime)
    if not refresh:
        if _CHARTKEY_INDEX_CACHE.fingerprint() == fp:
            cached = _CHARTKEY_INDEX_CACHE.load()
            if cached is not None:
                return cached
    if progress:
        progress('scanning Songs for chartkeys…')
    index = _build_chartkey_index(songs_dir, progress=progress)
    _CHARTKEY_INDEX_CACHE.save(index, fingerprint=fp)
    return index


def find_chart_by_key(chartkey, songs_dir, progress=None):
    """Fast lookup via cached chartkey index. Returns dict like
    find_chart_for_replay or None. The index now covers .sm files too, so
    the loader picks the right parser based on the stored path.

    `progress` is forwarded so the first-run scan can surface ticks to
    the caller instead of looking like a 30-second stall."""
    if not chartkey:
        return None
    idx = get_chartkey_index(songs_dir, progress=progress)
    hit = idx.get(chartkey)
    if hit is None:
        return None
    chart_file, chart_idx = hit
    try:
        data = (parse_ssc(chart_file) if chart_file.endswith('.ssc')
                else parse_sm(chart_file))
    except Exception:
        return None
    if chart_idx >= len(data['charts']):
        return None
    return {'file': chart_file, 'data': data,
            'chart': data['charts'][chart_idx]}


FINGERPRINT_N = 50
_FINGERPRINT_INDEX_VERSION = 2
_FINGERPRINT_INDEX_CACHE = Cache('fingerprint_index.pkl')


def _normalize_fingerprint(rows_cols, n=None):
    """Sort by (noterow, column) globally, then return the longest prefix
    made up of complete chord groups whose length is <= n.

    Two reorderings need fixing:
      1. Within a chord group, charts list columns ascending but replays
         record press order ; a sort within equal-row runs flattens that.
      2. Across chord groups, replays are *usually* monotonic by row but
         some charts (observed: Hall of Kings) produce brief row inversions
         where a later row records first (e.g. 6600 then 6588). Etterna's
         .bin writer isn't strictly time-ordered. Sorting globally covers
         both cases; charts are already sorted so the sort is a no-op for
         them.

    Stopping on a group boundary prevents a chord straddling `n` from
    reintroducing column-order ambiguity at the tail."""
    data = sorted(rows_cols, key=lambda p: (p[0], p[1]))
    out = []
    i = 0
    while i < len(data):
        j = i
        while j < len(data) and data[j][0] == data[i][0]:
            j += 1
        if n is not None and len(out) + (j - i) > n:
            break
        out.extend(data[i:j])
        i = j
    return tuple(out)


def _chart_replay_rows_cols(chart):
    notedata = chart.get('notedata', '') if isinstance(chart, dict) else chart
    notes, _ = parse_notes_block(notedata)
    rows_cols = []
    for nr, c, t in notes:
        if t not in (NT_TAP, NT_HOLD_HEAD, NT_ROLL_HEAD):
            continue
        rows_cols.append((nr, c))
    return rows_cols


def _chart_fingerprint(chart, n=FINGERPRINT_N):
    """Return the first `n` (noterow, column) tuples for a chart, with
    chord columns sorted. None if the chart has fewer notes than n.

    Only taps, hold heads, and roll heads count ; fakes, keysounds,
    lifts, and mines don't show up in replay noterow streams, so
    including them here would break matches against real replays."""
    rows_cols = _chart_replay_rows_cols(chart)
    if len(rows_cols) < n:
        return None
    return _normalize_fingerprint(rows_cols, n)


def _scan_one_chartfile_fp(p):
    """Return (path_str, [(chart_index, fingerprint, note_count), ...]) for one
    .ssc/.sm file. Pure CPU/IO, safe under ThreadPoolExecutor."""
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
        # Note count matches what replays see: taps + hold heads + roll heads.
        nc = len(_chart_replay_rows_cols(ch))
        fp = _chart_fingerprint(ch)
        if fp is not None:
            out.append((ci, fp, nc))
    return str(p), out


def _build_fingerprint_index(songs_dir, progress=None):
    """Walk Songs, fingerprint every chart. Returns
    `{fingerprint: [(file, chart_idx, note_count), ...]}`. Multiple charts in
    the same file (or different files) can share an intro-fingerprint ;
    common with packs that ship Beginner/Hard cuts of the same song. The
    resolver disambiguates by matching note count against the replay length.
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
            for chart_idx, fp, nc in entries:
                index.setdefault(fp, []).append((path_str, chart_idx, nc))
    return index


def get_fingerprint_index(songs_dir, refresh=False, progress=None):
    """Return {fingerprint: (path, chart_idx)}. Cached on disk, invalidated
    when Songs dir mtime changes."""
    songs_dir = str(songs_dir)
    try:
        songs_mtime = os.stat(songs_dir).st_mtime
    except OSError:
        songs_mtime = 0.0
    fp = (_FINGERPRINT_INDEX_VERSION, songs_dir, songs_mtime)
    if not refresh:
        if _FINGERPRINT_INDEX_CACHE.fingerprint() == fp:
            cached = _FINGERPRINT_INDEX_CACHE.load()
            if cached is not None:
                return cached
    if progress:
        progress('scanning Songs for fingerprints…')
    index = _build_fingerprint_index(songs_dir, progress=progress)
    _FINGERPRINT_INDEX_CACHE.save(index, fingerprint=fp)
    return index


def find_chart_for_replay(replay_noterows, replay_columns, songs_dir,
                          chartkey_hint=None, progress=None):
    """Fast fingerprint match using the cached fingerprint index. Builds the
    index on first use (parallelized .ssc/.sm scan); subsequent calls are
    O(1) dict lookups. When multiple charts share an intro fingerprint
    (common for packs shipping Beginner/Hard cuts of one song) we pick the
    candidate whose chart note count matches `len(replay_noterows)`, which
    is the exact shape the replay recorded against."""
    if len(replay_noterows) < FINGERPRINT_N:
        return None
    # Pass the full list so the normalizer can walk full chord groups and
    # stop when the next group would exceed FINGERPRINT_N. Truncating the
    # input first would chop a chord straddling n into incomplete halves.
    fp = _normalize_fingerprint(
        list(zip(replay_noterows.tolist(), replay_columns.tolist())),
        FINGERPRINT_N)
    idx = get_fingerprint_index(songs_dir, progress=progress)
    candidates = idx.get(fp)
    if not candidates:
        return None
    replay_n = len(replay_noterows)
    # Exact note-count match wins; otherwise closest-by-count. Identical
    # fingerprints on charts with identical note counts are assumed to be
    # the same chart content (safe ; first match wins by order).
    best = min(candidates, key=lambda c: abs(c[2] - replay_n))
    path_str, chart_idx, _ = best
    try:
        data = parse_ssc(path_str) if path_str.endswith('.ssc') else parse_sm(path_str)
    except Exception:
        return None
    if chart_idx >= len(data['charts']):
        return None
    return {'file': path_str, 'data': data, 'chart': data['charts'][chart_idx]}
