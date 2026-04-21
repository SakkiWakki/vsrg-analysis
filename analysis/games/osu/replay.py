"""osu!mania replay parser. Aligns .osr key events against .osu chart hitobjects
to produce the same shape of data as etterna_replay.parse_replay:
  {noterows, offsets, columns, notetypes, misses, holds, keycount, filepath}
"""
import os
import re
import sys
import numpy as np
from pathlib import Path
import osrparse
from osrparse import Key


def parse_osu_file(osu_path):
    """Parse a .osu file. Return dict with keycount, hitobjects, title, artist, diff, audio,
    plus sv_sections: list of (time_sec, sv_multiplier) derived from [TimingPoints]."""
    meta = {'title': '', 'artist': '', 'creator': '', 'version': '',
            'audio': '', 'keycount': None, 'hitobjects': [],
            'timing_points': [], 'sv_sections': [],
            'od': 8.0}
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
                if len(parts) < 2:
                    continue
                try:
                    t_ms = float(parts[0])
                    ms_per_beat = float(parts[1])
                    meta['timing_points'].append((t_ms, ms_per_beat))
                except ValueError:
                    continue
                continue
            if section in ('General', 'Metadata', 'Difficulty'):
                if ':' in line:
                    k, v = line.split(':', 1)
                    k, v = k.strip(), v.strip()
                    if k == 'Title':
                        meta['title'] = v
                    elif k == 'Artist':
                        meta['artist'] = v
                    elif k == 'Creator':
                        meta['creator'] = v
                    elif k == 'Version':
                        meta['version'] = v
                    elif k == 'AudioFilename':
                        meta['audio'] = v
                    elif k == 'CircleSize':
                        try:
                            meta['keycount'] = int(float(v))
                        except ValueError:
                            pass
                    elif k == 'OverallDifficulty':
                        try:
                            meta['od'] = float(v)
                        except ValueError:
                            pass
            elif section == 'HitObjects':
                parts = line.split(',')
                if len(parts) < 5:
                    continue
                x = int(parts[0])
                time = int(parts[2])
                obj_type = int(parts[3])
                # In mania, x maps to column: column = floor(x * keycount / 512)
                keycount = meta.get('keycount') or 4
                column = min(int(x * keycount / 512), keycount - 1)
                is_hold = bool(obj_type & 128)
                end_time = None
                if is_hold:
                    # extras field: endTime:hitSample  (mania hold)
                    extras = parts[5] if len(parts) > 5 else ''
                    head = extras.split(':', 1)[0]
                    try:
                        end_time = int(head)
                    except ValueError:
                        end_time = None
                meta['hitobjects'].append({
                    'time': time,
                    'column': column,
                    'is_hold': is_hold,
                    'end_time': end_time,
                })
    meta['hitobjects'].sort(key=lambda h: (h['time'], h['column']))
    meta['sv_sections'] = _compute_sv_sections(meta['timing_points'])
    return meta


def _compute_sv_sections(timing_points):
    """Port of pset6's SV parser.
    Uninherited TP (ms_per_beat > 0): SV = bpm / base_bpm.
    Inherited   TP (ms_per_beat < 0): SV = -100 / ms_per_beat.
    Returns list of (time_sec, sv) sorted by time. Empty if no TPs."""
    if not timing_points:
        return []
    uninherited = [(t, mpb) for t, mpb in timing_points if mpb > 0]
    inherited = [(t, mpb) for t, mpb in timing_points if mpb < 0]
    if len(uninherited) > 1:
        sorted_un = sorted(uninherited, key=lambda x: x[0])
        last_t = sorted_un[-1][0]
        durations = {}
        for i, (t_ms, mpb) in enumerate(sorted_un):
            end = sorted_un[i + 1][0] if i + 1 < len(sorted_un) else last_t
            durations[mpb] = durations.get(mpb, 0) + max(0, end - t_ms)
        dominant_ms = max(durations, key=durations.get)
        base_bpm = 60000.0 / dominant_ms
    elif uninherited:
        base_bpm = 60000.0 / uninherited[0][1]
    else:
        base_bpm = 120.0
    sections = []
    for t_ms, mpb in uninherited:
        sections.append((t_ms * 0.001, (60000.0 / mpb) / base_bpm))
    for t_ms, mpb in inherited:
        sections.append((t_ms * 0.001, -100.0 / mpb))
    sections.sort(key=lambda x: x[0])
    return sections


def _decode_osr_frames(osr_path):
    """Read the raw LZMA frame payload from the .osr and return
    (time_deltas, keys_values, rng_seed). All frames preserved in order
    (including the two leading placeholder frames), minus the RNG-seed
    sentinel (dt=-12345), whose keys value is returned as rng_seed.
    """
    import lzma, struct
    with open(osr_path, 'rb') as f:
        data = f.read()

    def _uleb(buf, off):
        result = 0; shift = 0
        while True:
            b = buf[off]; off += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result, off
            shift += 7

    def _string(buf, off):
        flag = buf[off]; off += 1
        if flag == 0x00:
            return '', off
        n, off = _uleb(buf, off)
        return buf[off:off+n].decode('utf-8', 'replace'), off + n

    off = 1 + 4                     # mode + version
    _, off = _string(data, off)     # beatmap hash
    _, off = _string(data, off)     # player
    _, off = _string(data, off)     # replay hash
    off += 6 + 6 + 4 + 2 + 1 + 4    # counts + score + combo + perfect + mods
    _, off = _string(data, off)     # lifebar
    off += 8                         # timestamp
    rlen = struct.unpack_from('<i', data, off)[0]; off += 4
    payload = data[off:off+rlen]

    text = lzma.decompress(payload).decode('ascii', 'replace')
    dts, keys, rng_seed = [], [], None
    for e in text.rstrip(',').split(','):
        if not e:
            continue
        parts = e.split('|')
        if len(parts) != 4:
            continue
        try:
            dt = int(parts[0])
            k = int(float(parts[1]))
        except ValueError:
            continue
        if dt == -12345:
            rng_seed = k
            continue
        dts.append(dt)
        keys.append(k)
    return dts, keys, rng_seed


def parse_osr_events(osr_path):
    """Return (keycount, events[(abs_time_ms, keys_bitmask)], meta).

    Decodes the LZMA replay payload directly. osrparse silently drops the
    first two placeholder frames but discards their time_deltas with them;
    for maps that open with a skip, the 2nd placeholder carries the skip
    duration (often ~8s) and losing it shifts every press earlier by that
    amount. See kszlim/osu-replay-parser#41.
    """
    r = osrparse.Replay.from_path(osr_path)
    dt_list, keys_list, rng_seed = _decode_osr_frames(osr_path)

    raw = []
    used_bits = 0
    t_acc = 0
    for dt, keys_val in zip(dt_list, keys_list):
        t_acc += dt
        used_bits |= keys_val
        raw.append([t_acc, keys_val])

    if len(raw) >= 2 and raw[1][0] < raw[0][0]:
        raw[1][0] = raw[0][0]
        raw[0][0] = 0
    if len(raw) >= 3 and raw[0][0] > raw[2][0]:
        raw[0][0] = raw[1][0] = raw[2][0]

    events = [(t, k) for t, k in raw]
    meta = {
        'beatmap_hash': r.beatmap_hash,
        'player': r.username,
        'score': r.score,
        'max_combo': r.max_combo,
        'count_300': getattr(r, 'count_300', 0),
        'count_100': getattr(r, 'count_100', 0),
        'count_50': getattr(r, 'count_50', 0),
        'count_miss': getattr(r, 'count_miss', 0),
        'count_geki': getattr(r, 'count_geki', 0),
        'count_katu': getattr(r, 'count_katu', 0),
        'mods': int(r.mods.value) if hasattr(r.mods, 'value') else int(r.mods),
        'timestamp': str(r.timestamp),
        'replay_hash': r.replay_hash,
        'used_bits': used_bits,
        'rng_seed': rng_seed,
    }
    # infer keycount from the highest bit used, clamp to sensible min.
    if used_bits:
        inferred = used_bits.bit_length()
    else:
        inferred = 4
    return inferred, events, meta


def _press_times(events, keycount):
    """Return list per column of press times (ms, int)."""
    presses = [[] for _ in range(keycount)]
    prev = 0
    for t, keys in events:
        newly = keys & ~prev
        for c in range(keycount):
            if newly & (1 << c):
                presses[c].append(t)
        prev = keys
    return presses


def _release_times(events, keycount):
    """Per-column release (key-up) times in ms."""
    releases = [[] for _ in range(keycount)]
    prev = 0
    for t, keys in events:
        gone = prev & ~keys
        for c in range(keycount):
            if gone & (1 << c):
                releases[c].append(t)
        prev = keys
    return releases


def stable_hit_windows(od):
    """osu!stable / lazer-Classic hit windows by OD, in ms.
    Order: [MAX(300g), 300, 200, 100, 50, MISS].
    Source: osu!lazer ManiaHitWindows.cs (non-convert Classic branch).
      perfect=16; great=34+3*(10-OD); good=67+3*(10-OD);
      ok=97+3*(10-OD); meh=121+3*(10-OD); miss=158+3*(10-OD).
    MISS is the maximum hit window — a press further away than that doesn't
    consume a note at all.
    """
    inv = max(0.0, min(10.0, 10.0 - float(od)))
    return [
        16.0,
        34.0 + 3.0 * inv,
        67.0 + 3.0 * inv,
        97.0 + 3.0 * inv,
        121.0 + 3.0 * inv,
        158.0 + 3.0 * inv,
    ]


TAIL_RELEASE_LENIENCE = 1.5  # osu!lazer TailNote.RELEASE_WINDOW_LENIENCE


def _extract_key_events(events, keycount):
    """Turn (t_ms, keys_bitmask) frames into per-column (t, is_press) events.
    Press = bit 0→1 transition; release = 1→0.
    Returns list-of-lists, one per column, chronologically ordered."""
    out = [[] for _ in range(keycount)]
    prev = 0
    for t, keys in events:
        changed = keys ^ prev
        if changed:
            for c in range(keycount):
                bit = 1 << c
                if changed & bit:
                    out[c].append((t, bool(keys & bit)))
        prev = keys
    return out


def simulate_mania(notes_by_col, key_events_by_col, windows):
    """Simulate osu!mania Classic judgment per column.

    Args:
        notes_by_col: list of list-of-note-dicts per column, each:
            {'time': int_ms, 'end_time': int_or_None (None = rice)}
            must be sorted by time ascending.
        key_events_by_col: list of (t_ms, is_press) per column, chronological.
        windows: [MAX, 300, 200, 100, 50, MISS] in ms.

    Returns list-of-dicts parallel to notes_by_col (flat): each has
        {col, time, end_time, is_hold, press_t, release_t,
         head_off, tail_off, judgement, broken, missed}
    where:
        judgement is one of 'MAX','300','200','100','50','miss'
        head_off = press_t - note.time (None if never pressed)
        tail_off = release_t - note.end_time (None if never released or rice)
        broken = LN released too early (before tail_window)
        missed = head missed entirely (or rice missed)
    """
    early_w = windows[5]       # hitWindowEarly (158+3*inv): earliest press accepted
    late_expire_w = windows[3]  # HitWindow100 (97+3*inv): late auto-expiry
    meh_w = windows[4]          # HitWindow50 (121+3*inv): max accuracy for a non-miss
    tail_windows = [w * TAIL_RELEASE_LENIENCE for w in windows]
    tail_miss_w = tail_windows[5]

    results = []

    for c in range(len(notes_by_col)):
        notes = notes_by_col[c]
        events = key_events_by_col[c] if c < len(key_events_by_col) else []
        # per-note result slot
        per = [{
            'col': c,
            'time': n['time'],
            'end_time': n.get('end_time'),
            'is_hold': n.get('end_time') is not None,
            'press_t': None,
            'release_t': None,
            'head_off': None,
            'tail_off': None,
            'judgement': None,
            'broken': False,
            'missed': False,
        } for n in notes]

        # holding state: index of currently-held LN, or None.
        held_idx = None
        next_unjudged = 0  # leftmost index whose head hasn't been decided

        def _advance_misses(upto_t):
            """Auto-expire heads whose late boundary has passed by upto_t."""
            nonlocal next_unjudged
            while next_unjudged < len(per):
                r = per[next_unjudged]
                if r['judgement'] is not None or r['missed']:
                    next_unjudged += 1
                    continue
                if upto_t - late_expire_w >= r['time']:
                    r['missed'] = True
                    r['judgement'] = 'miss'
                    if r['is_hold']:
                        r['broken'] = True
                    next_unjudged += 1
                else:
                    break

        for (t, is_press) in events:
            _advance_misses(t)

            if is_press:
                if held_idx is not None:
                    _judge_tail(per, held_idx, t, tail_windows, meh_w)
                    held_idx = None

                if next_unjudged >= len(per):
                    continue
                r = per[next_unjudged]
                diff = t - r['time']
                if diff < -early_w:
                    continue
                r['press_t'] = t
                r['head_off'] = diff
                if abs(diff) <= meh_w:
                    j = _judgement_for(abs(diff), windows)
                else:
                    j = 'miss'
                if r['is_hold']:
                    if j == 'miss':
                        r['judgement'] = 'miss'
                        r['missed'] = True
                        r['broken'] = True
                    else:
                        r['judgement'] = None
                        r['_head_j'] = j
                        held_idx = next_unjudged
                else:
                    r['judgement'] = j
                    if j == 'miss':
                        r['missed'] = True
                next_unjudged += 1
            else:
                if held_idx is not None:
                    _judge_tail(per, held_idx, t, tail_windows, meh_w)
                    held_idx = None

        # Final sweep: auto-miss anything past by the last event (or always).
        _advance_misses(10**18)

        # Dangling held LN (song ended while still holding): treat as tail miss at end_time+tail_miss_w.
        if held_idx is not None:
            r = per[held_idx]
            r['broken'] = True
            r['judgement'] = 'miss'

        results.extend(per)

    return results


def _judgement_for(abs_diff, windows):
    """Walk MAX,300,200,100,50,MISS; return first window >= abs_diff."""
    labels = ['MAX', '300', '200', '100', '50', 'miss']
    for i, w in enumerate(windows):
        if abs_diff <= w:
            return labels[i]
    return 'miss'


def _judge_tail(per, idx, release_t, tail_windows, meh_w):
    """Judge the tail of an LN that was being held."""
    r = per[idx]
    if not r['is_hold']:
        return
    et = r['end_time']
    diff = release_t - et
    r['release_t'] = release_t
    r['tail_off'] = diff
    abs_diff = abs(diff)
    tail_miss_w = tail_windows[5]
    if abs_diff > tail_miss_w:
        # Released outside tail miss window entirely — tail miss.
        r['broken'] = True
        r['judgement'] = 'miss'
        return
    tail_j = _judgement_for(abs_diff, tail_windows)
    # Cap: broken LN or missed head → at most meh (50).
    if r.get('_head_j') == 'miss' or diff < -tail_windows[5]:
        r['broken'] = True
        r['judgement'] = '50' if tail_j not in ('miss',) else 'miss'
    else:
        # Combine head & tail: LN result is typically min of the two in stable.
        head_j = r.get('_head_j', 'miss')
        order = ['MAX', '300', '200', '100', '50', 'miss']
        combined = order[max(order.index(head_j), order.index(tail_j))]
        r['judgement'] = combined


OSU_MOD_RANDOM = 1 << 11
OSU_MOD_DOUBLETIME = 1 << 6
OSU_MOD_HALFTIME = 1 << 8
OSU_MOD_NIGHTCORE = 1 << 9   # always set alongside DT; NC = DT | NC


def rate_for_mods(mods):
    """Return playback rate multiplier from osu mod bitfield.

    DoubleTime and Nightcore both run at 1.5x (NC also sets the DT bit and
    adds pitch-shift on top). HalfTime runs at 0.75x. Everything else is 1.0x."""
    m = int(mods)
    if m & (OSU_MOD_DOUBLETIME | OSU_MOD_NIGHTCORE):
        return 1.5
    if m & OSU_MOD_HALFTIME:
        return 0.75
    return 1.0


class _LegacyRandom:
    """Port of .NET System.Random (osu!stable's RNG). Ref: osu!framework
    LegacyRandom.cs — subtractive generator with seed array of 56 ints."""
    MBIG = 2147483647
    MSEED = 161803398

    def __init__(self, seed):
        ii = 0
        mj = self.MSEED - abs(int(seed))
        seed_arr = [0] * 56
        seed_arr[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55
            seed_arr[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += self.MBIG
            mj = seed_arr[ii]
        for _ in range(1, 5):
            for i in range(1, 56):
                seed_arr[i] -= seed_arr[1 + (i + 30) % 55]
                if seed_arr[i] < 0:
                    seed_arr[i] += self.MBIG
        self._seed_arr = seed_arr
        self._inext = 0
        self._inextp = 21

    def _sample(self):
        ni = self._inext + 1
        if ni >= 56:
            ni = 1
        np_ = self._inextp + 1
        if np_ >= 56:
            np_ = 1
        ret = self._seed_arr[ni] - self._seed_arr[np_]
        if ret == self.MBIG:
            ret -= 1
        if ret < 0:
            ret += self.MBIG
        self._seed_arr[ni] = ret
        self._inext = ni
        self._inextp = np_
        return ret

    def next_int(self, max_value):
        """Return integer in [0, max_value)."""
        return int(self._sample() * (1.0 / self.MBIG) * max_value)


def _random_column_permutation(keycount, seed):
    """osu!mania Random-mod column shuffle. Fisher-Yates using LegacyRandom,
    matching osu!stable's ManiaModRandom.ApplyToBeatmap shuffle logic."""
    rng = _LegacyRandom(seed)
    perm = list(range(keycount))
    for i in range(keycount - 1, 0, -1):
        j = rng.next_int(i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def parse_replay(osr_path, osu_path=None, songs_dir=None, hit_window_ms=None):
    """Parse .osr + matching .osu to offset-per-note list.
    Runs a per-column osu!mania Classic judgment simulation to assign
    each press/release to a specific note, matching how the game judges.

    hit_window_ms is retained for signature compatibility but ignored:
    windows now come from chart OD via stable_hit_windows().
    """
    keycount_from_osr, events, meta = parse_osr_events(osr_path)

    if osu_path is None:
        if songs_dir is None:
            raise ValueError("need osu_path or songs_dir")
        osu_path = find_osu_by_hash(meta['beatmap_hash'], songs_dir)
        if osu_path is None:
            raise FileNotFoundError(f"no .osu match for hash {meta['beatmap_hash']}")

    chart = parse_osu_file(osu_path)
    keycount = chart.get('keycount') or max(keycount_from_osr, 4)

    # Random-mod column remap. The replay's seed comes from the -12345
    # sentinel event; we re-run osu!stable's Fisher-Yates over LegacyRandom
    # to reproduce the same permutation that was active during the play.
    col_perm = None
    if (meta.get('mods', 0) & OSU_MOD_RANDOM) and meta.get('rng_seed') is not None:
        col_perm = _random_column_permutation(keycount, meta['rng_seed'])

    # Hitobjects grouped per-column as dicts with time + optional end_time.
    by_col_notes = [[] for _ in range(keycount)]
    holds_meta = []
    for h in chart['hitobjects']:
        c = h['column']
        if c >= keycount:
            continue
        if col_perm is not None:
            c = col_perm[c]
        by_col_notes[c].append({'time': int(h['time']),
                                'end_time': int(h['end_time']) if h['is_hold'] else None})
        if h['is_hold']:
            holds_meta.append((h['time'], c, h['end_time']))
    for col_notes in by_col_notes:
        col_notes.sort(key=lambda n: n['time'])

    key_events_by_col = _extract_key_events(events, keycount)

    windows = stable_hit_windows(float(chart.get('od', 8.0)))

    sim = simulate_mania(by_col_notes, key_events_by_col, windows)

    # Ghost taps: presses that didn't get assigned to any note (pressed too
    # early, or pressed after the last note, or just a random tap between
    # notes). Collect the set of press times actually consumed by the
    # simulator per column, then anything left over is a ghost.
    #
    # Presses that fall inside any LN's [head_time, end_time] interval on
    # the same column are NOT ghosts — the player was clearly attempting
    # that LN, even if they missed the head and re-pressed mid-hold. A
    # missed LN can be re-hit partway through, so we can't restrict this
    # to only missed LNs.
    used_press_ts = [set() for _ in range(keycount)]
    for r in sim:
        c = r['col']
        if r['press_t'] is not None:
            used_press_ts[c].add(int(r['press_t']))
    ln_intervals = [[] for _ in range(keycount)]
    for r in sim:
        if r['is_hold'] and r['end_time'] is not None:
            ln_intervals[r['col']].append((r['time'], r['end_time']))
    for iv in ln_intervals:
        iv.sort()
    import bisect as _bisect

    def _inside_any_ln(t_ms, col):
        iv = ln_intervals[col]
        if not iv:
            return False
        # Rightmost LN whose head is <= t. If its end_time >= t, we're
        # inside. Intervals don't overlap within a column, so checking the
        # single candidate is sufficient.
        idx = _bisect.bisect_right(iv, (t_ms, float('inf'))) - 1
        if idx < 0:
            return False
        head, end = iv[idx]
        return head <= t_ms <= end

    ghost_taps = []
    for c in range(keycount):
        for (t, is_press) in key_events_by_col[c]:
            if not is_press:
                continue
            if int(t) in used_press_ts[c]:
                continue
            if _inside_any_ln(int(t), c):
                continue
            ghost_taps.append((int(t), c))

    # Ghost holds: for any LN whose head was missed or released early, the
    # player can still mash the key inside the LN interval and "partially
    # hold" the note. Each continuous press→release span that overlaps a
    # missed LN interval is emitted so the renderer can draw a red hit line
    # showing the true held duration — unclipped on the release side so you
    # can see if they kept holding past the LN tail.
    # One emission per span per column; deduped so a single continuous
    # hold that straddles N consecutive missed LNs produces ONE entry, not
    # N identical ones. ln_head is kept in the tuple for downstream
    # debugging/analysis — the renderer ignores it.
    ghost_holds = []
    missed_lns_by_col = [[] for _ in range(keycount)]
    for r in sim:
        if r['is_hold'] and r['judgement'] == 'miss' and r['end_time'] is not None:
            missed_lns_by_col[r['col']].append((r['time'], r['end_time']))
    for col_lns in missed_lns_by_col:
        col_lns.sort()
    for c in range(keycount):
        if not missed_lns_by_col[c]:
            continue
        press_start = None
        for (t, is_press) in key_events_by_col[c]:
            if is_press:
                if press_start is None:
                    press_start = int(t)
            else:
                if press_start is None:
                    continue
                span_lo, span_hi = press_start, int(t)
                press_start = None
                # Emit at most one entry per span — tag it with the first
                # LN it overlaps. Multiple-LN overlaps would otherwise draw
                # the same red bar N times.
                for (ln_head, ln_end) in missed_lns_by_col[c]:
                    if ln_end < span_lo or ln_head > span_hi:
                        continue
                    if span_hi > span_lo:
                        ghost_holds.append((ln_head, c, span_lo, span_hi))
                        break
        if press_start is not None:
            span_lo = press_start
            for (ln_head, ln_end) in missed_lns_by_col[c]:
                if ln_end < span_lo:
                    continue
                if ln_end > span_lo:
                    ghost_holds.append((ln_head, c, span_lo, ln_end))
                    break

    # Flatten sim results into the legacy parallel arrays. Hold release
    # offsets are also carried separately for the renderer.
    # For misses, preserve the actual press offset when the player DID press
    # (too early/late to count, but pressed); miss_pressed flags that, so
    # the renderer knows to draw a red hit-line at the press offset instead
    # of just the X.
    all_rows, all_offs, all_cols, all_nt, all_miss = [], [], [], [], []
    miss_pressed = []
    hold_releases = []
    for r in sim:
        all_rows.append(r['time'])
        all_cols.append(r['col'])
        all_nt.append(0)
        is_miss = (r['head_off'] is None or r['judgement'] == 'miss')
        if is_miss:
            if r['head_off'] is not None:
                all_offs.append(r['head_off'] / 1000.0)
                miss_pressed.append(True)
            else:
                all_offs.append(1.0)
                miss_pressed.append(False)
            all_miss.append(True)
        else:
            all_offs.append(r['head_off'] / 1000.0)
            all_miss.append(False)
            miss_pressed.append(False)
        if r['is_hold']:
            hold_releases.append((r['time'], r['col'], r['end_time'], r['tail_off']))

    order = np.argsort(all_rows, kind='stable')
    noterows = np.array(all_rows, dtype=np.int64)[order]
    offsets = np.array(all_offs, dtype=np.float64)[order]
    columns = np.array(all_cols, dtype=np.int32)[order]
    notetypes = np.array(all_nt, dtype=np.int32)[order]
    misses = np.array(all_miss, dtype=bool)[order]
    miss_pressed_arr = np.array(miss_pressed, dtype=bool)[order]

    return {
        'noterows': noterows,
        'offsets': offsets,
        'columns': columns,
        'notetypes': notetypes,
        'misses': misses,
        'miss_pressed': miss_pressed_arr,  # parallel to misses — True = miss had an actual press
        'holds': holds_meta,
        'hold_releases': hold_releases,
        'ghost_taps': ghost_taps,   # list of (t_ms, column) — osu only
        'ghost_holds': ghost_holds,  # list of (ln_head_t_ms, col, press_t_ms, release_t_ms) — osu only
        'keycount': keycount,
        'filepath': str(osr_path),
        'chart_path': str(osu_path),
        'meta': meta,
        'chart_meta': {k: chart[k] for k in ('title', 'artist', 'creator', 'version', 'keycount')},
        'sv_sections': chart.get('sv_sections', []),
        'od': float(chart.get('od', 8.0)),
        'mods': int(meta.get('mods', 0)),
    }


def find_osu_by_hash(md5_hash, songs_dir):
    """Scan songs_dir for a .osu file whose MD5 matches."""
    import hashlib
    if not md5_hash:
        return None
    root = Path(songs_dir)
    for p in root.rglob('*.osu'):
        try:
            with open(p, 'rb') as f:
                h = hashlib.md5(f.read()).hexdigest()
            if h == md5_hash:
                return str(p)
        except OSError:
            continue
    return None


def _osu_songs_override():
    """User-set Songs dir override from the GUI settings layer, if any.
    Lazy-imported so the core module stays usable without Qt."""
    try:
        from analysis.gui.settings import get_osu_songs_override
    except Exception:
        return None
    try:
        return get_osu_songs_override()
    except Exception:
        return None


def _osu_replays_for(songs_dir):
    """Given an osu! Songs dir (or override), derive the sibling Data/r replay
    dir if present — osu! stores replays at <install>/Data/r. We also keep
    the traditional default Wine/native locations so a user who only set a
    custom Songs path still gets their replays picked up."""
    out = []
    if songs_dir:
        sibling = Path(songs_dir).parent / 'Data' / 'r'
        if sibling.exists():
            out.append(str(sibling))
    home = Path.home()
    for base in [home / 'osu!' / 'Data' / 'r',
                 home / '.local' / 'share' / 'osu-wine' / 'osu!' / 'Data' / 'r']:
        if base.exists() and str(base) not in out:
            out.append(str(base))
    return out


def find_osu_dirs():
    """Returns dict with songs_dir, replays_dirs. User override wins over
    autodetection; replay dirs are derived from the Songs dir's sibling and
    merged with the usual default locations."""
    override = _osu_songs_override()
    if override and Path(override).exists():
        return {'songs_dir': str(override),
                'replays_dirs': _osu_replays_for(override)}
    home = Path.home()
    candidates = [
        home / 'osu!' / 'Songs',
        home / 'Games' / 'osu!' / 'Songs',
        home / '.local' / 'share' / 'osu-wine' / 'osu!' / 'Songs',
        home / '.local' / 'share' / 'osu!' / 'Songs',
        home / 'Documents' / 'osu!' / 'Songs',
        home / '.wine' / 'drive_c' / 'users' / os.environ.get('USER', '') / 'AppData' / 'Local' / 'osu!' / 'Songs',
        Path('/mnt/c/Users') / os.environ.get('USER', '') / 'AppData' / 'Local' / 'osu!' / 'Songs',
    ]
    for c in candidates:
        if c.exists():
            return {'songs_dir': str(c), 'replays_dirs': _osu_replays_for(str(c))}
    return {'songs_dir': None, 'replays_dirs': _osu_replays_for(None)}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: osu_replay.py <replay.osr> [chart.osu]")
        print(find_osu_dirs())
        sys.exit(0)
    osr = sys.argv[1]
    osu = sys.argv[2] if len(sys.argv) > 2 else None
    songs = find_osu_dirs().get('songs_dir')
    rep = parse_replay(osr, osu_path=osu, songs_dir=songs)
    n = len(rep['offsets'])
    m = int(rep['misses'].sum())
    clean = rep['offsets'][~rep['misses']]
    print(f"keycount: {rep['keycount']}")
    print(f"chart: {rep['chart_meta']}")
    print(f"notes: {n}  hits: {n - m}  misses: {m}")
    if len(clean):
        print(f"mean: {clean.mean()*1000:+.2f}ms  std: {clean.std()*1000:.2f}ms")
