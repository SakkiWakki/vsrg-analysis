"""`.osr` replay decoding. Reads the LZMA frame payload directly because
osrparse drops the two leading placeholder frames *with* their deltas ;
on maps that open with a skip the 2nd placeholder carries the skip
duration (~8s), and losing it shifts every press earlier by that amount.
See kszlim/osu-replay-parser#41.
"""
import lzma
import struct

import osrparse


OSU_MOD_RANDOM = 1 << 11
OSU_MOD_DOUBLETIME = 1 << 6
OSU_MOD_HALFTIME = 1 << 8
OSU_MOD_NIGHTCORE = 1 << 9


def rate_for_mods(mods):
    """Return playback rate multiplier from the osu mod bitfield.
    DT/NC = 1.5x, HT = 0.75x, else 1.0x. (NC also sets the DT bit.)"""
    m = int(mods)
    if m & (OSU_MOD_DOUBLETIME | OSU_MOD_NIGHTCORE):
        return 1.5
    if m & OSU_MOD_HALFTIME:
        return 0.75
    return 1.0


def _read_uleb(buf, off):
    result = 0
    shift = 0
    while True:
        b = buf[off]
        off += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, off
        shift += 7


def _read_string(buf, off):
    flag = buf[off]
    off += 1
    if flag == 0x00:
        return '', off
    n, off = _read_uleb(buf, off)
    return buf[off:off + n].decode('utf-8', 'replace'), off + n


def _decode_osr_frames(osr_path):
    """Return (time_deltas, keys_values, rng_seed). Frames preserved in
    order including the two leading placeholders; the RNG-seed sentinel
    (dt=-12345) is split out into `rng_seed`."""
    with open(osr_path, 'rb') as f:
        data = f.read()

    off = 1 + 4                     # mode + version
    _, off = _read_string(data, off)  # beatmap hash
    _, off = _read_string(data, off)  # player
    _, off = _read_string(data, off)  # replay hash
    off += 6 + 6 + 4 + 2 + 1 + 4    # counts + score + combo + perfect + mods
    _, off = _read_string(data, off)  # lifebar
    off += 8                         # timestamp
    rlen = struct.unpack_from('<i', data, off)[0]
    off += 4
    payload = data[off:off + rlen]

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


def _fix_leading_placeholder_ordering(raw):
    """The first two frames in old replays can have out-of-order absolute
    times after accumulation ; stable's player treats them as t=0. Mirror
    that so the first real press doesn't get a bogus negative offset."""
    if len(raw) >= 2 and raw[1][0] < raw[0][0]:
        raw[1][0] = raw[0][0]
        raw[0][0] = 0
    if len(raw) >= 3 and raw[0][0] > raw[2][0]:
        raw[0][0] = raw[1][0] = raw[2][0]


def parse_osr_events(osr_path):
    """Return `(keycount, events, meta)` where events is a list of
    `(abs_time_ms, keys_bitmask)` frames and keycount is inferred from
    the highest key bit seen."""
    r = osrparse.Replay.from_path(osr_path)
    dt_list, keys_list, rng_seed = _decode_osr_frames(osr_path)

    raw = []
    used_bits = 0
    t_acc = 0
    for dt, keys_val in zip(dt_list, keys_list):
        t_acc += dt
        used_bits |= keys_val
        raw.append([t_acc, keys_val])
    _fix_leading_placeholder_ordering(raw)

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
    inferred_keycount = used_bits.bit_length() if used_bits else 4
    return inferred_keycount, events, meta


def extract_key_events(events, keycount):
    """Turn `(t_ms, keys_bitmask)` frames into per-column `(t, is_press)`
    events. Press = 0→1 bit transition, release = 1→0."""
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
