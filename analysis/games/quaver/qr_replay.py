"""`.qr` replay decoding.

Quaver replays are a C# `BinaryWriter` stream:
    string  ReplayVersion   (e.g. "0.0.2")
    string  MapMd5
    string  Md5             (replay hash)
    string  PlayerName
    string  Date            (CultureInfo.InvariantCulture format)
    int64   TimePlayed
    int32   Mode            (Quaver.API.Enums.GameMode)
    int32 | int64  Mods     (32-bit on "0.0.1"/"None", else 64-bit)
    int32   Score
    float   Accuracy
    int32   MaxCombo
    int32   CountMarv .. CountMiss     (6 ints)
    int32   PauseCount
    int32   RandomizeModifierSeed      (only on >= 0.0.1)
    bytes   LZMA-compressed `time|keys,time|keys,...` ASCII frames (to EOF)

C# strings use a 7-bit encoded length prefix (same wire format as the
osu replay strings ; `_read_uleb` lifted from `analysis/games/osu/replay/osr.py`).
"""
from __future__ import annotations

import lzma
import struct


# Quaver speed-mod bits (Quaver.API/Enums/ModIdentifier.cs). Each speed
# step is its own bit ; the SpeedMods union covers ~30 of them. We map
# each to its playback rate so `rate_for_mods` doesn't depend on string
# parsing.
_QUAVER_RATE_BITS = (
    (1 << 1,  0.5),  (1 << 24, 0.55), (1 << 2,  0.6),  (1 << 25, 0.65),
    (1 << 3,  0.7),  (1 << 26, 0.75), (1 << 4,  0.8),  (1 << 27, 0.85),
    (1 << 5,  0.9),  (1 << 28, 0.95),
    (1 << 33, 1.05), (1 << 6,  1.1),  (1 << 34, 1.15), (1 << 7,  1.2),
    (1 << 35, 1.25), (1 << 8,  1.3),  (1 << 36, 1.35), (1 << 9,  1.4),
    (1 << 37, 1.45), (1 << 10, 1.5),  (1 << 38, 1.55), (1 << 11, 1.6),
    (1 << 39, 1.65), (1 << 12, 1.7),  (1 << 40, 1.75), (1 << 13, 1.8),
    (1 << 41, 1.85), (1 << 14, 1.9),  (1 << 42, 1.95), (1 << 15, 2.0),
)

QUAVER_MOD_NO_SLIDER_VELOCITY = 1 << 0
QUAVER_MOD_MIRROR = 1 << 31
QUAVER_MOD_RANDOMIZE = 1 << 23


def rate_for_mods(mods):
    """Return playback rate multiplier from the Quaver mod bitfield.
    Quaver encodes each speed step as its own bit ; no flag = 1.0x."""
    m = int(mods)
    for bit, rate in _QUAVER_RATE_BITS:
        if m & bit:
            return rate
    return 1.0


# ----------------------------------------------------------------------
# Binary reader helpers
# ----------------------------------------------------------------------


def _read_uleb(buf, off):
    """C# 7-bit-encoded length prefix used by `BinaryWriter.Write(string)`.
    Identical to the osu replay format's ULEB128."""
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
    """C# `BinaryReader.ReadString` ; UTF-8 with a ULEB128 length prefix."""
    n, off = _read_uleb(buf, off)
    return buf[off:off + n].decode('utf-8', 'replace'), off + n


def _read_int32(buf, off):
    return struct.unpack_from('<i', buf, off)[0], off + 4


def _read_int64(buf, off):
    return struct.unpack_from('<q', buf, off)[0], off + 8


def _read_float(buf, off):
    return struct.unpack_from('<f', buf, off)[0], off + 4


# ----------------------------------------------------------------------
# Replay frames
# ----------------------------------------------------------------------


def _decode_frames(payload):
    """LZMA-decompress + parse `time|keys,...`. Returns
    `(times_ms, key_bitfields)`.

    Quaver uses standard `.lzma` framing (5-byte properties + 8-byte
    uncompressed size + data) ; Python's stdlib reads it via
    `FORMAT_ALONE`."""
    decoded = lzma.decompress(payload, format=lzma.FORMAT_ALONE)
    text = decoded.decode('ascii', 'replace').rstrip(',')
    times, keys = [], []
    for entry in text.split(','):
        if not entry:
            continue
        parts = entry.split('|')
        if len(parts) != 2:
            continue
        try:
            times.append(int(parts[0]))
            keys.append(int(parts[1]))
        except ValueError:
            continue
    return times, keys


def _key_bits_to_lanes(bits):
    """Quaver's `KeyPressStateToLanes`: a press state is a bitfield over
    K1..K12. Returns the list of lane indices (0-based, in low-bit-first
    order) that were down."""
    out = []
    while bits:
        lsb = bits & -bits
        out.append(lsb.bit_length() - 1)
        bits ^= lsb
    return out


def extract_key_events(events, keycount):
    """`(t_ms, keys_bitfield)` -> per-column `(t, is_press)` lists.
    Mirrors `analysis/games/osu/replay/osr.extract_key_events` so the
    judgement sim can be column-symmetric across games."""
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


# ----------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------


def parse_qr_events(qr_path):
    """Parse a `.qr` replay file. Returns `(keycount, events, meta)`.

    `events` is `[(abs_time_ms, keys_bitfield), ...]` ; keycount is
    inferred from the highest key bit observed across all frames so the
    column splitter doesn't trim active lanes."""
    with open(qr_path, 'rb') as f:
        data = f.read()

    off = 0
    replay_version, off = _read_string(data, off)
    map_md5, off = _read_string(data, off)
    md5, off = _read_string(data, off)
    player_name, off = _read_string(data, off)
    date_str, off = _read_string(data, off)
    time_played, off = _read_int64(data, off)
    mode, off = _read_int32(data, off)

    # Mods width depends on version (older replays write int32, newer int64).
    if replay_version in ('None', '0.0.1'):
        mods32, off = _read_int32(data, off)
        mods = _decode_legacy_mods(mods32)
    else:
        mods, off = _read_int64(data, off)

    score, off = _read_int32(data, off)
    accuracy, off = _read_float(data, off)
    max_combo, off = _read_int32(data, off)
    count_marv, off = _read_int32(data, off)
    count_perf, off = _read_int32(data, off)
    count_great, off = _read_int32(data, off)
    count_good, off = _read_int32(data, off)
    count_okay, off = _read_int32(data, off)
    count_miss, off = _read_int32(data, off)
    pause_count, off = _read_int32(data, off)

    rng_seed = -1
    if replay_version != 'None':
        rng_seed, off = _read_int32(data, off)

    times_ms, keys_list = _decode_frames(data[off:])

    used_bits = 0
    for k in keys_list:
        used_bits |= k

    events = list(zip(times_ms, keys_list))
    keycount = used_bits.bit_length() if used_bits else 4

    meta = {
        'replay_version': replay_version,
        'map_md5': map_md5,
        'md5': md5,
        'player_name': player_name,
        'date': date_str,
        'time_played': time_played,
        'mode': mode,
        'mods': int(mods),
        'score': score,
        'accuracy': accuracy,
        'max_combo': max_combo,
        'count_marv': count_marv,
        'count_perf': count_perf,
        'count_great': count_great,
        'count_good': count_good,
        'count_okay': count_okay,
        'count_miss': count_miss,
        'pause_count': pause_count,
        'rng_seed': rng_seed,
        'used_bits': used_bits,
        'rate': rate_for_mods(mods),
    }
    return keycount, events, meta


def _decode_legacy_mods(mods32):
    """Older Quaver replays write `Mods` as int32. -1 is the sentinel
    for `ModIdentifier.None`; otherwise a negative int32 has the 32nd
    bit set, which corresponds to the Mirror flag (`1L << 31`).
    Match Quaver/Replay.cs's read path."""
    if mods32 == -1:
        return 0
    if mods32 < 0:
        return (mods32 & 0x7FFFFFFF) | QUAVER_MOD_MIRROR
    return mods32
