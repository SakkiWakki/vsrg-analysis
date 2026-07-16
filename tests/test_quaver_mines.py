"""Quaver 1.7: mines (point + hold), scratch key, mine-hit detection."""
import os
import tempfile
import textwrap

import numpy as np

from analysis.games.quaver.qua_chart import parse_qua_file
from analysis.games.quaver.judge_sim import simulate_mines
from analysis.games.quaver.parse import _group_notes_by_col, _build_mine_arrays


MARV_W = 18.0


def _parse_src(src):
    with tempfile.NamedTemporaryFile('w', suffix='.qua', delete=False) as f:
        f.write(textwrap.dedent(src))
        path = f.name
    try:
        return parse_qua_file(path)
    finally:
        os.unlink(path)


def test_qua_parser_flags_mines_and_scratch():
    chart = _parse_src("""\
        Mode: Keys4
        HasScratchKey: true
        TimingPoints:
        - StartTime: 0
          Bpm: 120
        HitObjects:
        - StartTime: 1000
          Lane: 1
        - StartTime: 2000
          Lane: 2
          Type: Mine
        - StartTime: 3000
          Lane: 3
          EndTime: 3500
          Type: Mine
        - StartTime: 4000
          Lane: 5
    """)
    assert chart['keycount'] == 5   # Keys4 + scratch

    by_time = {h['time']: h for h in chart['hitobjects']}
    assert not by_time[1000]['is_mine']
    assert by_time[2000]['is_mine']
    assert by_time[3000]['is_mine'] and by_time[3000]['end_time'] == 3500
    # Scratch-lane note survives the lane bounds check.
    assert by_time[4000]['column'] == 4


def test_mines_split_out_of_note_stream():
    chart = _parse_src("""\
        Mode: Keys4
        TimingPoints:
        - StartTime: 0
          Bpm: 120
        HitObjects:
        - StartTime: 1000
          Lane: 1
        - StartTime: 2000
          Lane: 1
          Type: Mine
    """)
    by_col, holds, mines_by_col = _group_notes_by_col(
        chart['hitobjects'], chart['keycount'])
    assert [n['time'] for n in by_col[0]] == [1000]
    assert [m['time'] for m in mines_by_col[0]] == [2000]
    assert holds == []


def _mines(col_lists):
    return [[{'time': t, 'end_time': e} for t, e in col] for col in col_lists]


def test_press_inside_window_detonates():
    mines = _mines([[(1000, None)]])
    hits = simulate_mines(mines, [[(1010, True)]], MARV_W)
    assert len(hits) == 1
    assert hits[0] == {'col': 0, 'mine_time': 1000, 'end_time': 1000,
                       'press_time': 1010}


def test_press_outside_window_and_release_do_nothing():
    mines = _mines([[(1000, None)]])
    assert simulate_mines(mines, [[(1030, True)]], MARV_W) == []
    assert simulate_mines(mines, [[(960, True)]], MARV_W) == []
    assert simulate_mines(mines, [[(1000, False)]], MARV_W) == []


def test_hold_mine_armed_over_full_span():
    mines = _mines([[(1000, 2000)]])
    hits = simulate_mines(mines, [[(1900, True)]], MARV_W)
    assert len(hits) == 1 and hits[0]['end_time'] == 2000
    assert simulate_mines(mines, [[(2030, True)]], MARV_W) == []


def test_one_press_detonates_all_overlapping_mines_once():
    mines = _mines([[(1000, None), (1005, None)]])
    events = [[(1002, True), (1004, True)]]
    hits = simulate_mines(mines, events, MARV_W)
    assert [h['mine_time'] for h in hits] == [1000, 1005]


def test_mine_arrays_shape_and_hit_indexing():
    mines_by_col = _mines([[(2000, None)], [(1000, 1500)]])
    hits = simulate_mines(mines_by_col, [[(2005, True)], []], MARV_W)
    arrays = _build_mine_arrays(mines_by_col, hits)

    assert np.allclose(arrays['mine_times'], [1.0, 2.0])
    assert list(arrays['mine_cols']) == [1, 0]
    assert np.isclose(arrays['mine_end_times'][0], 1.5)
    assert np.isnan(arrays['mine_end_times'][1])
    assert np.all(np.isinf(arrays['mine_until']))
    # The single detonation points at the 2000ms mine (sorted index 1).
    assert list(arrays['mine_hit_idx']) == [1]
    assert np.allclose(arrays['mine_hit_press'], [2.005])


def test_no_mines_yields_no_arrays():
    assert _build_mine_arrays([[], []], []) == {}


def _cs_string(s: str) -> bytes:
    """C# BinaryWriter string: 7-bit-encoded length prefix + UTF-8."""
    raw = s.encode()
    n = len(raw)
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            break
    return bytes(out) + raw


def _synth_qr(version: str, *, mine_hit_count: int | None) -> bytes:
    import lzma
    import struct
    frames = lzma.compress(b'0|0,100|1,120|0', format=lzma.FORMAT_ALONE)
    parts = [
        _cs_string(version),
        _cs_string('m' * 32),            # map md5
        _cs_string('h' * 32),            # frame md5
        _cs_string('tester'),
        _cs_string('01/01/2026 00:00:00'),
        struct.pack('<q', 0),            # TimePlayed
        struct.pack('<i', 1),            # Mode (Keys4)
        struct.pack('<q', 0),            # Mods (int64 for >= 0.0.2)
        struct.pack('<i', 0),            # Score
        struct.pack('<f', 100.0),        # Accuracy
        struct.pack('<i', 0),            # MaxCombo
    ]
    parts += [struct.pack('<i', 0)] * 6  # marv..miss
    if mine_hit_count is not None:
        parts.append(struct.pack('<i', mine_hit_count))
    parts += [
        struct.pack('<i', 0),            # PauseCount
        struct.pack('<i', -1),           # rng seed
        frames,
    ]
    return b''.join(parts)


def test_qr_v003_reads_mine_hit_count():
    """Replay v0.0.3 (Quaver 1.7) inserts CountMineHit before
    PauseCount; without reading it the LZMA blob misaligns."""
    from analysis.games.quaver.qr_replay import parse_qr_events

    for version, mine_hits in (('0.0.3', 3), ('0.0.2', None)):
        with tempfile.NamedTemporaryFile('wb', suffix='.qr',
                                         delete=False) as f:
            f.write(_synth_qr(version, mine_hit_count=mine_hits))
            path = f.name
        try:
            _kc, events, meta = parse_qr_events(path)
        finally:
            os.unlink(path)
        assert meta['replay_version'] == version
        assert meta['count_mine_hit'] == (mine_hits or 0)
        assert [t for t, _k in events] == [0, 100, 120]


def test_copy_chart_streams_preserves_dtypes():
    from analysis.player.init.notes_model import NotesModel, copy_chart_streams
    m = NotesModel()
    copy_chart_streams(m, {
        'mine_times': [1.0, 2.0],
        'mine_cols': [0, 1],
        'mine_hit_idx': [1],
        'mine_hit_press': [2.005],
    })
    assert m.mine_times.dtype == np.float64
    assert m.mine_cols.dtype == np.int32
    assert m.mine_hit_idx.dtype == np.int64
    assert list(m.mine_hit_idx) == [1]
    assert m.lift_times.size == 0
