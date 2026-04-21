import hashlib

from analysis.games.etterna.sm_chart import generate_chartkey, _scan_one_chartfile


def _expected_chartkey(payload):
    return 'X' + hashlib.sha1(payload.encode('utf-8')).hexdigest()


def test_generate_chartkey_ignores_hold_tail_only_rows():
    notedata = """
2000
0000
3000
0000
"""

    # Row 0: HoldHead in track 0, BPM 120 -> "2000120".
    # Raw '3' cells only close the head duration in Etterna's loader; they
    # are not stored as TapNoteType_HoldTail in NoteData for chartkey.
    assert generate_chartkey(notedata, [(0.0, 120.0)], 'dance-single') == (
        _expected_chartkey('2000120')
    )


def test_generate_chartkey_hold_tail_cell_is_empty_on_nonempty_row():
    notedata = """
2000
0000
3100
0000
"""

    assert generate_chartkey(notedata, [(0.0, 120.0)], 'dance-single') == (
        _expected_chartkey('20001200100120')
    )


def test_generate_chartkey_preserves_duplicate_bpm_order():
    notedata = """
1000
0000
0000
0000
"""

    assert generate_chartkey(
        notedata,
        [(0.0, 150.0), (0.0, 120.0)],
        'dance-single',
    ) == _expected_chartkey('1000120')


def test_generate_chartkey_bpm_lookup_uses_segment_rows():
    notedata = """
0000
0100
0000
0000
"""

    assert generate_chartkey(
        notedata,
        [(0.0, 120.0), (1.00000001, 240.0)],
        'dance-single',
    ) == _expected_chartkey('0100240')


def test_generate_chartkey_empty_chart_returns_none():
    assert generate_chartkey('0000\n0000', [(0.0, 120.0)], 'dance-single') is None


def test_scan_ssc_indexes_stored_and_generated_keys_when_stored_is_stale(tmp_path):
    stored_key = 'X26b1ec2ccf13e307259795d2ff58a21199be54c7'
    generated_key = _expected_chartkey('1000120')
    chart = tmp_path / 'stale_chartkey.ssc'
    chart.write_text(
        f"""#TITLE:stale key;
#BPMS:0.000=120.000;
#NOTEDATA:;
#STEPSTYPE:dance-single;
#DESCRIPTION:;
#DIFFICULTY:Challenge;
#METER:1;
#CHARTKEY:{stored_key};
#NOTES:
1000
0000
0000
0000
;
""",
        encoding='utf-8',
    )

    _path, entries = _scan_one_chartfile(chart)

    assert (stored_key, 0) in entries
    assert (generated_key, 0) in entries
