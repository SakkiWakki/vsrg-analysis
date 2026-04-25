"""Regression tests for SCROLLS/SPEEDS parsing and post-death note injection."""
import numpy as np
import pytest

from analysis.games.etterna.sm_chart import (
    parse_ssc, parse_sm, sv_sections_from_chart,
    _chart_fingerprint, _parse_scrolls, _parse_speeds, beat_to_time,
)


# ---------------------------------------------------------------------------
# STOPS / DELAYS / WARPS in beat_to_time
# ---------------------------------------------------------------------------

def test_beat_to_time_no_events_matches_bpm_only():
    # 120 BPM: 1 beat = 0.5s
    assert beat_to_time(2.0, [(0.0, 120.0)], 0.0) == pytest.approx(1.0)


def test_stop_adds_its_duration_after_beat():
    # 120 BPM, 1 beat normally = 0.5s. Stop of 1s at beat 0.5 means time
    # advances by 1 extra second AFTER beat 0.5 is reached.
    t = beat_to_time(1.0, [(0.0, 120.0)], 0.0, stops=[(0.5, 1.0)])
    assert t == pytest.approx(1.5)


def test_stop_row_itself_is_pre_stop_time():
    # Etterna's target marker wins before STOP at the same row. The pause is
    # after notes/events on that row, not before them.
    t = beat_to_time(0.5, [(0.0, 120.0)], 0.0, stops=[(0.5, 1.0)])
    assert t == pytest.approx(0.25)


def test_stop_only_applies_if_passed():
    # Target beat before the stop ; stop doesn't contribute
    t = beat_to_time(0.4, [(0.0, 120.0)], 0.0, stops=[(0.5, 1.0)])
    assert t == pytest.approx(0.2)


def test_warp_teleports_forward_with_no_time():
    # At beat 0.5, warp forward by 0.5 beats. Target beat 1.0 is reached
    # at time 0.25s (first 0.5 beats) with no extra time for the warp.
    t = beat_to_time(1.0, [(0.0, 120.0)], 0.0, warps=[(0.5, 0.5)])
    assert t == pytest.approx(0.25)


def test_warp_target_inside_warp_collapses_to_warp_entry():
    # Target beat is inside the warp range ; time should be the warp entry
    t = beat_to_time(0.7, [(0.0, 120.0)], 0.0, warps=[(0.5, 0.5)])
    assert t == pytest.approx(0.25)


def test_stop_then_warp_same_row_matches_etterna_precedence():
    # FindEvent checks STOP before WARP. During the stop, beat stays at the
    # warp row; after the stop completes, the warp jumps beat-space forward.
    bpms = [(0.0, 120.0)]
    stops = [(1.0, 1.0)]
    warps = [(1.0, 1.0)]
    assert beat_to_time(1.0, bpms, 0.0, stops=stops, warps=warps) == pytest.approx(0.5)
    assert beat_to_time(1.5, bpms, 0.0, stops=stops, warps=warps) == pytest.approx(1.5)
    assert beat_to_time(2.0, bpms, 0.0, stops=stops, warps=warps) == pytest.approx(1.5)
    assert beat_to_time(3.0, bpms, 0.0, stops=stops, warps=warps) == pytest.approx(2.0)


def test_delay_adds_duration_before_beat():
    # Delay behaves like a pause before the beat's events
    t = beat_to_time(1.0, [(0.0, 120.0)], 0.0, delays=[(0.5, 1.0)])
    assert t == pytest.approx(1.5)


def test_ssc_parser_captures_stops_delays_warps(tmp_path):
    ssc = tmp_path / 'test.ssc'
    ssc.write_text(
        """#TITLE:Test;
#BPMS:0.000=120.000;
#OFFSET:0.0;
#STOPS:1.000=0.500;
#DELAYS:2.000=0.250;
#WARPS:3.000=1.000;
#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:Challenge;
#METER:10;
#NOTES:
1000
0000
0000
0000
;
""",
        encoding='utf-8',
    )
    data = parse_ssc(ssc)
    chart = data['charts'][0]
    assert chart['stops'] == [(1.0, 0.5)]
    assert chart['delays'] == [(2.0, 0.25)]
    assert chart['warps'] == [(3.0, 1.0)]


def test_ssc_old_version_warps_are_absolute_destinations(tmp_path):
    ssc = tmp_path / 'old_warp.ssc'
    ssc.write_text(
        """#VERSION:0.600;
#TITLE:Old Warp;
#BPMS:0.000=120.000;
#WARPS:3.000=5.000;
#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:Challenge;
#METER:10;
#NOTES:
1000
;
""",
        encoding='utf-8',
    )
    chart = parse_ssc(ssc)['charts'][0]
    assert chart['warps'] == [(3.0, 2.0)]


def test_timing_segments_are_quantized_to_etterna_rows(tmp_path):
    ssc = tmp_path / 'quantized_warp.ssc'
    ssc.write_text(
        """#VERSION:0.700;
#TITLE:Quantized Warp;
#BPMS:0.000=120.000;
#WARPS:1.011=0.030;
#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:Challenge;
#METER:10;
#NOTES:
1000
;
""",
        encoding='utf-8',
    )
    chart = parse_ssc(ssc)['charts'][0]
    assert chart['warps'] == [(49 / 48, 1 / 48)]


def test_chart_fingerprint_keeps_notes_inside_warps_for_analysis():
    chart = {
        'warps': [(0.0, 1.0)],
        'stops': [],
        'delays': [],
        'notedata': "\n".join([
            "1000",  # beat 0, inside warp but still useful for analysis render
            ",",
            "0100",  # beat 4
            ",",
            "0010",  # beat 8
            ",",
            "0001",  # beat 12
        ]),
    }
    assert _chart_fingerprint(chart, n=3) == ((0, 0), (192, 1), (384, 2))


def test_adapter_renders_warped_chart_taps_as_fakes():
    from analysis.games.etterna.adapter import EtternaAdapter

    replay = {
        'noterows': np.array([96], dtype=np.int64),
        'columns': np.array([1], dtype=np.int32),
        'offsets': np.array([0.0], dtype=np.float64),
        'notetypes': np.array([0], dtype=np.int32),
        'misses': np.array([False], dtype=bool),
        'holds': [],
    }
    found = {
        'file': 'synthetic.sm',
        'data': {'bpms': [(0.0, 120.0)], 'offset': 0.0},
        'chart': {
            'stepstype': 'dance-single',
            'bpms': [(0.0, 120.0)],
            'offset': 0.0,
            'warps': [(0.0, 1.0)],
            'stops': [],
            'delays': [],
            'notedata': "\n".join([
                "1000",  # warped tap, absent from replay stream
                ",",
                "0100",
            ]),
        },
    }
    EtternaAdapter._attach_chart_extras(replay, found)
    assert replay['chart_fakes'] == [(0, 0)]

    EtternaAdapter().prepare_replay_times(replay, [(0.0, 120.0)], 0.0)
    assert replay['fake_times'].tolist() == [0.0]
    assert replay['fake_cols'].tolist() == [0]


def test_etterna_player_kwargs_use_replay_xmod_modifier():
    from analysis.games.etterna.adapter import EtternaAdapter

    out = EtternaAdapter().player_tab_kwargs(
        {},
        {
            'modifiers': '1.23x, 20% Mini, Reverse, Overhead',
            'judgments': {'W1': 1},
            'keycount': 4,
        },
        ([(0.0, 120.0)], 0.0, None),
    )
    assert out['scroll_mode'] == 'xmod'
    assert out['xmod_value'] == pytest.approx(1.23)


def test_etterna_player_kwargs_use_replay_cmod_modifier():
    from analysis.games.etterna.adapter import EtternaAdapter

    out = EtternaAdapter().player_tab_kwargs(
        {},
        {'modifiers': 'C650, Reverse', 'judgments': {}, 'keycount': 4},
        ([(0.0, 120.0)], 0.0, None),
    )
    assert out['scroll_mode'] == 'cmod'
    assert out['cmod_bpm'] == pytest.approx(650.0)


def test_etterna_replay_mine_hits_are_not_main_notes(tmp_path):
    from analysis.games.etterna.replay import parse_replay

    replay_file = tmp_path / 'minehit'
    replay_file.write_text(
        "100 0.012 0\n"
        "144 -0.000 2 4\n"
        "192 -0.010 1 2\n",
        encoding='utf-8',
    )
    replay = parse_replay(replay_file)

    assert replay['noterows'].tolist() == [100, 192]
    assert replay['columns'].tolist() == [0, 1]
    assert replay['notetypes'].tolist() == [0, 2]
    assert replay['holds'] == [(192, 1)]
    assert replay['mine_hits'] == [(144, 2, -0.0)]


def test_sm_parser_captures_stops_delays_warps(tmp_path):
    sm = tmp_path / 'test.sm'
    sm.write_text(
        """#TITLE:Test;
#BPMS:0.000=120.000;
#OFFSET:0.0;
#STOPS:1.000=0.500;
#DELAYS:2.000=0.250;
#WARPS:3.000=1.000;
#NOTES:dance-single:desc:Challenge:10:0:
1000
0000
0000
0000
;
""",
        encoding='utf-8',
    )
    data = parse_sm(sm)
    chart = data['charts'][0]
    assert chart['stops'] == [(1.0, 0.5)]
    assert chart['delays'] == [(2.0, 0.25)]
    assert chart['warps'] == [(3.0, 1.0)]


# ---------------------------------------------------------------------------
# SCROLLS / SPEEDS parsing
# ---------------------------------------------------------------------------

def test_parse_scrolls_basic():
    result = _parse_scrolls('0.000=0.000,26.000=0.015,27.000=0.030')
    assert result == [(0.0, 0.0), (26.0, 0.015), (27.0, 0.030)]


def test_parse_speeds_preserves_duration_and_type():
    result = _parse_speeds('0.000=1.000=1.000=0,484.000=0.000=0.000=1')
    assert result == [(0.0, 1.0, 1.0, 0), (484.0, 0.0, 0.0, 1)]


def test_parse_speeds_two_fields_ok():
    # Some editors omit duration/type entirely
    result = _parse_speeds('0=1.0,10=2.0')
    assert result == [(0.0, 1.0), (10.0, 2.0)]


def test_parse_speeds_preserves_transition_fields():
    result = _parse_speeds('0=2.0=4=0,8=0.5=1.5=1')
    assert result == [(0.0, 2.0, 4.0, 0), (8.0, 0.5, 1.5, 1)]


def test_sv_sections_from_chart_empty_gives_empty():
    chart = {}
    assert sv_sections_from_chart(chart, [(0.0, 120.0)], 0.0) == []


def test_sv_sections_scrolls_only():
    bpms = [(0.0, 120.0)]  # 120 BPM: 1 beat = 0.5s
    chart = {'scrolls': [(0.0, 0.5), (4.0, 1.0)], 'speeds': []}
    sv = sv_sections_from_chart(chart, bpms, 0.0)
    assert len(sv) == 2
    t0, m0 = sv[0]
    t1, m1 = sv[1]
    assert m0 == pytest.approx(0.5)
    assert m1 == pytest.approx(1.0)
    assert t0 < t1


def test_sv_sections_speeds_only():
    bpms = [(0.0, 120.0)]
    chart = {'scrolls': [], 'speeds': [(0.0, 2.0), (8.0, 1.0)]}
    sv = sv_sections_from_chart(chart, bpms, 0.0)
    assert len(sv) == 2
    assert sv[0][1] == pytest.approx(2.0)
    assert sv[1][1] == pytest.approx(1.0)


def test_sv_sections_scrolls_times_speeds():
    bpms = [(0.0, 120.0)]
    # scroll=0.5 at beat 0, speed=2.0 at beat 0 -> combined = 1.0
    chart = {'scrolls': [(0.0, 0.5)], 'speeds': [(0.0, 2.0)]}
    sv = sv_sections_from_chart(chart, bpms, 0.0)
    assert len(sv) == 1
    assert sv[0][1] == pytest.approx(1.0)


def test_sv_sections_sorted_by_time():
    bpms = [(0.0, 120.0)]
    # Speeds change at beat 10, scrolls change at beat 5 -- result must be sorted
    chart = {'scrolls': [(5.0, 0.8)], 'speeds': [(10.0, 1.5)]}
    sv = sv_sections_from_chart(chart, bpms, 0.0)
    times = [t for t, _ in sv]
    assert times == sorted(times)


def test_parse_ssc_stores_scrolls_and_speeds(tmp_path):
    ssc = tmp_path / 'test.ssc'
    ssc.write_text(
        """#TITLE:Test;
#BPMS:0.000=120.000;
#OFFSET:0.0;
#NOTEDATA:;
#STEPSTYPE:dance-single;
#DESCRIPTION:;
#DIFFICULTY:Challenge;
#METER:10;
#SPEEDS:0.000=1.000=0.000=0,
8.000=2.000=0.000=0
;
#SCROLLS:0.000=1.000,
4.000=0.500
;
#NOTES:
1000
0000
0000
0000
;
""",
        encoding='utf-8',
    )
    data = parse_ssc(ssc)
    chart = data['charts'][0]
    assert chart['speeds'] == [(0.0, 1.0, 0.0, 0), (8.0, 2.0, 0.0, 0)]
    assert chart['scrolls'] == [(0.0, 1.0), (4.0, 0.5)]


def test_parse_sm_stores_scrolls_and_speeds(tmp_path):
    sm = tmp_path / 'test.sm'
    sm.write_text(
        """#TITLE:Test;
#BPMS:0.000=120.000;
#OFFSET:0.0;
#SPEEDS:0.000=1.500=0.000=0;
#SCROLLS:0.000=0.750;
#NOTES:dance-single:desc:Challenge:10:0:
1000
0000
0000
0000
;
""",
        encoding='utf-8',
    )
    data = parse_sm(sm)
    chart = data['charts'][0]
    assert chart['speeds'] == [(0.0, 1.5, 0.0, 0)]
    assert chart['scrolls'] == [(0.0, 0.75)]


# ---------------------------------------------------------------------------
# Post-death note injection
# ---------------------------------------------------------------------------

def _make_replay(noterows, cols, offsets=None, notetypes=None, holds=None):
    """Minimal replay dict matching what parse_replay returns."""
    n = len(noterows)
    if offsets is None:
        offsets = np.zeros(n, dtype=np.float64)
    if notetypes is None:
        notetypes = np.zeros(n, dtype=np.int32)
    return {
        'noterows':  np.array(noterows,  dtype=np.int64),
        'columns':   np.array(cols,      dtype=np.int32),
        'offsets':   np.array(offsets,   dtype=np.float64),
        'notetypes': np.array(notetypes, dtype=np.int32),
        'misses':    np.isclose(offsets if offsets is not None
                                else np.zeros(n), 1.0),
        'holds':     holds or [],
    }


def _make_found(notedata, bpms=None, offset=0.0):
    """Minimal `found` dict as returned by _find_chart."""
    return {
        'data': {'bpms': bpms or [(0.0, 120.0)], 'offset': offset},
        'chart': {
            'notedata': notedata,
            'stepstype': 'dance-single',
            'bpms': bpms or [(0.0, 120.0)],
            'offset': offset,
            'scrolls': [],
            'speeds': [],
        },
    }


def _attach(replay, found):
    from analysis.games.etterna.adapter import EtternaAdapter
    EtternaAdapter._attach_chart_extras(replay, found)


def test_no_injection_on_full_clear():
    # Chart: two taps at rows 0 and 192. Replay has both -> nothing injected.
    notedata = "1000\n0000\n0000\n0000\n,\n0001\n0000\n0000\n0000\n"
    # parse_notes_block gives rows 0 (col 0) and 192 (col 3)
    replay = _make_replay(noterows=[0, 192], cols=[0, 3])
    original_len = len(replay['noterows'])
    _attach(replay, _make_found(notedata))
    assert len(replay['noterows']) == original_len
    assert replay.get('death_time') is None


def test_death_injects_missing_notes():
    # Chart: taps at rows 0 (col 0) and 384 (col 3). Replay only has row 0.
    # Gap = 384 > 192 threshold -> death, row 384 injected as miss.
    notedata = ("1000\n0000\n0000\n0000\n"   # measure 0: row 0
                ",\n0000\n0000\n0000\n0000\n"  # measure 1: empty
                ",\n0001\n0000\n0000\n0000\n"  # measure 2: row 384 col 3
                )
    replay = _make_replay(noterows=[0], cols=[0])
    _attach(replay, _make_found(notedata))

    assert len(replay['noterows']) == 2
    assert replay['misses'][1]
    assert replay['offsets'][1] == 1.0  # MISS_SENTINEL
    assert replay.get('death_time') is not None


def test_injected_noterows_stay_sorted():
    # Chart: taps at rows 0, 192, 384, 576. Replay only has row 0.
    notedata = (
        "1000\n0000\n0000\n0000\n"
        ",\n1000\n0000\n0000\n0000\n"
        ",\n1000\n0000\n0000\n0000\n"
        ",\n1000\n0000\n0000\n0000\n"
    )
    replay = _make_replay(noterows=[0], cols=[0])
    _attach(replay, _make_found(notedata))
    rows = replay['noterows']
    assert np.all(rows[:-1] <= rows[1:]), "noterows must be sorted after injection"


def test_death_hold_injected_into_holds():
    # Chart: tap at row 0 (player survives this), then hold head at row 192
    # (col 0, tail at row 336), then tap at row 576. Replay only has row 0.
    # Gap = 576 - 0 = 576 > 192 -> death. Hold at row 192 is post-death.
    notedata = (
        "1000\n0000\n0000\n0000\n"   # tap at row 0
        ",\n2000\n0000\n3000\n0000\n"  # hold head row 192, tail row 288
        ",\n0000\n0000\n0000\n0000\n"  # empty
        ",\n1000\n0000\n0000\n0000\n"  # tap at row 576
    )
    replay = _make_replay(noterows=[0], cols=[0])
    _attach(replay, _make_found(notedata))

    injected_holds = [h for h in replay['holds'] if len(h) == 3]
    assert len(injected_holds) == 1
    head_row, col, tail_row = injected_holds[0]
    assert head_row == 192
    assert col == 0
    assert tail_row > head_row


def test_death_time_not_set_when_gap_is_small():
    # Chart ends 96 rows after last replay row (< 192 threshold) -> no death_time.
    notedata = "1000\n0000\n0000\n0000\n,\n0001\n0000\n0000\n0000\n"
    # chart_max_row=192, replay_max_row=96: gap=96 <= 192
    replay = _make_replay(noterows=[0, 96], cols=[0, 3])
    _attach(replay, _make_found(notedata))
    assert replay.get('death_time') is None


def test_death_time_set_when_gap_exceeds_threshold():
    # Chart: tap at row 0 and tap at row 576 (3 empty measures then one tap).
    # Replay: only row 0. Gap = 576 > 192.
    notedata = (
        "1000\n0000\n0000\n0000\n"
        ",\n0000\n0000\n0000\n0000\n"
        ",\n0000\n0000\n0000\n0000\n"
        ",\n0001\n0000\n0000\n0000\n"
    )
    replay = _make_replay(noterows=[0], cols=[0])
    _attach(replay, _make_found(notedata))
    assert replay.get('death_time') is not None


# ---------------------------------------------------------------------------
# chart-extras cull-space indexing
# ---------------------------------------------------------------------------
# `_sweep_chart_notes` was replaced by `_cull_indices` + `_draw_chart_bucket`
# in the chart_extras layer split. Testing the cull directly makes the
# semantics easier to verify without constructing a full draw context.

def test_cull_indices_sv_space():
    """In SV-space the cull bisects the sv-time array (not the real-time
    array), so notes are picked by their projected visual position."""
    from analysis.player.render.layers.chart_extras import _cull_indices

    sv_times = np.array([0.1, 0.3, 0.5], dtype=np.float64)
    indices = _cull_indices(sv_times, 0.2, 0.4)
    assert list(indices) == [1]  # only sv=0.3 is in [0.2, 0.4]


def test_cull_indices_real_time():
    from analysis.player.render.layers.chart_extras import _cull_indices

    times = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    indices = _cull_indices(times, 1.5, 2.5)
    assert list(indices) == [1]  # only t=2.0 is in [1.5, 2.5]


def test_chart_sprite_y_uses_row_space_sv_coordinate():
    from types import SimpleNamespace

    from analysis.player.render.layers.chart_extras import _chart_sprite_y

    ctx = SimpleNamespace(
        use_sv_space=True,
        judge_y=400.0,
        scroll_speed=10.0,
        frame=SimpleNamespace(visual_cum_now=100.0, render_multiplier=2.0),
        time_to_y=lambda _t: -999.0,
    )
    y = _chart_sprite_y(ctx, 31.94, np.array([112.0], dtype=np.float64), 0)
    assert y == pytest.approx(160.0)


def test_chart_sprite_y_falls_back_to_time_without_sv():
    from types import SimpleNamespace

    from analysis.player.render.layers.chart_extras import _chart_sprite_y

    ctx = SimpleNamespace(use_sv_space=False, time_to_y=lambda t: t + 10.0)
    y = _chart_sprite_y(ctx, 3.0, np.array([112.0], dtype=np.float64), 0)
    assert y == pytest.approx(13.0)
