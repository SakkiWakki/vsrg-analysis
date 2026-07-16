"""fluXis playback: windows math, replay decoding, judgement sim,
lane masks, landmines, ticks."""
import pytest

from analysis.games.fluxis.frp_replay import extract_key_events, keybind_offset
from analysis.games.fluxis.judge_sim import (hit_windows_ms,
                                              landmine_windows_ms,
                                              release_windows_ms,
                                              simulate_landmines,
                                              simulate_mania)
from analysis.games.fluxis.lane_switch import (build_lane_mask_timeline,
                                                lane_mask_for)


# ── windows -----------------------------------------------------------

def test_windows_difficulty_anchors():
    names_mid = dict(hit_windows_ms(5.0))
    assert names_mid['flawless'] == pytest.approx(19.0)
    assert names_mid['miss'] == pytest.approx(173.0)
    assert dict(hit_windows_ms(0.0))['flawless'] == pytest.approx(22.0)
    assert dict(hit_windows_ms(10.0))['flawless'] == pytest.approx(13.0)
    # Rate multiplies the window.
    assert dict(hit_windows_ms(5.0, rate=1.5))['flawless'] == pytest.approx(28.5)
    assert dict(release_windows_ms(5.0))['alright'] == pytest.approx(136.0)
    assert dict(landmine_windows_ms(5.0))['miss'] == pytest.approx(49.0)


# ── replay decoding ---------------------------------------------------

def test_keybind_offsets_match_enum_layout():
    # Key1k1=0; blocks are cumulative: 4k = 6..9, 6k = 15..20.
    assert keybind_offset(4) == 6
    assert keybind_offset(6) == 15
    assert keybind_offset(10) == 45


def test_extract_key_events_maps_actions_to_lanes():
    frames = [
        (0.0, frozenset({16})),         # 6k lane 1 down
        (10.0, frozenset({16, 20})),    # lane 5 joins
        (20.0, frozenset({20})),        # lane 1 up
        (30.0, frozenset()),            # lane 5 up
    ]
    events = extract_key_events(frames, 6)
    assert events[1] == [(0.0, True), (20.0, False)]
    assert events[5] == [(10.0, True), (30.0, False)]
    assert events[0] == []


# ── judgement sim ------------------------------------------------------

def _cols(keycount, **col_notes):
    out = [[] for _ in range(keycount)]
    for c, notes in col_notes.items():
        out[int(c)] = notes
    return out


def _note(t, end=None):
    return {'time': float(t), 'end_time': end, 'is_hold': end is not None}


def test_tap_hit_and_out_of_window_press_is_noop():
    notes = _cols(1, **{'0': [_note(1000)]})
    ticks = _cols(1)
    # Way-early press targets the note (IsFirst) but is outside the
    # window: no-op; the later in-window press still hits it.
    events = [[(500.0, True), (600.0, False), (1010.0, True), (1100.0, False)]]
    sim = simulate_mania(notes, ticks, events, 5.0)
    assert sim[0]['head_off'] == pytest.approx(10.0)
    assert sim[0]['judgement'] == 'flawless'


def test_unpressed_note_misses():
    sim = simulate_mania(_cols(1, **{'0': [_note(1000)]}), _cols(1),
                         [[]], 5.0)
    assert sim[0]['head_off'] is None
    assert sim[0]['judgement'] == 'miss'


def test_ln_broken_release_scores_alright_not_miss():
    notes = _cols(1, **{'0': [_note(1000, end=2000)]})
    events = [[(1000.0, True), (1300.0, False)]]   # released 700ms early
    sim = simulate_mania(notes, _cols(1), events, 5.0)
    assert sim[0]['judgement'] == 'alright'        # ReleaseWindows.Lowest


def test_ln_clean_release_grades():
    notes = _cols(1, **{'0': [_note(1000, end=2000)]})
    events = [[(1000.0, True), (2010.0, False)]]
    sim = simulate_mania(notes, _cols(1), events, 5.0)
    assert sim[0]['judgement'] == 'flawless'
    assert sim[0]['tail_off'] == pytest.approx(10.0)


def test_tick_held_through_is_flawless_late_press_grades():
    ticks = _cols(1, **{'0': [_note(1000), _note(3000)]})
    events = [[(900.0, True), (1100.0, False),      # held across tick 1
               (3040.0, True), (3100.0, False)]]    # 40ms late on tick 2
    sim = simulate_mania(_cols(1), ticks, events, 5.0)
    tick1, tick2 = sorted((r for r in sim if r['is_tick']),
                          key=lambda r: r['time'])
    assert tick1['judgement'] == 'flawless' and tick1['head_off'] == 0.0
    assert tick2['judgement'] == 'perfect'
    assert tick2['head_off'] == pytest.approx(40.0)


def test_tick_without_input_misses():
    sim = simulate_mania(_cols(1), _cols(1, **{'0': [_note(1000)]}),
                         [[]], 5.0)
    assert sim[0]['judgement'] == 'miss'


def test_landmine_held_through_detonates_avoided_does_not():
    mines = _cols(2, **{'0': [_note(1000)], '1': [_note(1000)]})
    events = [[(900.0, True), (1200.0, False)], []]   # lane 0 held through
    hits, avoided = simulate_landmines(mines, events,
                                       dict(landmine_windows_ms(5.0))['miss'])
    assert len(hits) == 1 and hits[0]['col'] == 0
    assert avoided == 1


# ── lane masks ---------------------------------------------------------

def test_lane_mask_matches_fluxis_tables():
    assert lane_mask_for(4, 2) == (0, 1, 1, 0)
    assert lane_mask_for(6, 3) == (0, 1, 1, 0, 1, 0)   # hand-authored, asymmetric
    assert lane_mask_for(4, 4) == (1, 1, 1, 1)


def test_lane_mask_generated_beyond_tables():
    assert lane_mask_for(12, 4) == (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0)


def test_lane_mask_timeline_dedupes_and_converts():
    events = [{'time': -2000, 'count': 4}, {'time': 5000, 'count': 2},
              {'time': 6000, 'count': 2}]
    tl = build_lane_mask_timeline(events, 4, v2=True)
    assert tl == [(-2.0, (1, 1, 1, 1)), (5.0, (0, 1, 1, 0))]


def test_static_chart_yields_empty_timeline():
    assert build_lane_mask_timeline([{'time': 0, 'count': 4}], 4,
                                    v2=True) == []
    assert build_lane_mask_timeline([], 4, v2=True) == []
