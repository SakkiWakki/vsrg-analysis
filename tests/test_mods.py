"""Per-note mod math + approach-channel compilation.

Formula tests carry hand-computed expected values traced from
OpenITG's ArrowEffects.cpp; channel tests exercise the approach-chase
compilation (snap / chase / mid-approach re-target); vectorization tests
assert the batch path equals a scalar loop; determinism tests assert the
song-time substitution makes repeated calls identical.
"""
import math

import numpy as np
import pytest

from analysis.player.render.mods import arrow_effects as ae
from analysis.player.render.mods import channels as ch
from analysis.player.render.mods import ModChannels, ModEvent, note_offsets


# --- channel approach compilation -----------------------------------

def test_snap_is_instant():
    mc = ModChannels.compile([ModEvent(0.0, 1.0, -1, 'drunk')])
    assert mc.value('drunk', 0.0) == 1.0
    assert mc.value('drunk', 5.0) == 1.0


def test_rest_before_first_event():
    mc = ModChannels.compile([ModEvent(2.0, 1.0, -1, 'drunk')])
    assert mc.value('drunk', 0.0) == 0.0
    assert mc.value('drunk', 1.999) == 0.0
    assert mc.value('drunk', 2.0) == 1.0


def test_missing_mod_is_rest():
    mc = ModChannels.compile([ModEvent(0.0, 1.0, -1, 'drunk')])
    assert mc.value('tornado', 3.0) == 0.0


def test_chase_reaches_target_at_computed_time():
    # 0.4 -> 1.0 at speed 5 => arrival at (1.0-0.4)/5 = 0.12s, linear.
    mc = ModChannels.compile([ModEvent(0.0, 0.4, -1, 'drunk'),
                              ModEvent(0.0, 1.0, 5.0, 'drunk')])
    assert mc.value('drunk', 0.0) == pytest.approx(0.4)
    assert mc.value('drunk', 0.06) == pytest.approx(0.7)
    assert mc.value('drunk', 0.12) == pytest.approx(1.0)
    assert mc.value('drunk', 0.5) == pytest.approx(1.0)


def test_retarget_mid_approach_starts_from_current_value():
    # chase 0 -> 1 speed 1 (arrival t=1). At t=0.5 (value 0.5) retarget to
    # 0 speed 1 => back to 0 at t=1.0, peaking at 0.5.
    mc = ModChannels.compile([ModEvent(0.0, 1.0, 1.0, 'drunk'),
                              ModEvent(0.5, 0.0, 1.0, 'drunk')])
    assert mc.value('drunk', 0.25) == pytest.approx(0.25)
    assert mc.value('drunk', 0.5) == pytest.approx(0.5)
    assert mc.value('drunk', 0.75) == pytest.approx(0.25)
    assert mc.value('drunk', 1.0) == pytest.approx(0.0)


def test_beat_to_time_conversion():
    # events keyed in beats, 2 beats/sec => t = beat / 2.
    mc = ModChannels.compile([ModEvent(2.0, 1.0, -1, 'drunk')],
                             beat_to_time=lambda b: b / 2.0)
    assert mc.value('drunk', 0.9) == 0.0
    assert mc.value('drunk', 1.0) == 1.0


def test_per_player_channels_are_separate():
    mc = ModChannels.compile([ModEvent(0.0, 1.0, -1, 'drunk', player=0),
                              ModEvent(0.0, 0.5, -1, 'drunk', player=1)])
    assert mc.value('drunk', 1.0, player=0) == 1.0
    assert mc.value('drunk', 1.0, player=1) == 0.5
    assert mc.players == (0, 1)


def test_values_at_returns_active_mods():
    mc = ModChannels.compile([ModEvent(0.0, 1.0, -1, 'drunk'),
                              ModEvent(0.0, 0.5, -1, 'tornado')])
    vals = mc.values_at(1.0)
    assert vals == {'drunk': 1.0, 'tornado': 0.5}


def test_values_over_matches_scalar():
    mc = ModChannels.compile([ModEvent(0.0, 0.0, 1.0, 'drunk'),
                              ModEvent(1.0, 1.0, -1, 'drunk')])
    ts = np.linspace(0.0, 2.0, 9)
    batch = mc.values_over(ts)['drunk']
    scalar = np.array([mc.value('drunk', t) for t in ts])
    np.testing.assert_allclose(batch, scalar)


# --- position formulas (hand-computed vs ArrowEffects.cpp) -----------

def test_drunk_x_known_values():
    # percent*cos(t + col*0.2 + yoff*10/480)*ARROW_SIZE*0.5, t=0, yoff=0.
    cols = np.array([0, 1, 2, 3])
    out = ae.drunk_x(1.0, cols, np.zeros(4), 0.0, 4)
    expect = np.array([math.cos(c * 0.2) * 32.0 for c in range(4)])
    np.testing.assert_allclose(out, expect)


def test_drunk_x_yoffset_phase():
    # single note, col 0, yoff 240 => phase = 240*10/480 = 5.0.
    out = ae.drunk_x(0.5, np.array([0]), np.array([240.0]), 0.0, 4)
    assert out[0] == pytest.approx(0.5 * math.cos(5.0) * 32.0)


def test_tipsy_y_known_values():
    # percent*cos(t*1.2 + col*1.8)*ARROW_SIZE*0.4, t=0.
    cols = np.array([0, 1])
    out = ae.tipsy_y(1.0, cols, 0.0)
    expect = np.array([math.cos(0.0) * 25.6, math.cos(1.8) * 25.6])
    np.testing.assert_allclose(out, expect)


def test_flip_permutation_and_shift():
    assert list(ae.flip_permutation(4)) == [3, 2, 1, 0]
    # col 0 (x=-96) -> col 3 (x=+96): distance +192 * percent.
    out = ae.flip_x(1.0, np.array([0]), 4)
    assert out[0] == pytest.approx(192.0)


def test_invert_permutation_odd_and_even():
    # 4k mirrors halves: [1,0,3,2]; 5k middle column fixed: [2,1,0,4,3].
    assert list(ae.invert_permutation(4)) == [1, 0, 3, 2]
    assert list(ae.invert_permutation(5)) == [2, 1, 0, 4, 3]


def test_movex_is_one_arrow_at_full():
    # NotITG: 100% movex = one ARROW_SIZE.
    out = ae.movex_x(np.array([1.0, 0.5]))
    np.testing.assert_allclose(out, [64.0, 32.0])


def test_bumpy_zoom_center_of_wave():
    # z = percent*40*sin(yoff/16); at yoff = 8*pi, sin(pi/2)... pick yoff so
    # sin = 1: yoff/16 = pi/2 => yoff = 8*pi. zoom = 1 + 40*percent/480.
    yoff = np.array([8.0 * math.pi])
    out = ae.bumpy_zoom(1.0, yoff)
    assert out[0] == pytest.approx(1.0 + 40.0 / 480.0)


def test_dizzy_rotation_wraps_to_degrees():
    # (note_beat - beat_now)*percent, mod 2pi, *180/pi. beat gap 1, percent
    # 1 => 1 rad => 180/pi deg.
    out = ae.dizzy_rotation(1.0, np.array([5.0]), 4.0)
    assert out[0] == pytest.approx(180.0 / math.pi)


def test_confusion_rotation_radians_times_beat():
    # percent (radians) * beat, in degrees.
    out = ae.confusion_rotation(0.5, 4.0)
    assert out == pytest.approx(0.5 * 4.0 * 180.0 / math.pi)


def test_zoom_from_mini():
    assert ae.zoom_from_mini(0.0) == 1.0
    assert ae.zoom_from_mini(1.0) == 0.5
    assert ae.zoom_from_mini(2.0) == 0.0


# --- alpha / visibility ---------------------------------------------

def test_stealth_lowers_alpha():
    p = {'stealth': 0.4}
    r = note_offsets(p, np.array([0]), np.array([100.0]),
                     t_now=0.0, beat_now=0.0, keycount=4)
    assert r.alpha_mult[0] == pytest.approx(0.6)


def test_notes_past_gray_arrows_fully_visible():
    # y_pos < 0 => visible 1 regardless of stealth (ArrowEffects.cpp:448).
    p = {'stealth': 1.0}
    r = note_offsets(p, np.array([0]), np.array([-50.0]),
                     t_now=0.0, beat_now=0.0, keycount=4)
    assert r.alpha_mult[0] == 1.0


def test_sudden_hides_far_notes():
    # A far note (large y_offset) under full sudden is invisible.
    p = {'sudden': 1.0}
    r = note_offsets(p, np.array([0]), np.array([400.0]),
                     t_now=0.0, beat_now=0.0, keycount=4)
    assert r.alpha_mult[0] == pytest.approx(0.0)


# --- per-column NotITG variants -------------------------------------

def test_numbered_drunk_variant_overrides_one_column():
    # drunk2 sets column 2 only; other columns take the global drunk.
    p = {'drunk': 0.0, 'drunk2': 1.0}
    cols = np.array([0, 1, 2, 3])
    r = note_offsets(p, cols, np.zeros(4), t_now=0.0, beat_now=0.0,
                     keycount=4)
    assert r.dx[0] == 0.0 and r.dx[1] == 0.0 and r.dx[3] == 0.0
    # col 2: cos(0 + 2*0.2)*32.
    assert r.dx[2] == pytest.approx(math.cos(0.4) * 32.0)


def test_movex_per_column():
    p = {'movex1': 1.0}
    cols = np.array([0, 1, 2, 3])
    r = note_offsets(p, cols, np.zeros(4), t_now=0.0, beat_now=0.0,
                     keycount=4)
    np.testing.assert_allclose(r.dx, [0.0, 64.0, 0.0, 0.0])


# --- vectorization: batch == scalar loop ----------------------------

def test_note_offsets_batch_equals_scalar_loop():
    p = {'drunk': 0.5, 'tornado': 0.3, 'tipsy': 0.4, 'movex': 0.2,
         'stealth': 0.25, 'mini': 0.5, 'bumpy': 0.3}
    cols = np.array([0, 1, 2, 3, 0, 2])
    y = np.array([120.0, 80.0, 40.0, 10.0, -30.0, 300.0])
    beats = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    batch = note_offsets(p, cols, y, t_now=1.7, beat_now=3.3,
                         keycount=4, note_beats=beats)
    for i in range(len(cols)):
        one = note_offsets(p, cols[i:i + 1], y[i:i + 1], t_now=1.7,
                           beat_now=3.3, keycount=4,
                           note_beats=beats[i:i + 1])
        assert batch.dx[i] == pytest.approx(one.dx[0])
        assert batch.dy[i] == pytest.approx(one.dy[0])
        assert batch.rotation_deg[i] == pytest.approx(one.rotation_deg[0])
        assert batch.alpha_mult[i] == pytest.approx(one.alpha_mult[0])
        assert batch.zoom[i] == pytest.approx(one.zoom[0])


def test_tornado_batch_equals_scalar():
    cols = np.array([0, 1, 2, 3])
    y = np.array([50.0, 100.0, 150.0, 200.0])
    batch = ae.tornado_x(0.7, cols, y, 4)
    for i in range(4):
        one = ae.tornado_x(0.7, cols[i:i + 1], y[i:i + 1], 4)
        assert batch[i] == pytest.approx(one[0])


# --- determinism (song-time substitution for wall clock) ------------

def test_determinism_same_time_same_result():
    p = {'drunk': 0.5, 'tipsy': 0.3, 'blink': 0.5}
    cols = np.array([0, 1, 2, 3])
    y = np.array([100.0, 50.0, 25.0, 10.0])
    a = note_offsets(p, cols, y, t_now=2.5, beat_now=4.0, keycount=4)
    b = note_offsets(p, cols, y, t_now=2.5, beat_now=4.0, keycount=4)
    np.testing.assert_array_equal(a.dx, b.dx)
    np.testing.assert_array_equal(a.dy, b.dy)
    np.testing.assert_array_equal(a.alpha_mult, b.alpha_mult)


def test_different_time_changes_periodic_mods():
    p = {'drunk': 1.0}
    cols = np.array([0])
    y = np.array([0.0])
    a = note_offsets(p, cols, y, t_now=0.0, beat_now=0.0, keycount=4)
    b = note_offsets(p, cols, y, t_now=1.0, beat_now=0.0, keycount=4)
    assert a.dx[0] != b.dx[0]


def test_receptor_offsets_have_zero_dizzy():
    # receptors sit at the current beat, so dizzy (beats-until-step) is 0.
    p = {'dizzy': 1.0, 'drunk': 0.5}
    cols = np.array([0, 1, 2, 3])
    r = ae.receptor_offsets(p, cols, t_now=1.0, beat_now=3.0, keycount=4)
    np.testing.assert_array_equal(r.rotation_deg, np.zeros(4))
    # drunk still displaces receptors (y_offset = 0).
    assert r.dx[0] != 0.0


# --- reverse family (GetReversePercentForColumn) --------------------

def test_reverse_all_columns_full():
    # reverse 100% => every column r = 1.
    cols = np.array([0, 1, 2, 3])
    r = ae.reverse_fractions({'reverse': 1.0}, cols, 4)
    np.testing.assert_allclose(r, [1.0, 1.0, 1.0, 1.0])


def test_split_only_right_half():
    # split hits iCol >= N/2 = 2 => cols 2,3 only (4k).
    cols = np.array([0, 1, 2, 3])
    r = ae.reverse_fractions({'split': 1.0}, cols, 4)
    np.testing.assert_allclose(r, [0.0, 0.0, 1.0, 1.0])


def test_alternate_odd_columns():
    cols = np.array([0, 1, 2, 3])
    r = ae.reverse_fractions({'alternate': 1.0}, cols, 4)
    np.testing.assert_allclose(r, [0.0, 1.0, 0.0, 1.0])


def test_cross_middle_half():
    # 4k: first = N/4 = 1, last = N-1-1 = 2 => cols 1,2.
    cols = np.array([0, 1, 2, 3])
    r = ae.reverse_fractions({'cross': 1.0}, cols, 4)
    np.testing.assert_allclose(r, [0.0, 1.0, 1.0, 0.0])


def test_reverse_wrap_folds_double_reverse():
    # reverse + split on col 3 = 2.0; > 1 folds via SCALE(f,1,2,1,0):
    # f=2 -> 0 (double reverse reads as none). col 0 stays at reverse=1.
    cols = np.array([0, 3])
    r = ae.reverse_fractions({'reverse': 1.0, 'split': 1.0}, cols, 4)
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx(0.0)


def test_reverse_fractions_odd_keycount():
    # 5k: N/2 = 2 (split cols 2,3,4); N/4 = 1, last = 3 (cross cols 1..3);
    # alternate odd cols 1,3. Check split membership on the middle col 2.
    cols = np.array([0, 1, 2, 3, 4])
    r = ae.reverse_fractions({'split': 1.0}, cols, 5)
    np.testing.assert_allclose(r, [0.0, 0.0, 1.0, 1.0, 1.0])


# --- accel family (GetYOffset second half) --------------------------

def test_accel_no_mods_is_identity():
    y = np.array([100.0, -50.0, 0.0])
    out = ae.accel_y_offset({}, y)
    np.testing.assert_array_equal(out, y)


def test_accel_leaves_past_notes_untouched():
    # y < 0 (past the receptor) returns unchanged for every accel.
    y = np.array([-30.0])
    out = ae.accel_y_offset({'boost': 1.0, 'brake': 1.0, 'wave': 1.0}, y)
    assert out[0] == -30.0


def test_boost_compresses_near_notes():
    # boost pulls close notes IN (smaller offset) and far notes OUT.
    # fNewY = y*1.5/((y + H/1.2)/H); H = 480. At y = 100:
    #   denom = (100 + 400)/480 = 500/480; fNewY = 150 / (500/480) = 144.
    #   adjust = 1*(144 - 100) = 44 => out = 144.
    y = np.array([100.0])
    out = ae.accel_y_offset({'boost': 1.0}, y)
    assert out[0] == pytest.approx(144.0)
    # a far note (y large) is pushed further than a near note relative gap:
    near = ae.accel_y_offset({'boost': 1.0}, np.array([20.0]))[0]
    assert near < out[0]


def test_brake_slows_near_receptor():
    # brake: scale = y/H, fNewY = y*scale; adjust = brake*(fNewY - y).
    # y = 240, H = 480: scale = 0.5, fNewY = 120, adjust = 1*(120-240) = -120.
    y = np.array([240.0])
    out = ae.accel_y_offset({'brake': 1.0}, y)
    assert out[0] == pytest.approx(120.0)


def test_wave_adds_sine():
    # wave adds wave*20*sin(y/38). At y = 38*pi/2, sin = 1 => +20*wave.
    y = np.array([38.0 * math.pi / 2.0])
    out = ae.accel_y_offset({'wave': 0.5}, y)
    assert out[0] == pytest.approx(y[0] + 0.5 * 20.0 * 1.0)


def test_accel_adjust_clamped():
    # boost adjust is clamped to +/- 400. At huge y, fNewY -> ~1.5*480 so
    # adjust = fNewY - y is a large negative, clamped to -400.
    y = np.array([100000.0])
    out = ae.accel_y_offset({'boost': 1.0}, y)
    assert out[0] == pytest.approx(100000.0 - 400.0)


def test_expand_scales_whole_offset():
    # expand at phase 0: cos(0)=1 => mult = SCALE(1,-1,1,0.75,1.75) = 1.75;
    # scroll factor = SCALE(1,0,1,1,1.75) = 1.75. out = y * 1.75.
    y = np.array([100.0])
    out = ae.accel_y_offset({'expand': 1.0, '_expand_phase': 0.0}, y)
    assert out[0] == pytest.approx(175.0)


# --- tiny / dark ----------------------------------------------------

def test_tiny_shrinks_notes():
    # tiny 100% = half size, 200% = zero, same curve as mini but zoom-only.
    p = {'tiny': 1.0}
    r = note_offsets(p, np.array([0]), np.array([100.0]),
                     t_now=0.0, beat_now=0.0, keycount=4)
    assert r.zoom[0] == pytest.approx(0.5)


def test_dark_hides_receptor_alpha():
    assert ae.receptor_alpha_from_dark(0.0) == 1.0
    assert ae.receptor_alpha_from_dark(1.0) == 0.0
    assert ae.receptor_alpha_from_dark(0.4) == pytest.approx(0.6)


# --- note_mods reverse/centered remap (fake-ctx integration) ---------

class _FakeNotes:
    def __init__(self, n):
        self.noterows_list = list(range(n))


class _FakePlayer:
    def __init__(self, cols, keycount):
        self.columns = np.asarray(cols, dtype=np.int64)
        self.keycount = keycount
        self.notes = _FakeNotes(len(cols))


class _FakeCtx:
    def __init__(self, player, heads, judge_y, chart_h):
        self.player = player
        self.candidates = list(range(len(heads)))
        self.t_now = 0.0
        self.lane_w = ae.ARROW_SIZE  # scale = 1
        self.judge_y = judge_y
        self.chart_rect = (0.0, 0.0, 400.0, float(chart_h))
        self.candidate_head_y = np.asarray(heads, dtype=np.float64)
        self.candidate_tail_y = np.asarray(heads, dtype=np.float64)
        self.candidate_press_y = np.asarray(heads, dtype=np.float64)


def _mods(events):
    from analysis.games.notitg.note_mods import NotitgNoteMods
    mc = ModChannels.compile(events)
    return NotitgNoteMods(mc, [(0.0, 120.0)])


def test_reverse_100_flips_positions_about_midline():
    # judge_y = 100, chart center = 200 => mirror_y = 300. A note at
    # head_y = 40 (60 above judge line) fully reversed sits at
    # mirror_y - (40 - 100) = 300 + 60 = 360 (60 below the mirrored line).
    player = _FakePlayer([0, 1, 2, 3], 4)
    ctx = _FakeCtx(player, [40.0, 40.0, 40.0, 40.0], judge_y=100, chart_h=400)
    _mods([ModEvent(0.0, 1.0, -1, 'reverse')]).apply(ctx)
    np.testing.assert_allclose(ctx.candidate_head_y, [360.0] * 4)


def test_centered_converges_receptor_to_midscreen():
    # centered 50% slides each receptor halfway to field center (200).
    # receptors at judge_y = 100 -> 100 + 0.5*(200 - 100) = 150. A note at
    # head_y = judge_y (y_offset 0) lands exactly on the receptor.
    player = _FakePlayer([0, 1, 2, 3], 4)
    ctx = _FakeCtx(player, [100.0] * 4, judge_y=100, chart_h=400)
    _mods([ModEvent(0.0, 0.5, -1, 'centered')]).apply(ctx)
    np.testing.assert_allclose(ctx.candidate_head_y, [150.0] * 4)
    np.testing.assert_allclose(ctx.receptor_offsets['dy'], [50.0] * 4)


def test_split_reverses_only_right_half():
    # split 100%: cols 2,3 fully reverse, cols 0,1 stay put.
    player = _FakePlayer([0, 1, 2, 3], 4)
    ctx = _FakeCtx(player, [40.0] * 4, judge_y=100, chart_h=400)
    _mods([ModEvent(0.0, 1.0, -1, 'split')]).apply(ctx)
    # left half unchanged (r = 0 => identity), right half flipped to 360.
    np.testing.assert_allclose(ctx.candidate_head_y, [40.0, 40.0, 360.0, 360.0])


def test_boost_compresses_via_apply():
    # boost remaps y_offset before positions rebuild. head_y = 60 =>
    # y_offset = 100 - 60 = 40 (scale 1). boost 100%: fNewY = 40*1.5/
    # ((40 + 400)/480) = 60/(440/480) = 65.4545; new head_y = 100 - 65.4545.
    player = _FakePlayer([0], 1)
    ctx = _FakeCtx(player, [60.0], judge_y=100, chart_h=400)
    _mods([ModEvent(0.0, 1.0, -1, 'boost')]).apply(ctx)
    expect_off = 40.0 * 1.5 / ((40.0 + 480.0 / 1.2) / 480.0)
    assert ctx.candidate_head_y[0] == pytest.approx(100.0 - expect_off)
