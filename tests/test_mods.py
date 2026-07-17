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
    # ReceptorGetRotationZ: (beat*percent mod 2pi) * -180/pi. beat 4, percent
    # 0.5 => 2.0 rad (< 2pi, no wrap) => 2.0 * -180/pi.
    out = ae.confusion_rotation(0.5, 4.0)
    assert out == pytest.approx(2.0 * -180.0 / math.pi)


def test_confusion_wraps_at_two_pi():
    # beat*percent = 8.0 rad wraps to 8 - 2pi ~ 1.717 before scaling.
    out = ae.confusion_rotation(1.0, 8.0)
    assert out == pytest.approx((8.0 % (2.0 * math.pi)) * -180.0 / math.pi)


def test_confusion_offset_adds_constant():
    # confusionoffset adds offset*180/pi regardless of beat.
    out = ae.confusion_rotation(0.0, 4.0, offset=0.5)
    assert out == pytest.approx(0.5 * 180.0 / math.pi)


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


def test_reverse_100_reads_as_native_downscroll():
    # Native candidate space IS engine reverse=1, so a chart pinning
    # reverse=1 (gat's baseline) leaves positions untouched.
    player = _FakePlayer([0, 1, 2, 3], 4)
    ctx = _FakeCtx(player, [40.0, 40.0, 40.0, 40.0], judge_y=100, chart_h=400)
    _mods([ModEvent(0.0, 1.0, -1, 'reverse')]).apply(ctx)
    np.testing.assert_allclose(ctx.candidate_head_y, [40.0] * 4)
    np.testing.assert_allclose(ctx.receptor_offsets['dy'], [0.0] * 4)


def test_zero_channels_flip_to_engine_default_upscroll():
    # No mods = engine reverse 0 = receptors on top. judge_y = 100,
    # chart center = 200 => mirror_y = 300. A note at head_y = 40 (60
    # above the judge line) mirrors to 300 + 60 = 360, and receptors
    # shift to the mirrored line (dy = +200).
    player = _FakePlayer([0, 1, 2, 3], 4)
    ctx = _FakeCtx(player, [40.0, 40.0, 40.0, 40.0], judge_y=100, chart_h=400)
    _mods([]).apply(ctx)
    np.testing.assert_allclose(ctx.candidate_head_y, [360.0] * 4)
    np.testing.assert_allclose(ctx.receptor_offsets['dy'], [200.0] * 4)


def test_centered_converges_receptor_to_midscreen():
    # centered 50% slides each receptor halfway to field center (200).
    # With no reverse channels the receptor starts at the engine-default
    # mirrored line (300) -> 300 + 0.5*(200 - 300) = 250. A note at
    # head_y = judge_y (y_offset 0) lands exactly on the receptor.
    player = _FakePlayer([0, 1, 2, 3], 4)
    ctx = _FakeCtx(player, [100.0] * 4, judge_y=100, chart_h=400)
    _mods([ModEvent(0.0, 0.5, -1, 'centered')]).apply(ctx)
    np.testing.assert_allclose(ctx.candidate_head_y, [250.0] * 4)
    np.testing.assert_allclose(ctx.receptor_offsets['dy'], [150.0] * 4)


def test_split_reverses_only_right_half():
    # split 100%: cols 2,3 engine-reverse (= our native downscroll, stay
    # put); cols 0,1 remain engine-default and mirror to 360.
    player = _FakePlayer([0, 1, 2, 3], 4)
    ctx = _FakeCtx(player, [40.0] * 4, judge_y=100, chart_h=400)
    _mods([ModEvent(0.0, 1.0, -1, 'split')]).apply(ctx)
    np.testing.assert_allclose(ctx.candidate_head_y, [360.0, 360.0, 40.0, 40.0])


def test_boost_compresses_via_apply():
    # boost remaps y_offset before positions rebuild. head_y = 60 =>
    # y_offset = 100 - 60 = 40 (scale 1). boost 100%: fNewY = 40*1.5/
    # ((40 + 400)/480) = 60/(440/480) = 65.4545. The zero-reverse baseline
    # then mirrors about the judge line: mirror_y + off = 300 + 65.4545.
    player = _FakePlayer([0], 1)
    ctx = _FakeCtx(player, [60.0], judge_y=100, chart_h=400)
    _mods([ModEvent(0.0, 1.0, -1, 'boost')]).apply(ctx)
    expect_off = 40.0 * 1.5 / ((40.0 + 480.0 / 1.2) / 480.0)
    assert ctx.candidate_head_y[0] == pytest.approx(300.0 + expect_off)


# --- digital / waveform warp family (ITGmania ArrowEffects.cpp) -------

def _digital_angle(yoff, offset, period):
    return math.pi * (yoff + offset) / (ae.ARROW_SIZE + period * ae.ARROW_SIZE)


def test_rage_triangle_shape():
    # RageTriangle (u = angle/PI): rises 0->1 over u in [0,0.5], falls
    # 1->-1 over [0.5,1.5], rises -1->0 over [1.5,2]. So the peak +1 is at
    # angle=PI/2 (u=0.5), zero at angle=PI (u=1), trough -1 at 3PI/2 (u=1.5).
    ang = np.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    np.testing.assert_allclose(ae.rage_triangle(ang), [0.0, 1.0, 0.0, -1.0])


def test_rage_square_shape():
    # +1 for [0, PI), -1 for [PI, 2PI). Wraps and handles negatives.
    ang = np.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2, -math.pi / 2])
    np.testing.assert_allclose(ae.rage_square(ang), [1.0, 1.0, -1.0, -1.0, -1.0])


def test_digital_x_steps_quantize_sine():
    # percent*ARROW_SIZE*0.5 * round((steps+1)*sin(angle))/(steps+1).
    # steps=0 => levels 1 => round(sin(angle)) in {-1,0,1}. Pick yoff so
    # angle = PI/2 => sin=1 => round(1)/1 = 1 => 1*64*0.5*1 = 32.
    yoff = np.array([ae.ARROW_SIZE / 2.0])  # angle = PI*(32)/64 = PI/2
    out = ae.digital_x(1.0, yoff, 0.0, 0.0, 0.0)
    assert out[0] == pytest.approx(32.0)


def test_digital_x_more_steps_recovers_sine():
    # steps large => levels large => round(levels*sin)/levels ~ sin.
    yoff = np.array([20.0])
    angle = _digital_angle(20.0, 0.0, 0.0)
    out = ae.digital_x(1.0, yoff, 0.0, 0.0, 999.0)
    levels = 1000.0
    expect = 32.0 * round(levels * math.sin(angle)) / levels
    assert out[0] == pytest.approx(expect)


def test_digital_x_offset_and_period_modulate():
    # offset shifts phase by 1 px/percent; period stretches the base period.
    yoff = np.array([15.0])
    out = ae.digital_x(0.5, yoff, 20.0, 0.5, 3.0)
    angle = _digital_angle(15.0, 20.0, 0.5)
    levels = 4.0
    expect = 0.5 * 64.0 * 0.5 * round(levels * math.sin(angle)) / levels
    assert out[0] == pytest.approx(expect)


def test_zigzag_x_triangle_scaled():
    # percent*(ARROW_SIZE/2)*RageTriangle(PI*(1/(period+1))*((yoff+100*off)/AS)).
    yoff = np.array([32.0])
    out = ae.zigzag_x(1.0, yoff, 0.0, 0.0, ae.ARROW_SIZE)
    angle = math.pi * (32.0 / 64.0)  # PI/2
    expect = 32.0 * ae.rage_triangle(np.array([angle]))[0]
    assert out[0] == pytest.approx(expect)


def test_zigzag_x_period_stretches():
    yoff = np.array([48.0])
    out = ae.zigzag_x(0.5, yoff, 10.0, 1.0, ae.ARROW_SIZE)
    angle = math.pi * (1.0 / 2.0) * ((48.0 + 100.0 * 10.0) / 64.0)
    expect = 0.5 * 32.0 * ae.rage_triangle(np.array([angle]))[0]
    assert out[0] == pytest.approx(expect)


def test_sawtooth_x_fractional_ramp():
    # percent*ARROW_SIZE*frac(0.5/(period+1)*yoff/ARROW_SIZE). period 0:
    # yoff=64 => 0.5*64/64 = 0.5, frac = 0.5 => 1*64*0.5 = 32.
    out = ae.sawtooth_x(1.0, np.array([64.0]), 0.0)
    assert out[0] == pytest.approx(32.0)
    # wraps: yoff = 3*64 => 0.5*3 = 1.5, frac = 0.5 => still 32.
    out2 = ae.sawtooth_x(1.0, np.array([192.0]), 0.0)
    assert out2[0] == pytest.approx(32.0)


def test_sawtooth_x_period_slows_ramp():
    # period 1 halves the slope: 0.5/2*yoff/AS. yoff=128 => 0.25*128/64=0.5.
    out = ae.sawtooth_x(1.0, np.array([128.0]), 1.0)
    assert out[0] == pytest.approx(32.0)


def test_square_x_wave_scaled():
    # percent*ARROW_SIZE*0.5*RageSquare(digital_angle). Pick yoff so angle
    # is in [0,PI) => +1 => +32; and one in [PI,2PI) => -1 => -32.
    pos = ae.square_x(1.0, np.array([16.0]), 0.0, 0.0)  # angle=PI/4
    assert pos[0] == pytest.approx(32.0)
    neg = ae.square_x(1.0, np.array([96.0]), 0.0, 0.0)  # angle=3PI/2
    assert neg[0] == pytest.approx(-32.0)


def test_square_x_offset_period():
    yoff = np.array([10.0])
    out = ae.square_x(0.5, yoff, 5.0, 0.5)
    angle = _digital_angle(10.0, 5.0, 0.5)
    expect = 0.5 * 64.0 * 0.5 * ae.rage_square(np.array([angle]))[0]
    assert out[0] == pytest.approx(expect)


def test_bounce_x_rectified_sine():
    # percent*ARROW_SIZE*0.5*abs(sin((yoff+off)/(60+period*60))). period 0:
    # yoff = 60*PI/2 => sin(PI/2)=1 => 1*64*0.5*1 = 32.
    yoff = np.array([60.0 * math.pi / 2.0])
    out = ae.bounce_x(1.0, yoff, 0.0, 0.0)
    assert out[0] == pytest.approx(32.0)
    # rectified: a phase that gives sin<0 still yields +abs.
    neg_phase = np.array([60.0 * 3.0 * math.pi / 2.0])
    out2 = ae.bounce_x(1.0, neg_phase, 0.0, 0.0)
    assert out2[0] == pytest.approx(32.0)


def test_bounce_x_period_and_offset():
    yoff = np.array([30.0])
    out = ae.bounce_x(0.5, yoff, 12.0, 1.0)
    amt = abs(math.sin((30.0 + 12.0) / (60.0 + 60.0)))
    assert out[0] == pytest.approx(0.5 * 64.0 * 0.5 * amt)


def test_waveform_z_zoom_reprojection():
    # z push maps to 1 + z/SCREEN_HEIGHT, matching bumpy's reprojection.
    out = ae.waveform_z_zoom(np.array([48.0]))
    assert out[0] == pytest.approx(1.0 + 48.0 / 480.0)


# --- digital family via note_offsets (channels-through smoke) --------

def test_digital_note_offsets_displaces_dx():
    # a 'digital' percent moves dx off-lane; zero percent leaves dx at 0.
    cols = np.array([0, 1, 2, 3])
    yoff = np.array([32.0, 32.0, 32.0, 32.0])
    on = note_offsets({'digital': 1.0}, cols, yoff, t_now=0.0, beat_now=0.0,
                      keycount=4)
    assert np.any(on.dx != 0.0)
    off = note_offsets({'digital': 0.0}, cols, yoff, t_now=0.0, beat_now=0.0,
                       keycount=4)
    np.testing.assert_array_equal(off.dx, np.zeros(4))


def test_digital_companions_change_dx():
    # digitalperiod / digitaloffset / digitalsteps modulate the result:
    # changing a companion changes dx (they are read from the percents dict).
    # Use a fine step base (100 levels) so the sine, not the quantizer,
    # dominates - period/offset shifts then reliably change the sample.
    cols = np.array([0])
    yoff = np.array([37.0])
    fine = {'digital': 1.0, 'digitalsteps': 99.0}
    base = note_offsets(fine, cols, yoff, t_now=0.0, beat_now=0.0,
                        keycount=4).dx[0]
    period = note_offsets({**fine, 'digitalperiod': 0.7}, cols, yoff,
                          t_now=0.0, beat_now=0.0, keycount=4).dx[0]
    offset = note_offsets({**fine, 'digitaloffset': 25.0}, cols, yoff,
                          t_now=0.0, beat_now=0.0, keycount=4).dx[0]
    # steps changes the quantization: coarse (0) vs fine (99) differ.
    coarse = note_offsets({'digital': 1.0, 'digitalsteps': 0.0}, cols, yoff,
                          t_now=0.0, beat_now=0.0, keycount=4).dx[0]
    assert period != base
    assert offset != base
    assert coarse != base


def test_digitalz_note_offsets_changes_zoom():
    # the Z sibling pushes zoom off 1.0 via the reprojection; X stays clean.
    cols = np.array([0])
    yoff = np.array([32.0])
    r = note_offsets({'digitalz': 1.0}, cols, yoff, t_now=0.0, beat_now=0.0,
                     keycount=4)
    assert r.zoom[0] != pytest.approx(1.0)
    np.testing.assert_array_equal(r.dx, np.zeros(1))


def test_zigzag_per_column_variant():
    # numbered zigzag2 fires on column 2 only (rides auto-detection).
    p = {'zigzag2': 1.0}
    cols = np.array([0, 1, 2, 3])
    yoff = np.array([48.0, 48.0, 48.0, 48.0])
    r = note_offsets(p, cols, yoff, t_now=0.0, beat_now=0.0, keycount=4)
    assert r.dx[0] == 0.0 and r.dx[1] == 0.0 and r.dx[3] == 0.0
    assert r.dx[2] != 0.0


def test_waveform_batch_equals_scalar_loop():
    p = {'digital': 0.6, 'digitalperiod': 0.3, 'digitaloffset': 10.0,
         'digitalsteps': 2.0, 'zigzag': 0.5, 'zigzagperiod': 0.4,
         'sawtooth': 0.3, 'square': 0.4, 'squareoffset': 5.0,
         'bounce': 0.5, 'bounceperiod': 0.2, 'digitalz': 0.4,
         'zigzagz': 0.3, 'bouncez': 0.2}
    cols = np.array([0, 1, 2, 3, 0, 2])
    y = np.array([120.0, 80.0, 41.0, 13.0, -30.0, 305.0])
    batch = note_offsets(p, cols, y, t_now=1.7, beat_now=3.3, keycount=4)
    for i in range(len(cols)):
        one = note_offsets(p, cols[i:i + 1], y[i:i + 1], t_now=1.7,
                           beat_now=3.3, keycount=4)
        assert batch.dx[i] == pytest.approx(one.dx[0])
        assert batch.zoom[i] == pytest.approx(one.zoom[0])


# --- boomerang (position parabola + visibility approximation) --------

def test_boomerang_y_offset_parabola():
    # y' = -y*y/H + 1.5*y, H = 480. At y = 240: -240^2/480 + 360 = 240.
    out = ae.boomerang_y_offset(np.array([240.0]))
    assert out[0] == pytest.approx(240.0)
    # at the fold peak y = 0.75*480 = 360: -360^2/480 + 540 = 270 (the max).
    peak_raw, peak_y = ae.boomerang_peak()
    assert peak_raw == pytest.approx(360.0)
    assert peak_y == pytest.approx(270.0)
    assert ae.boomerang_y_offset(np.array([360.0]))[0] == pytest.approx(270.0)


def test_boomerang_applies_via_accel():
    # boomerang runs inside accel_y_offset (after boost/brake/wave). Alone it
    # is just the parabola of the raw offset.
    y = np.array([120.0, 360.0, 480.0])
    out = ae.accel_y_offset({'boomerang': 1.0}, y)
    expect = -1.0 * y * y / 480.0 + 1.5 * y
    np.testing.assert_allclose(out, expect)


def test_boomerang_visibility_fades_past_fold():
    # visibility mirrors bIsPastPeak: alpha 1 at the fold p = 360, ramping to
    # 0 one ARROW_SIZE (64) later. percent gates the strength.
    p = 360.0
    assert ae.boomerang_visibility(1.0, np.array([p]))[0] == pytest.approx(1.0)
    assert ae.boomerang_visibility(1.0, np.array([p + 32.0]))[0] == pytest.approx(0.5)
    assert ae.boomerang_visibility(1.0, np.array([p + 64.0]))[0] == pytest.approx(0.0)
    # notes before the fold are fully visible.
    assert ae.boomerang_visibility(1.0, np.array([100.0]))[0] == pytest.approx(1.0)
    # percent 0 => no fade at all.
    assert ae.boomerang_visibility(0.0, np.array([p + 64.0])) == 1.0


def test_boomerang_visibility_enters_alpha():
    # a far note (past the fold) under boomerang fades in note_offsets.alpha.
    r = note_offsets({'boomerang': 1.0}, np.array([0]), np.array([440.0]),
                     t_now=0.0, beat_now=0.0, keycount=4)
    assert r.alpha_mult[0] == pytest.approx(0.0)
    near = note_offsets({'boomerang': 1.0}, np.array([0]), np.array([100.0]),
                        t_now=0.0, beat_now=0.0, keycount=4)
    assert near.alpha_mult[0] == pytest.approx(1.0)


# --- pulse / shrink zoom --------------------------------------------

def test_pulse_zoom_sine_swing():
    # sine = sin((yoff+100*off)/(0.4*(AS+period*AS))). Pick yoff so sine = 1:
    # yoff/(0.4*64) = pi/2 => yoff = 0.4*64*pi/2. inner 0, outer 1:
    # zoom = 1*(1*0.5) + (0*0.5 + 1) = 1.5.
    yoff = np.array([0.4 * 64.0 * math.pi / 2.0])
    out = ae.pulse_zoom(0.0, 1.0, yoff)
    assert out[0] == pytest.approx(1.5)


def test_pulse_zoom_off_when_both_zero():
    assert ae.pulse_zoom(0.0, 0.0, np.array([50.0])) == 1.0


def test_pulse_inner_sets_rest_scale():
    # sine term 0 (yoff = 0), inner 1 => rest = inner*0.5 + 1 = 1.5.
    out = ae.pulse_zoom(1.0, 0.0, np.array([0.0]))
    assert out[0] == pytest.approx(1.5)


def test_pulse_via_note_offsets_zoom():
    r = note_offsets({'pulseouter': 1.0}, np.array([0]),
                     np.array([0.4 * 64.0 * math.pi / 2.0]),
                     t_now=0.0, beat_now=0.0, keycount=4)
    assert r.zoom[0] != pytest.approx(1.0)


def test_shrink_mult_shrinks_far_notes():
    # zoom *= 1/(1 + yoff*(mult/100)). yoff=100, mult=100 => 1/101.
    m, a = ae.shrink_zoom(100.0, 0.0, np.array([100.0]))
    assert m[0] == pytest.approx(1.0 / 101.0)
    assert a[0] == 0.0


def test_shrink_linear_adds_by_distance():
    # zoom += yoff*(0.5*linear/AS). yoff=64, linear=1 => +0.5.
    m, a = ae.shrink_zoom(0.0, 1.0, np.array([64.0]))
    assert a[0] == pytest.approx(0.5)
    assert m[0] == 1.0


def test_shrink_only_affects_approaching_notes():
    # y_offset < 0 (past receptor) is untouched by either shrink.
    m, a = ae.shrink_zoom(100.0, 1.0, np.array([-50.0]))
    assert m[0] == 1.0 and a[0] == 0.0


# --- tan / cosec family ---------------------------------------------

def test_select_tan_matches_math():
    ang = np.array([0.3, 0.8])
    np.testing.assert_allclose(ae._select_tan(ang), np.tan(ang))
    np.testing.assert_allclose(ae._select_tan(ang, cosecant=True), 1.0 / np.sin(ang))


def test_tan_drunk_uses_tan_kernel():
    # tandrunk is drunk with the tan kernel: percent*tan(angle)*AS*0.5.
    # angle at t=1, col0, yoff0 = 1. Expect 1*tan(1)*64*0.5.
    cols = np.array([0])
    r = note_offsets({'tandrunk': 1.0}, cols, np.array([0.0]),
                     t_now=1.0, beat_now=0.0, keycount=4)
    assert r.dx[0] == pytest.approx(math.tan(1.0) * 64.0 * 0.5)


def test_tan_tornado_differs_from_tornado():
    cols = np.array([0])
    y = np.array([50.0])
    plain = ae.tornado_x(1.0, cols, y, 4)
    tanv = ae.tan_tornado_x(1.0, cols, y, 4)
    assert plain[0] != pytest.approx(tanv[0])


def test_tan_tipsy_uses_tan_kernel():
    # tantipsy = percent*tan(angle)*AS*0.4; angle at t=0.5, col0 = 0.5*1.2 = 0.6.
    r = note_offsets({'tantipsy': 1.0}, np.array([0]), np.array([0.0]),
                     t_now=0.5, beat_now=0.0, keycount=4)
    assert r.dy[0] == pytest.approx(math.tan(0.6) * 64.0 * 0.4)


def test_tan_digital_uses_tan_kernel():
    # tandigital shares digital's angle but tan-kernels the sine. steps large
    # so quantization is fine.
    yoff = np.array([15.0])
    r = note_offsets({'tandigital': 1.0, 'tandigitalsteps': 999.0},
                     np.array([0]), yoff, t_now=0.0, beat_now=0.0, keycount=4)
    angle = math.pi * 15.0 / 64.0
    levels = 1000.0
    expect = 32.0 * round(levels * math.tan(angle)) / levels
    assert r.dx[0] == pytest.approx(expect)


def test_tan_bumpy_pushes_zoom():
    # tanbumpy is a z-push (-> zoom) using the tan kernel.
    r = note_offsets({'tanbumpy': 0.5}, np.array([0]), np.array([20.0]),
                     t_now=0.0, beat_now=0.0, keycount=4)
    assert r.zoom[0] != pytest.approx(1.0)


def test_tan_bumpyx_pushes_dx():
    r = note_offsets({'tanbumpyx': 0.5}, np.array([0]), np.array([20.0]),
                     t_now=0.0, beat_now=0.0, keycount=4)
    assert r.dx[0] != 0.0


# --- companion sweep (drunk/tipsy/tornado/bumpy) --------------------

def test_drunk_speed_offset_period_companions():
    cols = np.array([0])
    y = np.array([48.0])
    base = ae.drunk_x(1.0, cols, y, 1.5, 4)
    speed = ae.drunk_x(1.0, cols, y, 1.5, 4, speed=0.5)
    offset = ae.drunk_x(1.0, cols, y, 1.5, 4, offset=0.3)
    period = ae.drunk_x(1.0, cols, y, 1.5, 4, period=0.4)
    assert speed[0] != base[0]
    # offset scales the per-column term; at col 0 it multiplies 0 -> no change.
    assert offset[0] == pytest.approx(base[0])
    assert period[0] != base[0]


def test_drunk_companions_via_note_offsets():
    cols = np.array([0, 1])
    y = np.array([40.0, 40.0])
    base = note_offsets({'drunk': 1.0}, cols, y, t_now=1.0, beat_now=0.0,
                        keycount=4).dx
    withspeed = note_offsets({'drunk': 1.0, 'drunkspeed': 1.0}, cols, y,
                             t_now=1.0, beat_now=0.0, keycount=4).dx
    assert not np.allclose(base, withspeed)


def test_tipsy_speed_offset_companions():
    cols = np.array([1])
    base = ae.tipsy_y(1.0, cols, 0.5)
    speed = ae.tipsy_y(1.0, cols, 0.5, speed=1.0)
    offset = ae.tipsy_y(1.0, cols, 0.5, offset=0.5)
    assert speed[0] != base[0]
    assert offset[0] != base[0]


def test_tornado_offset_period_companions():
    cols = np.array([0])
    y = np.array([60.0])
    base = ae.tornado_x(1.0, cols, y, 4)
    period = ae.tornado_x(1.0, cols, y, 4, period=0.5)
    offset = ae.tornado_x(1.0, cols, y, 4, offset=30.0)
    assert period[0] != base[0]
    assert offset[0] != base[0]


def test_bumpy_offset_period_companions():
    y = np.array([20.0])
    base = ae.bumpy_z(1.0, y)
    offset = ae.bumpy_z(1.0, y, offset=0.5)
    period = ae.bumpy_z(1.0, y, period=0.5)
    assert offset[0] != base[0]
    assert period[0] != base[0]


# --- beat siblings (beaty / beatz) + companions ---------------------

def test_beat_x_shift_shape():
    # factor at beat_now=0 is 20; shift = factor*sin(yoff/15 + pi/2).
    # yoff=0 => sin(pi/2)=1 => 20.
    out = ae.beat_x(1.0, np.array([0.0]), 0.0)
    assert out[0] == pytest.approx(20.0)


def test_beaty_matches_beat_shape_on_y():
    r = note_offsets({'beaty': 1.0}, np.array([0]), np.array([0.0]),
                     t_now=0.0, beat_now=0.0, keycount=4)
    assert r.dy[0] == pytest.approx(20.0)


def test_beatz_reprojects_to_zoom():
    # beatz pushes z; zoom = 1 + z/480. yoff=0, factor 20 => z=20 => 1+20/480.
    r = note_offsets({'beatz': 1.0}, np.array([0]), np.array([0.0]),
                     t_now=0.0, beat_now=0.0, keycount=4)
    assert r.zoom[0] == pytest.approx(1.0 + 20.0 / 480.0)


def test_beat_mult_companion_speeds_pulse():
    # beatmult doubles the beat phase, changing the factor.
    plain = ae.beat_factor(0.0)
    fast = ae.beat_factor(0.0, mult=1.0)
    assert plain != pytest.approx(fast)


def test_beat_offset_companion_shifts_phase():
    # beat_now = 1.0 => phase frac 0.2 (in the active 0..0.5 window); a small
    # offset shifts the phase to a different active amount.
    plain = ae.beat_factor(1.0)
    shifted = ae.beat_factor(1.0, offset=0.1)
    assert plain != pytest.approx(shifted)
    assert plain != 0.0 and shifted != 0.0


# --- attenuate / parabola -------------------------------------------

def test_parabola_quadratic():
    # percent*(yoff/AS)^2. yoff=128 => (2)^2 = 4.
    out = ae.parabola(1.0, np.array([128.0]))
    assert out[0] == pytest.approx(4.0)


def test_attenuate_scaled_by_column_xoffset():
    # percent*(yoff/AS)^2*(xoff/AS). 4k col 0 xoff = -96, yoff = 64:
    # 1 * (1)^2 * (-96/64) = -1.5.
    out = ae.attenuate(1.0, np.array([0]), np.array([64.0]), 4)
    assert out[0] == pytest.approx(-1.5)
    # center-ish columns flip sign; col 3 xoff = +96 => +1.5.
    out3 = ae.attenuate(1.0, np.array([3]), np.array([64.0]), 4)
    assert out3[0] == pytest.approx(1.5)


def test_attenuate_x_y_z_route_to_axes():
    y = np.array([64.0])
    r = note_offsets({'attenuatex': 1.0, 'attenuatey': 1.0, 'attenuatez': 1.0},
                     np.array([0]), y, t_now=0.0, beat_now=0.0, keycount=4)
    assert r.dx[0] == pytest.approx(-1.5)
    assert r.dy[0] == pytest.approx(-1.5)
    # z push -1.5 px -> zoom 1 + (-1.5)/480.
    assert r.zoom[0] == pytest.approx(1.0 - 1.5 / 480.0)


def test_parabola_x_y_z_route_to_axes():
    y = np.array([128.0])
    r = note_offsets({'parabolax': 1.0, 'parabolay': 1.0, 'parabolaz': 1.0},
                     np.array([0]), y, t_now=0.0, beat_now=0.0, keycount=4)
    assert r.dx[0] == pytest.approx(4.0)
    assert r.dy[0] == pytest.approx(4.0)
    assert r.zoom[0] == pytest.approx(1.0 + 4.0 / 480.0)


# --- xmode / confusionoffset ----------------------------------------

def test_xmode_shoves_by_y_offset():
    # single-side field: dx = percent * y_offset.
    r = note_offsets({'xmode': 0.5}, np.array([0]), np.array([200.0]),
                     t_now=0.0, beat_now=0.0, keycount=4)
    assert r.dx[0] == pytest.approx(100.0)


def test_confusion_offset_via_note_offsets():
    # confusionoffset alone spins the whole field a constant amount.
    r = note_offsets({'confusionoffset': 0.5}, np.array([0]), np.array([50.0]),
                     t_now=0.0, beat_now=3.0, keycount=4)
    assert r.rotation_deg[0] == pytest.approx(0.5 * 180.0 / math.pi)


# --- new-mod vectorization parity -----------------------------------

def test_new_mods_batch_equals_scalar_loop():
    p = {'tandrunk': 0.4, 'tandrunkspeed': 0.2, 'tantornado': 0.3,
         'tantipsy': 0.5, 'beaty': 0.6, 'beatz': 0.4, 'beatmult': 1.0,
         'attenuatex': 0.3, 'attenuatey': 0.2, 'attenuatez': 0.25,
         'parabolax': 0.2, 'bumpyx': 0.3, 'bumpyxoffset': 0.4,
         'tandigital': 0.3, 'tandigitalsteps': 3.0, 'pulseouter': 0.5,
         'shrinkmult': 20.0, 'shrinklinear': 0.5, 'drunk': 0.4,
         'drunkperiod': 0.3, 'tornado': 0.3, 'tornadooffset': 10.0}
    cols = np.array([0, 1, 2, 3, 0, 2])
    y = np.array([120.0, 83.0, 41.0, 13.0, -30.0, 305.0])
    beats = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    batch = note_offsets(p, cols, y, t_now=1.7, beat_now=3.3, keycount=4,
                         note_beats=beats)
    for i in range(len(cols)):
        one = note_offsets(p, cols[i:i + 1], y[i:i + 1], t_now=1.7,
                           beat_now=3.3, keycount=4, note_beats=beats[i:i + 1])
        assert batch.dx[i] == pytest.approx(one.dx[0])
        assert batch.dy[i] == pytest.approx(one.dy[0])
        assert batch.zoom[i] == pytest.approx(one.zoom[0])
        assert batch.alpha_mult[i] == pytest.approx(one.alpha_mult[0])


def test_new_mods_determinism():
    p = {'tandrunk': 0.5, 'beaty': 0.3, 'pulseouter': 0.4, 'boomerang': 0.5}
    cols = np.array([0, 1, 2, 3])
    y = np.array([100.0, 200.0, 300.0, 400.0])
    a = note_offsets(p, cols, y, t_now=2.5, beat_now=4.0, keycount=4)
    b = note_offsets(p, cols, y, t_now=2.5, beat_now=4.0, keycount=4)
    np.testing.assert_array_equal(a.dx, b.dx)
    np.testing.assert_array_equal(a.zoom, b.zoom)
    np.testing.assert_array_equal(a.alpha_mult, b.alpha_mult)


# --- hold-body warp (item 39) ----------------------------------------

class _LnNotes:
    def __init__(self, n, ln_tail_times):
        self.noterows_list = list(range(n))
        self.ln_tail_times = np.asarray(ln_tail_times, dtype=np.float64)


class _LnPlayer:
    def __init__(self, cols, keycount, ln_tail_times):
        self.columns = np.asarray(cols, dtype=np.int64)
        self.keycount = keycount
        self.notes = _LnNotes(len(cols), ln_tail_times)


class _LnCtx:
    """Fake ctx with LN tail y, lane_x, and a hold_body_samples slot."""
    def __init__(self, player, heads, tails, judge_y, chart_h, lane_w=ae.ARROW_SIZE):
        self.player = player
        self.candidates = list(range(len(heads)))
        self.t_now = 0.0
        self.lane_w = lane_w
        self.judge_y = judge_y
        self.chart_rect = (0.0, 0.0, 400.0, float(chart_h))
        self.candidate_head_y = np.asarray(heads, dtype=np.float64)
        self.candidate_tail_y = np.asarray(tails, dtype=np.float64)
        self.candidate_press_y = np.asarray(heads, dtype=np.float64)

    def lane_x(self, col):
        return 100.0 + col * self.lane_w


def test_hold_body_samples_absent_without_mods():
    # No dx-producing channel active => rect fallback: no samples stashed.
    player = _LnPlayer([0], 1, [1.0])
    ctx = _LnCtx(player, [40.0], [200.0], judge_y=300, chart_h=600)
    _mods([ModEvent(0.0, 1.0, -1, 'reverse')]).apply(ctx)
    assert getattr(ctx, 'hold_body_samples', None) is None


def test_hold_body_samples_none_for_taps():
    # A tap (NaN ln_tail_time) never gets a body polyline even with drunk on.
    player = _LnPlayer([0], 1, [float('nan')])
    ctx = _LnCtx(player, [40.0], [float('nan')], judge_y=300, chart_h=600)
    _mods([ModEvent(0.0, 1.2, -1, 'drunk')]).apply(ctx)
    assert getattr(ctx, 'hold_body_samples', None) is None


def test_hold_body_passes_through_head_and_tail():
    # Under drunk the body's first sample sits at the head, the last at the
    # tail (both in x AND y), so the bent strip stays attached. reverse=1
    # pins the native (downscroll) space so head/tail y are untouched.
    player = _LnPlayer([1], 4, [5.0])
    ctx = _LnCtx(player, [50.0], [400.0], judge_y=300, chart_h=600)
    _mods([ModEvent(0.0, 1.0, -1, 'reverse'),
           ModEvent(0.0, 1.2, -1, 'drunk')]).apply(ctx)
    xs, ys = ctx.hold_body_samples[0]
    assert ys[0] == pytest.approx(ctx.candidate_head_y[0])
    assert ys[-1] == pytest.approx(ctx.candidate_tail_y[0])
    # x at the endpoints matches the head's own displaced x (lane_x + dx).
    head_x = ctx.lane_x(1) + ctx.candidate_dx[0]
    assert xs[0] == pytest.approx(head_x)


def test_hold_body_bends_under_drunk():
    # A straight rect would keep xs constant; drunk warps the body so at
    # least one interior sample's x differs from the endpoints' x.
    player = _LnPlayer([2], 4, [6.0])
    ctx = _LnCtx(player, [40.0], [500.0], judge_y=300, chart_h=600)
    _mods([ModEvent(0.0, 1.0, -1, 'reverse'),
           ModEvent(0.0, 1.5, -1, 'drunk')]).apply(ctx)
    xs, _ys = ctx.hold_body_samples[0]
    assert len(xs) >= 3
    assert not np.allclose(xs, xs[0])


def test_hold_body_sample_x_matches_direct_note_offsets():
    # Each body sample's dx is exactly note_offsets(drunk) at that sample's
    # engine y_offset. reverse=1 keeps native space so y_offset = judge_y -
    # screen_y (scale = lane_w/64 = 1 here). Verify one interior sample.
    player = _LnPlayer([0], 4, [4.0])
    lane_w = ae.ARROW_SIZE
    ctx = _LnCtx(player, [100.0], [300.0], judge_y=300, chart_h=600,
                 lane_w=lane_w)
    _mods([ModEvent(0.0, 1.0, -1, 'reverse'),
           ModEvent(0.0, 1.3, -1, 'drunk')]).apply(ctx)
    xs, ys = ctx.hold_body_samples[0]
    k = len(ys) // 2
    y_off = (ctx.judge_y - ys[k]) / (lane_w / ae.ARROW_SIZE)
    off = note_offsets({'drunk': 1.3}, np.array([0]), np.array([y_off]),
                       t_now=0.0, beat_now=0.0, keycount=4,
                       note_beats=np.array([0.0]))
    expect_x = ctx.lane_x(0) + off.dx[0] * (lane_w / ae.ARROW_SIZE)
    assert xs[k] == pytest.approx(expect_x, abs=1e-6)


def test_hold_body_batches_multiple_holds():
    # Two holds in one apply => two polylines, each attached to its own
    # head/tail (the batched note_offsets call splits back correctly).
    player = _LnPlayer([0, 3], 4, [4.0, 4.0])
    ctx = _LnCtx(player, [40.0, 80.0], [300.0, 450.0], judge_y=300,
                 chart_h=600)
    _mods([ModEvent(0.0, 1.0, -1, 'reverse'),
           ModEvent(0.0, 1.2, -1, 'drunk')]).apply(ctx)
    assert set(ctx.hold_body_samples.keys()) == {0, 1}
    for pos in (0, 1):
        xs, ys = ctx.hold_body_samples[pos]
        assert ys[0] == pytest.approx(ctx.candidate_head_y[pos])
        assert ys[-1] == pytest.approx(ctx.candidate_tail_y[pos])
