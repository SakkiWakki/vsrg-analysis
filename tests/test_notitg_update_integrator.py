"""Per-frame Update integrator: window parsing, sampling mirrors, the
screen-expr resolver, the field-copy composition, and a guarded
end-to-end pass against the real gat pilot."""
from pathlib import Path

import pytest

pytest.importorskip('lupa')

from analysis.games.notitg import update_integrator
from analysis.games.notitg.recording_actor import (
    RecordingActor, _resolve_screen_expr)
from analysis.player.render.effects.timeline import EventTimeline, Keyframe

_GAT_SM = Path('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/'
               'UKSRT8/5. gat/gat.sm')


# -- window parsing -------------------------------------------------------

def test_live_windows_merge_and_default_end():
    body = ('%function(self) if perframe(10,20) then end '
            'if perframe(18,30) then end if perframe(50) then end end')
    windows = update_integrator._live_windows(body)
    # (10,20) and (18,30) overlap -> merged; perframe(50) defaults to
    # a one-beat [50,51] window.
    assert windows == [(10.0, 30.0), (50.0, 51.0)]


def test_live_windows_ignore_commented_drivers():
    body = ('%function(self) if perframe(10,20) then end '
            '--[[ if perframe(100,200) then end ]] '
            '-- if perframe(300,400) then end\n end')
    # only the live driver bounds the range; commented blocks are dropped.
    assert update_integrator._live_windows(body) == [(10.0, 20.0)]


def test_live_windows_from_raw_beat_guards():
    # gat 2's Update body dispatches sections with `if beat > a and beat < b`
    # guards and ZERO perframe() calls; the window set must come from the
    # guards or the whole body never integrates.
    body = ('%function(self) '
            'if beat > 0 and beat < 127 then intro(beat) end '
            'if beat>127 and beat<352 then revolt(beat) end '
            'if beat > 601 and beat < 760 then afthell(beat) end end')
    # 0-127 and 127-352 are adjacent -> merged.
    assert update_integrator._live_windows(body) == [(0.0, 352.0),
                                                     (601.0, 760.0)]


def test_beat_guard_over_a_variable_is_not_a_window():
    # a guard whose bound is a Lua variable cannot resolve statically; it is
    # skipped, not guessed (no phantom window).
    body = '%function(self) if beat > a and beat < b then x() end end'
    assert update_integrator._live_windows(body) == []


def test_perframe_and_beat_guards_union():
    body = ('%function(self) if perframe(10,20) then end '
            'if beat > 40 and beat < 60 then end end')
    assert update_integrator._live_windows(body) == [(10.0, 20.0),
                                                     (40.0, 60.0)]


def test_beat_inverter_round_trips_a_linear_clock():
    # to_seconds beat -> time at a constant 2 s/beat; the inverter should
    # recover the beat from the time.
    to_seconds = lambda beat: beat * 2.0
    to_beats = update_integrator._beat_inverter([(0.0, 100.0)], to_seconds)
    for beat in (0.0, 12.5, 40.0, 100.0):
        assert to_beats(to_seconds(beat)) == pytest.approx(beat, abs=0.05)


# -- screen-expr resolver -------------------------------------------------

def test_resolve_screen_constant_and_arithmetic():
    assert _resolve_screen_expr('SCREEN_CENTER_X') == 320.0
    assert _resolve_screen_expr('-SCREEN_WIDTH/2') == -320.0
    assert _resolve_screen_expr('SCREEN_HEIGHT*0.55') == pytest.approx(264.0)
    assert _resolve_screen_expr('112*(SCREEN_HEIGHT/480)') == pytest.approx(112.0)
    assert _resolve_screen_expr('not_a_screen_thing') is None


# -- sampling mirror ------------------------------------------------------

def test_sampling_mirror_reads_baseline_timeline_until_poked():
    actor = RecordingActor(clock=0.0)
    # a compiled x tween 0 -> 100 over [0, 10]
    actor.poke('linear', ['10'])
    actor.poke('x', ['100'])

    clock = [0.0]
    actor.begin_sampling(clock)
    # before the pass pokes x, get(x) samples the baseline timeline.
    clock[0] = 5.0
    assert actor.get('x') == pytest.approx(50.0)
    clock[0] = 10.0
    assert actor.get('x') == pytest.approx(100.0)

    # once the pass pokes x it becomes a live accumulator; get(x) returns
    # the accumulated value, NOT the baseline sample.
    actor.reset_clock(10.0)
    actor.poke('addx', ['5'])          # 100 (baseline) + 5
    assert actor.get('x') == pytest.approx(105.0)
    actor.poke('addx', ['5'])          # accumulates on the live value
    assert actor.get('x') == pytest.approx(110.0)
    actor.end_sampling()


def test_sampling_mirror_off_by_default_reads_last_set():
    actor = RecordingActor(clock=0.0)
    actor.poke('x', ['42'])
    # no begin_sampling: get is the plain last-set snapshot.
    assert actor.get('x') == 42.0
    assert actor.get('y') == 0.0       # rest for an untouched prop


def test_live_poke_resets_a_live_accumulator_without_recording():
    # A per-frame driver accumulating y (`addy`) - the toss-quad fall -
    # runs away until a frozen reset message re-anchors it. live_poke
    # resets the running value so the next read sees the anchor, and adds
    # no keyframe to the compiled stream (the replay owns the timeline).
    actor = RecordingActor(clock=0.0)
    clock = [0.0]
    actor.begin_sampling(clock)
    actor.poke('addy', ['10'])         # tick 1: y = 10 (now a live prop)
    actor.poke('addy', ['10'])         # tick 2: y = 20
    assert actor.get('y') == pytest.approx(20.0)
    before = len(actor.keyframes().get('y', []))

    actor.live_poke('y', ['100'])      # the frozen reset re-anchors y
    assert actor.get('y') == pytest.approx(100.0)
    actor.poke('addy', ['10'])         # tick 3 accumulates on the anchor
    assert actor.get('y') == pytest.approx(110.0)
    # the reset added no keyframe (only the three real ticks did).
    assert len(actor.keyframes().get('y', [])) == before + 1
    actor.end_sampling()


def test_live_poke_leaves_an_unpoked_property_on_its_baseline_curve():
    # A quad the per-frame body only READS (gat's slam quads sampled by the
    # split loop) is reset by its own message tween, whose curve the replay
    # captured. live_poke must NOT freeze it to an endpoint - the property
    # was never a live accumulator, so it keeps sampling the baseline.
    actor = RecordingActor(clock=0.0)
    actor.poke('linear', ['10'])
    actor.poke('x', ['100'])           # baseline tween 0 -> 100 over [0, 10]
    clock = [5.0]
    actor.begin_sampling(clock)

    actor.live_poke('x', ['0'])        # a frozen tween-endpoint poke
    # x was never poked by the pass, so it stays on the baseline curve.
    assert actor.get('x') == pytest.approx(50.0)
    actor.end_sampling()


# -- field-copy composition ----------------------------------------------

def test_sum_timeline_adds_children():
    from analysis.games.notitg.modfile import _SumTimeline
    a = EventTimeline([Keyframe(0.0, (10.0,), 0.0, 0)], rest=(0.0,))
    b = EventTimeline([Keyframe(0.0, (5.0,), 0.0, 0)], rest=(0.0,))
    total = _SumTimeline((a, b))
    assert total.sample(1.0) == (15.0,)


# -- end to end (guarded on the local pilot) ------------------------------









def _iter(elements):
    for element in elements:
        yield element
        yield from _iter(element.children)
