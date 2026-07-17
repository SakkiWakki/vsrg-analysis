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

@pytest.mark.skipif(not _GAT_SM.exists(), reason='gat pilot not installed')
def test_gat_integration_populates_proxy_grid_and_per_frame_mods():
    from analysis.games.notitg.modfile import compile_modfile
    compiled = compile_modfile(str(_GAT_SM))
    assert compiled is not None

    # the 3x3 proxy grid per player -> 18 field copies sourced to the
    # player notefields (the t=42 gat_updateproxies scatter).
    grid = [c for c in compiled['field_copies']
            if c['name'].startswith('gat_proxy_')]
    assert len(grid) == 18
    assert {c['source'] for c in grid} == {'P1p', 'P2p'}

    # a grid copy scatters away from the screen centre at t=42 (the
    # accumulator drove it), not resting at identity.
    at_42 = [c['timelines']['x'].sample(42.0)[0] for c in grid]
    assert any(abs(x - 320.0) > 100.0 for x in at_42)

    # the per-frame drivers' ApplyGameCommand mods (the walking movey
    # family) landed as windowed events.
    def last_name(modstring):
        tokens = str(modstring).split(',')[-1].split()
        return tokens[-1] if tokens else ''
    perframe_moveys = [e for e in compiled['mod_events']
                       if e.get('apply_type') == 'perframe'
                       and last_name(e['modstring']).startswith('movey')]
    assert perframe_moveys


@pytest.mark.skipif(not _GAT_SM.exists(), reason='gat pilot not installed')
def test_gat_toss_proxies_stay_on_screen():
    # The toss section (perframe(784,848), t~245-260) integrates a
    # gravity fall per frame; the `Toss` message re-anchors each toss quad
    # so the copies arc instead of flying off. Without the frozen-reset
    # reaching the live mirror the y accumulator ran away to ~6000 against
    # a 480-tall design box - here the P1p..P4p copies stay bounded.
    from analysis.games.notitg.modfile import compile_modfile
    compiled = compile_modfile(str(_GAT_SM))
    proxies = [c for c in compiled['field_copies']
               if c['name'] in ('P1p', 'P2p', 'P3p', 'P4p')]
    assert proxies
    for t in range(245, 261):
        for copy in proxies:
            tl = copy['timelines']
            if tl['hidden'].sample(float(t))[0] >= 0.5:
                continue
            x = tl['x'].sample(float(t))[0]
            y = tl['y'].sample(float(t))[0]
            assert -700.0 <= x <= 1200.0, (copy['name'], t, x)
            assert -500.0 <= y <= 1100.0, (copy['name'], t, y)


@pytest.mark.skipif(not _GAT_SM.exists(), reason='gat pilot not installed')
def test_gat_aft_target_hidden_outside_showaft_window():
    # gat_aft_target is a full-screen AFT copy shown by ShowAFT (beat 128,
    # t~37.8) and hidden by HideAFT (beat 252, t~74.1). The compiled hidden
    # timeline must gate it to exactly that window - shown inside the grid
    # section, hidden before and after (so it is not a phantom copy at other
    # times).
    from analysis.games.notitg.modfile import compile_modfile
    compiled = compile_modfile(str(_GAT_SM))
    target = next(c for c in compiled['field_copies']
                  if c['name'] == 'gat_aft_target')
    hidden = target['timelines']['hidden']
    assert hidden.sample(50.0)[0] < 0.5     # inside the ShowAFT window
    assert hidden.sample(20.0)[0] >= 0.5    # before ShowAFT
    assert hidden.sample(120.0)[0] >= 0.5   # after HideAFT


@pytest.mark.skipif(not _GAT_SM.exists(), reason='gat pilot not installed')
def test_gat_integration_animates_shame_clones():
    from analysis.games.notitg.modfile import compile_modfile
    from analysis.player.render.storyboard.model import build_timelines

    compiled = compile_modfile(str(_GAT_SM))
    tree = compiled['tree']

    def find(name_asset_stem):
        for element in _iter(tree):
            asset = getattr(element, 'asset', None)
            if asset and name_asset_stem in Path(asset).stem.casefold():
                yield element

    # the shame dark-circle clones (perframe(469/502) drivers) gained an
    # animated alpha - they fade in during their sections, so at least one
    # dark-circle element carries a non-trivial alpha timeline.
    darks = list(find('darkcircle'))
    assert darks
    assert any(len(el.timelines['alpha']._kf) > 2 for el in darks)


def _iter(elements):
    for element in elements:
        yield element
        yield from _iter(element.children)
