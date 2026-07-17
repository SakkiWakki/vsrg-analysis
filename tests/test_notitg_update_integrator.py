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
