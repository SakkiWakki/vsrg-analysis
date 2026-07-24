"""Tree-order static-scene compiler (drawable_doc): the STATIC field-instance
topology compiled into a channel-backed DrawableDoc.

The core deliverable is the parity harness: a SYNTHETIC compiled chart - a
hand-built provider of `field_compose.instance` dicts - is compiled into a doc
with `build_static_doc`, and the evaluator's per-frame BLIT stream is compared,
in order, against `NotitgFieldInstances.at` (the same effect the renderer
samples). Order, alpha (1e-3) and the design-space mat3 (1e-2 relative) must
agree at N sample times.

`export_channel` is exercised directly: EXACT reconstruction for an
`EventTimeline` (instants + linear ramps), and the documented dense-sampling
interim for a curved ease and a lazy (`.sample`-only) curve.

A skippable gat 2 smoke compiles the real chart via the sim, polls the lazy
provider, builds the static doc, and prints the parity report.
"""
import math
import os

import numpy as np
import pytest

sn = pytest.importorskip('storyboard_native')

from analysis.games.notitg import drawable_doc as dd
from analysis.games.notitg import field_compose as fc
from analysis.games.notitg.field_instances import (
    NotitgFieldInstances, PlayerFieldsSpec)
from analysis.player.render.effects.timeline import EventTimeline, Keyframe


_EASE_LINEAR = 0
_EASE_IN_QUAD = 2


def _instant(value, t=0.0):
    """One immediate keyframe holding `value` (a scalar) from `t`."""
    return [Keyframe(float(t), (float(value),), 0.0, _EASE_LINEAR)]


def _link(**pokes):
    """A field link resting at the SM defaults with `pokes` written as immediate
    keyframes at t=0."""
    return fc.link_timelines({p: _instant(v) for p, v in pokes.items()})


def _compiled(instances, base_hidden=None, player_fields=None):
    payload = {'field_instances': list(instances), 'base_field_hidden': base_hidden}
    if player_fields is not None:
        payload['player_fields'] = player_fields
    return payload


def _effect(compiled):
    return NotitgFieldInstances(
        compiled['field_instances'],
        base_hidden=compiled.get('base_field_hidden'),
        player_fields=compiled.get('player_fields'))


# --------------------------------------------------------------------------
# export_channel - the shared primitive
# --------------------------------------------------------------------------

def _channel_sampler(timeline, t0, t1, prop=0):
    """Round-trip a timeline through export_channel + a real doc channel, and
    return a `sample(t)` reading the native channel back (via a linkless item's
    x lane). This is the exact substrate the static doc feeds link props into."""
    ts, vals, durs, rest = dd.export_channel(timeline, t0, t1, prop)
    builder = sn.DocBuilder(640.0, 480.0)
    chan_id = builder.channel([float(v) for v in ts], [float(v) for v in vals],
                              [float(v) for v in durs], float(rest))
    builder.item(0, sn.SRC_FILL, 0, x_id=chan_id, x_rest=rest)
    evaluator = builder.finish()

    def sample(t):
        u_bytes, f_bytes, _uf, n = evaluator.frame(float(t))
        u = np.frombuffer(u_bytes, dtype=np.uint32).reshape(n, evaluator.u_stride)
        f = np.frombuffer(f_bytes, dtype=np.float32).reshape(n, evaluator.f_stride)
        for i in range(n):
            if u[i, 0] == dd._OP_BLIT:
                return float(f[i, 2])  # the x lane (tx) of a linkless item
        return None

    return sample


def test_export_channel_event_timeline_exact_for_instants_and_linear():
    # Instant to 10 at t=1, then a linear ramp 10 -> 30 over [2, 3], holding.
    tl = EventTimeline(
        [Keyframe(1.0, (10.0,), 0.0, _EASE_LINEAR),
         Keyframe(2.0, (30.0,), 1.0, _EASE_LINEAR)],
        rest=(5.0,))
    native = _channel_sampler(tl, 0.0, 6.0)
    for t in np.linspace(0.0, 6.0, 61):
        assert abs(native(t) - tl.sample(t)[0]) < 1e-3, t


def test_export_channel_curved_ease_densified_within_tolerance():
    # A curved (in-quad) ease cannot be one linear breakpoint; export_channel
    # densifies it at 1/30s. The reconstruction tracks the curve to a small
    # design-pixel tolerance (NOT bit-exact - the documented interim).
    tl = EventTimeline([Keyframe(1.0, (0.0,), 0.0, _EASE_LINEAR),
                        Keyframe(1.0, (100.0,), 2.0, _EASE_IN_QUAD)],
                       rest=(0.0,))
    native = _channel_sampler(tl, 0.0, 4.0)
    errs = [abs(native(t) - tl.sample(t)[0]) for t in np.linspace(1.0, 3.0, 41)]
    assert max(errs) < 0.5  # sub-visible, densified
    assert max(errs) > 0.0  # genuinely approximated, not bit-exact


def test_export_channel_lazy_curve_dense_sampled():
    # A duck-typed curve with only `.sample` (a LiveCurve/SegCurve stand-in) is
    # dense-sampled across [t0, t1]; no keyframe structure is assumed.
    class _Curve:
        _rest = (0.0,)

        def sample(self, t):
            return (math.sin(t),)

    native = _channel_sampler(_Curve(), 0.0, 6.0)
    errs = [abs(native(t) - math.sin(t)) for t in np.linspace(0.0, 6.0, 61)]
    assert max(errs) < 0.05  # dense 1/30s piecewise-linear over a sine


# --------------------------------------------------------------------------
# build_static_doc + parity harness
# --------------------------------------------------------------------------

def _proxy(name, player=1, **pokes):
    return fc.instance(name, 'proxy', player, [_link(**pokes)])


def _synthetic_scene():
    """A single-player scene covering every static kind: capture, player, an
    animated proxy, an aft sampler (with its slot), and a fill curtain."""
    cap = fc.instance('nodeX', 'capture', 0, [_link(x=320.0)])
    player = fc.player_instance(1, {'x': _instant(200.0), 'y': _instant(240.0)})
    proxy_link = fc.link_timelines({
        'x': [Keyframe(0.0, (100.0,), 0.0, _EASE_LINEAR),
              Keyframe(1.0, (500.0,), 3.0, _EASE_LINEAR)],
        'y': _instant(240.0),
        'scale_x': _instant(0.5), 'scale_y': _instant(0.5),
    })
    proxy = fc.instance('proxyA', 'proxy', 1, [proxy_link])
    aft = fc.instance('aftA', 'aft', 0, [_link(x=320.0, y=240.0)],
                      aft_order='post')
    aft['aft_node'] = 'nodeX'
    fill = fc.instance('fillA', 'fill', 0, [_link(x=320.0, y=240.0)])
    return [cap, player, proxy, aft, fill]


_SAMPLE_TIMES = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0)


def test_synthetic_parity_exact_over_sample_times():
    compiled = _compiled(_synthetic_scene())
    evaluator, id_maps, report = dd.build_static_doc(compiled)

    # Topology counts land as expected.
    assert report['captures'] == 1 and report['fills'] == 1
    assert report['aft'] == 1 and report['proxy'] == 2
    assert report['slots'] == 1 and report['fields'] == 1

    rep = dd.parity_report(evaluator, id_maps, _effect(compiled), _SAMPLE_TIMES)
    assert rep['all_ok'], dd.format_parity_report(rep)
    # The linear/instant transforms reconstruct to float precision; the one
    # curved-eased proxy stays well within the 1e-2 relative mat3 bar.
    assert rep['max_mat_err'] < 1e-2
    assert rep['max_alpha_err'] < 1e-3


def test_capture_emits_snapshot_not_a_blit():
    compiled = _compiled(_synthetic_scene())
    evaluator, id_maps, _report = dd.build_static_doc(compiled)
    u_bytes, _f, _uf, n = evaluator.frame(0.0)
    u = np.frombuffer(u_bytes, dtype=np.uint32).reshape(n, evaluator.u_stride)
    ops = [int(u[i, 0]) for i in range(n)]
    assert sn.OP_COPY in ops  # the capture became a Snapshot -> COPY
    # The capture is NOT among the BLITs (it draws nothing of its own).
    blit_srcs = [(int(u[i, 1]), int(u[i, 2])) for i in range(n)
                 if u[i, 0] == sn.OP_BLIT]
    slot_id = id_maps['slots']['nodeX']
    # The slot is BLITted by the aft sampler, but the capture itself is a COPY.
    assert (sn.SRC_DRAWABLE, slot_id) in blit_srcs


def test_z_group_run_wrapped_in_sort_span():
    # Two proxies in one z_group whose z ordering crosses over time; the
    # SortSpan must reorder their blits to match the effect's stable z sort.
    a = _proxy('a', x=100.0, y=240.0)
    b = _proxy('b', x=500.0, y=240.0)
    a['z_group'] = 'g'
    a['z_sort'] = EventTimeline(
        [Keyframe(0.0, (0.0,), 4.0, _EASE_LINEAR),
         Keyframe(4.0, (10.0,), 0.0, _EASE_LINEAR)], rest=(0.0,))
    b['z_group'] = 'g'
    b['z_sort'] = EventTimeline(
        [Keyframe(0.0, (10.0,), 4.0, _EASE_LINEAR),
         Keyframe(4.0, (0.0,), 0.0, _EASE_LINEAR)], rest=(10.0,))
    compiled = _compiled([a, b])
    evaluator, id_maps, report = dd.build_static_doc(compiled)
    assert report['z_groups'] == 1
    rep = dd.parity_report(evaluator, id_maps, _effect(compiled),
                           list(np.linspace(0.0, 5.0, 11)))
    assert rep['all_ok'], dd.format_parity_report(rep)


def test_base_hidden_gate_drops_the_base_field():
    # base_field_hidden rises then falls; the base-field blit must appear only
    # while the chart shows the real field (the inverted visible gate).
    hidden = EventTimeline([Keyframe(2.0, (1.0,), 0.0, _EASE_LINEAR),
                            Keyframe(4.0, (0.0,), 0.0, _EASE_LINEAR)],
                           rest=(0.0,))
    compiled = _compiled([_proxy('p', x=400.0, y=240.0)], base_hidden=hidden)
    evaluator, id_maps, _report = dd.build_static_doc(compiled)
    rep = dd.parity_report(evaluator, id_maps, _effect(compiled),
                           [0.0, 1.0, 2.5, 3.0, 4.5, 5.0])
    assert rep['all_ok'], dd.format_parity_report(rep)


def test_aft_flip_swaps_leaf_crop():
    # A flipped aft leaf swaps top/bottom crop; the composed mat3 + crop must
    # match the effect's flipped homography exactly.
    aft = fc.instance('aftA', 'aft', 0,
                      [_link(x=320.0, y=240.0, crop_top=0.1, crop_bottom=0.3)],
                      aft_order='post')
    aft['aft_node'] = 'n'
    cap = fc.instance('n', 'capture', 0, [_link(x=320.0)])
    compiled = _compiled([cap, aft])
    evaluator, id_maps, _report = dd.build_static_doc(compiled)
    rep = dd.parity_report(evaluator, id_maps, _effect(compiled), [0.0, 1.0])
    assert rep['all_ok'], dd.format_parity_report(rep)


def test_dual_player_fields_get_distinct_drawables():
    # Two player instances + a proxy of player 2: the dual path draws no base
    # original, and player 2's field routes to its own 'field2' drawable.
    p1 = fc.player_instance(1, {'x': _instant(160.0), 'y': _instant(240.0)})
    p2 = fc.player_instance(2, {'x': _instant(480.0), 'y': _instant(240.0)})
    pr2 = fc.instance('pr2', 'proxy', 2, [_link(x=300.0, y=240.0)])
    spec = PlayerFieldsSpec({2: object()})
    compiled = _compiled([p1, p2, pr2], player_fields=spec)
    evaluator, id_maps, _report = dd.build_static_doc(compiled)
    assert set(id_maps['fields']) == {'field', 'field2'}
    rep = dd.parity_report(evaluator, id_maps, _effect(compiled),
                           [0.0, 1.0, 5.0])
    assert rep['all_ok'], dd.format_parity_report(rep)


# --------------------------------------------------------------------------
# gat 2 smoke (skippable)
# --------------------------------------------------------------------------

_GAT2 = ('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/'
         'UKSRT9/5. getfucked2/get_fucked_2.sm')


@pytest.mark.skipif(not os.path.exists(_GAT2), reason='gat 2 chart not on disk')
def test_gat2_smoke_static_doc_parity():
    """Compile the real gat 2 chart via the sim, poll the lazy provider until
    the topology fills in (~20s sweep), build the static doc, and print the
    parity report against NotitgFieldInstances.at at a spread of sample times.

    The static doc's link chains are 2D-affine only, so instances driven by the
    camera-area math (rotation_x/y, quat, z, skew, fov) diverge BY DESIGN - the
    report SURFACES those rather than asserting zero; the assertion is only that
    the compile + evaluate + compare runs cleanly and produces a report."""
    import time

    from analysis.games.notitg.sim.producers import compile_via_sim

    compiled = compile_via_sim(_GAT2)
    assert compiled is not None
    provider = compiled.get('field_instances')
    assert provider is not None

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        current = list(provider() if callable(provider) else provider)
        if len(current) > 150:
            break
        time.sleep(0.5)
    count = len(list(provider() if callable(provider) else provider))
    print(f'[gat2] provider instance count: {count}')

    evaluator, id_maps, report = dd.build_static_doc(compiled)
    print(f'[gat2] static doc: drawables={evaluator.drawable_count()} '
          f'fields={report["fields"]} slots={report["slots"]} '
          f'captures={report["captures"]} fills={report["fills"]} '
          f'aft={report["aft"]} proxy={report["proxy"]} '
          f'z_groups={report["z_groups"]}')

    effect = _effect(compiled)
    sample_times = [50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0]
    rep = dd.parity_report(evaluator, id_maps, effect, sample_times)
    print('[gat2] ' + dd.format_parity_report(rep).replace('\n', '\n[gat2] '))
    # The harness must run end-to-end and produce a per-time report.
    assert len(rep['times']) == len(sample_times)
    assert all('n_blit' in r for r in rep['times'])
