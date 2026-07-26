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
from types import SimpleNamespace

import numpy as np
import pytest

sn = pytest.importorskip('storyboard_native')

from analysis.games.notitg import drawable_doc as dd
from analysis.games.notitg import field_compose as fc
from analysis.games.notitg.field_instances import (
    NotitgFieldInstances, PlayerFieldsSpec)
from analysis.player.render.effects.timeline import EventTimeline, Keyframe
from analysis.player.render.storyboard.model import Element, build_timelines


_EASE_LINEAR = 0
_EASE_IN_QUAD = 2


def _instant(value, t=0.0):
    """One immediate keyframe holding `value` (a scalar) from `t`."""
    return [Keyframe(float(t), (float(value),), 0.0, _EASE_LINEAR)]


def _link(**pokes):
    """A field link resting at the SM defaults with `pokes` written as immediate
    keyframes at t=0."""
    return fc.link_timelines({p: _instant(v) for p, v in pokes.items()})


def _compiled(instances, base_hidden=None, player_fields=None, tree=None):
    payload = {'field_instances': list(instances), 'base_field_hidden': base_hidden}
    if player_fields is not None:
        payload['player_fields'] = player_fields
    if tree is not None:
        payload['tree'] = list(tree)
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
    ts, vals, durs, rest, eases = dd.export_channel(timeline, t0, t1, prop)
    builder = sn.DocBuilder(640.0, 480.0)
    chan_id = builder.channel(ts, vals, durs, float(rest), eases)
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



@pytest.fixture
def doc_elements(monkeypatch):
    """Pin the doc's storyboard-element items ON.

    They are the default now (`drawable_doc.elements_in_doc`), but these tests
    ARE the element path's contract, so they state the requirement rather than
    inheriting it - the flag exists to turn them OFF for differential testing,
    and a suite that silently followed it would stop testing this path."""
    monkeypatch.setenv('VSRG_DRAWABLE_ELEMENTS', '1')


@pytest.fixture
def captured_notefield(monkeypatch):
    """Pin the base field to the CAPTURED-notefield representation.

    Field-instance parity is defined against `NotitgFieldInstances.at`, whose
    model is one blit of the captured notefield. The default path draws the
    base field's notes as inline items instead (drawable_doc.notes_inline), so
    it legitimately has no such blit - these tests pin the representation the
    parity contract is written against."""
    monkeypatch.setenv('VSRG_DRAWABLE_NOTES', '0')


def test_synthetic_parity_exact_over_sample_times(captured_notefield):
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


def test_z_group_run_wrapped_in_sort_span(captured_notefield):
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


def test_base_hidden_gate_drops_the_base_field(captured_notefield):
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


def test_dual_player_fields_get_distinct_drawables(captured_notefield):
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


def test_inline_notes_players_feed_and_per_player_scopes_blit():
    # The default (inline-notes) path: a 'field'-scope consumer RE-RENDERS
    # the shared fed note items under its own chain (no capture-boxed
    # drawable to clip at), while a per-player 'field{N}' scope keeps the
    # capture blit - the feed carries player 1's items only.
    p1 = fc.player_instance(1, {'x': _instant(160.0), 'y': _instant(240.0)})
    p2 = fc.player_instance(2, {'x': _instant(480.0), 'y': _instant(240.0)})
    pr2 = fc.instance('pr2', 'proxy', 2, [_link(x=300.0, y=240.0)])
    spec = PlayerFieldsSpec({2: object()})
    compiled = _compiled([p1, p2, pr2], player_fields=spec)
    _evaluator, id_maps, _report = dd.build_static_doc(compiled)
    assert set(id_maps['fields']) == {'field2'}
    assert set(id_maps['note_feeds']) == {'field'}
    assert id_maps['notes_slot'] == id_maps['note_feeds']['field']


# --------------------------------------------------------------------------
# storyboard elements banded into the static doc
# --------------------------------------------------------------------------

_BACKGROUND_Z = -100
_PRE_FIELD_Z = -75


def _element(kind, z, *, asset=None, z_index=0, t_start=0.0, keyframes=None,
             children=(), **fields):
    """A synthetic storyboard Element at band `z`, resting at the SM defaults
    with `keyframes` (property -> Keyframe list) written on top."""
    return Element(
        kind=kind, z=z, z_index=z_index, t_start=t_start, t_end=float('inf'),
        anchor=(0.0, 0.0), origin=(0.5, 0.5),
        timelines=build_timelines(keyframes=keyframes or {}),
        asset=asset, children=tuple(children), **fields)


def _sprite(z, path, **kw):
    return _element('sprite', z, asset=path, **kw)


def test_an_element_draws_between_two_field_instances_by_tree_order(
        captured_notefield, doc_elements):
    # THE CASE BANDING CANNOT EXPRESS. A curtain quad sits between the AFT
    # node that captured the scene and the sampler that redraws it, and all
    # three are z=0 - so a below-band/above-band split can only put the quad
    # before both or after both. After both, it paints over the freeze, which
    # is what blacked out gat 1 at 5:48.
    first = _proxy('first', x=100.0, y=240.0)
    second = _proxy('second', x=500.0, y=240.0)
    first['tree_index'], second['tree_index'] = 10, 30
    middle = _sprite(0, '/tmp/curtain.png', keyframes={'x': _instant(320.0)})
    object.__setattr__(middle, 'tree_index', 20)

    compiled = _compiled([first, second], tree=[middle])
    evaluator, id_maps, _report = dd.build_static_doc(compiled)

    kinds = [k for (k, _sid, _m, _a) in dd._blit_stream(evaluator, 0.0)]
    image_at = kinds.index(dd._SRC_IMAGE)
    drawables = [i for i, k in enumerate(kinds) if k == sn.SRC_DRAWABLE]
    assert drawables, 'expected the field instances to blit'
    assert drawables[0] < image_at < drawables[-1], (
        f'element must land BETWEEN the instances, got kinds={kinds}')


def test_below_and_above_bands_split_around_the_field_stream(captured_notefield,
                                                             doc_elements):
    # A background sprite (z<0) and a foreground sprite (z>=0) around one proxy:
    # the below sprite draws first, the field instance next, the above last.
    below = _sprite(_BACKGROUND_Z, '/tmp/bg.png',
                    keyframes={'x': _instant(100.0), 'y': _instant(100.0)})
    above = _sprite(50, '/tmp/fg.png',
                    keyframes={'x': _instant(500.0), 'y': _instant(400.0)})
    compiled = _compiled([_proxy('p', x=320.0, y=240.0)],
                         tree=[below, above])
    evaluator, id_maps, report = dd.build_static_doc(compiled)

    assert report['elements_below'] == 1 and report['elements_above'] == 1
    # Two distinct image ids collected, mapped to their absolute paths.
    assert set(id_maps['images'].values()) == {'/tmp/bg.png', '/tmp/fg.png'}
    assert report['images'] == 2

    stream = dd._blit_stream(evaluator, 0.0)
    kinds = [k for (k, _sid, _m, _a) in stream]
    # image, then the field-instance blit (a drawable), then image.
    assert kinds[0] == dd._SRC_IMAGE
    assert kinds[-1] == dd._SRC_IMAGE
    assert sn.SRC_DRAWABLE in kinds[1:-1]


def test_field_instance_subsequence_unchanged_by_elements(captured_notefield,
                                                          doc_elements):
    # The field-instance parity must hold IDENTICALLY whether or not storyboard
    # elements are present: the harness compares the field-instance subsequence.
    scene = _synthetic_scene()
    plain = _compiled(scene)
    with_elems = _compiled(_synthetic_scene(),
                           tree=[_sprite(_BACKGROUND_Z, '/tmp/a.png',
                                         keyframes={'x': _instant(50.0)}),
                                 _sprite(20, '/tmp/b.png',
                                         keyframes={'x': _instant(600.0)})])

    ev_plain, im_plain, _r = dd.build_static_doc(plain)
    ev_elem, im_elem, rep_elem = dd.build_static_doc(with_elems)

    # Parity still passes with elements present (subsequence comparison).
    parity = dd.parity_report(ev_elem, im_elem, _effect(with_elems), _SAMPLE_TIMES)
    assert parity['all_ok'], dd.format_parity_report(parity)

    # And the field subsequence is byte-for-byte the plain doc's blit stream.
    for t in _SAMPLE_TIMES:
        plain_stream = dd._blit_stream(ev_plain, t)
        sub = dd._field_blit_subsequence(ev_elem, t)
        assert len(sub) == len(plain_stream)
        for (pk, pid, pm, pa), (sk, sid, sm, sa) in zip(plain_stream, sub):
            assert (pk, pid) == (sk, sid)
            assert abs(pa - sa) < 1e-6
            assert np.allclose(pm, sm, atol=1e-6)
    assert rep_elem['elements_below'] == 1 and rep_elem['elements_above'] == 1


def test_within_band_sorted_by_z_then_index_then_start(doc_elements):
    # Three below-band sprites out of z/z_index/t_start order; the emitted image
    # order must be the renderer's (z, z_index, t_start) sort.
    a = _sprite(-100, '/tmp/a.png', z_index=1, keyframes={'x': _instant(10.0)})
    b = _sprite(-100, '/tmp/b.png', z_index=0, keyframes={'x': _instant(20.0)})
    c = _sprite(-75, '/tmp/c.png', z_index=0, keyframes={'x': _instant(30.0)})
    compiled = _compiled([], tree=[a, b, c])
    evaluator, id_maps, report = dd.build_static_doc(compiled)

    # Sorted order: b(z=-100,zi=0), a(z=-100,zi=1), c(z=-75).
    path_by_id = id_maps['images']
    image_blits = [(sid) for (k, sid, _m, _a) in dd._blit_stream(evaluator, 0.0)
                   if k == dd._SRC_IMAGE]
    ordered_paths = [path_by_id[sid] for sid in image_blits]
    assert ordered_paths == ['/tmp/b.png', '/tmp/a.png', '/tmp/c.png']
    assert report['elements_below'] == 3 and report['elements_above'] == 0


def test_unsupported_kinds_skipped_with_per_kind_counts(doc_elements):
    # Video and an asset-less sprite are skipped, each tallied by kind. Rects
    # draw as tinted fills and text draws as a deferred raster, so both emit.
    tree = [
        _sprite(_BACKGROUND_Z, '/tmp/real.png', keyframes={'x': _instant(1.0)}),
        _element('rect', _BACKGROUND_Z),
        _element('rect', -50),
        _element('text', 10, text='hi'),
        _element('video', -100),
        _sprite(-100, None),  # a Sprite with no texture -> 'untextured'
    ]
    compiled = _compiled([], tree=tree)
    evaluator, id_maps, report = dd.build_static_doc(compiled)

    assert report['elements_below'] == 3 and report['elements_above'] == 1
    assert report['element_skips'] == {'video': 1, 'untextured': 1}
    assert report['images'] == 1


def test_group_children_band_by_the_top_level_z(doc_elements):
    # A group draws nothing itself (tallied as a skip); its sprite children
    # emit as items but band by the TOP-LEVEL element's z - the legacy
    # StoryboardEffect bands top-level elements only, and the compiler puts a
    # hoisted subtree's band z on its root (modfile._with_z), so reading the
    # leaf's own z would strand a background hoist above the field.
    child_a = _sprite(-100, '/tmp/ga.png', keyframes={'x': _instant(5.0)})
    child_b = _sprite(30, '/tmp/gb.png', keyframes={'x': _instant(6.0)})
    group = _element('group', 0, children=[child_a, child_b])
    compiled = _compiled([], tree=[group])
    evaluator, id_maps, report = dd.build_static_doc(compiled)

    assert report['elements_below'] == 0  # group z=0 bands both above
    assert report['elements_above'] == 2
    assert report['element_skips'].get('group') == 1
    assert set(id_maps['images'].values()) == {'/tmp/ga.png', '/tmp/gb.png'}

    hoisted = _element('group', _BACKGROUND_Z,
                       children=[_sprite(0, '/tmp/bg.png',
                                         keyframes={'x': _instant(1.0)})])
    compiled = _compiled([], tree=[hoisted])
    _evaluator, _id_maps, report = dd.build_static_doc(compiled)
    assert report['elements_below'] == 1  # the background hoist


def test_sheet_sprite_gets_a_frame_channel(doc_elements):
    # A frame-animated sheet sprite carries a frame lane that steps through its
    # states over time; a plain 1x1 sprite does not.
    sheet = _element('frames', -100, z_index=0, t_start=0.0,
                     frames=('/tmp/s.png',), sheet_cols=2, sheet_rows=1,
                     sheet_states=((0, 0.5), (1, 0.5)),
                     keyframes={'x': _instant(320.0)})
    compiled = _compiled([], tree=[sheet])
    evaluator, id_maps, report = dd.build_static_doc(compiled)

    assert report['elements_below'] == 1
    # The frame lane advances 0 -> 1 across the state list (evaluator carries a
    # frame value per image blit; sampling before/after the first delay differs).
    curve = dd._FrameCurve(sheet)
    assert curve.sample(0.1)[0] == 0.0
    assert curve.sample(0.6)[0] == 1.0


# --------------------------------------------------------------------------
# gat 2 smoke (skippable)
# --------------------------------------------------------------------------

_GAT2 = ('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/'
         'UKSRT9/5. getfucked2/get_fucked_2.sm')


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(_GAT2), reason='gat 2 chart not on disk')
def test_gat2_smoke_static_doc_parity():
    """Compile the real gat 2 chart via the sim, wait for the lazy compile's
    background upgrade to hand over the complete topology, build the static
    doc, and print the parity report against NotitgFieldInstances.at at a
    spread of sample times.

    The static doc's link chains are 2D-affine only, so instances driven by the
    camera-area math (rotation_x/y, quat, z, skew, fov) diverge BY DESIGN - the
    report SURFACES those rather than asserting zero; the assertion is only that
    the compile + evaluate + compare runs cleanly and produces a report."""
    from analysis.games.notitg.sim.producers import (
        compile_via_sim, wait_for_upgrade)

    compiled = compile_via_sim(_GAT2)
    assert compiled is not None
    provider = compiled.get('field_instances')
    assert provider is not None

    assert wait_for_upgrade(compiled)

    count = len(list(provider() if callable(provider) else provider))
    print(f'[gat2] provider instance count: {count}')

    evaluator, id_maps, report = dd.build_static_doc(compiled)
    print(f'[gat2] static doc: drawables={evaluator.drawable_count()} '
          f'fields={report["fields"]} slots={report["slots"]} '
          f'captures={report["captures"]} fills={report["fills"]} '
          f'aft={report["aft"]} proxy={report["proxy"]} '
          f'z_groups={report["z_groups"]}')
    print(f'[gat2] elements: below={report["elements_below"]} '
          f'above={report["elements_above"]} images={report["images"]} '
          f'skips-by-kind={report["element_skips"]}')

    effect = _effect(compiled)
    sample_times = [50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0]
    rep = dd.parity_report(evaluator, id_maps, effect, sample_times)
    print('[gat2] ' + dd.format_parity_report(rep).replace('\n', '\n[gat2] '))
    # The harness must run end-to-end and produce a per-time report.
    assert len(rep['times']) == len(sample_times)
    assert all('n_blit' in r for r in rep['times'])


def test_export_channel_prefers_a_curves_own_breakpoints():
    # A curve that knows its closed form hands the breakpoints over; the
    # exporter must take them and never fall back to sampling it.
    class _Structural:
        _rest = (5.0,)

        def __init__(self):
            self.sampled = 0

        def sample(self, t):
            self.sampled += 1
            return (5.0,)

        def breakpoints(self, t0, t1, prop=0):
            # Hold 10 from t=1, then ramp 10 -> 30 over [2, 3].
            return [1.0, 2.0, 3.0], [10.0, 10.0, 30.0], [0.0, 1.0, 0.0], [0, 0, 0]

    curve = _Structural()
    native = _channel_sampler(curve, 0.0, 6.0)
    assert curve.sampled == 0
    for t, want in ((0.5, 5.0), (1.5, 10.0), (2.5, 20.0), (5.0, 30.0)):
        assert abs(native(t) - want) < 1e-3, t


def test_export_channel_samples_a_curve_that_declines_to_export():
    # None means "I cannot answer for this window" - the exporter samples.
    class _Declines:
        _rest = (0.0,)

        def sample(self, t):
            return (math.sin(t),)

        def breakpoints(self, t0, t1, prop=0):
            return None

    native = _channel_sampler(_Declines(), 0.0, 6.0)
    assert max(abs(native(t) - math.sin(t))
               for t in np.linspace(0.0, 6.0, 61)) < 0.05


def test_dense_export_collapses_a_held_curve_to_two_breakpoints():
    # A curve that never moves must not cost one breakpoint per sample.
    class _Held:
        _rest = (0.0,)

        def sample(self, t):
            return (7.0,)

    ts, vals, _durs, _rest, _eases = dd.export_channel(_Held(), 0.0, 600.0)
    assert len(ts) == 2 and vals == [7.0, 7.0]


def test_horizon_falls_back_to_the_live_sims_end():
    # Nothing else names the chart's length, so exporting to a fixed default
    # past the sim's end is pure cost.
    live = SimpleNamespace(_end_seconds=123.5)
    assert dd._horizon({'_live_sim': live}) == 123.5
    assert dd._horizon({'_live_sim': live, 'duration': 60.0}) == 60.0
    assert dd._horizon({}) == 600.0


def test_each_sheet_element_gets_its_own_frame_lane():
    # `_channel` memoizes on `id(timeline)`, but `_element_frame_kwarg` passes a
    # TEMPORARY `_FrameCurve` that dies as soon as the call returns - and
    # CPython recycles the address immediately, so the next element hit the
    # cache and inherited the previous one's frame lane. Every sheet sprite
    # then animated on one shared timeline, and a chart freezing one sprite on
    # a frame had the freeze overwritten.
    class _Rec:
        def __init__(self):
            self.channels = []

        def channel(self, ts, vals, durs, rest, eases=()):
            self.channels.append(list(vals))
            return len(self.channels) - 1

    def sheet(states):
        return SimpleNamespace(sheet_states=states, t_start=0.0,
                               t_end=float('inf'), state_pin=None,
                               sheet_cols=2, sheet_rows=1)

    rec = _Rec()
    builder = dd._Builder({}, 640.0, 480.0, builder=rec)
    slow = builder._element_frame_kwarg(sheet(((0, 1.0), (1, 1.0))))
    fast = builder._element_frame_kwarg(sheet(((7, 0.1), (9, 0.1))))

    assert slow['frame_id'] != fast['frame_id'], 'sheets share one frame lane'
    # And the lanes carry each sheet's OWN frames, not the first one's.
    assert set(rec.channels[slow['frame_id']]) == {0.0, 1.0}
    assert set(rec.channels[fast['frame_id']]) == {7.0, 9.0}


def test_out_of_plane_chain_matches_the_legacy_homography(captured_notefield):
    # A field chain that leaves the z=0 plane must carry the perspective
    # divide, and carry it the RIGHT way round. Two failure modes this pins:
    #   - no camera at all -> a flat squash (both edges the same height);
    #     gat measured 72-235px of corner error against legacy.
    #   - the camera applied as `world @ _TO_CONTENT` instead of
    #     `_TO_CONTENT @ world` -> still a trapezoid, wrong place, 2780px.
    # The Rust unit tests cannot see either: none of them compose a
    # non-planar chain WITH a camera, and a shape assertion cannot tell the
    # two multiplication orders apart. Comparing the whole homography against
    # TransformChannel can.
    tipped = fc.link_timelines({
        'rotation_y': [Keyframe(0.0, (55.0,), 0.0, _EASE_LINEAR)],
        'x': _instant(320.0), 'y': _instant(240.0),
    })
    inst = fc.instance('tippedA', 'proxy', 1, [tipped])
    compiled = _compiled([inst])
    evaluator, id_maps, _report = dd.build_static_doc(compiled)

    rep = dd.parity_report(evaluator, id_maps, _effect(compiled),
                           (0.0, 1.0, 2.5))
    assert rep['all_ok'], dd.format_parity_report(rep)
    assert rep['max_mat_err'] < 1e-3, (
        f"out-of-plane chain diverges from legacy: {rep['max_mat_err']:.2e}")


def _blit_lanes(evaluator, t):
    """(u, f) records for one frame, as (n, stride) arrays."""
    u_bytes, f_bytes, _uf, n = evaluator.frame(t)
    u = np.frombuffer(u_bytes, dtype=np.uint32).reshape(n, evaluator.u_stride)
    f = np.frombuffer(f_bytes, dtype=np.float32).reshape(n, evaluator.f_stride)
    return u, f


def test_fill_carries_its_diffuse_rgb_not_white():
    # The AFT-rig idiom parks a `diffuse,0,0,0,1` curtain between a capture
    # node and its samplers. Composited WHITE it paints over the rig instead
    # of masking it, which is what an untinted item does: Item::of rests the
    # tint at 1,1,1 and nothing overrode it.
    fill = fc.instance('curtain', 'fill', 0, [_link(x=320.0, y=240.0)])
    fill['color'] = EventTimeline(
        [Keyframe(0.0, (0.0, 0.0, 0.0), 0.0, _EASE_LINEAR)],
        rest=(0.0, 0.0, 0.0))
    evaluator, _id_maps, _report = dd.build_static_doc(_compiled([fill]))

    u, f = _blit_lanes(evaluator, 0.0)
    fills = [i for i in range(len(u))
             if u[i, 0] == sn.OP_BLIT and u[i, 1] == sn.SRC_FILL]
    assert len(fills) == 1, 'expected exactly one fill blit'
    assert tuple(f[fills[0], 10:13]) == (0.0, 0.0, 0.0)


def test_blend_add_is_sampled_per_frame_not_baked_at_t0():
    # `actor:blend('add')` fires inside a section body, so at t=0 the curve
    # still rests at 0. Sampling it once at build time therefore bakes
    # SourceOver for the whole chart - which composited gat 2's recursion
    # samplers as opaque occluders instead of summing them.
    aft = fc.instance('sampler', 'aft', 0, [_link(x=320.0, y=240.0)])
    aft['aft_node'] = 'nodeX'
    aft['blend_add'] = EventTimeline(
        [Keyframe(0.0, (0.0,), 0.0, _EASE_LINEAR),
         Keyframe(2.0, (1.0,), 0.0, _EASE_LINEAR)], rest=(0.0,))
    cap = fc.instance('nodeX', 'capture', 0, [_link()])
    evaluator, _id_maps, _report = dd.build_static_doc(_compiled([cap, aft]))

    def blend_at(t):
        u, _f = _blit_lanes(evaluator, t)
        blits = [i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT]
        assert len(blits) == 1, 'expected exactly one sampler blit'
        return int(u[blits[0], 4])

    assert blend_at(0.0) == 0, 'rests source-over before the blend call'
    assert blend_at(3.0) == 1, 'blend(add) after t=2 must reach the record'


def test_capture_sorts_by_its_z_inside_a_sort_span():
    # A capture node is an ordinary child of its frame, so a z-sorted parent
    # orders it against its siblings. Reading only Item::z sank every
    # non-Item command to the span's z=0 slot.
    #
    # The sibling's z sits BETWEEN 0 and the capture's, which is what makes
    # this test able to fail: with the capture pinned at 0 it sorts FIRST
    # (0 < 5), and only its real z=10 puts it last. A sibling at negative z
    # would order identically either way and prove nothing.
    cap = fc.instance('nodeX', 'capture', 0, [_link()])
    cap['z_group'] = 'g'
    cap['z_sort'] = EventTimeline(
        [Keyframe(0.0, (10.0,), 0.0, _EASE_LINEAR)], rest=(10.0,))
    below = fc.instance('below', 'fill', 0, [_link(x=100.0, y=240.0)])
    below['z_group'] = 'g'
    below['z_sort'] = EventTimeline(
        [Keyframe(0.0, (5.0,), 0.0, _EASE_LINEAR)], rest=(5.0,))
    evaluator, _id_maps, report = dd.build_static_doc(_compiled([cap, below]))
    assert report['z_groups'] == 1

    u, _f = _blit_lanes(evaluator, 0.0)
    ops = [int(u[i, 0]) for i in range(len(u))]
    copy_at = ops.index(sn.OP_COPY)
    blit_at = ops.index(sn.OP_BLIT)
    assert blit_at < copy_at, (
        'the z=5 sibling must draw before the z=10 capture; '
        f'ops={ops}')


def test_prepare_replay_matches_a_direct_build():
    # The async path records every DocBuilder call on a worker thread and
    # replays it where the unsendable Evaluator lives. Nothing covered it,
    # so a builder method added for the direct path (item_tint) recorded
    # fine in tests and AttributeError'd in the app.
    scene = _synthetic_scene()
    direct, _id_maps, _report = dd.build_static_doc(_compiled(scene))
    ops, _rec_maps, _rec_report = dd.prepare_static_doc(_compiled(scene))
    replayed = dd.assemble_static_doc(ops)

    for t in (0.0, 1.0, 2.5):
        want_u, want_f = _blit_lanes(direct, t)
        got_u, got_f = _blit_lanes(replayed, t)
        assert got_u.shape == want_u.shape, f'op count differs at t={t}'
        assert np.array_equal(got_u, want_u), f'u lanes differ at t={t}'
        assert np.allclose(got_f, want_f, atol=1e-6), f'f lanes differ at t={t}'


def test_recording_builder_rejects_a_name_docbuilder_lacks():
    recorder = dd._RecordingBuilder()
    recorder.item_tint(0, r_id=-1, r_rest=0.0)  # a real method records
    assert recorder.ops[-1][0] == 'item_tint'
    with pytest.raises(AttributeError, match='neither does DocBuilder'):
        recorder.item_tnit(0)


def test_frag_sampler_registers_a_shader_and_binds_its_uniforms(tmp_path):
    # A NotITG `Frag=` on a sampler is a per-actor program over that actor's
    # own texture. The doc emitted none, so every shaded sampler blitted the
    # raw capture and a rig whose whole visual IS the shader degraded to a
    # plain copy of the screen.
    frag = tmp_path / 'monitor.frag'
    frag.write_text('uniform float fAmt = 200.0;\nvoid main(){}\n')

    aft = fc.instance('shaded', 'aft', 0, [_link(x=320.0, y=240.0)])
    aft['aft_node'] = 'nodeX'
    aft['frag'] = str(frag)
    aft['frag_uniforms'] = {
        'fAmt': EventTimeline([Keyframe(0.0, (200.0,), 0.0, _EASE_LINEAR)],
                              rest=(200.0,)),
        'glitch': EventTimeline([Keyframe(2.0, (0.75,), 0.0, _EASE_LINEAR)],
                                rest=(0.0,)),
    }
    cap = fc.instance('nodeX', 'capture', 0, [_link()])
    evaluator, id_maps, _report = dd.build_static_doc(_compiled([cap, aft]))

    # The source travels in the doc, so the render thread never reads a file.
    assert len(id_maps['shaders']) == 1
    frag_src, vert_src, names = id_maps['shaders'][0]
    assert 'uniform float fAmt' in frag_src
    assert vert_src is None
    assert names == ['fAmt', 'glitch']  # sorted: item_uniform's index order

    u_bytes, _f, uf_bytes, n = evaluator.frame(3.0)
    u = np.frombuffer(u_bytes, dtype=np.uint32).reshape(n, evaluator.u_stride)
    uf = np.frombuffer(uf_bytes, dtype=np.float32)
    blits = [i for i in range(n) if u[i, 0] == sn.OP_BLIT]
    assert len(blits) == 1
    row = blits[0]
    assert int(u[row, 5]) == 1, 'shader lane is id+1 (0 means unshaded)'
    offset, count = int(u[row, 8]), int(u[row, 9])
    assert count == 2
    # Sampled at t=3, i.e. AFTER glitch's keyframe - the values must be live,
    # not the rests, and must pair with `names` positionally.
    assert list(uf[offset:offset + count]) == [200.0, 0.75]


def test_a_sampler_without_a_frag_stays_unshaded():
    aft = fc.instance('plain', 'aft', 0, [_link(x=320.0, y=240.0)])
    aft['aft_node'] = 'nodeX'
    cap = fc.instance('nodeX', 'capture', 0, [_link()])
    evaluator, id_maps, _report = dd.build_static_doc(_compiled([cap, aft]))

    assert id_maps['shaders'] == []
    u, _f = _blit_lanes(evaluator, 0.0)
    blits = [i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT]
    assert all(int(u[i, 5]) == 0 for i in blits)


def test_unreadable_frag_degrades_to_unshaded_rather_than_failing_the_build():
    aft = fc.instance('missing', 'aft', 0, [_link(x=320.0, y=240.0)])
    aft['aft_node'] = 'nodeX'
    aft['frag'] = '/nonexistent/does_not_exist.frag'
    cap = fc.instance('nodeX', 'capture', 0, [_link()])
    evaluator, id_maps, _report = dd.build_static_doc(_compiled([cap, aft]))

    assert id_maps['shaders'] == []
    u, _f = _blit_lanes(evaluator, 0.0)
    assert all(int(u[i, 5]) == 0 for i in range(len(u))
               if u[i, 0] == sn.OP_BLIT)


def test_recording_builder_mints_the_same_ids_as_a_real_builder(tmp_path):
    # Every id-returning DocBuilder method must mint on the recorder too. A
    # method missing from _MINTING records fine and returns None, which then
    # travels into replay as a null id - `shader` did exactly that, and only
    # the async path would have shown it.
    calls = {
        'channel': (([0.0], [1.0], [0.0], 1.0, [0]), {}),
        'drawable': ((64.0, 64.0, False, False), {}),
        'feed_slot': ((), {}),
        'shader': (('void main(){}',), {}),
        'mesh': (([0.0] * 6, 0, -1), {}),
        'clip_rect': ((0.0, 0.0, 1.0, 1.0), {}),
        'clip_polygon': (([0.0, 0.0, 1.0, 0.0, 1.0, 1.0],), {}),
    }
    assert set(calls) == set(dd._RecordingBuilder._MINTING), (
        'a minting method changed - update this test AND _MINTING')

    real = sn.DocBuilder(640.0, 480.0)
    recorder = dd._RecordingBuilder()
    for name, (args, kwargs) in calls.items():
        want = getattr(real, name)(*args, **kwargs)
        got = getattr(recorder, name)(*args, **kwargs)
        assert got == want, f'{name}: recorder minted {got}, builder {want}'


def test_prepare_replay_carries_a_shaded_sampler(tmp_path):
    # The round-trip over a scene whose sampler is SHADED: this is what
    # catches an id-minting method the recorder does not know about, since a
    # null shader id only fails once it reaches the real builder at replay.
    frag = tmp_path / 'lumikey.frag'
    frag.write_text('uniform float pixelSize = 0.00001;\nvoid main(){}\n')
    scene = _synthetic_scene()
    shaded = fc.instance('shadedA', 'aft', 0, [_link(x=320.0, y=240.0)])
    shaded['aft_node'] = 'nodeX'
    shaded['frag'] = str(frag)
    shaded['frag_uniforms'] = {
        'pixelSize': EventTimeline(
            [Keyframe(1.0, (0.02,), 0.0, _EASE_LINEAR)], rest=(0.00001,)),
    }
    scene.append(shaded)

    direct, direct_maps, _r = dd.build_static_doc(_compiled(scene))
    ops, rec_maps, _r2 = dd.prepare_static_doc(_compiled(scene))
    replayed = dd.assemble_static_doc(ops)

    assert rec_maps['shaders'] == direct_maps['shaders']
    assert len(rec_maps['shaders']) == 1
    for t in (0.0, 2.0):
        want_u, want_f = _blit_lanes(direct, t)
        got_u, got_f = _blit_lanes(replayed, t)
        assert np.array_equal(got_u, want_u), f'u lanes differ at t={t}'
        assert np.allclose(got_f, want_f, atol=1e-6), f'f lanes differ at t={t}'


def test_element_origin_and_size_reach_the_record(doc_elements, tmp_path):
    # SM draws an actor about its `origin` (0.5, 0.5 by default) while a bare
    # item quad spans (0,0)-(w,h), so an unforwarded origin hangs every
    # element down-right by half its own size. `size_x/y` carry zoomto, which
    # REPLACES the natural basis rather than scaling it.
    asset = tmp_path / 'bar.png'
    asset.write_bytes(b'')
    el = _sprite(0, str(asset))
    el.timelines['size_x'] = EventTimeline(
        [Keyframe(0.0, (20.0,), 0.0, _EASE_LINEAR)], rest=(20.0,))
    evaluator, _id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[el]))

    u, f = _blit_lanes(evaluator, 0.0)
    blits = [i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT]
    assert len(blits) == 1
    row = blits[0]
    assert tuple(f[row, 17:19]) == (0.5, 0.5), 'element origin not forwarded'
    assert float(f[row, 19]) == 20.0, 'zoomto width not forwarded'
    assert float(f[row, 20]) < 0.0, 'unset height must stay natural'


def test_glow_emits_a_second_additive_item_tinted_to_the_glow_colour(
        doc_elements, tmp_path):
    # SM's glow pass is the same sprite drawn again, additively, tinted to the
    # glow colour at the glow alpha (Sprite.cpp:536-541) - so it is a second
    # item, not a record lane.
    asset = tmp_path / 'g.png'
    asset.write_bytes(b'')
    el = _sprite(0, str(asset))
    el.timelines['glow'] = EventTimeline(
        [Keyframe(1.0, (1.0, 0.0, 0.0, 0.5), 0.0, _EASE_LINEAR)],
        rest=(1.0, 1.0, 1.0, 0.0))
    evaluator, _id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[el]))

    u, f = _blit_lanes(evaluator, 2.0)
    blits = [i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT]
    assert len(blits) == 2, 'content then glow'
    content, glow = blits
    assert content < glow, 'glow draws over the content'
    assert int(u[content, 4]) == 0, 'content stays source-over'
    assert int(u[glow, 4]) == 1, 'glow composites additively'
    assert tuple(f[glow, 10:13]) == (1.0, 0.0, 0.0), 'glow tint'
    assert float(f[glow, 9]) == pytest.approx(0.5), 'glow alpha as opacity'


def test_an_unglowed_element_emits_no_second_item(doc_elements, tmp_path):
    # glow rests at alpha 0, so a static rest must not double every element's
    # op count for a pass that draws nothing.
    asset = tmp_path / 'p.png'
    asset.write_bytes(b'')
    evaluator, _id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[_sprite(0, str(asset))]))

    u, _f = _blit_lanes(evaluator, 1.0)
    assert sum(1 for i in range(len(u)) if u[i, 0] == sn.OP_BLIT) == 1


def test_scale_to_cover_sends_the_rect_extent_not_its_corners(doc_elements,
                                                              tmp_path):
    # The engine's fit zoom is rect/natural per axis, then the larger (cover)
    # or smaller (fit-inside) of the two - the rect's POSITION never reaches
    # the draw, so the doc sends spans and the executor keeps the natural size.
    asset = tmp_path / 'f.png'
    asset.write_bytes(b'')
    el = _sprite(0, str(asset))
    for prop, value in (('fit_mode', 1.0), ('fit_left', 100.0),
                        ('fit_right', 420.0), ('fit_top', 40.0),
                        ('fit_bottom', 280.0)):
        el.timelines[prop] = EventTimeline(
            [Keyframe(0.0, (value,), 0.0, _EASE_LINEAR)], rest=(value,))
    evaluator, _id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[el]))

    u, f = _blit_lanes(evaluator, 1.0)
    row = next(i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT)
    assert float(f[row, 21]) == 1.0, 'cover mode'
    assert float(f[row, 22]) == 320.0, 'right - left'
    assert float(f[row, 23]) == 240.0, 'bottom - top'


def test_span_timeline_is_static_only_when_both_edges_are():
    lo = EventTimeline([Keyframe(0.0, (10.0,), 0.0, _EASE_LINEAR)],
                       rest=(10.0,))
    moving = EventTimeline([Keyframe(0.0, (50.0,), 0.0, _EASE_LINEAR),
                            Keyframe(1.0, (90.0,), 2.0, _EASE_LINEAR)],
                           rest=(50.0,))
    still = EventTimeline([], rest=(90.0,))

    assert dd._SpanTimeline(lo, still).sample(0.0) == (80.0,)
    assert dd._SpanTimeline(lo, moving).sample(0.0) == (40.0,)
    # A moving edge must NOT collapse to a constant channel.
    assert dd._SpanTimeline(lo, moving).is_static() is False


def test_a_corner_gradient_is_counted_not_silently_dropped(doc_elements,
                                                           tmp_path):
    # Neither reference chart uses one, so per-vertex colour waits for a chart
    # that does - but once legacy stops drawing elements an unimplemented verb
    # is invisible, so it has to show up in the report.
    asset = tmp_path / 'grad.png'
    asset.write_bytes(b'')
    el = _sprite(0, str(asset))
    el.timelines['color_ul'] = EventTimeline(
        [Keyframe(0.0, (1.0, 0.0, 0.0), 0.0, _EASE_LINEAR)],
        rest=(1.0, 1.0, 1.0))
    _evaluator, _id_maps, report = dd.build_static_doc(
        _compiled([], tree=[el]))

    assert report['element_skips'].get('corner_gradient') == 1


def test_edge_fades_reach_the_record_only_when_poked(doc_elements, tmp_path):
    # SetFade* is used 21 times across the two reference charts. A non-resting
    # fade routes the blit through a second GL program, so an element that
    # never fades must leave the lanes at 0 rather than pay for a hard edge.
    asset = tmp_path / 'fade.png'
    asset.write_bytes(b'')
    faded = _sprite(0, str(asset))
    faded.timelines['fade_top'] = EventTimeline(
        [Keyframe(0.0, (0.25,), 0.0, _EASE_LINEAR)], rest=(0.0,))
    plain = _sprite(0, str(asset))

    evaluator, _id_maps, report = dd.build_static_doc(
        _compiled([], tree=[faded]))
    assert 'edge_fade' not in report['element_skips'], 'fades are implemented'
    u, f = _blit_lanes(evaluator, 1.0)
    row = next(i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT)
    # lanes are l, r, t, b - only the top edge fades.
    assert tuple(f[row, 24:28]) == (0.0, 0.0, 0.25, 0.0)

    evaluator, _id_maps, _r = dd.build_static_doc(_compiled([], tree=[plain]))
    u, f = _blit_lanes(evaluator, 1.0)
    row = next(i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT)
    assert tuple(f[row, 24:28]) == (0.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# element parity harness
# --------------------------------------------------------------------------

def _parity_element(tmp_path, name, **timelines):
    asset = tmp_path / f'{name}.png'
    asset.write_bytes(b'')
    el = _sprite(0, str(asset))
    for prop, value in timelines.items():
        el.timelines[prop] = EventTimeline(
            [Keyframe(0.0, (value,), 0.0, _EASE_LINEAR)], rest=(value,))
    return el


def test_element_parity_harness_agrees_with_the_legacy_painter(doc_elements,
                                                               tmp_path):
    # The harness compares the DRAWN QUAD's corners, because origin, the
    # absolute-size override and scale-to-fit all move the box the mat3 acts
    # on - a mat3-only comparison passes with every one of them wrong.
    el = _parity_element(tmp_path, 'a', x=120.0, y=200.0, rotation=30.0,
                         scale_x=1.5, scale_y=0.75)
    evaluator, _id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[el]))

    rep = dd.element_parity_report(evaluator, _id_maps['element_order'],
                                   lambda _el: (64.0, 32.0), (0.0, 1.0, 2.5))
    assert rep['all_ok'], dd.format_element_parity_report(rep)
    assert rep['max_corner_err'] < 1e-3


def test_a_grouped_element_composites_at_its_alpha_not_its_square(
        doc_elements, tmp_path):
    # The leaf is BOTH the item and its chain's last link. Carrying alpha on
    # both multiplied it in twice, so an element at diffusealpha 0.5 under an
    # opaque group composited at 0.25 - and a veil at 0.05 came out at 0.0025,
    # invisible. Placement was exact throughout, which is why the quad harness
    # never saw it.
    leaf = _parity_element(tmp_path, 'l', alpha=0.5)
    group = _element('group', 0, children=[leaf],
                     keyframes={'alpha': _instant(1.0)})
    evaluator, id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[group]))

    order = id_maps['element_order']
    index = next(i for i, (el, role, _a) in enumerate(order)
                 if el is leaf and role == 'content')
    drawn = {tag: f for _k, _s, tag, f in dd._blit_records(evaluator, 0.0)}
    assert drawn[index + 1][dd._rec.F_OPACITY] == pytest.approx(0.5)


def test_the_element_harness_compares_opacity_not_only_placement(
        doc_elements, tmp_path):
    # The harness is only worth having if it FAILS on the difference it exists
    # to catch, and a compositing difference moves no corner at all.
    el = _parity_element(tmp_path, 'a', x=100.0, y=100.0, alpha=0.5)
    evaluator, id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[el]))

    rep = dd.element_parity_report(evaluator, id_maps['element_order'],
                                   lambda _el: (64.0, 32.0), (0.0, 1.0))
    assert rep['all_ok'], dd.format_element_parity_report(rep)

    el.timelines['alpha'] = EventTimeline(
        [Keyframe(0.0, (0.25,), 0.0, _EASE_LINEAR)], rest=(0.25,))
    rep = dd.element_parity_report(evaluator, id_maps['element_order'],
                                   lambda _el: (64.0, 32.0), (0.0, 1.0))
    assert not rep['all_ok']
    assert any(d[1] == 'opacity' for r in rep['times'] for d in r['diffs'])


def test_the_element_harness_reports_an_element_legacy_culls(doc_elements,
                                                             tmp_path):
    # An element the doc draws and legacy does not is invisible to a
    # placement comparison over drawn items - and a full-screen rect drawn a
    # frame too long blacks out the whole composite while every measured
    # element still reports 0.000px.
    el = _parity_element(tmp_path, 'a', x=100.0, y=100.0)
    evaluator, id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[el]))
    natural = lambda _el: (64.0, 32.0)  # noqa: E731
    assert dd.element_parity_report(
        evaluator, id_maps['element_order'], natural, (0.0,))['all_ok']

    el.timelines['hidden'] = EventTimeline(
        [Keyframe(0.0, (1.0,), 0.0, _EASE_LINEAR)], rest=(1.0,))
    rep = dd.element_parity_report(evaluator, id_maps['element_order'],
                                   natural, (0.0,))
    assert rep['n_extra'] == 1, dd.format_element_parity_report(rep)


def test_element_parity_harness_catches_a_dropped_origin(doc_elements,
                                                         tmp_path):
    # The harness is only worth having if it FAILS on the bug it exists to
    # catch: an element drawn top-left-anchored instead of about its origin
    # lands half its own size off, which is exactly what shipped before
    # `item_box`.
    el = _parity_element(tmp_path, 'b', x=100.0, y=100.0)
    evaluator, _id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[el]))
    natural = (64.0, 32.0)

    blits = [f for kind, _sid, _tag, f in dd._blit_records(evaluator, 0.0)
             if kind == sn.SRC_IMAGE]
    assert len(blits) == 1
    want = dd._legacy_element_quad(el, 0.0, natural, ())
    got = dd._record_quad(blits[0], natural)
    assert max(max(abs(a - b) for a, b in zip(wc, gc))
               for wc, gc in zip(want, got)) < 1e-3

    # Zero the origin lanes: the quad must now be off by exactly the origin
    # offset (half the 64x32 box), and the harness must see it.
    stripped = blits[0].copy()
    stripped[dd._rec.F_ORIGIN:dd._rec.F_ORIGIN + 2] = 0.0
    moved = dd._record_quad(stripped, natural)
    err = max(max(abs(a - b) for a, b in zip(wc, gc))
              for wc, gc in zip(want, moved))
    assert err == pytest.approx(32.0), f'expected half the 64px width, got {err}'


# --------------------------------------------------------------------------
# field-instance parity harness
# --------------------------------------------------------------------------

def test_field_parity_harness_agrees_with_the_legacy_effect(captured_notefield):
    compiled = _compiled(_synthetic_scene())
    evaluator, id_maps, _report = dd.build_static_doc(compiled)

    rep = dd.field_parity_report(evaluator, compiled,
                                 id_maps['instance_order'], _SAMPLE_TIMES)
    assert rep['all_ok'], dd.format_field_parity_report(rep)
    # Not vacuous: a run where every instance is 3D or uncomparable reports OK
    # having compared nothing at all, which is the failure mode a "0 diffs"
    # summary hides best.
    assert all(r['n_compared'] > 0 for r in rep['times'])
    assert rep['n_missing'] == 0 and rep['n_extra'] == 0


def test_the_legacy_field_reference_is_read_row_vector(captured_notefield):
    # A field transform is a ROW-vector homography; the record mat3 and
    # `_apply_h` are column-vector. Read untransposed the reference collapses
    # to a degenerate point, and every instance reports hundreds of pixels off
    # against a doc that is exactly right.
    scene = _synthetic_scene()
    proxy = next(i for i in scene if i.get('name') == 'proxyA')
    quad = dd._legacy_field_draws(scene, False, 0.0)[dd._instance_key(proxy)][0]

    width = max(x for x, _y in quad) - min(x for x, _y in quad)
    height = max(y for _x, y in quad) - min(y for _x, y in quad)
    assert width == pytest.approx(320.0)   # 640 design px at the proxy's 0.5x
    assert height == pytest.approx(240.0)


def test_field_parity_survives_a_provider_that_rebuilds(captured_notefield):
    # The lazy provider rebuilds its instance list whenever the topology
    # signature changes, so the list the doc was built from and the one the
    # harness samples are EQUAL BUT NOT IDENTICAL. Keyed on `id()` the harness
    # matched nothing and reported all 21 of LINARIA's comparisons as phantom
    # extras.
    scene = _synthetic_scene()
    evaluator, id_maps, _report = dd.build_static_doc(_compiled(scene))
    # Fresh dicts, same transforms - what a rebuild actually produces.
    rebuilt = _compiled([])
    rebuilt['field_instances'] = lambda: [dict(inst) for inst in scene]

    rep = dd.field_parity_report(evaluator, rebuilt,
                                 id_maps['instance_order'], _SAMPLE_TIMES)
    assert rep['n_extra'] == 0 and rep['n_missing'] == 0
    assert rep['n_compared'] > 0


def test_field_parity_harness_catches_a_moved_instance(captured_notefield,
                                                       monkeypatch):
    # The harness is only worth having if it FAILS on the difference it exists
    # to catch. A curtain that drew one design pixel wide instead of covering
    # its target passed the mat3-level `parity_report` for exactly as long as
    # that report was the only one.
    compiled = _compiled(_synthetic_scene())
    evaluator, id_maps, _report = dd.build_static_doc(compiled)

    real_quad = dd._record_quad
    monkeypatch.setattr(dd, '_record_quad',
                        lambda frec, natural: [(x + 7.0, y)
                                               for x, y in real_quad(frec, natural)])
    rep = dd.field_parity_report(evaluator, compiled,
                                 id_maps['instance_order'], _SAMPLE_TIMES)
    assert not rep['all_ok']
    # The harness's own 1e-3 bar: record lanes are f32, so a 640px quad
    # reconstructs to about 1e-5.
    assert rep['max_corner_err'] == pytest.approx(7.0, abs=1e-3)


_GAT1 = ('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/'
         'UKSRT8/5. gat/gat.sm')


def _natural_lookup():
    """`element -> (w, h)` logical frame size, read from the asset header the
    way the executor reads it from the uploaded texture.

    Returns None for an element with no readable asset, which the harness
    skips - a missing file is a fixture problem, not a placement diff."""
    from PySide6.QtGui import QImageReader
    from analysis.player.render.storyboard.asset_size import (
        AssetSizeSpec, resolve)

    cache = {}

    def natural(element):
        path = element.asset or (element.frames[0] if element.frames else None)
        if not path:
            return None
        if path not in cache:
            size = QImageReader(str(path)).size()
            if not size.isValid():
                cache[path] = None
            else:
                spec = element.size_spec or AssetSizeSpec(
                    cols=element.sheet_cols, rows=element.sheet_rows)
                logical = resolve(size.width(), size.height(), spec)
                cache[path] = logical.natural
        return cache[path]

    return natural


@pytest.mark.slow
@pytest.mark.parametrize('chart_path,label', [
    pytest.param(_GAT1, 'gat1', marks=pytest.mark.skipif(
        not os.path.exists(_GAT1), reason='gat 1 chart not on disk')),
    pytest.param(_GAT2, 'gat2', marks=pytest.mark.skipif(
        not os.path.exists(_GAT2), reason='gat 2 chart not on disk')),
])
def test_real_chart_element_parity(chart_path, label, doc_elements):
    """Measure the doc's storyboard-element placement against the legacy
    painter on a REAL chart, in design pixels.

    This is the gate for turning `VSRG_DRAWABLE_ELEMENTS` on: the doc copy
    replaces a legacy copy that renders correctly today, so it has to be
    shown equivalent rather than eyeballed. The assertion is deliberately
    loose - it reports the number and fails only on a gross divergence, so
    the measurement lands in CI output either way."""
    from analysis.games.notitg.sim.producers import (
        compile_via_sim, wait_for_upgrade)

    compiled = compile_via_sim(chart_path)
    assert compiled is not None
    assert wait_for_upgrade(compiled)

    evaluator, id_maps, report = dd.build_static_doc(compiled)
    order = id_maps['element_order']
    roles = {}
    for _el, role, _ancestors in order:
        roles[role] = roles.get(role, 0) + 1
    print(f'[{label}] element blits: {roles} '
          f'images={report["images"]} skips={report["element_skips"]}')

    # Off round seconds on purpose: channel breakpoint times are f32, so a
    # segment starting at 60.00000000000378 rounds to exactly 60.0 and wins
    # the lookup at t=60.0 that the f64 timeline does not. The window is
    # ~4e-12s - unreachable from an audio clock, but round sample times sit
    # in it exactly.
    rep = dd.element_parity_report(
        evaluator, order, _natural_lookup(),
        [30.037, 60.037, 120.037, 180.037, 240.037, 300.037, 400.037])
    print(f'[{label}] ' +
          dd.format_element_parity_report(rep).replace('\n', f'\n[{label}] '))
    summary = dd.format_element_parity_report(rep)
    assert rep['max_corner_err'] < 1e-2, (
        'element placement diverges from the legacy painter:\n' + summary)
    assert rep['n_missing'] == 0, (
        'the doc drops an element legacy draws:\n' + summary)
    assert rep['n_extra'] == 0, (
        'the doc draws an element legacy culls:\n' + summary)
    for r in rep['times']:
        assert not r['diffs'], summary


def test_a_rect_is_gated_by_its_w_h_and_sized_by_its_zoomto(doc_elements):
    # The legacy painter asks TWO questions and the doc mirrors both:
    # `_element_size` refuses a shape whose own w/h are not positive - it is
    # not drawable, whatever else is set - and only THEN does `_draw_size`
    # prefer an absolute size_x/size_y over that box.
    #
    # Reading size_x alone drew a Quad that only ever set `zoomto`. gat 2
    # closes on two full-screen black ones: they covered the background for
    # the rest of the chart, where the reference video shows it bright.
    zoomed_only = _element('rect', 0)
    zoomed_only.timelines['size_x'] = EventTimeline([], rest=(640.0,))
    zoomed_only.timelines['size_y'] = EventTimeline([], rest=(480.0,))
    evaluator, _id_maps, _r = dd.build_static_doc(
        _compiled([], tree=[zoomed_only]))
    u, f = _blit_lanes(evaluator, 1.0)
    row = next(i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT)
    assert tuple(f[row, 19:21]) == (0.0, 0.0), 'no w/h: not drawable'

    # With a real box it draws, and the zoomto sizes it.
    el = _element('rect', 0)
    el.timelines['w'] = EventTimeline([], rest=(8.0,))
    el.timelines['h'] = EventTimeline([], rest=(8.0,))
    el.timelines['size_x'] = EventTimeline([], rest=(640.0,))
    el.timelines['size_y'] = EventTimeline([], rest=(480.0,))
    el.timelines['color'] = EventTimeline([], rest=(1.0, 0.0, 0.0))
    evaluator, id_maps, report = dd.build_static_doc(_compiled([], tree=[el]))

    assert 'rect' not in report['element_skips'], 'rects draw now'
    u, f = _blit_lanes(evaluator, 1.0)
    row = next(i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT)
    assert int(u[row, 1]) == sn.SRC_FILL
    assert tuple(f[row, 19:21]) == (640.0, 480.0), 'zoomto wins over w/h'
    assert tuple(f[row, 10:13]) == (1.0, 0.0, 0.0)

    rep = dd.element_parity_report(
        evaluator, id_maps['element_order'],
        lambda el_: (el_.sample('w', 0.0)[0], el_.sample('h', 0.0)[0]),
        (0.0, 1.0))
    assert rep['all_ok'], dd.format_element_parity_report(rep)


def test_a_zero_size_rect_draws_nothing_like_legacy(doc_elements):
    # legacy `_element_size` returns None for a shape with w<=0 or h<=0, so it
    # never reaches the painter. The doc must not fall back to the executor's
    # unit quad and paint a 1x1 dot.
    el = _element('rect', 0)
    evaluator, _id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[el]))
    u, f = _blit_lanes(evaluator, 1.0)
    rows = [i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT]
    assert rows, 'the item is still emitted'
    assert tuple(f[rows[0], 19:21]) == (0.0, 0.0), (
        'a zero-size fill must reach the executor as zero, which it skips')


def _bitmap_font(tmp_path, cols=16, rows=16, advance=10.0):
    from analysis.player.render.storyboard.bitmap_font import BitmapFont
    atlas = tmp_path / 'font.png'
    atlas.write_bytes(b'')
    return BitmapFont(texture_path=str(atlas), cols=cols, rows=rows,
                      line_spacing=0.0, default_advance=advance, advances={})


def test_bitmaptext_emits_one_glyph_item_per_character(doc_elements, tmp_path):
    # An SM bitmap font is a GRID atlas, so a glyph is a sheet cell: the
    # codepoint IS the frame index, and no machinery a sheet sprite lacks is
    # needed.
    font = _bitmap_font(tmp_path)
    el = _element('bitmaptext', 0, text='AB', font=font)
    evaluator, id_maps, report = dd.build_static_doc(_compiled([], tree=[el]))

    assert 'bitmaptext' not in report['element_skips'], 'bitmaptext draws now'
    assert [role for _e, role, _a in id_maps['element_order']] == ['glyph'] * 2

    u, f = _blit_lanes(evaluator, 1.0)
    blits = [i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT]
    assert len(blits) == 2
    assert [int(u[i, 3]) for i in blits] == [ord('A'), ord('B')], 'frame = codepoint'
    # Centred on the actor: two 10px advances span [-10, +10], so the glyph
    # centres sit at -5 and +5.
    assert [round(float(f[i, 2]), 3) for i in blits] == [-5.0, 5.0]
    assert all(round(float(f[i, 5]), 3) == 0.0 for i in blits), 'centres on y=0'
    assert all(tuple(f[i, 17:19]) == (0.5, 0.5) for i in blits)


def test_bitmaptext_glyph_offsets_ride_inside_the_element_rotation(
        doc_elements, tmp_path):
    # The per-glyph offset is a LINK, not an x lane, so it composes inside the
    # element's own rotation - a rotated string must rotate as a string, not
    # smear along the unrotated axis.
    font = _bitmap_font(tmp_path)
    el = _element('bitmaptext', 0, text='AB', font=font)
    el.timelines['rotation'] = EventTimeline([], rest=(90.0,))
    evaluator, _id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[el]))

    u, f = _blit_lanes(evaluator, 1.0)
    blits = [i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT]
    assert len(blits) == 2
    # Rotated 90 degrees, the run advances along Y instead of X.
    xs = [round(float(f[i, 2]), 3) for i in blits]
    ys = [round(float(f[i, 5]), 3) for i in blits]
    assert all(abs(x) < 1e-3 for x in xs), f'x should not advance: {xs}'
    assert ys == [-5.0, 5.0] or ys == [5.0, -5.0], f'y should advance: {ys}'


def test_a_codepoint_outside_the_atlas_grid_draws_nothing_but_advances(
        doc_elements, tmp_path):
    # legacy's `font.cell` returns None outside the grid and draws nothing,
    # but the pen still moved - so the REST of the string must not shift.
    font = _bitmap_font(tmp_path, cols=8, rows=8)  # 64 cells; 'A' is 65
    el = _element('bitmaptext', 0, text='A!', font=font)
    evaluator, _id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[el]))

    u, f = _blit_lanes(evaluator, 1.0)
    blits = [i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT]
    assert len(blits) == 1, "'A' (65) is outside an 8x8 grid; '!' (33) is not"
    assert int(u[blits[0], 3]) == ord('!')
    assert round(float(f[blits[0], 2]), 3) == 5.0, 'the dropped glyph still advanced'


def test_a_run_crop_is_redistributed_across_the_glyphs(doc_elements, tmp_path):
    # SM crops the whole TEXT ACTOR, but the doc draws one item per glyph and
    # `compose_links` takes crop from the LEAF - so a run crop is dropped
    # unless each glyph gets its share. Windfall hides its seizure warning
    # with `cropright(1)` and nothing else, which is why it covered the screen.
    font = _bitmap_font(tmp_path, advance=10.0)
    el = _element('bitmaptext', 0, text='ABCD', font=font)
    el.timelines['crop_right'] = EventTimeline([], rest=(0.5,))
    evaluator, _id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[el]))

    u, f = _blit_lanes(evaluator, 1.0)
    blits = [i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT]
    assert len(blits) == 4
    # Half the 40px run is cropped: the first two glyphs are untouched, the
    # last two entirely gone.
    assert [round(float(f[i, 15]), 3) for i in blits] == [0.0, 0.0, 1.0, 1.0]


def test_a_full_run_crop_removes_every_glyph(doc_elements, tmp_path):
    font = _bitmap_font(tmp_path, advance=10.0)
    el = _element('bitmaptext', 0, text='ABCD', font=font)
    el.timelines['crop_right'] = EventTimeline([], rest=(1.0,))
    evaluator, _id_maps, _report = dd.build_static_doc(
        _compiled([], tree=[el]))

    u, f = _blit_lanes(evaluator, 1.0)
    blits = [i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT]
    assert all(round(float(f[i, 15]), 3) == 1.0 for i in blits), (
        'cropright(1) must leave nothing visible')


def test_text_elements_defer_their_raster_to_the_image_table(doc_elements):
    # Laying out a system font needs Qt, and this compiler runs on a worker
    # thread and stays Qt-free - so the doc records the spec and the consumer
    # rasterises it.
    el = _element('text', 0, text='hello', font_px=24.0)
    evaluator, id_maps, report = dd.build_static_doc(_compiled([], tree=[el]))

    assert 'text' not in report['element_skips'], 'text draws now'
    assert id_maps['text_images'] == {0: ('hello', 24.0)}
    u, _f = _blit_lanes(evaluator, 1.0)
    blits = [i for i in range(len(u)) if u[i, 0] == sn.OP_BLIT]
    assert len(blits) == 1
    assert int(u[blits[0], 1]) == sn.SRC_IMAGE and int(u[blits[0], 2]) == 0


def test_the_same_caption_at_the_same_size_uploads_once(doc_elements):
    tree = [_element('text', 0, text='same', font_px=20.0),
            _element('text', 1, text='same', font_px=20.0),
            _element('text', 2, text='same', font_px=40.0)]
    _evaluator, id_maps, _report = dd.build_static_doc(_compiled([], tree=tree))
    assert len(id_maps['text_images']) == 2, 'keyed by (text, size)'


def test_an_empty_text_element_is_skipped_not_drawn(doc_elements):
    _evaluator, _id_maps, report = dd.build_static_doc(
        _compiled([], tree=[_element('text', 0, text='')]))
    assert report['element_skips'].get('text') == 1


def test_legacy_storyboard_is_suppressed_when_the_doc_owns_elements(
        monkeypatch, tmp_path):
    # Both paths read the same compiled['tree'] and neither knows about the
    # other, so leaving legacy's on composites every element TWICE - and the
    # doubling reads as a placement bug, since the copies agree to a fraction
    # of a pixel.
    from analysis.games.notitg.adapter import NotitgAdapter

    element = _element('rect', 0)
    compiled = {'tree': [element], 'field_instances': [],
                'base_field_hidden': None}
    adapter = NotitgAdapter.__new__(NotitgAdapter)
    monkeypatch.setattr(adapter, '_compiled_modfile', lambda _r: compiled,
                        raising=False)

    monkeypatch.setenv('VSRG_DRAWABLE_ELEMENTS', '0')
    assert adapter.storyboard(object()) is not None, 'legacy draws when the doc does not'

    monkeypatch.setenv('VSRG_DRAWABLE_ELEMENTS', '1')
    monkeypatch.setenv('VSRG_DRAWABLE_PIPELINE', '1')
    assert adapter.storyboard(object()) is None, 'exactly one owner'


def test_the_screen_clear_is_declared_and_switchable(monkeypatch):
    """The doc says what its own surface is, so the pipeline does not have to
    assume. `VSRG_DRAWABLE_OPAQUE_SCREEN=0` is the bisect switch: an opaque
    surface hides whatever the renderer painted under the chart region."""
    from analysis.games.notitg import drawable_doc as dd

    monkeypatch.delenv('VSRG_DRAWABLE_OPAQUE_SCREEN', raising=False)
    assert dd.opaque_screen() is True
    monkeypatch.setenv('VSRG_DRAWABLE_OPAQUE_SCREEN', '0')
    assert dd.opaque_screen() is False
