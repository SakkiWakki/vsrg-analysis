"""The NotITG -> Drawable bridge (drawable_bridge): the first game producer
for the Drawable core.

The parity harness builds a SYNTHETIC compiled chart - a hand-built field
provider of `field_compose.instance` dicts - so the translation runs with no
sim: it asserts the op stream's blit order matches the sampled entry order and
the alphas survive to 1e-4. The snapshot-topology golden test runs the real
`RasterExecutor` on a `[blit, capture, fill-curtain, slot-sampler]` chart and
asserts the slot blit shows PRE-curtain content (the monitor class, now through
the real bridge). A skippable smoke test exercises the real gat 2 chart when it
is present on disk.

Feed v2 (wave-3 C1): every entry crosses as a mat3 in the record's
column-vector layout (f32 stride 18); there is no affine decomposition and no
projective skip, so coverage is `{translated, total}` with translated == total.
"""
import os

import numpy as np
import pytest

sn = pytest.importorskip('storyboard_native')

from analysis.games.notitg import drawable_bridge
from analysis.games.notitg import field_compose as fc
from analysis.player.render.effects.timeline import Keyframe


_EASE_LINEAR = 0


def _instant(value):
    """One immediate keyframe holding `value` (a scalar) from t=0."""
    return [Keyframe(0.0, (float(value),), 0.0, _EASE_LINEAR)]


def _link(**pokes):
    """A field link resting at the SM defaults with `pokes` (prop -> scalar)
    written as immediate keyframes at t=0."""
    keyframes = {prop: _instant(value) for prop, value in pokes.items()}
    return fc.link_timelines(keyframes)


def _synthetic_compiled(instances):
    """A minimal compiled dict carrying a fixed field-instance provider and no
    base-hidden gate (single-player, base visible: the identity original is
    prepended to the entry stream)."""
    return {'field_instances': list(instances), 'base_field_hidden': None}


def _seg_f(feed, drawable_bridge_mod=drawable_bridge):
    """The concatenated f32 feed rows as an (N, 18) view."""
    _ids, counts, _u, feed_f_bytes, _cov = feed
    n = int(sum(counts))
    return np.frombuffer(feed_f_bytes, dtype=np.float32).reshape(
        n, drawable_bridge_mod._FEED_F_STRIDE)


def _seg_u(feed, drawable_bridge_mod=drawable_bridge):
    _ids, counts, feed_u_bytes, _f, _cov = feed
    n = int(sum(counts))
    return np.frombuffer(feed_u_bytes, dtype=np.uint32).reshape(
        n, drawable_bridge_mod._FEED_U_STRIDE)


def _frames(evaluator, feed):
    feed_ids, counts, feed_u, feed_f, coverage = feed
    u_raw, f_raw, _uf, n = evaluator.frame_with_feeds(
        0.0, feed_ids, counts, feed_u, feed_f)
    u = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, evaluator.u_stride)
    f = np.frombuffer(f_raw, dtype=np.float32).reshape(n, evaluator.f_stride)
    return u, f, coverage


def test_module_imports_headless():
    """The bridge must import with NO Qt pulled at module load - checked in a
    clean subprocess so an already-imported PySide6 (pulled by sibling modules
    in this session) cannot mask a top-level Qt import in the bridge itself."""
    import subprocess
    import sys

    code = ('import sys\n'
            'import analysis.games.notitg.drawable_bridge\n'
            'qt = [m for m in sys.modules if m.startswith("PySide6")]\n'
            'print(";".join(qt))\n')
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '', f'Qt pulled at load: {result.stdout!r}'


def test_blit_order_matches_entry_order_and_alpha_survives():
    """Three proxy copies at distinct positions + the identity original: the
    fed blit order matches the sampled entry order, and every alpha survives to
    1e-4 into the feed's opacity lane."""
    instances = [
        fc.instance('copyA', 'proxy', 1, [_link(x=100.0, y=50.0, alpha=0.5)]),
        fc.instance('copyB', 'proxy', 1, [_link(x=-80.0, y=20.0, alpha=0.25)]),
        fc.instance('copyC', 'proxy', 1, [_link(x=0.0, y=-30.0, alpha=1.0)]),
    ]
    compiled = _synthetic_compiled(instances)
    ev, id_maps = drawable_bridge.build_doc(compiled)
    feed = drawable_bridge.feed_frame(compiled, 0.0, id_maps)
    _u, f, coverage = _frames(ev, feed)

    # The single-player effect prepends the identity original (alpha 1.0) to
    # the three copies -> four entries, all proxy field blits. Every entry
    # crosses as a mat3, so translated == total (no skips).
    assert coverage['total'] == 4
    assert coverage['translated'] == 4
    assert 'skipped_projective' not in coverage
    assert coverage['stale'] is False

    # The fed alphas in sampled ENTRY order (identity original first). Feed v2
    # opacity is lane 9 (after the 9 mat3 lanes).
    feed_f = _seg_f(feed)
    assert feed_f[:, 9].tolist() == pytest.approx([1.0, 0.5, 0.25, 1.0], abs=1e-4)

    # The evaluated op stream: the (single) segment dynamic drawable composes
    # the four fed blits in order, then the screen root blits it.
    field_id = id_maps['fields']['field']
    u, _f, _c = _frames(ev, feed)
    blit_rows = u[u[:, 0] == sn.OP_BLIT]
    # Four fed field blits (SRC_DRAWABLE of the field drawable) + the screen's
    # segment blit.
    fed = [(int(r[1]), int(r[2])) for r in blit_rows[:4]]
    assert fed == [(sn.SRC_DRAWABLE, field_id)] * 4
    # The blit opacities are the entry alphas, in order.
    fed_op = _f[u[:, 0] == sn.OP_BLIT][:4, 9]
    assert fed_op.tolist() == pytest.approx([1.0, 0.5, 0.25, 1.0], abs=1e-4)


def test_fed_transform_reaches_the_mat3_translate_lanes():
    """A positioned proxy's poke reaches the feed mat3 translate lanes (2/5),
    offset by the capture-centering the transform channel applies (the capture
    holds content centered on the design centre 320x240, so a copy poked to
    design (x, y) blits its capture's top-left at (x - 320, y - 240) under unit
    scale). Cross-check the feed mat3 against the effect's own screen QTransform
    so the translation is faithful, not a guessed constant."""
    inst = fc.instance('copy', 'proxy', 1, [_link(x=123.0, y=-45.0)])
    compiled = _synthetic_compiled([inst])
    ev, id_maps = drawable_bridge.build_doc(compiled)
    feed = drawable_bridge.feed_frame(compiled, 0.0, id_maps)
    feed_f = _seg_f(feed)

    # The copy is entry 1 (entry 0 is the identity original). Its feed mat3
    # translate lands in lanes 2 (m02, dx) and 5 (m12, dy), which must equal
    # the effect's own screen QTransform dx/dy for that entry.
    from analysis.games.notitg.field_instances import NotitgFieldInstances
    frame = NotitgFieldInstances(list(compiled['field_instances'])).at(
        drawable_bridge._Ctx(0.0, drawable_bridge._DESIGN_RECT))
    copy_qt = frame.fields[1][0]
    assert feed_f[1, 2] == pytest.approx(copy_qt.dx(), abs=1e-3)
    assert feed_f[1, 5] == pytest.approx(copy_qt.dy(), abs=1e-3)
    # And that faithful translate is the poke minus the design centre.
    assert feed_f[1, 2] == pytest.approx(123.0 - 320.0, abs=1e-3)
    assert feed_f[1, 5] == pytest.approx(-45.0 - 240.0, abs=1e-3)


def test_mat3_matches_executor_qtransform_reading():
    """The fed mat3 must reproduce the entry's QTransform under the executor's
    read `QTransform(m[0], m[3], m[1], m[4], m[2], m[5])` - i.e. the record is
    the Qt matrix transposed. Verify against a rotate+scale copy."""
    inst = fc.instance('rot', 'proxy', 1, [_link(rotation=25.0, x=30.0, y=40.0)])
    compiled = _synthetic_compiled([inst])
    _ev, id_maps = drawable_bridge.build_doc(compiled)
    feed = drawable_bridge.feed_frame(compiled, 0.0, id_maps)
    feed_f = _seg_f(feed)
    m = feed_f[1, :9]  # the copy's mat3 (entry 1)

    from analysis.games.notitg.field_instances import NotitgFieldInstances
    frame = NotitgFieldInstances(list(compiled['field_instances'])).at(
        drawable_bridge._Ctx(0.0, drawable_bridge._DESIGN_RECT))
    qt = frame.fields[1][0]
    # Executor reads QTransform(m[0], m[3], m[1], m[4], m[2], m[5]); that must
    # equal the entry's own affine block.
    assert m[0] == pytest.approx(qt.m11(), abs=1e-4)
    assert m[3] == pytest.approx(qt.m12(), abs=1e-4)
    assert m[1] == pytest.approx(qt.m21(), abs=1e-4)
    assert m[4] == pytest.approx(qt.m22(), abs=1e-4)
    assert m[2] == pytest.approx(qt.dx(), abs=1e-3)
    assert m[5] == pytest.approx(qt.dy(), abs=1e-3)


def test_fill_scope_translates_to_src_fill_with_tint():
    """An AFT-rig fill entry becomes an SRC_FILL feed item carrying its rgb as
    the tint (lanes 10..13 in feed v2)."""
    fill = fc.instance('curtain', 'fill', 0, [_link()])
    from analysis.player.render.effects.timeline import EventTimeline
    fill['color'] = EventTimeline([Keyframe(0.0, (0.2, 0.4, 0.6), 0.0, _EASE_LINEAR)],
                                  rest=(1.0, 1.0, 1.0))
    compiled = _synthetic_compiled([fill])
    _ev, id_maps = drawable_bridge.build_doc(compiled)
    feed = drawable_bridge.feed_frame(compiled, 0.0, id_maps)
    feed_u = _seg_u(feed)
    feed_f = _seg_f(feed)
    # The identity original (proxy field) is prepended, then the fill.
    fill_row = feed_u[:, 0].tolist().index(sn.SRC_FILL)
    assert feed_u[fill_row, 0] == sn.SRC_FILL
    assert feed_f[fill_row, 10:13].tolist() == pytest.approx([0.2, 0.4, 0.6], abs=1e-4)


def test_sheared_entry_crosses_as_mat3():
    """A skew (non-orthogonal linear block) now crosses as a mat3 verbatim - no
    skip. Its feed mat3 reproduces the entry's QTransform affine block, and
    coverage counts it as translated."""
    inst = fc.instance('sheared', 'proxy', 1, [_link(skew_x=0.6, x=10.0)])
    compiled = _synthetic_compiled([inst])
    _ev, id_maps = drawable_bridge.build_doc(compiled)
    feed = drawable_bridge.feed_frame(compiled, 0.0, id_maps)
    _ids, counts, _u, _f, coverage = feed
    # The identity original + the skewed copy, both translated (no skip).
    assert coverage['translated'] == coverage['total'] == 2

    feed_f = _seg_f(feed)
    m = feed_f[1, :9]
    from analysis.games.notitg.field_instances import NotitgFieldInstances
    frame = NotitgFieldInstances(list(compiled['field_instances'])).at(
        drawable_bridge._Ctx(0.0, drawable_bridge._DESIGN_RECT))
    qt = frame.fields[1][0]
    # The off-diagonal shear term survives into the record (lane 1 = m01 =
    # qt.m21()), which a decomposition would have rejected.
    assert m[1] == pytest.approx(qt.m21(), abs=1e-4)
    assert abs(m[1]) > 1e-3


def test_qt_to_mat3_none_is_identity():
    """A None transform (the centered original blit) crosses as the identity
    mat3 in the record's column-vector layout."""
    m = drawable_bridge._qt_to_mat3(None)
    assert m == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def test_snapshot_topology_shows_pre_curtain_content():
    """The monitor class through the real bridge: a chart of
    [blit, capture, fill-curtain, slot-sampler] must show PRE-curtain content
    in the slot blit. The capture snapshots the composite before the curtain
    fill; the sampler that blits the slot therefore shows the first blit's
    content, not the curtain that covers it. Run the real RasterExecutor and
    read the composed pixels.
    """
    pytest.importorskip('PySide6')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from analysis.player.render.storyboard.executor import RasterExecutor
    from analysis.player.render.effects.timeline import EventTimeline

    # A proxy blit (green field), then a capture node (snapshots the composite
    # into slot 'mon'), then a full-screen red fill curtain, then an aft
    # sampler reading slot 'mon' back. Because the slot froze BEFORE the
    # curtain, sampling it must reveal the pre-curtain green, not the red.
    blit = fc.instance('proxy', 'proxy', 1, [_link()])
    capture = fc.instance('mon', 'capture', 0, [_link()])
    curtain = fc.instance('curtain', 'fill', 0, [_link()])
    curtain['color'] = EventTimeline(
        [Keyframe(0.0, (1.0, 0.0, 0.0), 0.0, _EASE_LINEAR)],
        rest=(1.0, 1.0, 1.0))
    # The sampler reads slot 'mon' (a 'screen' aft whose freeze key is 'mon'),
    # placed after the curtain in tree order.
    sampler = fc.instance('read', 'aft', 0, [_link()], aft_order='post')
    sampler['aft_node'] = 'mon'
    compiled = _synthetic_compiled([blit, capture, curtain, sampler])

    ev, id_maps = drawable_bridge.build_doc(compiled)
    # Two inter-capture segments (one capture) -> two dynamic drawables.
    assert len(id_maps['segments']) == 2
    assert id_maps['captures'] == ['mon']
    assert 'mon' in id_maps['slots']

    feed = drawable_bridge.feed_frame(compiled, 0.0, id_maps)
    feed_ids, counts, feed_u, feed_f, _cov = feed
    u_raw, f_raw, _uf, n = ev.frame_with_feeds(0.0, feed_ids, counts, feed_u, feed_f)
    u = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, ev.u_stride)
    f = np.frombuffer(f_raw, dtype=np.float32).reshape(n, ev.f_stride)

    # The op stream: segment 0 (the green proxy blit) composes into its own
    # target, the screen blits it, the Snapshot copies the screen INTO 'mon'
    # (green, pre-curtain), then segment 1 (the red curtain) composes on top.
    # The Snapshot's at-position freeze is the monitor-class fix: 'mon' holds
    # the composite AS OF its tree index, never the curtained finished frame.
    from PySide6.QtGui import QImage, QColor

    green = QImage(640, 480, QImage.Format.Format_ARGB32_Premultiplied)
    green.fill(QColor(0, 200, 0, 255))
    field_id = id_maps['fields']['field']
    sizes = [(640.0, 480.0)] * ev.drawable_count()
    execu = RasterExecutor({}, sizes)
    # Pre-populate the field source drawable with green so the proxy blit and
    # thus segment-0's composite (and the snapshot) carry green (the field
    # source has no doc commands, so it retains this content unchanged).
    execu._targets[field_id] = green.copy()
    execu.execute(u, f)

    # The definitive monitor-class assertion: the slot 'mon' froze after
    # segment-0's green proxy blit and BEFORE the red curtain, so reading its
    # texture back shows PRE-curtain green and NOT the curtain's red - the
    # snapshot is an at-position capture, never the finished (curtained) frame.
    slot_img = execu._targets[id_maps['slots']['mon']]
    slot_px = slot_img.pixelColor(320, 240)
    assert slot_px.green() > 120, (
        f'snapshot should hold pre-curtain green, got {slot_px.getRgb()}')
    assert slot_px.red() < 80, (
        f'snapshot must NOT hold the post-curtain red, got {slot_px.getRgb()}')


def test_topology_signature_changes_with_capture_set():
    """The signature changes when the capture set grows, so a doc built for the
    old topology reports stale on the next feed."""
    base = [fc.instance('proxy', 'proxy', 1, [_link()])]
    grown = base + [fc.instance('mon', 'capture', 0, [_link()])]
    sig_a = drawable_bridge.topology_signature(_synthetic_compiled(base))
    sig_b = drawable_bridge.topology_signature(_synthetic_compiled(grown))
    assert sig_a != sig_b

    # A doc built for `base` is stale once the compiled grows to `grown`.
    compiled = _synthetic_compiled(base)
    _ev, id_maps = drawable_bridge.build_doc(compiled)
    feed = drawable_bridge.feed_frame(
        _synthetic_compiled(grown), 0.0, id_maps)
    assert feed[4]['stale'] is True


_GAT2 = ('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/'
         'UKSRT9/5. getfucked2/get_fucked_2.sm')


@pytest.mark.skipif(not os.path.exists(_GAT2), reason='gat 2 chart not on disk')
def test_gat2_smoke_build_and_feed():
    """Build the doc + feed three times against the real gat 2 chart: no
    exception, nonzero coverage, translated == total, and print the coverage
    stats. The lazy provider grows as the background sweep runs, so poll it
    until the topology is populated before sampling."""
    import time

    from analysis.games.notitg.sim.producers import compile_via_sim

    compiled = compile_via_sim(_GAT2)
    assert compiled is not None
    provider = compiled.get('field_instances')
    assert provider is not None

    # The lazy topology fills in over ~20s (the background sweep). Poll until
    # the provider carries a substantial instance set.
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        current = list(provider() if callable(provider) else provider)
        if len(current) > 150:
            break
        time.sleep(0.5)
    count = len(list(provider() if callable(provider) else provider))
    print(f'[gat2] provider instance count: {count}')

    ev, id_maps = drawable_bridge.build_doc(compiled)
    print(f'[gat2] doc drawables: {ev.drawable_count()} '
          f'(segments={len(id_maps["segments"])}, '
          f'slots={len(id_maps["slots"])}, fields={len(id_maps["fields"])})')

    total_translated = 0
    for t in (100.0, 300.0, 500.0):
        feed = drawable_bridge.feed_frame(compiled, t, id_maps)
        feed_ids, counts, feed_u, feed_f, coverage = feed
        # Must not raise, and the evaluator must ingest the feed cleanly.
        u_raw, _f_raw, _uf, n = ev.frame_with_feeds(
            t, feed_ids, counts, feed_u, feed_f)
        assert n >= 0
        # Feed v2: every resolved entry crosses as a mat3.
        assert coverage['translated'] == coverage['total']
        total_translated += coverage['translated']
        print(f'[gat2] t={t}: translated={coverage["translated"]} '
              f'total={coverage["total"]} stale={coverage["stale"]}')
    assert total_translated > 0
