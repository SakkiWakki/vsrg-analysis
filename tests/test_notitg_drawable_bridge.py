"""The NotITG -> Drawable bridge (drawable_bridge): the first game producer
for the Drawable core.

The parity harness builds a SYNTHETIC compiled chart - a hand-built field
provider of `field_compose.instance` dicts - so the translation runs with no
sim: it asserts the op stream's blit order matches the sampled entry order and
the alphas survive to 1e-4. A skippable smoke test exercises the real gat 2
chart when it is present on disk.
"""
import math
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
    _feed_ids, counts, feed_u_bytes, feed_f_bytes, _cov = feed

    # The single-player effect prepends the identity original (alpha 1.0) to
    # the three copies -> four entries, all proxy field blits, all affine.
    assert coverage['total'] == 4
    assert coverage['translated'] == 4
    assert coverage['skipped_projective'] == 0

    # The fed alphas in sampled ENTRY order (identity original first).
    feed_f = np.frombuffer(feed_f_bytes, dtype=np.float32).reshape(
        counts[0], drawable_bridge._FEED_F_STRIDE)
    assert feed_f[:, 5].tolist() == pytest.approx([1.0, 0.5, 0.25, 1.0], abs=1e-4)

    # The evaluated op stream: the dynamic drawable composes the four fed blits
    # in order, then the screen root blits the dynamic drawable.
    field_id = id_maps['fields']['field']
    blits = f  # opacity lane 9 on the u=BLIT rows
    u, _f, _c = _frames(ev, feed)
    blit_rows = u[u[:, 0] == sn.OP_BLIT]
    # Four fed field blits (SRC_DRAWABLE of the field drawable) + the screen's
    # dynamic-drawable blit.
    fed = [(int(r[1]), int(r[2])) for r in blit_rows[:4]]
    assert fed == [(sn.SRC_DRAWABLE, field_id)] * 4
    # The blit opacities are the entry alphas, in order.
    fed_op = _f[u[:, 0] == sn.OP_BLIT][:4, 9]
    assert fed_op.tolist() == pytest.approx([1.0, 0.5, 0.25, 1.0], abs=1e-4)


def test_fed_position_reaches_the_translate_lanes():
    """A positioned proxy's poke reaches the feed mat3 translate lanes, offset
    by the capture-centering the transform channel applies (the capture holds
    content centered on the design centre 320x240, so a copy poked to design
    (x, y) blits its capture's top-left at (x - 320, y - 240) under unit scale).
    Cross-check the feed against the effect's own screen QTransform so the
    translation is faithful, not a guessed constant."""
    from PySide6.QtGui import QTransform

    inst = fc.instance('copy', 'proxy', 1, [_link(x=123.0, y=-45.0)])
    compiled = _synthetic_compiled([inst])
    ev, id_maps = drawable_bridge.build_doc(compiled)
    feed = drawable_bridge.feed_frame(compiled, 0.0, id_maps)
    _ids, counts, _u, feed_f_bytes, _cov = feed
    feed_f = np.frombuffer(feed_f_bytes, dtype=np.float32).reshape(
        counts[0], drawable_bridge._FEED_F_STRIDE)

    # The copy is entry 1 (entry 0 is the identity original). Its feed translate
    # must equal the effect's own screen QTransform dx/dy for that entry.
    from analysis.games.notitg.field_instances import NotitgFieldInstances
    frame = NotitgFieldInstances(list(compiled['field_instances'])).at(
        drawable_bridge._Ctx(0.0, drawable_bridge._DESIGN_RECT))
    copy_qt = frame.fields[1][0]
    assert feed_f[1, 0] == pytest.approx(copy_qt.dx(), abs=1e-3)
    assert feed_f[1, 1] == pytest.approx(copy_qt.dy(), abs=1e-3)
    # And that faithful translate is the poke minus the design centre.
    assert feed_f[1, 0] == pytest.approx(123.0 - 320.0, abs=1e-3)
    assert feed_f[1, 1] == pytest.approx(-45.0 - 240.0, abs=1e-3)


def test_fill_scope_translates_to_src_fill_with_tint():
    """An AFT-rig fill entry becomes an SRC_FILL feed item carrying its rgb as
    the tint."""
    fill = fc.instance('curtain', 'fill', 0, [_link()])
    from analysis.player.render.effects.timeline import EventTimeline
    fill['color'] = EventTimeline([Keyframe(0.0, (0.2, 0.4, 0.6), 0.0, _EASE_LINEAR)],
                                  rest=(1.0, 1.0, 1.0))
    compiled = _synthetic_compiled([fill])
    ev, id_maps = drawable_bridge.build_doc(compiled)
    feed = drawable_bridge.feed_frame(compiled, 0.0, id_maps)
    _ids, counts, feed_u_bytes, feed_f_bytes, coverage = feed
    feed_u = np.frombuffer(feed_u_bytes, dtype=np.uint32).reshape(
        counts[0], drawable_bridge._FEED_U_STRIDE)
    feed_f = np.frombuffer(feed_f_bytes, dtype=np.float32).reshape(
        counts[0], drawable_bridge._FEED_F_STRIDE)
    # The identity original (proxy field) is prepended, then the fill.
    fill_row = feed_u[:, 0].tolist().index(sn.SRC_FILL)
    assert feed_u[fill_row, 0] == sn.SRC_FILL
    assert feed_f[fill_row, 6:9].tolist() == pytest.approx([0.2, 0.4, 0.6], abs=1e-4)


def test_sheared_entry_is_skipped_and_counted():
    """A skew (non-orthogonal linear block) cannot fit the TRS feed lanes, so
    the entry is skipped and counted in the coverage stats."""
    inst = fc.instance('sheared', 'proxy', 1, [_link(skew_x=0.6, x=10.0)])
    compiled = _synthetic_compiled([inst])
    ev, id_maps = drawable_bridge.build_doc(compiled)
    feed = drawable_bridge.feed_frame(compiled, 0.0, id_maps)
    _ids, counts, _u, _f, coverage = feed
    # The skewed copy is projective/sheared -> skipped; the identity original
    # (also present) is affine -> translated.
    assert coverage['skipped_projective'] >= 1
    assert coverage['translated'] >= 1
    assert coverage['total'] == coverage['translated'] + coverage['skipped_projective']


def test_decompose_rejects_shear_and_perspective():
    """The affine decomposition accepts a rotate+scale, rejects a shear and a
    projective transform."""
    from PySide6.QtGui import QTransform

    qt = QTransform()
    qt.translate(30.0, 40.0)
    qt.rotate(25.0)
    qt.scale(2.0, -1.5)
    trs = drawable_bridge._decompose_affine(qt)
    assert trs is not None
    x, y, sx, sy, rot = trs
    # Rebuild the rust mat3 from the recovered TRS and compare to the Qt
    # accessor block the executor reads.
    r = math.radians(rot)
    c, s = math.cos(r), math.sin(r)
    rebuilt = np.array([[c * sx, -s * sy, x], [s * sx, c * sy, y]])
    qt_block = np.array([[qt.m11(), qt.m21(), qt.dx()],
                         [qt.m12(), qt.m22(), qt.dy()]])
    assert np.allclose(rebuilt, qt_block, atol=1e-4)

    shear = QTransform(1.0, 0.0, 0.5, 1.0, 0.0, 0.0)
    assert drawable_bridge._decompose_affine(shear) is None

    persp = QTransform(1.0, 0.0, 0.001, 0.0, 1.0, 0.002, 0.0, 0.0, 1.0)
    assert drawable_bridge._decompose_affine(persp) is None


_GAT2 = ('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/'
         'UKSRT9/5. getfucked2/get_fucked_2.sm')


@pytest.mark.skipif(not os.path.exists(_GAT2), reason='gat 2 chart not on disk')
def test_gat2_smoke_build_and_feed():
    """Build the doc + feed three times against the real gat 2 chart: no
    exception, nonzero coverage, and print the coverage stats. The lazy
    provider grows as the background sweep runs, so poll it until the topology
    is populated before sampling."""
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
          f'(slots={len(id_maps["slots"])}, fields={len(id_maps["fields"])})')

    total_translated = 0
    for t in (100.0, 300.0, 500.0):
        feed = drawable_bridge.feed_frame(compiled, t, id_maps)
        feed_ids, counts, feed_u, feed_f, coverage = feed
        # Must not raise, and the evaluator must ingest the feed cleanly.
        u_raw, _f_raw, _uf, n = ev.frame_with_feeds(
            t, feed_ids, counts, feed_u, feed_f)
        assert n >= 0
        total_translated += coverage['translated']
        print(f'[gat2] t={t}: translated={coverage["translated"]} '
              f'skipped_projective={coverage["skipped_projective"]} '
              f'total={coverage["total"]}')
    assert total_translated > 0
