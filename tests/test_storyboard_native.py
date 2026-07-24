"""The Drawable core's Python surface (storyboard_native): Seam A doc
building, Seam B flat-buffer schedules, and the ordering semantics the
type sheet (.claude/plans/drawable-ir.md) freezes. Rust unit tests own
the fine-grained ordering rules; these guard the boundary contract.
"""
import numpy as np
import pytest

sn = pytest.importorskip('storyboard_native')


def _frames(evaluator, t):
    u_raw, f_raw, _uf_raw, n = evaluator.frame(t)
    u = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, evaluator.u_stride)
    f = np.frombuffer(f_raw, dtype=np.float32).reshape(n, evaluator.f_stride)
    return u, f


def _frames3(evaluator, t):
    """Like _frames but also returns the flat uniform-value buffer."""
    u_raw, f_raw, uf_raw, n = evaluator.frame(t)
    u = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, evaluator.u_stride)
    f = np.frombuffer(f_raw, dtype=np.float32).reshape(n, evaluator.f_stride)
    uf = np.frombuffer(uf_raw, dtype=np.float32)
    return u, f, uf


def test_schedule_round_trip_orders_and_samples():
    b = sn.DocBuilder(640.0, 480.0)
    fade = b.channel([0.0, 2.0], [0.0, 1.0], [2.0, 0.0], 0.0)
    slot = b.drawable(640.0, 480.0, True, False)
    b.item(0, sn.SRC_IMAGE, 3, opacity_id=fade, opacity_rest=0.0)
    b.snapshot(0, slot)
    b.item(0, sn.SRC_DRAWABLE, slot)
    ev = b.finish()

    u, f = _frames(ev, 1.0)
    kinds = u[:, 0].tolist()
    assert kinds == [sn.OP_BEGIN, sn.OP_BLIT, sn.OP_COPY, sn.OP_BLIT,
                     sn.OP_END]
    # The image blit samples the fade ramp (0 -> 1 over 2s) at t=1.
    assert f[1][9] == pytest.approx(0.5)
    # The slot blit reads the snapshotted drawable.
    assert (u[3][1], u[3][2]) == (sn.SRC_DRAWABLE, slot)


def test_referenced_drawable_composes_before_the_screen():
    b = sn.DocBuilder(640.0, 480.0)
    sub = b.drawable(64.0, 64.0, False, False)
    b.item(sub, sn.SRC_IMAGE, 7)
    b.item(0, sn.SRC_DRAWABLE, sub)
    ev = b.finish()

    u, _f = _frames(ev, 0.0)
    begins = u[u[:, 0] == sn.OP_BEGIN][:, 1].tolist()
    assert begins == [sub, 0]


def test_hidden_item_emits_no_op():
    b = sn.DocBuilder(640.0, 480.0)
    b.item(0, sn.SRC_FILL, 0, visible_id=-1, visible_rest=0.0)
    ev = b.finish()
    u, _f = _frames(ev, 0.0)
    assert u[:, 0].tolist() == [sn.OP_BEGIN, sn.OP_END]


def test_shader_uniform_values_ride_the_side_buffer():
    b = sn.DocBuilder(640.0, 480.0)
    sh = b.shader('frag-src', None, ['strength', 'phase'])
    # A plain fill binds no shader; a shaded fill binds two uniforms, one
    # a ramp (0 -> 4 over 2s) sampled at t=1, one a constant.
    b.item(0, sn.SRC_FILL, 0)
    ramp = b.channel([0.0, 2.0], [0.0, 4.0], [2.0, 0.0], 0.0)
    b.item(0, sn.SRC_FILL, 0)
    b.item_shader(0, sh)
    b.item_uniform(0, 0, ramp, 0.0)   # strength <- ramp
    b.item_uniform(0, 1, -1, 9.0)     # phase <- constant 9
    ev = b.finish()

    u, _f, uf = _frames3(ev, 1.0)
    blits = u[u[:, 0] == sn.OP_BLIT]
    assert len(blits) == 2
    # Plain blit: no shader, zero uniform count.
    assert blits[0][5] == 0 and blits[0][9] == 0
    # Shaded blit: shader+1, offset/count into the uniform buffer.
    assert blits[1][5] == sh + 1
    off, cnt = int(blits[1][8]), int(blits[1][9])
    assert cnt == 2
    assert uf[off] == pytest.approx(2.0)   # ramp at t=1
    assert uf[off + 1] == pytest.approx(9.0)


def test_clip_id_rides_lane_six():
    b = sn.DocBuilder(640.0, 480.0)
    rect = b.clip_rect(0.0, 0.0, 320.0, 240.0)
    poly = b.clip_polygon([0.0, 0.0, 10.0, 0.0, 10.0, 10.0])
    b.item(0, sn.SRC_FILL, 0)
    b.item_clip(0, rect)
    b.item(0, sn.SRC_FILL, 0)
    b.item_clip(0, poly)
    ev = b.finish()

    u, _f = _frames(ev, 0.0)
    clips = u[u[:, 0] == sn.OP_BLIT][:, 6].tolist()
    assert clips == [rect + 1, poly + 1]


def test_frame_with_feeds_ingests_soa_and_matches_static():
    b = sn.DocBuilder(640.0, 480.0)
    notes = b.drawable(640.0, 480.0, False, True)  # dynamic
    b.item(0, sn.SRC_DRAWABLE, notes)
    ev = b.finish()

    # Two fed items in the frozen feed v2 SoA layout: u32 stride 4, f32
    # stride 18 (mat3 lanes 0..9, then opacity, tint, crop, z).
    fu = ev.feed_u_stride
    ff = ev.feed_f_stride
    assert (fu, ff) == (4, 18)
    ADDITIVE = 1
    u = np.zeros((2, fu), dtype=np.uint32)
    f = np.zeros((2, ff), dtype=np.float32)
    # item 0: image 5, additive, opacity 0.5, mat3 translate(10, 20) in the
    # record's column-vector layout (tx/ty in lanes 2/5).
    u[0] = [sn.SRC_IMAGE, 5, 0, ADDITIVE]
    f[0, :9] = [1.0, 0.0, 10.0, 0.0, 1.0, 20.0, 0.0, 0.0, 1.0]
    f[0, 9] = 0.5   # opacity
    # item 1: image 6, plain, identity mat3, opacity 1.0.
    u[1] = [sn.SRC_IMAGE, 6, 0, 0]
    f[1, :9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    f[1, 9] = 1.0   # opacity

    u_raw, f_raw, _uf_raw, n = ev.frame_with_feeds(
        0.0, [notes], [2], u.tobytes(), f.tobytes())
    uu = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, ev.u_stride)
    fu_out = np.frombuffer(f_raw, dtype=np.float32).reshape(n, ev.f_stride)

    blits = uu[uu[:, 0] == sn.OP_BLIT]
    # Two fed image blits, then the screen's drawable blit consuming them.
    assert [(int(r[1]), int(r[2])) for r in blits] == [
        (sn.SRC_IMAGE, 5), (sn.SRC_IMAGE, 6), (sn.SRC_DRAWABLE, notes)]
    # The additive flag reached the blit's blend lane (lane 4).
    assert blits[0][4] == 1 and blits[1][4] == 0
    # The fed mat3/opacity survived verbatim: item 0's mat3 tx/ty = 10/20
    # (record lanes 2/5), opacity 0.5 (f lane 9).
    fed0 = fu_out[uu[:, 0] == sn.OP_BLIT][0]
    assert fed0[2] == pytest.approx(10.0)   # mat[0][2] = tx
    assert fed0[5] == pytest.approx(20.0)   # mat[1][2] = ty
    assert fed0[9] == pytest.approx(0.5)    # opacity

    # frame() with no feeds leaves the dynamic drawable empty (only the
    # screen's consuming blit, which reads the retained/empty target).
    u2, _f2 = _frames(ev, 0.0)
    static_blits = u2[u2[:, 0] == sn.OP_BLIT]
    assert [(int(r[1]), int(r[2])) for r in static_blits] == [
        (sn.SRC_DRAWABLE, notes)]


# --- wave 2: full link chain, camera projection, schedule lowering -------

def test_linkless_item_transform_is_bit_identical():
    # Wiring the full-transform path must not perturb linkless items: a
    # plain translate(10,20) scale(2,3) yields the frozen first-cut mat3.
    b = sn.DocBuilder(640.0, 480.0)
    b.item(0, sn.SRC_FILL, 0, x_rest=10.0, y_rest=20.0, sx_rest=2.0, sy_rest=3.0)
    ev = b.finish()
    _u, f = _frames(ev, 0.0)
    blit = f[1]  # after BEGIN
    # Column-vector record mat3 [a b tx; c d ty; 0 0 1].
    assert blit[:9].tolist() == [2.0, 0.0, 10.0, 0.0, 3.0, 20.0, 0.0, 0.0, 1.0]


def test_item_link_uses_full_compose_chain():
    # One default link folds through compose_links to the centered
    # _TO_CONTENT translate; the record carries its column-vector transpose
    # (tx=-320 in lane 2, ty=-240 in lane 5), and the link alpha multiplies.
    b = sn.DocBuilder(640.0, 480.0)
    b.item(0, sn.SRC_FILL, 0, opacity_rest=0.6)
    b.item_link(0, alpha_rest=0.5)
    ev = b.finish()
    u, f = _frames(ev, 0.0)
    blits = u[u[:, 0] == sn.OP_BLIT]
    assert len(blits) == 1
    blit = f[u[:, 0] == sn.OP_BLIT][0]
    assert blit[0] == pytest.approx(1.0)   # scale x
    assert blit[4] == pytest.approx(1.0)   # scale y
    assert blit[2] == pytest.approx(-320.0, abs=1e-2)  # tx
    assert blit[5] == pytest.approx(-240.0, abs=1e-2)  # ty
    assert blit[9] == pytest.approx(0.3)   # 0.6 opacity * 0.5 link alpha


def test_item_link_hidden_drops_the_item():
    b = sn.DocBuilder(640.0, 480.0)
    b.item(0, sn.SRC_FILL, 0)
    b.item_link(0, hidden_rest=1.0)
    ev = b.finish()
    u, _f = _frames(ev, 0.0)
    assert u[:, 0].tolist() == [sn.OP_BEGIN, sn.OP_END]


def test_item_link_flip_swaps_leaf_crop():
    # A flipped leaf swaps top/bottom crop into the record crop lanes.
    b = sn.DocBuilder(640.0, 480.0)
    b.item(0, sn.SRC_FILL, 0)
    b.item_link(0, crop_t_rest=0.1, crop_b_rest=0.3, flip_base_y=True)
    ev = b.finish()
    u, f = _frames(ev, 0.0)
    blit = f[u[:, 0] == sn.OP_BLIT][0]
    assert blit[14] == pytest.approx(0.3)  # crop top <- bottom
    assert blit[16] == pytest.approx(0.1)  # crop bottom <- top


def test_item_projection_centered_fov45_is_identity():
    # A centered fov-45 projection folds to identity on an untransformed
    # fullscreen item: the record mat3 stays the identity homography.
    b = sn.DocBuilder(640.0, 480.0)
    b.item(0, sn.SRC_FILL, 0)
    # far = eye_distance(45, 640) + 1000 ~= 1772.7 (the item_projection default).
    b.item_projection(0, fov_rest=45.0, vanish_x_rest=320.0, vanish_y_rest=240.0)
    ev = b.finish()
    u, f = _frames(ev, 0.0)
    m = f[u[:, 0] == sn.OP_BLIT][0][:9]
    m = m / m[8]  # homographies are scale-free
    assert m.tolist() == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], abs=1e-4)


def _seg(dur, prop, value, mode='abs', ease=0):
    return {'kind': 'Seg', 'dur': dur, 'ease': ease,
            'targets': [{'prop': prop, 'mode': mode, 'value': value}],
            'effect': None, 'fire_id': -1}


def test_schedule_channel_lowers_a_ramp_and_samples_it():
    # A single 0 -> 100 ramp over 1s, lowered to a doc channel; sampling it
    # through an item's opacity-like use is the same fold the Rust lower()
    # produces (0 at t=0, 50 at t=0.5, 100 held after).
    b = sn.DocBuilder(640.0, 480.0)
    node = _seg(1.0, 0, 100.0)
    cid = b.schedule_channel(node, 0.0, -1.0, 0, state=[(0, 0.0)])
    assert cid is not None
    # Bind the channel to an item's x and read it back at three times.
    b.item(0, sn.SRC_FILL, 0, x_id=cid, x_rest=0.0)
    ev = b.finish()
    for t, want in [(0.0, 0.0), (0.5, 50.0), (2.0, 100.0)]:
        u, f = _frames(ev, t)
        blit = f[u[:, 0] == sn.OP_BLIT][0]
        assert blit[2] == pytest.approx(want)  # tx lane = sampled x


def test_schedule_channel_absent_prop_returns_none():
    b = sn.DocBuilder(640.0, 480.0)
    node = _seg(1.0, 0, 100.0)
    # The schedule touches prop 0, never prop 5.
    assert b.schedule_channel(node, 0.0, -1.0, 5, state=[(0, 0.0)]) is None


def test_schedule_channel_unrolls_a_loop_to_the_horizon():
    b = sn.DocBuilder(640.0, 480.0)
    loop = {'kind': 'Loop', 'period': 2.0, 'body': _seg(1.0, 0, 10.0)}
    cid = b.schedule_channel(loop, 0.0, 7.0, 0, state=[(0, 0.0)])
    assert cid is not None
    b.item(0, sn.SRC_FILL, 0, x_id=cid, x_rest=0.0)
    ev = b.finish()
    # First pass ramps 0 -> 10 over [0,1]; midpoint is 5.
    u, f = _frames(ev, 0.5)
    assert f[u[:, 0] == sn.OP_BLIT][0][2] == pytest.approx(5.0)


def test_schedule_fires_returns_effect_times():
    b = sn.DocBuilder(640.0, 480.0)
    # The nested_with_fire fixture: a 2s seg whose effect (a Seq: hibernate
    # 1s, then a fire seg) joins the queue tail, so it fires after the outer
    # seg completes (t=2) plus the hibernate (t=3). Matches lower() exactly.
    fire_seg = {'kind': 'Seg', 'dur': 0.0, 'ease': 0, 'targets': [],
                'effect': None, 'fire_id': 0}
    effect = {'kind': 'Seq', 'parts': [{'kind': 'Hibernate', 'dur': 1.0}, fire_seg]}
    node = {'kind': 'Seg', 'dur': 2.0, 'ease': 0,
            'targets': [{'prop': 0, 'mode': 'abs', 'value': 5.0}],
            'effect': effect, 'fire_id': -1}
    fires = b.schedule_fires(node, 0.0, -1.0, state=[(0, 0.0)])
    assert fires == pytest.approx([3.0])


# --- wave 4: event-driven reactions (spliced Schedule fragments) ---------


def _opacity(evaluator, t, kinds, times, columns, strengths):
    """The single BLIT op's opacity lane through frame_with_events."""
    u_raw, f_raw, _uf, n = evaluator.frame_with_events(
        t, kinds, times, columns, strengths)
    u = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, evaluator.u_stride)
    f = np.frombuffer(f_raw, dtype=np.float32).reshape(n, evaluator.f_stride)
    return f[u[:, 0] == sn.OP_BLIT][0][9]


def _opacity_ramp():
    # A 0 -> 1 opacity ramp over 1s (eases off the seeded base at te).
    return {'kind': 'Seg', 'dur': 1.0, 'ease': 0,
            'targets': [{'prop': sn.PROP_OPACITY, 'mode': 'abs', 'value': 1.0}],
            'effect': None, 'fire_id': -1}


def test_reaction_press_ramps_opacity_after_the_event():
    b = sn.DocBuilder(640.0, 480.0)
    b.item(0, sn.SRC_FILL, 0, opacity_rest=0.4)
    b.item_reaction(0, sn.EV_PRESS, -1, _opacity_ramp(), sn.PROP_OPACITY)
    ev = b.finish()

    press = ([sn.EV_PRESS], [1.0], [-1], [1.0])
    # Before the press: the base opacity (0.4) rules.
    assert _opacity(ev, 0.5, *press) == pytest.approx(0.4)
    # After: the ramp eases 0.4 -> 1.0 over [1, 2]; midpoint 1.5 -> 0.7.
    assert _opacity(ev, 1.5, *press) == pytest.approx(0.7)
    assert _opacity(ev, 2.5, *press) == pytest.approx(1.0)


def test_reaction_base_channel_rules_with_no_event():
    b = sn.DocBuilder(640.0, 480.0)
    b.item(0, sn.SRC_FILL, 0, opacity_rest=0.4)
    b.item_reaction(0, sn.EV_PRESS, -1, _opacity_ramp(), sn.PROP_OPACITY)
    ev = b.finish()
    # No events this frame: the base channel is untouched.
    assert _opacity(ev, 1.5, [], [], [], []) == pytest.approx(0.4)


def test_reaction_second_event_resplices():
    b = sn.DocBuilder(640.0, 480.0)
    b.item(0, sn.SRC_FILL, 0, opacity_rest=0.4)
    b.item_reaction(0, sn.EV_PRESS, -1, _opacity_ramp(), sn.PROP_OPACITY)
    ev = b.finish()
    # Two presses; the later (te=5) wins - the fresh ramp reads 0.7 at 5.5,
    # not the held 1.0 of the first press.
    two = ([sn.EV_PRESS, sn.EV_PRESS], [1.0, 5.0], [-1, -1], [1.0, 1.0])
    assert _opacity(ev, 5.5, *two) == pytest.approx(0.7)


def test_reaction_column_filter_respected():
    b = sn.DocBuilder(640.0, 480.0)
    b.item(0, sn.SRC_FILL, 0, opacity_rest=0.4)
    b.item_reaction(0, sn.EV_PRESS, 3, _opacity_ramp(), sn.PROP_OPACITY)
    ev = b.finish()
    # A column-1 press is filtered out; a column-3 press fires.
    assert _opacity(ev, 2.0, [sn.EV_PRESS], [1.0], [1], [1.0]) == pytest.approx(0.4)
    assert _opacity(ev, 2.0, [sn.EV_PRESS], [1.0], [3], [1.0]) == pytest.approx(1.0)


# -- C12 ease-aware channels --------------------------------------------

def test_channel_eases_match_the_python_timeline():
    """A curved-ease channel samples exactly what the repo's EventTimeline
    plays for the same keyframe - so a Python-compiled eased keyframe
    crosses Seam A losslessly (no 1/30s densification)."""
    from analysis.player.render.effects.timeline import EventTimeline, Keyframe

    easing = 13  # OutQuint
    # Positive opacity range so the item never trips the transparent-item
    # drop gate; opacity is the probed BLIT lane.
    t0, dur, v0, v1 = 2.0, 4.0, 0.1, 0.9
    tl = EventTimeline(
        [Keyframe(t=t0, values=(v1,), duration=dur, easing=easing, start=(v0,))],
        rest=(v0,),
    )

    b = sn.DocBuilder(640.0, 480.0)
    ramp = b.channel([t0, t0 + dur], [v0, v1], [dur, 0.0], v0, eases=[easing, 0])
    b.item(0, sn.SRC_FILL, 0, opacity_id=ramp, opacity_rest=v0)
    ev = b.finish()

    for t in (1.0, 2.0, 3.0, 4.5, 6.0, 7.5):
        u, f = _frames(ev, t)
        got = f[u[:, 0] == sn.OP_BLIT][0][9]  # BLIT opacity lane
        assert got == pytest.approx(tl.sample(t)[0], abs=1e-4), f't={t}'


def test_channel_default_is_linear_and_curved_diverges():
    b = sn.DocBuilder(640.0, 480.0)
    lin = b.channel([0.0, 4.0], [0.0, 4.0], [4.0, 0.0], 0.0)
    curved = b.channel([0.0, 4.0], [0.0, 4.0], [4.0, 0.0], 0.0, eases=[2, 0])  # InQuad
    b.item(0, sn.SRC_FILL, 0, opacity_id=lin, opacity_rest=0.0)
    b.item(0, sn.SRC_FILL, 0, opacity_id=curved, opacity_rest=0.0)
    ev = b.finish()
    _u, f = _frames(ev, 2.0)  # midpoint
    lin_val, curved_val = f[1][9], f[2][9]
    assert lin_val == pytest.approx(2.0)             # linear midpoint
    assert curved_val == pytest.approx(1.0, abs=1e-4)  # InQuad(0.5)*4 = 1.0


# -- B8 mesh crate-side -------------------------------------------------

def test_mesh_item_record_lanes():
    """A registered mesh is referenced by a Source::Mesh item, whose BLIT
    record carries (SRC_MESH, mesh_id) in the source lanes."""
    b = sn.DocBuilder(640.0, 480.0)
    quad = b.mesh([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1], mode=0)
    other = b.mesh([0, 0, 0, 0], mode=1, vert_shader_id=-1, vert_source='void main(){}')
    b.item(0, sn.SRC_MESH, quad)
    b.item(0, sn.SRC_MESH, other)
    ev = b.finish()
    assert ev.mesh_count() == 2

    u, _f = _frames(ev, 0.0)
    blits = u[u[:, 0] == sn.OP_BLIT]
    srcs = [(int(row[1]), int(row[2])) for row in blits]
    assert srcs == [(sn.SRC_MESH, quad), (sn.SRC_MESH, other)]


# -- C15 nested SortSpan ruling -----------------------------------------

def test_nested_sort_span_is_rejected_at_build():
    b = sn.DocBuilder(640.0, 480.0)
    b.sort_span(0, 3)          # outer span covers the next 3 commands
    b.item(0, sn.SRC_FILL, 0)
    b.sort_span(0, 1)          # inner span inside the outer window -> error
    b.item(0, sn.SRC_FILL, 0)
    with pytest.raises(ValueError, match='nested SortSpan'):
        b.finish()


def test_snapshot_inside_sort_span_is_allowed():
    b = sn.DocBuilder(640.0, 480.0)
    slot = b.drawable(640.0, 480.0, True, False)
    b.sort_span(0, 2)
    b.item(0, sn.SRC_IMAGE, 5, z_rest=1.0, has_z=True)
    b.snapshot(0, slot)        # a snapshot within the span sorts with it
    ev = b.finish()            # no error
    u, _f = _frames(ev, 0.0)
    # The snapshot (z 0) sorts before the z=1 item.
    span_ops = [k for k in u[:, 0].tolist() if k in (sn.OP_BLIT, sn.OP_COPY)]
    assert span_ops == [sn.OP_COPY, sn.OP_BLIT]
