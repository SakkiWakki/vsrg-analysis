"""Completeness + per-mechanism tests for the generated verb surface.

The completeness test is the point: every name in the vendored 221-method
list (analysis/games/notitg/actor_api_names.txt) must resolve to a
(mechanism, target) or an explicit IGNORED / DEFERRED entry with a reason,
so a registered actor method with no mapping fails CI instead of silently
no-oping. The per-mechanism tests exercise each mechanism against a real
SimActor, engine-cited to openitg Actor.h (the LunaActor ADD_METHOD
block) where the semantics are stock. No gat / local-chart dependence.
"""
from pathlib import Path

import pytest

from analysis.games.notitg import sim as sim_pkg
from analysis.games.notitg.sim import SimActor, verb_surface as vs
from analysis.games.notitg.sim.verb_surface import (
    READ_CURRENT, READ_DEST, all_targets)

_NAME_LIST = (Path(sim_pkg.__file__).resolve().parent.parent
              / 'actor_api_names.txt')


def _spec_names() -> list:
    lines = _NAME_LIST.read_text().splitlines()
    return [ln.strip() for ln in lines
            if ln.strip() and not ln.strip().startswith('#')]


# -- completeness: the CI firewall -------------------------------------------

def test_name_list_is_the_full_surface():
    """221 unique names, no duplicates - the vendored spec matches the
    decompile's Actor::PushSelf count (ACTOR_LUA_API.md)."""
    names = _spec_names()
    assert len(names) == 221
    assert len(set(names)) == 221


def test_every_registered_name_resolves():
    """Zero unmapped names: a registered actor method with no mechanism
    mapping is the failure this test exists to catch."""
    targets = all_targets()
    unmapped = [n for n in _spec_names() if n not in targets]
    assert unmapped == [], f'unmapped actor methods: {unmapped}'


def test_ignored_and_deferred_carry_a_reason():
    for name, reason in {**vs.IGNORED, **vs.DEFERRED}.items():
        assert reason and reason.strip(), f'{name} has an empty reason'


def test_no_name_maps_to_two_mechanisms():
    """Each name lands in exactly one mechanism table - a name in both a
    setter table and DEFERRED would silently pick one."""
    tables = (vs.SCALAR_SETTERS, vs.ADD_SETTERS, vs.BULK_SETTERS,
              vs.BULK_ADD_SETTERS, vs.SIZE_PAIR_SETTERS, vs.SIZE_AXIS_SETTERS,
              vs.GETTERS, vs.TUPLE_GETTERS, vs.HANDLED_BY_NAME,
              vs.IGNORED, vs.DEFERRED)
    seen: dict = {}
    for table in tables:
        for name in table:
            assert name not in seen, f'{name} in two mechanism tables'
            seen[name] = True


def test_getter_read_kinds_are_valid():
    for name, (_, read) in vs.GETTERS.items():
        assert read in (READ_CURRENT, READ_DEST), name


# Per-verb args where the generic single float would not exercise the
# real route (a command string, a name, a mode token, a rect).
_POKE_ARGS = {
    'queuecommand': ['Foo'], 'queuemessage': ['Foo'],
    'effectclock': ['music'], 'blend': ['add'],
    'SetRotationOrder': ['zyx'],
    'stretchto': [0.0, 0.0, 640.0, 480.0],
    'scaletocover': [0.0, 0.0, 640.0, 480.0],
    'scaletofit': [0.0, 0.0, 640.0, 480.0],
}

# Handled-by-name verbs that never reach SimActor.poke: the recorder
# bridge routes playcommand/queuecommand named-dispatch and Actor:cmd
# through the environment (env._actor_command / env._actor_poke's cmd
# branch, covered by tests/test_notitg_sim_load.py).
_ENV_ROUTED = {'playcommand', 'cmd',
               # note-shader binds resolve their actor-table argument in
               # env._actor_poke; SimActor sees set_note_shader directly.
               'SetArrowShader', 'SetHoldShader', 'SetReceptorShader',
               'ClearArrowShader', 'ClearHoldShader',
               'ClearReceptorShader',
               # command-registry writes resolve their Lua-function
               # argument in env._actor_poke.
               'addcommand', 'removecommand', 'luaeffect',
               'PushNoteData'}


def test_every_handled_claim_actually_routes():
    """A verb HANDLED_BY_NAME claims is handled must not fall through to
    the drop tail - Actor:cmd carried the 'command' tag for months while
    every call silently vanished. Getter-mechanism names verify through
    read(); the rest poke through the real dispatch with dropped_notify
    armed."""
    for name in vs.HANDLED_BY_NAME:
        if name in _ENV_ROUTED:
            continue
        if name.lower().startswith('get'):
            actor = SimActor()
            assert actor.read(name) is not None, \
                f'{name} claimed handled but read() has no answer'
            continue
        actor = SimActor()
        dropped = []
        actor.dropped_notify = dropped.append
        actor.poke(name, _POKE_ARGS.get(name, [1.0]))
        assert not dropped, \
            f'{name} claimed handled but fell through to the drop tail'


def test_unmapped_verb_reports_via_dropped_notify():
    actor = SimActor()
    dropped = []
    actor.dropped_notify = dropped.append
    actor.poke('no_such_verb', [1.0])
    assert dropped == ['no_such_verb']


def test_documented_ignored_and_deferred_verbs_stay_silent():
    actor = SimActor()
    dropped = []
    actor.dropped_notify = dropped.append
    ignored = next(iter(vs.IGNORED))
    deferred = 'x2'
    actor.poke(ignored, [1.0])
    actor.poke(deferred, [1.0])
    assert dropped == []


def test_trap_families_are_deferred_not_guessed():
    """The '*2' second-slot family stays DEFERRED (Actor.clean.c is
    COMDAT-folded and openitg has no analogue, so its dual-transform role
    cannot be pinned), never mapped to a scalar write. The rotation-order,
    skew-order, and spherical families are NO LONGER traps - the fork's
    SetRotationOrder swizzle, the pre/post-rotation skew gates, and the
    RageQuat* spherical adds are all pinned from Actor.clean.c BeginDraw +
    openitg RageMath (see test_transform_order_family below)."""
    for name in ('x2', 'zoom2', 'rotationz2', 'GetX2', 'skewx2'):
        assert name in vs.DEFERRED, name
        for setter in (vs.SCALAR_SETTERS, vs.ADD_SETTERS, vs.BULK_SETTERS):
            assert name not in setter, f'{name} was mapped to a write'


# -- fork transform-order / spherical rotation / skew-order ------------------

def test_transform_order_family_is_no_longer_deferred():
    """The rotation-order, skew-order, and spherical verbs moved out of
    DEFERRED into the handled-by-name surface."""
    for name in ('SetRotationOrder', 'GetRotationOrder', 'skewto',
                 'skewx_before_rotation', 'skewy_before_rotation',
                 'GetSkewXBeforeRotation', 'GetSkewYBeforeRotation',
                 'heading', 'pitch', 'roll'):
        assert name not in vs.DEFERRED, name
        assert name in vs.HANDLED_BY_NAME, name


def test_set_rotation_order_records_and_reads_back():
    a = SimActor()
    assert a.read('GetRotationOrder') == 'xyz'  # engine default
    a.poke('SetRotationOrder', ['zyx'])
    assert a.read('GetRotationOrder') == 'zyx'
    # An unknown token logs-and-ignores (engine 'Invalid Rotation mode').
    a.poke('SetRotationOrder', ['bogus'])
    assert a.read('GetRotationOrder') == 'zyx'


def test_rotation_order_changes_the_composed_matrix():
    # rotate_ordered('xyz') == rotate_xyz; a non-xyz order differs when two
    # axes are non-zero, but the default order MUST stay byte-identical.
    from analysis.player.render import transform3d as t3
    import numpy as np
    m_default = t3.rotate_ordered(20.0, 40.0, 0.0, 'xyz')
    assert np.array_equal(m_default, t3.rotate_xyz(20.0, 40.0, 0.0))
    m_zyx = t3.rotate_ordered(20.0, 40.0, 0.0, 'zyx')
    assert not np.allclose(m_default, m_zyx)


def test_align_verbs_record_the_anchor():
    # halign/valign move the actor's anchor fraction; AFT band rigs poke
    # them at runtime (cropbottom 0.5 + valign 0.75 = top half in place),
    # so they are recorded scalars, not load-time layout.
    a = SimActor()
    a.poke('halign', [0.25])
    a.poke('valign', [0.75])
    assert a.get('halign') == pytest.approx(0.25)
    assert a.get('valign') == pytest.approx(0.75)


def test_align_shorthand_sets_both_axes():
    a = SimActor()
    a.poke('align', [0.0])
    assert a.get('halign') == 0.0
    assert a.get('valign') == 0.0


def test_align_rests_centered():
    a = SimActor()
    assert a.get('halign') == 0.5
    assert a.get('valign') == 0.5


def test_skewto_sets_both_skew_axes():
    a = SimActor()
    a.poke('skewto', [0.3, -0.2])
    assert a.get('skew_x') == pytest.approx(0.3)
    assert a.get('skew_y') == pytest.approx(-0.2)


def test_skew_before_rotation_flag_records_and_reads():
    a = SimActor()
    assert a.read('GetSkewXBeforeRotation') == 0.0  # skew-after default
    a.poke('skewx_before_rotation', [1])
    assert a.read('GetSkewXBeforeRotation') == 1.0
    a.poke('skewy_before_rotation', [1])
    assert a.read('GetSkewYBeforeRotation') == 1.0


def test_spherical_adds_accumulate_a_quaternion():
    # A single roll(90) about z should match the z-axis quat; a rest actor
    # holds the identity quat.
    from analysis.player.render import transform3d as t3
    a = SimActor()
    assert a._current.get('quat') is None  # untouched -> identity at compose
    a.poke('roll', [90.0])
    q = a._current['quat']
    expected = t3.quat_from_axis('z', 90.0)
    assert q == pytest.approx(expected)


def test_spherical_add_is_a_true_rotation_matrix():
    # matrix_from_quat of a heading(90) quat is a proper rotation (det 1,
    # orthonormal), and identity quat -> identity matrix (parity anchor).
    from analysis.player.render import transform3d as t3
    import numpy as np
    ident = t3.matrix_from_quat((0.0, 0.0, 0.0, 1.0))
    assert np.allclose(ident, np.eye(4))
    q = t3.quat_from_axis('y', 90.0)
    m = t3.matrix_from_quat(q)
    assert np.linalg.det(m) == pytest.approx(1.0)
    assert np.allclose(m @ m.T, np.eye(4), atol=1e-9)


# -- mechanism 1: dest-state scalar write ------------------------------------

def test_scalar_write_lands_on_the_property():
    # x -> SetX -> DestTweenState().pos.x (Actor.h:113/481).
    a = SimActor()
    a.poke('x', [42])
    assert a.get('x') == 42
    a.poke('rotationz', [90])
    assert a.get('rotation') == 90


def test_uniform_zoom_sets_both_axes():
    # zoom(f) -> SetZoom -> both scale axes (Actor.h:148/487).
    a = SimActor()
    a.poke('zoom', [2.0])
    assert a.get('scale_x') == 2.0
    assert a.get('scale_y') == 2.0


def test_scalar_setter_table_targets_match_rest_keys():
    """Every scalar-setter property has a rest value - an unknown property
    would sample to a wrong default."""
    from analysis.games.notitg.lua_api import _REST
    for prop in vs.SCALAR_SETTERS.values():
        for p in (prop if isinstance(prop, tuple) else (prop,)):
            assert p in _REST, p


# -- mechanism 2: relative add -----------------------------------------------

def test_add_stacks_on_destination():
    # AddX(v) = SetX(GetDestX()+v) (Actor.h:117/484).
    a = SimActor()
    a.poke('x', [10])
    a.poke('addx', [5])
    assert a.get_dest('x') == 15


def test_rotation_add_targets_the_axis():
    a = SimActor()
    a.poke('rotationz', [30])
    a.poke('addrotationz', [15])
    assert a.get_dest('rotation') == 45


# -- mechanism 3: bulk expansion ---------------------------------------------

def test_bulk_xyz_writes_three_axes():
    a = SimActor()
    for verb, args in [('xy', [1, 2]), ('xyz', [4, 5, 6])]:
        a.poke(verb, args)
    assert (a.get('x'), a.get('y'), a.get('z')) == (4, 5, 6)


def test_bulk_xyza_includes_alpha():
    a = SimActor()
    a.poke('xyza', [1, 2, 3, 0.5])
    assert (a.get('x'), a.get('y'), a.get('z'), a.get('alpha')) == (1, 2, 3, 0.5)


def test_bulk_rotationxyz_writes_all_rotation_axes():
    a = SimActor()
    a.poke('rotationxyz', [10, 20, 30])
    assert a.getrotation() == (10, 20, 30)


def test_bulk_add_stacks_each_axis_on_destination():
    a = SimActor()
    a.poke('rotationxyz', [1, 2, 3])
    a.poke('addrotationxyz', [10, 10, 10])
    assert (a.get_dest('rotation_x'), a.get_dest('rotation_y'),
            a.get_dest('rotation')) == (11, 12, 13)


# -- mechanism 6: getter dest vs current read --------------------------------

def test_getx_reads_in_flight_current():
    # GetX -> m_current (Actor.h:107) - mid-tween interpolation.
    a = SimActor()
    a.poke('linear', [1.0])
    a.poke('x', [100])
    a.update_to(0.5)
    prop, read = vs.GETTERS['GetX']
    assert read == READ_CURRENT
    assert a.get(prop) == pytest.approx(50.0)


def test_getzoomx_reads_the_destination():
    # GetZoomX -> DestTweenState (Actor.h:145) - the settled target, not
    # the mid-flight value.
    a = SimActor()
    a.poke('linear', [1.0])
    a.poke('zoomx', [3.0])
    a.update_to(0.5)
    prop, read = vs.GETTERS['GetZoomX']
    assert read == READ_DEST
    assert a.get_dest(prop) == pytest.approx(3.0)
    assert a.get(prop) == pytest.approx(2.0)  # current is mid-flight


def test_get_current_rotation_reads_current_not_dest():
    # The fork GetCurrentRotationZ reads m_current, unlike GetRotationZ.
    a = SimActor()
    a.poke('linear', [1.0])
    a.poke('rotationz', [90])
    a.update_to(0.5)
    cur_prop, cur_read = vs.GETTERS['GetCurrentRotationZ']
    dest_prop, dest_read = vs.GETTERS['GetRotationZ']
    assert (cur_read, dest_read) == (READ_CURRENT, READ_DEST)
    assert a.get(cur_prop) == pytest.approx(45.0)
    assert a.get_dest(dest_prop) == pytest.approx(90.0)


# -- mechanism 12: ignore / defer sweep --------------------------------------

def test_ignored_render_hint_is_a_no_op():
    """An ignored render-state hint pokes nothing - no keyframe, no state
    change (ztest is a depth toggle the 2D compositor has no use for)."""
    a = SimActor()
    a.poke('ztest', [1])
    a.poke('cullmode', ['back'])
    assert a.keyframes() == {}


def test_deferred_verb_does_not_silently_write():
    """A deferred trap verb (x2) must not land as an x write - deferring
    means the sim leaves the property at rest, not that it guesses."""
    a = SimActor()
    a.poke('x2', [999])
    assert a.get('x') == 0.0
    assert a.keyframes() == {}


# -- gat 2 backlog: Load / texcoordvelocity / SetAwake ------------------------

def test_small_sprite_verbs_are_no_longer_deferred():
    for name in ('Load', 'texcoordvelocity', 'SetAwake'):
        assert name not in vs.DEFERRED, name
        assert name in vs.HANDLED_BY_NAME, name


def test_texcoordvelocity_records_a_closed_form_anchor():
    """The recorded anchor (t0, offset, velocity) reproduces the engine's
    per-Update UV accumulation: a re-poke folds the offset scrolled so
    far into the new anchor, so offset(t) stays continuous across
    velocity changes (Sprite.cpp:346-359)."""
    a = SimActor()
    a.poke('texcoordvelocity', [0.0, 0.25])
    a.update_to(4.0)
    a.poke('texcoordvelocity', [1.0, 0.0])
    t0, ou, ov, vu, vv = a.keyframes()['texcoord_scroll'][-1].values
    assert (t0, vu, vv) == (4.0, 1.0, 0.0)
    assert ou == pytest.approx(0.0)
    assert ov == pytest.approx(1.0)  # 0.25 UV/s for 4s


def test_set_awake_records_the_gate_and_rests_awake():
    a = SimActor()
    a.poke('SetAwake', [False])
    assert a.keyframes()['awake'][-1].values == (0.0,)
    a.poke('SetAwake', [True])
    assert a.keyframes()['awake'][-1].values == (1.0,)


def test_load_records_resolved_path_with_sheet_grid(tmp_path):
    img = tmp_path / 'judgent labil 2x6.png'
    img.write_bytes(b'\x89PNG')
    a = SimActor()
    a.asset_resolver = lambda raw: str(img) if raw.endswith('.png') else None
    a.poke('Load', ['lua/judgent labil 2x6.png'])
    assert a.keyframes()['asset_swap'][-1].values == (str(img), 2.0, 6.0)


def test_load_drops_nonpath_and_unresolvable_arguments():
    """The engine no-ops on an empty path; a permissive-stub argument
    (THEME:GetPath under the sandbox) is not a string; an unresolvable
    path must keep the previous texture instead of recording a swap."""
    a = SimActor()
    a.asset_resolver = lambda raw: None
    a.poke('Load', [object()])
    a.poke('Load', [''])
    a.poke('Load', ['no/such/file.png'])
    assert 'asset_swap' not in a.keyframes()


def test_uniform_texture_is_a_handled_static_bind():
    assert 'uniformTexture' not in vs.DEFERRED
    assert vs.HANDLED_BY_NAME['uniformTexture'] == 'shader-sampler'
    a = SimActor()
    marker = type('Tex', (), {'marker': 'file:/charts/asciitable.png'})()
    a.poke('uniformTexture', ['samplerAscii', marker])
    assert a.sampler_binds == {'samplerAscii': 'file:/charts/asciitable.png'}
    # A non-texture second argument (permissive stub, plain float) is
    # ignored rather than recorded as a garbage bind.
    a.poke('uniformTexture', ['samplerOther', 3.0])
    assert 'samplerOther' not in a.sampler_binds


def test_get_texture_answers_a_file_marker_for_plain_sprites():
    a = SimActor()
    assert a.read('GetTexture') is None
    a.texture_file = '/charts/asciitable.png'
    assert a.read('GetTexture').marker == 'file:/charts/asciitable.png'
    # An AFT identity outranks the file (a capture target stays aft:).
    a.mark_aft('cap')
    assert a.read('GetTexture').marker == 'aft:cap'
