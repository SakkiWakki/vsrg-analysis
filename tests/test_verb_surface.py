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


def test_trap_families_are_deferred_not_guessed():
    """The known traps - the '*2' second-slot family, rotation-order verbs,
    skew-before-rotation - must be DEFERRED (Actor.clean.c is COMDAT-folded;
    their semantics cannot be pinned), never mapped to a scalar write."""
    for name in ('x2', 'zoom2', 'rotationz2', 'GetX2', 'skewx2',
                 'SetRotationOrder', 'heading', 'pitch', 'roll',
                 'skewx_before_rotation'):
        assert name in vs.DEFERRED, name
        for setter in (vs.SCALAR_SETTERS, vs.ADD_SETTERS, vs.BULK_SETTERS):
            assert name not in setter, f'{name} was mapped to a write'


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
