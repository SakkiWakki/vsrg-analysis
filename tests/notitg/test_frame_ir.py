"""Frame IR builder: nested windows/bindings, chain resolution, live windows.

`build_frames` walks an Update body's statement AST into a Frame tree; the
tests assert the tree shape (child windows, unrolled loop bindings,
non-window gates), that `resolve` walks the parent chain, that
`effective_window` intersects the chain, and that `iter_updates` finds every
poke. The gat 1 test builds the real Update body's tree without error (the
byte-parity floor is proven elsewhere; here we only require it compiles).
"""
from pathlib import Path

import pytest

from analysis.games.notitg import frame_ir
from analysis.games.notitg.frame_ir import (
    Frame, VarUpdate, build_frames, effective_window, iter_updates, resolve)
from analysis.player.render.expr import ast
from analysis.player.render.expr.surface import ConstSurface


def _only_child(frame: Frame) -> Frame:
    frames = [c for c in frame.children if isinstance(c, Frame)]
    assert len(frames) == 1
    return frames[0]


# -- windowed frames from beat/perframe guards -------------------------------

def test_perframe_guard_windows_child_frame():
    root = build_frames(
        '%function(self) if perframe(128,252) then self:rotationz(beat) end end')
    child = _only_child(root)
    assert child.window == (128.0, 252.0)
    assert child.gate is None


def test_beat_range_guard_windows_child_frame():
    root = build_frames(
        '%function(self) if beat > 88 and beat < 90 then self:zoom(1) end end')
    child = _only_child(root)
    assert child.window == (88.0, 90.0)


def test_perframe_wrapper_is_normalized_to_window():
    root = build_frames(
        '%function(self) if floral_perframe(104,164) then self:x(1) end end')
    assert _only_child(root).window == (104.0, 164.0)


def test_two_sided_bounds_resolve_through_const_surface():
    root = build_frames(
        '%function(self) if beat > e[1] and beat < e[2] then self:x(1) end end',
        ConstSurface({'e': [10, 40]}))
    assert _only_child(root).window == (10.0, 40.0)


# -- non-window gate: window inherited, guard kept as gate -------------------

def test_flag_guard_inherits_window_and_records_gate():
    root = build_frames(
        '%function(self) if disabled then self:x(1) end end')
    child = _only_child(root)
    assert child.window is None
    assert isinstance(child.gate, ast.Sym) and child.gate.name == 'disabled'


def test_nested_flag_under_window_intersects_to_outer_window():
    root = build_frames('%function(self) '
                        'if perframe(10,20) then '
                        'if disabled then self:x(1) end '
                        'end end')
    outer = _only_child(root)
    inner = _only_child(outer)
    assert outer.window == (10.0, 20.0)
    assert inner.window is None                 # inherits
    assert effective_window(inner) == (10.0, 20.0)


# -- numeric-for unrolls to per-value frames; loop var bound -----------------

def test_literal_for_unrolls_and_binds_loop_var():
    root = build_frames('%function(self) '
                        'for pn = 1, 3 do self:x(pn) end end')
    frames = [c for c in root.children if isinstance(c, Frame)]
    assert len(frames) == 3
    assert [f.bindings['pn'] for f in frames] == [1.0, 2.0, 3.0]


def test_literal_for_with_step():
    root = build_frames('%function(self) '
                        'for i = 0, 4, 2 do self:x(i) end end')
    frames = [c for c in root.children if isinstance(c, Frame)]
    assert [f.bindings['i'] for f in frames] == [0.0, 2.0, 4.0]


def test_nonliteral_for_leaves_loop_var_unbound():
    root = build_frames('%function(self) '
                        'for pn = 1, n do self:x(pn) end end')
    child = _only_child(root)
    assert 'pn' not in child.bindings
    assert resolve('pn', child) is frame_ir.UNBOUND


# -- local bindings + chain resolution ---------------------------------------

def test_local_binds_in_current_frame():
    root = build_frames('%function(self) local off = 5 end')
    assert 'off' in root.bindings


def test_resolve_walks_parent_chain_for_loop_var_and_local():
    root = build_frames('%function(self) '
                        'local base = 7 '
                        'for pn = 1, 2 do '
                        'if perframe(0,4) then self:x(pn) end '
                        'end end')
    loop_frame = [c for c in root.children if isinstance(c, Frame)][0]
    window_frame = _only_child(loop_frame)
    assert resolve('pn', window_frame) == 1.0        # from the loop frame
    assert resolve('base', window_frame) is not frame_ir.UNBOUND  # from root
    assert resolve('missing', window_frame) is frame_ir.UNBOUND


def test_inner_binding_shadows_outer():
    inner = Frame(window=None, bindings={'x': 2})
    outer = Frame(window=None, bindings={'x': 1}, children=[inner])
    inner.parent = outer
    assert resolve('x', inner) == 2
    assert resolve('x', outer) == 1


# -- effective_window intersects the chain -----------------------------------

def test_effective_window_intersects_nested_windows():
    root = build_frames('%function(self) '
                        'if perframe(10,40) then '
                        'if perframe(20,60) then self:x(1) end '
                        'end end')
    outer = _only_child(root)
    inner = _only_child(outer)
    assert effective_window(inner) == (20.0, 40.0)


def test_effective_window_none_when_unwindowed():
    root = build_frames('%function(self) self:x(1) end')
    assert effective_window(root) is None


def test_effective_window_empty_span_when_disjoint():
    outer = Frame(window=(0.0, 10.0))
    inner = Frame(window=(20.0, 30.0), parent=outer)
    start, end = effective_window(inner)
    assert end <= start                          # never live


# -- iter_updates enumerates every poke with its frame -----------------------

def test_iter_updates_finds_every_poke_with_owning_frame():
    root = build_frames('%function(self) '
                        'if perframe(0,4) then '
                        'for pn = 1, 2 do self:rotationz(pn) end '
                        'end '
                        'if beat > 5 and beat < 9 then self:zoom(1) end end')
    pairs = list(iter_updates(root))
    names = sorted(u.name for u, _f in pairs)
    assert names == ['self.rotationz', 'self.rotationz', 'self.zoom']
    windows = {frame_ir.effective_window(f) for _u, f in pairs}
    assert (0.0, 4.0) in windows and (5.0, 9.0) in windows


def test_assignment_becomes_varupdate():
    root = build_frames('%function(self) splitm = splitm * -1 end')
    updates = [c for c in root.children if isinstance(c, VarUpdate)]
    assert len(updates) == 1
    assert updates[0].name == 'splitm'
    assert updates[0].closed_form is None        # router fills it later


def test_unmodeled_statement_becomes_attention_varupdate():
    root = build_frames('%function(self) update_proxies() end')
    updates = [c for c in root.children if isinstance(c, VarUpdate)]
    assert len(updates) == 1
    assert updates[0].name == 'update_proxies'
    assert updates[0].closed_form is None        # attention


# -- gat 1 real body builds a tree (compiles; parity proven elsewhere) -------

_GAT1 = Path('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/'
             'UKSRT8/5. gat/gat.sm')


def _update_body_of(sm_path: Path):
    pytest.importorskip('lupa')
    from analysis.games.notitg.modfile import (
        _load_document, _resolve_lua_dir, _sm_background_name, parse_fgchanges)
    from analysis.games.notitg.update_integrator import _update_body
    entries = parse_fgchanges(sm_path)
    lua_dir = _resolve_lua_dir(sm_path, entries)
    root, _c, _cc = _load_document(
        lua_dir, Path(_sm_background_name(sm_path)).stem.casefold())
    return _update_body(root)


@pytest.mark.skipif(not _GAT1.exists(), reason='gat 1 chart not present')
def test_gat1_body_builds_frame_tree():
    body = _update_body_of(_GAT1)
    assert body is not None
    root = build_frames(body)
    pairs = list(iter_updates(root))
    assert pairs                                  # found writes
    assert all(isinstance(u, VarUpdate) and isinstance(f, Frame)
               for u, f in pairs)
