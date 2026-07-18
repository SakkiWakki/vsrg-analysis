"""NotITG guard-window extraction: AST path vs the frozen regex oracle.

The differential parity gate for phase 1 - the AST `_live_windows` must
reproduce the legacy regex output float-for-float on synthetic bodies and
(when the pilot charts are present) on gat 1's and gat 2's real Update
bodies. gat 1 parity is the byte-identical requirement.
"""
from pathlib import Path

import pytest

from analysis.games.notitg import guard_windows
from analysis.games.notitg.update_integrator import _live_windows_regex


# -- synthetic bodies: AST == regex, both forms ------------------------------

@pytest.mark.parametrize('body', [
    '%function(self) if perframe(128,252) then a() end end',
    '%function(self) if perframe(10,20) then end if perframe(18,30) then end end',
    '%function(self) if beat > 0 and beat < 127 then a() end end',
    '%function(self) if beat>127 and beat<352 then a() end '
    'if beat > 601 and beat < 760 then b() end end',
    '%function(self) if perframe(1240, 1240+128) then a() end end',
    '%function(self) if perframe(50) then a() end end',
    # commented-out driver must not widen either path
    '%function(self) --[[ if perframe(900,999) then end ]]\n'
    'if perframe(10,20) then end end',
])
def test_ast_matches_regex_on_synthetic_bodies(body):
    assert guard_windows.windows_from_body(body) == _live_windows_regex(body)


def test_ast_extends_beyond_regex_with_table_bounds():
    # A two-sided range whose bounds are compiled constants: the regex
    # cannot resolve v[]/e[]; the AST can (superset capability).
    body = '%function(self) if beat > e[1] and beat < e[2] then a() end end'
    from analysis.player.render.expr.surface import ConstSurface
    win = guard_windows.windows_from_body(body, ConstSurface({'e': [10, 40]}))
    assert win == [(10.0, 40.0)]
    assert _live_windows_regex(body) == []      # regex gets nothing


# -- real pilot charts (skip when absent) ------------------------------------

_GAT1 = Path('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/'
             'UKSRT8/5. gat/gat.sm')
_GAT2 = Path('/mnt/Yucky/Rhythm Games/Players/NotITG/Songs/'
             'UKSRT9/5. getfucked2/get_fucked_2.sm')


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
def test_gat1_window_parity_byte_identical():
    body = _update_body_of(_GAT1)
    assert body is not None
    assert guard_windows.windows_from_body(body) == _live_windows_regex(body)


@pytest.mark.skipif(not _GAT2.exists(), reason='gat 2 chart not present')
def test_gat2_window_parity():
    body = _update_body_of(_GAT2)
    assert body is not None
    assert guard_windows.windows_from_body(body) == _live_windows_regex(body)
