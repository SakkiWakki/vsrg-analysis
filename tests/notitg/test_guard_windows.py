"""NotITG guard-window extraction: AST path vs the frozen regex oracle.

The differential parity gate for phase 1 - the AST `_live_windows` must
reproduce the legacy regex output float-for-float on synthetic bodies and
(when the pilot charts are present) on gat 1's and gat 2's real Update
bodies. gat 1 parity is the byte-identical requirement.
"""
import re
from pathlib import Path

import pytest

from analysis.games.notitg import guard_windows


# Frozen copy of the retired regex window extractor - the parity ORACLE the
# AST path is checked against. Production no longer carries this; the AST is
# a proven superset across the local modfile corpus.
_PERFRAME_RE = re.compile(
    r'perframe\s*\(\s*([0-9][0-9.+\-*/ ()]*?)\s*'
    r'(?:,\s*([0-9][0-9.+\-*/ ()]*?)\s*)?\)')
_BEAT_GUARD_RE = re.compile(
    r'beat\s*>\s*([0-9][0-9.+\-*/ ()]*?)\s*and\s+'
    r'beat\s*<\s*([0-9][0-9.+\-*/ ()]*?)\s*(?:then|\))')
_BEAT_ARITH_RE = re.compile(r'(?!.*\*\*)[0-9.+\-*/ ()]+$')


def _beat_arg(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        pass
    if isinstance(text, str) and _BEAT_ARITH_RE.fullmatch(text.strip()):
        try:
            return float(eval(text, {'__builtins__': None}, {}))
        except (SyntaxError, ZeroDivisionError, TypeError, ValueError):
            return None
    return None


def _strip_comments(body):
    without_blocks = re.sub(r'--\[\[.*?\]\]', '', body, flags=re.DOTALL)
    return re.sub(r'--[^\n]*', '', without_blocks)


def _merge_spans(spans):
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _live_windows_regex(body):
    stripped = _strip_comments(body)
    spans = []
    for match in _PERFRAME_RE.finditer(stripped):
        start = _beat_arg(match.group(1))
        if start is None:
            continue
        end = _beat_arg(match.group(2)) if match.group(2) else start + 1.0
        if end is not None and end > start:
            spans.append((start, end))
    for match in _BEAT_GUARD_RE.finditer(stripped):
        start = _beat_arg(match.group(1))
        end = _beat_arg(match.group(2))
        if start is not None and end is not None and end > start:
            spans.append((start, end))
    return _merge_spans(sorted(spans))


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
        _load_document, _resolve_entry_xml, _sm_background_name, parse_fgchanges)
    from analysis.games.notitg.update_integrator import _update_body
    entries = parse_fgchanges(sm_path)
    entry = _resolve_entry_xml(sm_path, entries)
    root, _c, _cc = _load_document(
        entry, Path(_sm_background_name(sm_path)).stem.casefold())
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


# -- rearm period (self:sleep + self:queuecommand) ---------------------------

@pytest.mark.parametrize('body,expected', [
    ("%function(self) self:sleep(0.02); self:queuecommand('Update') end", 0.02),
    ("%function(self) self:sleep(0.05) self:queuecommand('Update') end", 0.05),
    ("%function(self) a() end", None),                       # no re-arm
    ("%function(self) self:sleep(0.02) end", None),          # sleep, no queue
    ("%function(self) self:queuecommand('Update') end", None),  # queue, no sleep
])
def test_rearm_period(body, expected):
    assert guard_windows.rearm_period(body) == expected


# -- guards survive intervening unmodeled constructs -------------------------

def test_guard_found_after_generic_for():
    # a generic-for (for k,v in ipairs) must parse so a later guard is still
    # found - the regression that motivated adding GenericFor to the grammar.
    body = ('%function(self) '
            'for i,v in ipairs(t) do x(v) end '
            'if beat > 88 and beat < 90 then a() end end')
    assert guard_windows.windows_from_body(body) == [(88.0, 90.0)]


def test_perframe_wrapper_names_are_recognized():
    # chart wrappers like floral_perframe(a,b) are perframe-family.
    body = '%function(self) if floral_perframe(104, 164) then a() end end'
    assert guard_windows.windows_from_body(body) == [(104.0, 164.0)]


def test_guard_inside_closure_call_arg_is_found():
    # A scheduled callback `mm(0, function(self) if perframe(a,b) ... end)`:
    # the guard lives inside a FuncExpr passed as a call arg. The walker must
    # descend into the closure body to find its window.
    body = ('%function(self) mm(0, function(self) '
            'if perframe(10, 20) then self:x(1) end end) end')
    assert guard_windows.windows_from_body(body) == [(10.0, 20.0)]


def test_guard_inside_closure_in_action_table_is_found():
    # The {beat, closure} action-table shape: a guard inside a FuncExpr that
    # is a table entry. The walker must descend into table values too.
    body = ('%function(self) actions = { {5, function() '
            'if perframe(50, 60) then x() end end}, {7, true} } end')
    assert guard_windows.windows_from_body(body) == [(50.0, 60.0)]


# -- bound global name (NAME = self) -----------------------------------------

@pytest.mark.parametrize('body,expected', [
    ('%function(self) my_actor = self end', 'my_actor'),
    ('%function(self) x = self:GetShader() end', None),   # not a bare self
    ('%function(self) a() end', None),
])
def test_bound_global_name(body, expected):
    assert guard_windows.bound_global_name(body) == expected
