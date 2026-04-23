"""Tests for the plugin symbolic verifier.

Covers: static sink detection, call graph reachability, alias chains,
Z3 config-key prefix check, and the load-time hard-block integration.
"""
from __future__ import annotations

import ast
import textwrap

import pytest

from unittest.mock import patch

from analysis.plugins.verifier import (
    CALL_GRAPH_TIMEOUT_MS,
    Z3_TOTAL_TIMEOUT_MS,
    VerificationError,
    VerificationTimeout,
    verify,
)
from analysis.plugins.verifier.call_graph import reachable_calls
from analysis.plugins.verifier.sinks import (
    SinkViolation,
    check_config_set,
    check_static,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _parse(src: str) -> ast.Module:
    return ast.parse(textwrap.dedent(src))


def _calls(src: str) -> list[ast.Call]:
    return reachable_calls(_parse(src))


def _verify_src(src: str, plugin_key: str = 'test:plugin',
                tmp_path=None) -> None:
    """Write src to a temp file and run verify() on it. Re-raises
    VerificationError if the plugin fails."""
    path = tmp_path / 'plugin.py'
    path.write_text(textwrap.dedent(src))
    verify(path, plugin_key)


# ── Call graph reachability ───────────────────────────────────────────

def test_module_level_code_is_always_included():
    calls = _calls('''
        import math
        x = some_func()  # module level -- always reachable
        def draw(ctx):
            pass
    ''')
    names = [c.func.id for c in calls if isinstance(c.func, ast.Name)]
    assert 'some_func' in names


def test_entry_point_calls_are_included():
    calls = _calls('''
        def draw(ctx):
            result = helper()
    ''')
    names = [c.func.id for c in calls if isinstance(c.func, ast.Name)]
    assert 'helper' in names


def test_unreachable_function_calls_excluded():
    calls = _calls('''
        def draw(ctx):
            pass
        def _never_called():
            dangerous()
    ''')
    names = [c.func.id for c in calls if isinstance(c.func, ast.Name)]
    assert 'dangerous' not in names


def test_transitively_reachable_calls_included():
    calls = _calls('''
        def draw(ctx):
            step_one()
        def step_one():
            step_two()
        def step_two():
            sink_call()
    ''')
    names = [c.func.id for c in calls if isinstance(c.func, ast.Name)]
    assert 'sink_call' in names


def test_register_components_is_entry_point():
    calls = _calls('''
        def register_components(add):
            setup()
        def setup():
            inner()
    ''')
    names = [c.func.id for c in calls if isinstance(c.func, ast.Name)]
    assert 'inner' in names


# ── Static sink detection ─────────────────────────────────────────────

def _first_call(src: str) -> ast.Call:
    tree = _parse(src)
    return next(n for n in ast.walk(tree) if isinstance(n, ast.Call))


def test_open_is_a_static_sink():
    call = _first_call("open('/tmp/x')")
    assert check_static(call) is not None


def test_eval_is_a_static_sink():
    call = _first_call("eval('1+1')")
    assert check_static(call) is not None


def test_exec_is_a_static_sink():
    call = _first_call("exec('pass')")
    assert check_static(call) is not None


def test_setattr_is_a_static_sink():
    call = _first_call("setattr(obj, 'x', 1)")
    assert check_static(call) is not None


def test_getattr_is_a_static_sink():
    call = _first_call("getattr(obj, '__class__')")
    assert check_static(call) is not None


def test_dunder_class_call_is_a_sink():
    # ctx.__class__() -- calling the class constructor via __class__
    # is an object-model escape pattern
    call = _first_call("ctx.__class__()")
    v = check_static(call)
    assert v is not None


def test_safe_math_call_is_not_a_sink():
    call = _first_call("math.sqrt(4)")
    assert check_static(call) is None


def test_safe_host_api_call_is_not_a_sink():
    call = _first_call("ctx.draw_heading('Title')")
    assert check_static(call) is None


def test_object_setattr_is_a_sink():
    call = _first_call("object.__setattr__(snap, 'combo', 999)")
    assert check_static(call) is not None


# ── Config prefix Z3 check ────────────────────────────────────────────

def _config_call(src: str) -> ast.Call:
    tree = _parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    return next(c for c in calls
                if isinstance(c.func, ast.Attribute)
                and c.func.attr == 'set')


def test_literal_path_traversal_is_caught():
    call = _config_call("cfg.set('../../other_plugin/key', 'val')")
    v = check_config_set(call, 'test:plugin')
    assert v is not None
    assert 'escape' in v.description.lower() or 'traversal' in v.description.lower()


def test_literal_dot_prefix_is_caught():
    call = _config_call("cfg.set('.hidden', 'val')")
    v = check_config_set(call, 'test:plugin')
    assert v is not None


def test_non_literal_field_is_flagged():
    call = _config_call("cfg.set(dynamic_field, 'val')")
    v = check_config_set(call, 'test:plugin')
    assert v is not None


def test_non_set_method_is_not_flagged():
    tree = _parse("cfg.get('key')")
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    get_call = calls[0]
    assert check_config_set(get_call, 'test:plugin') is None


# ── End-to-end verify() ───────────────────────────────────────────────

def test_clean_plugin_passes_verification(tmp_path):
    _verify_src('''
        from analysis.plugins.host_api import plugin_config
        import math

        _cfg = plugin_config('test:plugin')

        def draw(ctx):
            ctx.draw_heading(f'Combo: {math.floor(0.0)}')

        def register_components(add):
            pass
    ''', tmp_path=tmp_path)


def test_open_call_in_draw_is_hard_blocked(tmp_path):
    with pytest.raises(VerificationError) as exc_info:
        _verify_src('''
            def draw(ctx):
                open('/tmp/x').read()
        ''', tmp_path=tmp_path)
    assert any(v.kind == 'static' for v in exc_info.value.violations)


def test_eval_in_draw_is_hard_blocked(tmp_path):
    with pytest.raises(VerificationError):
        _verify_src('''
            def draw(ctx):
                eval("__import__('os').system('rm -rf /')")
        ''', tmp_path=tmp_path)


def test_setattr_on_frozen_object_is_blocked(tmp_path):
    with pytest.raises(VerificationError):
        _verify_src('''
            def draw(ctx):
                setattr(ctx.data, 'combo', 9999)
        ''', tmp_path=tmp_path)


def test_transitive_sink_through_helper_is_blocked(tmp_path):
    with pytest.raises(VerificationError):
        _verify_src('''
            def _do_evil():
                open('/tmp/x', 'w')

            def draw(ctx):
                _do_evil()
        ''', tmp_path=tmp_path)


def test_sink_in_unreachable_function_is_not_blocked(tmp_path):
    # Dead code should not cause a hard block
    _verify_src('''
        def _dead():
            open('/tmp/x')  # never called from any entry point

        def draw(ctx):
            ctx.draw_heading('safe')
    ''', tmp_path=tmp_path)


def test_path_traversal_in_config_set_is_blocked(tmp_path):
    with pytest.raises(VerificationError) as exc_info:
        _verify_src('''
            def draw(ctx):
                cfg = object()
                cfg.set('../../paths.songs_dir', '/evil')
        ''', tmp_path=tmp_path)
    assert any(v.kind == 'constraint' for v in exc_info.value.violations)


def test_module_level_open_is_blocked(tmp_path):
    with pytest.raises(VerificationError):
        _verify_src('''
            open('/tmp/x')  # module level, runs at import time

            def draw(ctx):
                pass
        ''', tmp_path=tmp_path)


def test_verification_error_message_includes_line_number(tmp_path):
    with pytest.raises(VerificationError) as exc_info:
        _verify_src('''
            def draw(ctx):
                eval("bad")
        ''', tmp_path=tmp_path)
    msg = str(exc_info.value)
    assert 'line' in msg
    assert 'eval' in msg


def test_syntax_error_does_not_raise_verification_error(tmp_path):
    path = tmp_path / 'bad.py'
    path.write_text('def (((')
    # Should not raise -- syntax errors are exec_module's job
    verify(path, 'test:plugin')


def test_file_not_found_does_not_raise(tmp_path):
    from pathlib import Path
    verify(Path('/nonexistent/plugin.py'), 'test:plugin')


# ── Timeout behaviour ─────────────────────────────────────────────────

def _slow_call_graph(*_):
    """Simulate a call graph that never finishes."""
    import time
    time.sleep(1)
    return []


def test_call_graph_timeout_hard_blocks(tmp_path):
    """If the call graph phase takes too long, VerificationTimeout is raised
    and the plugin is refused -- same outcome as a detected violation."""
    path = tmp_path / 'slow.py'
    path.write_text('def draw(ctx): pass')

    with patch('analysis.plugins.verifier.reachable_calls', _slow_call_graph), \
         patch('analysis.plugins.verifier.CALL_GRAPH_TIMEOUT_MS', 50):
        with pytest.raises(VerificationTimeout):
            verify(path, 'test:plugin')


def test_z3_timeout_hard_blocks(tmp_path):
    """If the Z3/sink-check phase takes too long, VerificationTimeout is raised."""
    path = tmp_path / 'slow.py'
    path.write_text('def draw(ctx): pass')

    import time

    def _slow_static(*_):
        time.sleep(1)
        return None

    # Patch at the verifier module's import site (where the closure captures
    # the name), not at the sinks module level.
    with patch('analysis.plugins.verifier.check_static', _slow_static), \
         patch('analysis.plugins.verifier.reachable_calls',
               return_value=[ast.parse('f()').body[0].value]), \
         patch('analysis.plugins.verifier.Z3_TOTAL_TIMEOUT_MS', 50):
        with pytest.raises(VerificationTimeout):
            verify(path, 'test:plugin')


def test_timeout_error_message_identifies_cause(tmp_path):
    path = tmp_path / 'p.py'
    path.write_text('def draw(ctx): pass')

    with patch('analysis.plugins.verifier.reachable_calls', _slow_call_graph), \
         patch('analysis.plugins.verifier.CALL_GRAPH_TIMEOUT_MS', 50):
        with pytest.raises(VerificationTimeout) as exc_info:
            verify(path, 'test:plugin')
    msg = str(exc_info.value)
    assert 'timed out' in msg
    assert 'test:plugin' in msg


def test_timeout_is_subclass_of_verification_error():
    """VerificationTimeout is a VerificationError so callers that catch
    the base class still see the timeout as a hard block."""
    assert issubclass(VerificationTimeout, VerificationError)


def test_timeout_constants_are_sane():
    """Sanity check that the timeouts are within reasonable bounds --
    not 0 (would block everything) and not absurdly large (would allow DoS)."""
    assert 50 <= CALL_GRAPH_TIMEOUT_MS <= 5000
    assert 50 <= Z3_TOTAL_TIMEOUT_MS <= 5000
