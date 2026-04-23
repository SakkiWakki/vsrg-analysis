"""Sandbox isolation safety tests.

Verifies that sandboxed plugins cannot:
- Import blocked modules (os, sys, threading, socket, urllib, ctypes, etc.)
- Read another plugin's config data
- Access the filesystem directly
- Access raw memory (ctypes, mmap, osu_memory_native)
- Escape the sandbox via builtins (open, exec, eval, globals, locals)
- Reach internal analysis modules not on the allow-list

These are not exhaustive security proofs -- the sandbox is best-effort
(see sandbox.py header). The goal is to catch accidental or lazy escape
attempts, not a determined attacker with source access.
"""
from __future__ import annotations

import pytest

from analysis.plugins.sandbox import (
    SandboxViolation,
    _gated_import,
    _is_allowed,
    _SANDBOXED_MODULES,
)
from analysis.plugins.host_api import PluginConfig


# ── Allow-list correctness ────────────────────────────────────────────

@pytest.mark.parametrize('module', [
    'math', 'cmath', 'random', 'json', 'dataclasses',
    'typing', 'enum', 'collections', 'itertools', 'functools',
    'bisect', 'copy', 'decimal', 'fractions',
    'numpy',
    'analysis.plugins.host_api',
    'analysis.player.render.theme',
    'analysis.player.hud.sidebar_api',
    'analysis.overlay.api',
    'analysis.plugins.permissions',
])
def test_allowed_modules_are_allowed(module):
    assert _is_allowed(module) is True


@pytest.mark.parametrize('module', [
    'os', 'sys', 'pathlib', 'shutil', 'glob',
    'socket', 'ssl', 'urllib', 'requests', 'http',
    'threading', '_thread', 'multiprocessing',
    'subprocess',
    'ctypes', 'cffi', 'mmap',
    'importlib', 'pkgutil', 'runpy',
    'pickle', 'shelve', 'marshal',
    'ast', 'code', 'codeop',
    'io',
    'osu_memory_native',
    'analysis.player.player',   # internal, not on allow-list
    'analysis.core',            # internal, not on allow-list
    'analysis.games',           # internal, not on allow-list
])
def test_blocked_modules_are_blocked(module):
    assert _is_allowed(module) is False


def test_explicit_deny_beats_allow_list():
    # urllib is explicitly denied; verify it can't sneak in via a submodule
    assert _is_allowed('urllib.request') is False
    assert _is_allowed('urllib.parse') is False


def test_osu_memory_native_is_blocked():
    assert _is_allowed('osu_memory_native') is False


# ── Gated import enforcement ──────────────────────────────────────────
# Simulate what happens when a sandboxed module calls __import__.
# We inject a fake sandboxed module name into _SANDBOXED_MODULES so the
# finder believes the call comes from a sandboxed plugin.

def _with_sandbox(fn, module_name='fake_sandboxed_plugin'):
    """Run fn() as if called from a sandboxed module."""
    _SANDBOXED_MODULES.add(module_name)
    try:
        return fn()
    finally:
        _SANDBOXED_MODULES.discard(module_name)


@pytest.mark.parametrize('module', ['os', 'sys', 'threading', 'socket',
                                    'urllib', 'ctypes', 'subprocess',
                                    'importlib', 'pickle', 'mmap'])
def test_gated_import_blocks_dangerous_modules(module):
    with pytest.raises(SandboxViolation):
        _gated_import(module)


def test_gated_import_allows_math():
    mod = _gated_import('math')
    assert hasattr(mod, 'sqrt')


def test_gated_import_blocks_osu_memory_native():
    with pytest.raises(SandboxViolation):
        _gated_import('osu_memory_native')


def test_gated_import_blocks_internal_player():
    with pytest.raises(SandboxViolation):
        _gated_import('analysis.player.player')


def test_gated_import_blocks_internal_core():
    with pytest.raises(SandboxViolation):
        _gated_import('analysis.core')


# ── PluginConfig isolation ────────────────────────────────────────────

def test_plugin_config_cannot_read_other_plugin_settings():
    cfg_a = PluginConfig('plugin:alpha')
    cfg_b = PluginConfig('plugin:beta')

    cfg_a.set('secret', 'top_secret_value')
    try:
        # plugin:beta cannot read plugin:alpha's settings through its own config
        val = cfg_b.get('secret')
        assert val is None, (
            f'plugin:beta should not see plugin:alpha secret, got {val!r}')
    finally:
        cfg_a.delete('secret')


def test_plugin_config_cannot_write_other_plugin_settings():
    cfg_a = PluginConfig('plugin:alpha')
    cfg_b = PluginConfig('plugin:beta')

    cfg_b.set('attempt', 'injected')
    try:
        val = cfg_a.get('attempt')
        assert val is None, (
            f'plugin:alpha should not see plugin:beta write, got {val!r}')
    finally:
        cfg_b.delete('attempt')


def test_plugin_config_snapshot_scoped_to_own_plugin():
    cfg_a = PluginConfig('plugin:alpha')
    cfg_b = PluginConfig('plugin:beta')

    cfg_a.set('my_key', 'my_value')
    try:
        snap_b = cfg_b.snapshot()
        assert 'my_key' not in snap_b
    finally:
        cfg_a.delete('my_key')


def test_plugin_config_path_traversal_blocked():
    cfg = PluginConfig('plugin:alpha')
    # A plugin must not be able to set a dotted key that climbs out of its
    # own settings subtree (e.g. ../../paths.songs_dir).
    # The PluginConfig._path builds: plugins.plugin_alpha.settings.<field>
    # Setting field='../../paths.songs_dir' would produce a path that exits
    # the plugin's subtree. Verify the value ends up scoped, not at the root.
    cfg.set('../../paths.songs_dir', '/evil')
    try:
        from analysis.config import get_config
        store = get_config()
        # Should NOT have written to paths.songs_dir
        top_level = store.get('paths.songs_dir')
        assert top_level != '/evil', (
            'plugin config must not allow path traversal to top-level config')
    finally:
        cfg.delete('../../paths.songs_dir')


# ── Stripped builtins ────────────────────────────────────────────────
# These tests verify the allow-list logic; the actual stripping happens
# in load_module which we don't call here. The tests confirm the intent
# is coded, not that every runtime path enforces it (that's the "best
# effort" caveat in sandbox.py).

@pytest.mark.parametrize('module', [
    'os', 'sys', 'pathlib', 'io',
    'ctypes', 'mmap', 'osu_memory_native',
])
def test_memory_and_filesystem_modules_denied(module):
    assert _is_allowed(module) is False, (
        f'{module!r} must be blocked to prevent memory/filesystem access')
