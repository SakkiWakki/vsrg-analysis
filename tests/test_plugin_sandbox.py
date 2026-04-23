"""Tests for the plugin sandbox.

Two layers are tested:

  1. ``_is_allowed`` decisions — the policy table.
  2. End-to-end loader behavior via a temp-dir bundle — does a real
     ``import`` statement in a sandboxed module actually get blocked?
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from analysis.plugins import discover_bundles
from analysis.plugins.sandbox import (
    SandboxViolation,
    _is_allowed,
    is_trusted_bundle,
)


# ─── Policy table ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('module', [
    'math', 'random', 'collections', 'collections.abc',
    'itertools', 'functools', 're', 'json', 'dataclasses',
    'numpy', 'numpy.linalg',
    'analysis.player.render.theme', 'analysis.player.hud.sidebar_api',
    'analysis.plugins.host_api',
    'analysis.overlay.api',
    # Parent of an allow-listed submodule — required for
    # ``from analysis.player.render import theme`` (__import__ fetches parent).
    'analysis.player',
])
def test_allowed_modules(module):
    assert _is_allowed(module)


@pytest.mark.parametrize('module', [
    # FS / process
    'os', 'os.path', 'sys', 'pathlib', 'subprocess', 'shutil',
    'tempfile', 'io', 'mmap',
    # Network
    'socket', 'ssl', 'http', 'http.client', 'urllib', 'urllib.request',
    'requests', 'httpx', 'aiohttp', 'ftplib', 'smtplib',
    # Memory / FFI
    'ctypes', 'ctypes.util', 'cffi',
    # Threading / async
    'threading', '_thread', 'multiprocessing', 'asyncio',
    # Code-loading
    'importlib', 'importlib.util', 'pickle', 'marshal', 'ast', 'code',
    # Frame-walking (denied even though some are stdlib)
    'gc', 'inspect', 'traceback', 'linecache',
    # NumPy escape vectors (denied even though 'numpy' is allowed)
    'numpy.ctypeslib', 'numpy.ctypes', 'numpy.distutils',
    'numpy.f2py', 'numpy.testing',
    # matplotlib escape vector
    'matplotlib.testing',
    # Random non-allow-listed stdlib
    'logging', 'unittest', 'argparse',
    # Third-party not on the allow-list
    'pytest', 'PySide6',
    # Overlay host runtime touches shm/threads and is not sandbox-safe.
    'analysis.overlay.publisher',
    # Global config store -- plugins must use ctx.config (scoped) instead.
    'analysis.config', 'analysis.config.store',
])
def test_denied_modules(module):
    assert not _is_allowed(module)


def test_numpy_submodule_deny_beats_parent_allow():
    """numpy is allowed but its escape-vector submodules are explicitly
    denied -- the deny-list must be checked before the allow-list."""
    assert _is_allowed('numpy') is True
    assert _is_allowed('numpy.ctypeslib') is False
    assert _is_allowed('numpy.ctypes') is False
    assert _is_allowed('numpy.f2py') is False
    assert _is_allowed('numpy.testing') is False


def test_frame_walking_modules_are_denied():
    assert _is_allowed('gc') is False
    assert _is_allowed('inspect') is False
    assert _is_allowed('traceback') is False
    assert _is_allowed('linecache') is False


def test_explicit_deny_beats_allow():
    """If someone accidentally adds ``urllib`` to the allow-list, the
    explicit deny should still refuse it. (Regression guard.)"""
    from analysis.plugins import sandbox
    original = sandbox._STDLIB_ALLOW
    try:
        sandbox._STDLIB_ALLOW = frozenset(original | {'urllib'})
        assert not _is_allowed('urllib')
        assert not _is_allowed('urllib.request')
    finally:
        sandbox._STDLIB_ALLOW = original


# ─── Trust classification ──────────────────────────────────────────────────

def test_builtin_is_not_trusted():
    # builtin/ is sandboxed -- see SECURITY.md. Only unsafe/ is trusted.
    repo_root = Path(__file__).resolve().parents[1]
    assert not is_trusted_bundle(repo_root / 'plugins' / 'builtin')


def test_unsafe_is_trusted():
    repo_root = Path(__file__).resolve().parents[1]
    assert is_trusted_bundle(repo_root / 'plugins' / 'unsafe' / 'anything')


def test_arbitrary_user_bundle_is_not_trusted():
    repo_root = Path(__file__).resolve().parents[1]
    assert not is_trusted_bundle(repo_root / 'plugins' / 'example_sandboxed')


def test_outside_repo_is_not_trusted(tmp_path):
    assert not is_trusted_bundle(tmp_path / 'random_dir')


# ─── End-to-end loader behavior ────────────────────────────────────────────

def _write_bundle(root: Path, name: str, sidebar_files: dict[str, str]):
    """Build a bundle at ``root/name`` with the given sidebar files."""
    bundle = root / name
    (bundle / 'sidebar').mkdir(parents=True)
    (bundle / 'manifest.toml').write_text(
        f'name = "{name}"\nkey = "{name}"\nversion = "0.0"\n')
    for fname, body in sidebar_files.items():
        (bundle / 'sidebar' / fname).write_text(textwrap.dedent(body))
    return bundle


@pytest.fixture
def sandboxed_root(tmp_path, monkeypatch):
    """Point discovery at a temp bundle root so tests stay hermetic."""
    monkeypatch.setenv('EA_PLUGINS_PATH', str(tmp_path))
    return tmp_path


def test_sandboxed_plugin_allowed_imports_succeed(sandboxed_root, capsys):
    _write_bundle(sandboxed_root, 'good_bundle', {
        'ok.py': '''
            import math
            from analysis.player.render import theme

            def register_sidebar(add):
                pass
        ''',
    })
    bundles = discover_bundles()
    good = next(b for b in bundles if b.key == 'good_bundle')
    assert good.trusted is False
    assert good.load_errors == []
    assert len(good.sidebar_modules) == 1


def test_sandboxed_plugin_os_import_refused(sandboxed_root, capsys):
    _write_bundle(sandboxed_root, 'evil_bundle', {
        'evil.py': '''
            import os

            def register_sidebar(add):
                pass
        ''',
    })
    bundles = discover_bundles()
    evil = next(b for b in bundles if b.key == 'evil_bundle')
    assert evil.trusted is False
    assert len(evil.load_errors) == 1
    role, fname, exc = evil.load_errors[0]
    assert role == 'sidebar'
    assert fname == 'evil.py'
    assert isinstance(exc, SandboxViolation)
    assert "'os'" in str(exc)
    assert len(evil.sidebar_modules) == 0


@pytest.mark.parametrize('forbidden', ['socket', 'urllib.request',
                                        'subprocess', 'ctypes', 'threading',
                                        'pickle'])
def test_sandboxed_plugin_refuses_dangerous_imports(sandboxed_root, forbidden):
    _write_bundle(sandboxed_root, f'bad_{forbidden.replace(".", "_")}', {
        'bad.py': f'''
            import {forbidden}

            def register_sidebar(add):
                pass
        ''',
    })
    bundles = discover_bundles()
    bad = next(b for b in bundles
               if b.key == f'bad_{forbidden.replace(".", "_")}')
    assert len(bad.load_errors) == 1
    _role, _fname, exc = bad.load_errors[0]
    assert isinstance(exc, SandboxViolation)


def test_sandboxed_plugin_cannot_call_open(sandboxed_root):
    """``open`` is caught at verification time (static sink) before the
    module even loads -- the verifier hard-blocks it."""
    _write_bundle(sandboxed_root, 'fs_bundle', {
        'fs.py': '''
            def register_sidebar(add):
                open('/tmp/should_not_happen', 'w').close()
        ''',
    })
    bundles = discover_bundles()
    fs = next(b for b in bundles if b.key == 'fs_bundle')
    assert len(fs.load_errors) > 0
    assert fs.sidebar_modules == []


def test_sandboxed_plugin_cannot_call_eval(sandboxed_root):
    """``eval`` is caught at verification time (static sink) before the
    module even loads -- the verifier hard-blocks it."""
    _write_bundle(sandboxed_root, 'eval_bundle', {
        'ev.py': '''
            def register_sidebar(add):
                eval("1 + 1")
        ''',
    })
    bundles = discover_bundles()
    b = next(x for x in bundles if x.key == 'eval_bundle')
    assert len(b.load_errors) > 0
    assert b.sidebar_modules == []


def test_sandboxed_relative_import_within_bundle_works(sandboxed_root):
    """Sibling helpers (``from ._helper import X``) must still resolve —
    the sandbox vets ``import`` targets, but within-bundle relative
    imports are pre-vetted at discovery time."""
    _write_bundle(sandboxed_root, 'rel_bundle', {
        '_helper.py': '''
            CONSTANT = 42
        ''',
        'main.py': '''
            from ._helper import CONSTANT

            def register_sidebar(add):
                pass
        ''',
    })
    bundles = discover_bundles()
    rel = next(b for b in bundles if b.key == 'rel_bundle')
    assert rel.load_errors == []
    assert len(rel.sidebar_modules) == 1
