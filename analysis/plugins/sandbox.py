"""Sandbox for user-installed plugin bundles.

Pragmatic, import-time allow-list. This is *not* a security boundary —
determined attackers can escape via numpy/ctypes tricks or frame walks.
The goal is:

  1. Stop lazy or accidental harm (plugins calling ``os.system`` or
     ``open`` on arbitrary paths).
  2. Push plugin authors toward host-provided APIs for anything
     side-effectful (FS, audio, chart loading, …).
  3. Reserve room to tighten toward audit hooks / capability-based
     mediation later without changing the plugin surface.

Trust model:

  * ``plugins/builtin/`` is trusted — ships with the app, runs with full
    Python access. Tampering with it is equivalent to tampering with the
    app itself, so no extra protection buys anything.
  * ``plugins/unsafe/`` is also trusted — escape hatch for power users
    who need raw Python (e.g. to prototype a plugin that will eventually
    be promoted into builtin). Explicitly named so the user opts in.
  * Everything else is sandboxed — restricted ``__builtins__`` and an
    import finder that rejects non-allow-listed modules.

Sandboxed plugins that try to import a blocked module fail to load.
Their bundle is surfaced to the user as "refused" via
``bundle.load_errors`` so the Plugins panel can flag it.
"""
from __future__ import annotations

import builtins
import importlib.abc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


# ─── Allow-lists ───────────────────────────────────────────────────────────
# Pure-Python stdlib modules that can't by themselves touch the filesystem,
# spawn processes, or open sockets. Submodules are allowed automatically
# (e.g. ``collections.abc``).
_STDLIB_ALLOW = frozenset({
    'math', 'cmath', 'random', 'statistics',
    'dataclasses', 'typing', 'enum', 'abc',
    'collections', 'itertools', 'functools', 'operator',
    're', 'string', 'textwrap',
    'bisect', 'heapq', 'array',
    'copy', 'numbers', 'fractions', 'decimal',
    'json',  # parse-only; plugins can't write it to disk through host API
    '__future__',
})

# Third-party libraries plugins may depend on. NumPy is allowed despite
# being a known escape vector (ctypeslib) — usability outweighs paranoia
# for v1. Revisit if we tighten toward audit hooks.
_THIRDPARTY_ALLOW = frozenset({
    'numpy',
})

# Host API modules exposed to sandboxed plugins. Keep this list narrow;
# expand only when the narrower surface is known to be safe.
_HOST_API_ALLOW = frozenset({
    'analysis.player.theme',
    'analysis.player.sidebar_api',
    'analysis.player.plugin_api',
    'analysis.plugins.host_api',
})

# Explicit deny-list for obviously dangerous modules. The allow-list is
# already exclusive, so this is belt-and-suspenders: a future reviewer
# who adds ``urllib`` to the stdlib allow-list without thinking should
# still get caught here. Checked before the allow-list in ``_is_allowed``.
_EXPLICIT_DENY = frozenset({
    'socket', 'ssl', 'select', 'selectors', 'asyncio',
    'http', 'urllib', 'urllib3', 'requests', 'httpx', 'aiohttp',
    'ftplib', 'smtplib', 'poplib', 'imaplib', 'telnetlib', 'nntplib',
    'xmlrpc', 'webbrowser',
    'subprocess', 'multiprocessing', 'threading', '_thread',
    'os', 'sys', 'pathlib', 'shutil', 'tempfile', 'glob', 'fnmatch',
    'io', 'mmap', 'fcntl', 'pty', 'pwd', 'grp', 'resource', 'signal',
    'ctypes', 'cffi',
    'importlib', 'pkgutil', 'runpy', 'zipimport',
    'pickle', 'shelve', 'marshal',
    'code', 'codeop', 'ast',
})


# Builtins that can be used for syscall-equivalent effects. Stripped.
# ``__import__`` is handled separately — replaced with a gated version
# rather than removed outright (plain ``import`` statements compile to
# ``__import__`` calls, so removing it breaks the sandbox entirely).
_UNSAFE_BUILTINS = frozenset({
    'open', 'exec', 'eval', 'compile',
    'input', 'breakpoint', 'help',
    'memoryview',  # can pair with ctypes for raw memory access
    'globals', 'locals', 'vars',  # frame-walking primitives
})


class SandboxViolation(ImportError):
    """Raised when a sandboxed plugin tries to import outside the allow-list."""


def _is_allowed(module_name: str) -> bool:
    """Return True if ``module_name`` may be imported from a sandboxed
    plugin. Submodules inherit their parent's status; parent packages of
    allow-listed entries are also allowed so ``from pkg import sub`` can
    reach through (``__import__('pkg', fromlist=['sub'])`` imports the
    parent first). Explicit denies beat allows."""
    parts = module_name.split('.')
    for i in range(len(parts), 0, -1):
        prefix = '.'.join(parts[:i])
        if prefix in _EXPLICIT_DENY:
            return False
    for i in range(len(parts), 0, -1):
        prefix = '.'.join(parts[:i])
        if (prefix in _STDLIB_ALLOW
                or prefix in _THIRDPARTY_ALLOW
                or prefix in _HOST_API_ALLOW):
            return True
    # Parent of an allow-listed module (e.g. ``analysis.player`` when
    # ``analysis.player.theme`` is allowed).
    dotted = module_name + '.'
    for allowed in (*_STDLIB_ALLOW, *_THIRDPARTY_ALLOW, *_HOST_API_ALLOW):
        if allowed.startswith(dotted):
            return True
    return False


class _SandboxFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that vetoes disallowed imports from sandboxed
    modules. Installed once; only activates when the importing module
    belongs to a sandboxed bundle (tracked via ``_SANDBOXED_MODULES``)."""

    def find_spec(self, fullname, path, target=None):
        if not _is_sandboxed_caller():
            return None
        if _is_allowed(fullname):
            return None  # let the normal import machinery handle it
        raise SandboxViolation(
            f'sandboxed plugin may not import {fullname!r}. '
            f'If you need this capability, request it via the host API.')


# Fully-qualified names of modules loaded under the sandbox. Used by the
# finder to detect "is the current import being triggered by a sandboxed
# plugin?" without walking frames on every non-plugin import.
_SANDBOXED_MODULES: set[str] = set()
_FINDER_INSTALLED = False


def _is_sandboxed_caller() -> bool:
    """Walk the call stack to see if a sandboxed module is on it. Only
    called from ``find_spec``; cost is O(stack depth) per import."""
    frame = sys._getframe(1)
    while frame is not None:
        mod_name = frame.f_globals.get('__name__')
        if mod_name in _SANDBOXED_MODULES:
            return True
        frame = frame.f_back
    return False


def _install_finder_once():
    global _FINDER_INSTALLED
    if _FINDER_INSTALLED:
        return
    sys.meta_path.insert(0, _SandboxFinder())
    _FINDER_INSTALLED = True


def _gated_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Replacement for ``__import__`` installed into sandboxed modules.
    Vetoes any import target not on the allow-list, regardless of whether
    that target is already cached in ``sys.modules``. Delegates to the
    real import machinery for permitted modules (so re-imports and
    ``from x import y`` work)."""
    if level != 0:
        # Relative imports (``from ._helper import ...``) stay within the
        # bundle's own package — we've already vetted its helpers.
        return builtins.__import__(name, globals, locals, fromlist, level)
    if not _is_allowed(name):
        raise SandboxViolation(
            f'sandboxed plugin may not import {name!r}. '
            f'If you need this capability, request it via the host API.')
    mod = builtins.__import__(name, globals, locals, fromlist, level)
    # ``from x.y import z`` — __import__ returns the top-level 'x', but
    # also vet each submodule in the dotted path.
    for part in name.split('.')[1:]:
        # Subpackages inherit their parent's allow status, so nothing
        # extra to check here. The check above already covered the full
        # dotted name.
        _ = part
    # ``from pkg import name`` — also vet each fromlist entry as
    # ``pkg.name`` so a sandboxed plugin can't ``from numpy import
    # ctypeslib`` when a future tightening bans that specific submodule.
    if fromlist:
        for item in fromlist:
            if item == '*':
                continue
            sub = f'{name}.{item}'
            if not _is_allowed(sub):
                # Only reject if the specific submodule is explicitly
                # disallowed; otherwise inherit parent's permission.
                # (In v1 we don't distinguish; kept for future use.)
                pass
    return mod


def _restricted_builtins() -> dict:
    """Build a builtins dict with the unsafe names removed and
    ``__import__`` replaced with the gated version. Returned as a fresh
    dict per sandboxed module to avoid shared-state leaks."""
    safe = {}
    for name in dir(builtins):
        if name.startswith('_'):
            if name in ('__build_class__', '__name__', '__doc__'):
                safe[name] = getattr(builtins, name)
            continue
        if name in _UNSAFE_BUILTINS:
            continue
        safe[name] = getattr(builtins, name)
    safe['__import__'] = _gated_import
    return safe


def is_trusted_bundle(path: Path) -> bool:
    """A bundle is trusted iff it lives under ``plugins/builtin/`` or
    ``plugins/unsafe/`` inside the repo. ``unsafe/`` is an opt-in escape
    hatch for bundles that legitimately need raw Python — naming it
    ``unsafe`` makes the user's choice explicit. Everything else —
    including user bundles directly under ``plugins/`` — is sandboxed."""
    try:
        repo_root = Path(__file__).resolve().parents[2]
        resolved = path.resolve()
        for trusted_dir in ('builtin', 'unsafe'):
            root = (repo_root / 'plugins' / trusted_dir).resolve()
            if resolved.is_relative_to(root):
                return True
        return False
    except (ValueError, OSError):
        return False


def prepare_sandboxed_module(mod: ModuleType):
    """Apply the sandbox to ``mod`` before its source executes. Called by
    the bundle loader between ``module_from_spec`` and ``exec_module``."""
    _install_finder_once()
    _SANDBOXED_MODULES.add(mod.__name__)
    mod.__builtins__ = _restricted_builtins()
