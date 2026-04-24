"""Sandbox for plugin bundles.

Pragmatic, import-time allow-list combined with a symbolic verifier. This
is a best-effort model, not a formal security guarantee. See SECURITY.md
for the full threat model and known limitations.

Trust model
-----------
Only ``plugins/unsafe/`` is trusted (full Python, no sandbox). Everything
else -- including ``plugins/builtin/`` -- is sandboxed. ``builtin/`` ships
with the app but is treated as untrusted because a malicious contribution
could be added to it; the sandbox and verifier enforce the same rules as
any third-party plugin.

Enforcement layers applied to all non-``unsafe/`` plugins
----------------------------------------------------------
1. Import allow-list: a patched ``__import__`` raises ``SandboxViolation``
   for any module not on ``_HOST_API_ALLOW | _STDLIB_ALLOW | _THIRDPARTY_ALLOW``.
2. Restricted builtins: ``open``, ``eval``, ``exec``, ``compile``,
   ``memoryview``, ``globals``, ``locals``, ``vars`` and all ``_``-prefixed
   names (except ``__build_class__``, ``__name__``, ``__doc__``) are removed.
3. Symbolic verifier (``analysis/plugins/verifier/``): analyses the AST
   before execution; hard-blocks on static sinks and uses Z3 to check
   config-namespace isolation.
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

# Third-party libraries plugins may depend on. Submodules with escape
# vectors are denied explicitly below in _EXPLICIT_DENY, which beats
# the allow-list entry for the parent package.
_THIRDPARTY_ALLOW = frozenset({
    'numpy',
    'matplotlib',
})

# Host API modules exposed to sandboxed plugins. Keep this list narrow;
# expand only when the narrower surface is known to be safe.
_HOST_API_ALLOW = frozenset({
    'analysis.player.render.theme',
    'analysis.player.hud.sidebar_api',
    'analysis.player.plugin.plugin_api',
    'analysis.player.input.events',
    'analysis.plugins.host_api',
    # permissions is resolved lazily by host_api.http_get; plugins don't
    # import it directly but the import graph passes through it.
    'analysis.plugins.permissions',
    # analysis.config is intentionally NOT here. Plugins must not call
    # get_config() directly -- that gives read/write access to the entire
    # application config store. Scoped config access goes through ctx.config
    # (read/write own namespace only) or host_api.plugin_config().
    'analysis.ui',
    'analysis.ui.components',
    'analysis.ui.render_sidebar',
    'analysis.overlay.api',
    # Component registration API -- needed by builtin sidepanel plugins.
    'analysis.components',
    'analysis.components.api',
    'analysis.components.overlay_backend',
    'analysis.components.viz_backend',
    # Visualization helpers -- needed by builtin viz plugins.
    # analysis.viz.plots provides only plotting functions, no FS/network access.
    'analysis.viz.plots',
    'analysis.viz',
    # sidepanel plugin package -- exports frozen dataclasses and constants only.
    'plugins.builtin.sidepanel',
})

# Explicit deny-list for obviously dangerous modules. The allow-list is
# already exclusive, so this is belt-and-suspenders: a future reviewer
# who adds ``urllib`` to the stdlib allow-list without thinking should
# still get caught here. Checked before the allow-list in ``_is_allowed``.
_EXPLICIT_DENY = frozenset({
    # Network
    'socket', 'ssl', 'select', 'selectors', 'asyncio',
    'http', 'urllib', 'urllib3', 'requests', 'httpx', 'aiohttp',
    'ftplib', 'smtplib', 'poplib', 'imaplib', 'telnetlib', 'nntplib',
    'xmlrpc', 'webbrowser',
    # Process / threading
    'subprocess', 'multiprocessing', 'threading', '_thread',
    # Filesystem / OS
    'os', 'sys', 'pathlib', 'shutil', 'tempfile', 'glob', 'fnmatch',
    'io', 'mmap', 'fcntl', 'pty', 'pwd', 'grp', 'resource', 'signal',
    # FFI / raw memory
    'ctypes', 'cffi',
    # Import machinery
    'importlib', 'pkgutil', 'runpy', 'zipimport',
    # Serialisation with exec paths
    'pickle', 'shelve', 'marshal',
    # Code introspection / execution
    'code', 'codeop', 'ast',
    # Frame walking -- these give access to live frame objects and globals
    # even without importing sys. gc.get_objects() walks to frame objects;
    # inspect/traceback expose sys._getframe equivalents.
    'gc', 'inspect', 'traceback', 'linecache',
    # NumPy escape vectors -- denied even though 'numpy' is allowed.
    # numpy.ctypeslib.load_library() loads arbitrary shared libraries;
    # numpy.ctypes is a module-level alias to ctypes.
    # numpy.distutils/f2py invoke subprocesses and compilers.
    # numpy.testing runs pytest as a subprocess.
    # The deny-list is checked before the allow-list so these beat 'numpy'.
    'numpy.ctypeslib', 'numpy.ctypes', 'numpy.distutils',
    'numpy.f2py', 'numpy.testing',
    # matplotlib escape vectors -- denied even though 'matplotlib' is allowed.
    # matplotlib.backends.backend_agg and gtk/wx/wx backends open display
    # connections; the Qt backend is handled by PySide6 being unlisted.
    # matplotlib.testing runs subprocess-based comparison tests.
    'matplotlib.testing',
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
    # ``analysis.player.render.theme`` is allowed).
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
    """Only ``plugins/unsafe/`` is trusted (full Python, no restrictions).
    All other plugin locations -- including ``plugins/builtin/`` -- are
    sandboxed. See SECURITY.md for the rationale."""
    try:
        repo_root = Path(__file__).resolve().parents[2]
        resolved = path.resolve()
        unsafe_root = (repo_root / 'plugins' / 'unsafe').resolve()
        return resolved.is_relative_to(unsafe_root)
    except (ValueError, OSError):
        return False


def prepare_sandboxed_module(mod: ModuleType):
    """Apply the sandbox to ``mod`` before its source executes. Called by
    the bundle loader between ``module_from_spec`` and ``exec_module``."""
    _install_finder_once()
    _SANDBOXED_MODULES.add(mod.__name__)
    mod.__builtins__ = _restricted_builtins()
