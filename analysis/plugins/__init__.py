"""Unified plugin-bundle discovery.

A *bundle* is a directory with this layout::

    <bundle>/
      manifest.toml        # optional: {name, key, version, author}
      sidebar/             # sidebar sections (register_sidebar)
      replay/              # lane-space draw plugins (register, Stage-based)
      viz/                 # visualizations (register)
      overlay/             # in-game overlay feeds (register_overlay)
      theme/               # optional; when active, overrides theme tokens

Bundles are discovered at these locations (later paths override earlier):

  1. ``plugins/`` at the repo root — ships the built-in bundle
  2. ``$EA_PLUGINS_PATH`` — colon-separated list of extra bundle roots
  3. ``~/.config/vsrg-analysis/plugins/`` — per-user bundles

Each *bundle root* contains one or more bundle subdirectories. The folder
name is the default bundle key (lowercased) if ``manifest.toml`` does not
override it.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib                 # 3.11+
except ImportError:  # pragma: no cover
    tomllib = None


@dataclass
class Bundle:
    key: str
    name: str
    path: Path
    manifest: dict = field(default_factory=dict)
    trusted: bool = False
    sidebar_modules: list = field(default_factory=list)
    replay_modules: list = field(default_factory=list)
    viz_modules: list = field(default_factory=list)
    overlay_modules: list = field(default_factory=list)
    theme_module: object | None = None
    load_errors: list = field(default_factory=list)  # [(role, filename, exc)]


_SUBDIR_ROLES = ('sidebar', 'replay', 'viz', 'overlay')
# Subdirectory aliases: additional folder names that populate the same
# bundle attribute as an existing role. 'sidepanel' populates sidebar_modules
# since it's the plugin folder name for the builtin sidepanel plugin.
_SUBDIR_ALIASES = {'sidepanel': 'sidebar'}


def _bundle_roots(extra_paths=None):
    roots = []
    repo_root = Path(__file__).resolve().parents[2]
    roots.append(repo_root / 'plugins')
    # ``plugins/unsafe/`` is a container of trusted bundles (opt-in escape
    # hatch for raw-Python plugins). Treated as an extra bundle root so
    # its children are discoverable as bundles in their own right.
    roots.append(repo_root / 'plugins' / 'unsafe')
    env = os.environ.get('EA_PLUGINS_PATH', '')
    for raw in env.split(os.pathsep):
        if raw.strip():
            roots.append(Path(raw).expanduser())
    if extra_paths:
        roots.extend(Path(p).expanduser() for p in extra_paths)
    roots.append(Path.home() / '.config' / 'vsrg-analysis' / 'plugins')
    seen = set()
    for r in roots:
        rp = str(r)
        if rp in seen:
            continue
        seen.add(rp)
        yield r


def _load_manifest(path: Path) -> dict:
    manifest_path = path / 'manifest.toml'
    if not manifest_path.is_file() or tomllib is None:
        return {}
    try:
        with open(manifest_path, 'rb') as f:
            return tomllib.load(f)
    except Exception as exc:
        print(f'plugin manifest failed: {manifest_path}: {exc}')
        return {}


def _load_module(file_path: Path, bundle_key: str, role: str,
                 sandboxed: bool = False):
    """Import a plugin module from an absolute path with a stable, unique
    fully-qualified name so sibling-relative imports (``from ._common
    import ...``) still work within a bundle's subfolder.

    When ``sandboxed`` is True the module is loaded with a restricted
    ``__builtins__`` dict and its imports are vetted by the sandbox
    finder.

    Returns ``(module_or_none, error_or_none)``.
    """
    from analysis.plugins.sandbox import prepare_sandboxed_module
    pkg_name = f'_ea_bundle.{bundle_key}.{role}'
    _ensure_package(pkg_name, (file_path.parent,))
    mod_name = f'{pkg_name}.{file_path.stem}'
    if mod_name in sys.modules:
        return sys.modules[mod_name], None
    spec = importlib.util.spec_from_file_location(
        mod_name, file_path,
        submodule_search_locations=None,
    )
    if spec is None or spec.loader is None:
        return None, ImportError(f'no loader for {file_path}')
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    if sandboxed:
        prepare_sandboxed_module(mod)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        return None, exc
    return mod, None


def _ensure_package(fqname: str, search_paths):
    """Register an in-memory namespace package so relative imports
    (``from ._common import ...``) resolve correctly inside a bundle
    subfolder without polluting the regular import graph."""
    parts = fqname.split('.')
    for i in range(1, len(parts) + 1):
        name = '.'.join(parts[:i])
        if name in sys.modules:
            continue
        spec = importlib.util.spec_from_loader(name, loader=None,
                                               is_package=True)
        pkg = importlib.util.module_from_spec(spec)
        pkg.__path__ = (list(search_paths)
                        if i == len(parts) else [])
        sys.modules[name] = pkg


def _iter_py_files(folder: Path):
    if not folder.is_dir():
        return
    for entry in sorted(folder.glob('*.py')):
        if entry.name == '__init__.py' or entry.name.startswith('.'):
            continue
        yield entry


def _preload_helpers(folder: Path, bundle_key: str, role: str,
                     sandboxed: bool, errors: list):
    """Import underscore-prefixed ``_helper.py`` siblings first so modules
    in the same folder can do ``from ._helper import …``."""
    if not folder.is_dir():
        return
    for entry in sorted(folder.glob('_*.py')):
        if entry.name == '__init__.py':
            continue
        _, err = _load_module(entry, bundle_key, role, sandboxed=sandboxed)
        if err is not None:
            errors.append((role, entry.name, err))


def _load_theme(bundle_path: Path, bundle_key: str, sandboxed: bool):
    theme_dir = bundle_path / 'theme'
    if theme_dir.is_dir() and (theme_dir / '__init__.py').is_file():
        mod, _err = _load_module(theme_dir / '__init__.py', bundle_key,
                                 'theme', sandboxed=sandboxed)
        return mod
    theme_py = bundle_path / 'theme.py'
    if theme_py.is_file():
        mod, _err = _load_module(theme_py, bundle_key, 'theme',
                                 sandboxed=sandboxed)
        return mod
    return None


def discover_bundles(extra_paths=None) -> list[Bundle]:
    from analysis.plugins.sandbox import is_trusted_bundle
    bundles: list[Bundle] = []
    for root in _bundle_roots(extra_paths):
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith('_'):
                continue
            if not any((entry / sub).is_dir() for sub in _SUBDIR_ROLES):
                continue
            manifest = _load_manifest(entry)
            key = str(manifest.get('key') or entry.name).lower()
            name = str(manifest.get('name') or entry.name)
            trusted = is_trusted_bundle(entry)
            sandboxed = not trusted
            bundle = Bundle(key=key, name=name, path=entry,
                            manifest=manifest, trusted=trusted)
            all_roles = list(_SUBDIR_ROLES) + list(_SUBDIR_ALIASES)
            for role in all_roles:
                target_attr = _SUBDIR_ALIASES.get(role, role)
                sub = entry / role
                target = getattr(bundle, f'{target_attr}_modules')
                _preload_helpers(sub, key, target_attr, sandboxed,
                                 bundle.load_errors)
                for fp in _iter_py_files(sub):
                    if fp.name.startswith('_'):
                        continue
                    mod, err = _load_module(fp, key, target_attr,
                                            sandboxed=sandboxed)
                    if mod is not None:
                        target.append(mod)
                    elif err is not None:
                        bundle.load_errors.append((target_attr, fp.name, err))
            bundle.theme_module = _load_theme(entry, key, sandboxed)
            for role, fname, exc in bundle.load_errors:
                tag = 'sandboxed' if sandboxed else 'trusted'
                print(f'plugin [{tag}] {key}/{role}/{fname} refused: {exc}')
            bundles.append(bundle)
    return bundles
