"""Discovery and dispatch for replay-player draw plugins."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import pkgutil
from dataclasses import dataclass
from pathlib import Path

from analysis.player.plugin_api import Stage, normalize_stage
from analysis.player.sidebar_api import SidebarSectionRegistry


@dataclass
class DrawPlugin:
    key: str
    name: str
    draw: object
    stages: tuple[Stage, ...]
    priority: int = 100
    module: str = ''
    enabled: bool = True


class PluginManager:
    def __init__(self):
        self._plugins: list[DrawPlugin] = []
        self._disabled_keys = self._load_disabled_keys()
        self.sidebar = SidebarSectionRegistry()

    def add(self, name, draw, stages=None, priority=100, enabled=True,
            module='', key=None):
        if stages is None:
            stages = (Stage.POST_FRAME,)
        if isinstance(stages, (str, Stage)):
            stages = (stages,)
        key = str(key or f'{module}:{name}')
        spec = DrawPlugin(
            key=key,
            name=str(name),
            draw=draw,
            stages=tuple(normalize_stage(s) for s in stages),
            priority=int(priority),
            module=str(module),
            enabled=bool(enabled) and key not in self._disabled_keys,
        )
        self._plugins.append(spec)
        self._plugins.sort(key=lambda p: (p.priority, p.name))

    def draw(self, stage, ctx):
        stage = normalize_stage(stage)
        for plugin in list(self._plugins):
            if not plugin.enabled or stage not in plugin.stages:
                continue
            try:
                plugin.draw(ctx, stage)
            except Exception as exc:
                plugin.enabled = False
                src = f' ({plugin.module})' if plugin.module else ''
                print(f'player plugin disabled: {plugin.name}{src}: {exc}')

    def all_plugins(self):
        return list(self._plugins)

    def enabled_count(self):
        return sum(1 for p in self._plugins if p.enabled)

    def set_enabled(self, key, enabled, persist=True):
        key = str(key)
        changed = False
        for plugin in self._plugins:
            if plugin.key == key:
                plugin.enabled = bool(enabled)
                changed = True
        if not changed:
            return False
        if persist:
            if enabled:
                self._disabled_keys.discard(key)
            else:
                self._disabled_keys.add(key)
            self._save_disabled_keys()
        return True

    def toggle_enabled(self, key):
        for plugin in self._plugins:
            if plugin.key == key:
                return self.set_enabled(key, not plugin.enabled)
        return False

    @classmethod
    def discover(cls, extra_paths=None):
        mgr = cls()
        mgr._discover_builtin()
        mgr._discover_paths(extra_paths)
        return mgr

    def _register_module(self, mod):
        module_name = getattr(mod, '__player_plugin_source__',
                              getattr(mod, '__name__', ''))
        handled = False

        if hasattr(mod, 'register'):
            def add(name, draw, stages=None, priority=100, enabled=True,
                    key=None):
                self.add(name, draw, stages=stages, priority=priority,
                         enabled=enabled, module=module_name, key=key)
            mod.register(add)
            handled = True

        if hasattr(mod, 'register_sidebar'):
            def add_section(name, draw, *, priority=1000, key=None,
                            pin_bottom=False):
                self.sidebar.add(name, draw, priority=priority, key=key,
                                 module=module_name, pin_bottom=pin_bottom)
            mod.register_sidebar(add_section)
            handled = True

        if not handled:
            return

    def _discover_builtin(self):
        pkg_name = 'analysis.player.draw_plugins'
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as exc:
            print(f'player plugin package unavailable: {exc}')
            return
        pkg_dir = Path(pkg.__file__).parent
        for info in pkgutil.iter_modules([str(pkg_dir)]):
            if info.name.startswith('_'):
                continue
            try:
                mod = importlib.import_module(f'{pkg_name}.{info.name}')
                self._register_module(mod)
            except Exception as exc:
                print(f'player plugin import failed: {info.name}: {exc}')

    def _discover_paths(self, extra_paths=None):
        for path in self._plugin_paths(extra_paths):
            if not path.is_dir():
                continue
            for file_path in sorted(path.glob('*.py')):
                if file_path.name.startswith('_'):
                    continue
                self._load_file(file_path)

    def _load_file(self, file_path):
        mod_name = f'_ea_player_plugin_{abs(hash(str(file_path.resolve())))}'
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        try:
            mod.__player_plugin_source__ = str(file_path.resolve())
            spec.loader.exec_module(mod)
            self._register_module(mod)
        except Exception as exc:
            print(f'player plugin import failed: {file_path}: {exc}')

    @staticmethod
    def _state_path():
        return Path.home() / '.config' / 'vsrg-analysis' / 'player_plugins.json'

    def _load_disabled_keys(self):
        path = self._state_path()
        try:
            data = json.loads(path.read_text())
        except Exception:
            return set()
        return set(str(k) for k in data.get('disabled', []))

    def _save_disabled_keys(self):
        path = self._state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {'disabled': sorted(self._disabled_keys)}
            path.write_text(json.dumps(data, indent=2) + '\n')
        except Exception as exc:
            print(f'player plugin state save failed: {exc}')

    @staticmethod
    def _plugin_paths(extra_paths=None):
        paths = []
        env = os.environ.get('ETTERNA_ANALYSIS_PLAYER_PLUGINS', '')
        for raw in env.split(os.pathsep):
            if raw.strip():
                paths.append(Path(raw).expanduser())
        if extra_paths:
            paths.extend(Path(p).expanduser() for p in extra_paths)
        cwd = Path.cwd()
        repo_root = Path(__file__).resolve().parents[2]
        paths.extend([
            cwd / 'draw_extensions',
            cwd / 'player_plugins',
            repo_root / 'draw_extensions',
            repo_root / 'player_plugins',
            Path.home() / '.config' / 'vsrg-analysis' / 'player_plugins',
        ])
        seen = set()
        for p in paths:
            rp = str(p)
            if rp in seen:
                continue
            seen.add(rp)
            yield p
