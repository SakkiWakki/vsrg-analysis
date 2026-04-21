"""Discovery and dispatch for replay-player draw plugins.

Draw plugins and sidebar sections are loaded from bundles (see
``analysis.plugins`` for the bundle layout). A bundle's ``replay/`` folder
contributes lane-space draw plugins; its ``sidebar/`` folder contributes
HUD sections. Legacy single-file plugin paths are still honored for
backwards compatibility.
"""
from __future__ import annotations

import json
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
        self.bundles = []

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
    def discover(cls, extra_paths=None, active_theme_key=None):
        from analysis.plugins import discover_bundles
        from analysis.player import theme as theme_mod
        mgr = cls()
        mgr.bundles = discover_bundles(extra_paths)
        for bundle in mgr.bundles:
            for mod in bundle.replay_modules:
                mgr._register_replay(mod, bundle)
            for mod in bundle.sidebar_modules:
                mgr._register_sidebar(mod, bundle)
        # Activate the user-chosen theme if the owning bundle was found.
        for bundle in mgr.bundles:
            if bundle.theme_module and bundle.key == active_theme_key:
                theme_mod.set_active(bundle.theme_module, bundle.key)
                break
        else:
            theme_mod.set_active(None)
        return mgr

    def available_themes(self):
        """Return [(bundle_key, bundle_name)] for bundles that ship a theme,
        plus the built-in default entry."""
        themes = [('builtin', 'Built-in')]
        for bundle in self.bundles:
            if bundle.theme_module and bundle.key != 'builtin':
                themes.append((bundle.key, bundle.name))
        return themes

    def _register_replay(self, mod, bundle):
        if not hasattr(mod, 'register'):
            return
        module_name = f'{bundle.key}/{getattr(mod, "__name__", "")}'

        def add(name, draw, stages=None, priority=100, enabled=True,
                key=None):
            self.add(name, draw, stages=stages, priority=priority,
                     enabled=enabled, module=module_name, key=key)
        try:
            mod.register(add)
        except Exception as exc:
            print(f'replay plugin register failed: {module_name}: {exc}')

    def _register_sidebar(self, mod, bundle):
        if not hasattr(mod, 'register_sidebar'):
            return
        module_name = f'{bundle.key}/{getattr(mod, "__name__", "")}'

        def add_section(name, draw, *, priority=1000, key=None,
                        pin_bottom=False):
            self.sidebar.add(name, draw, priority=priority, key=key,
                             module=module_name, pin_bottom=pin_bottom)
        try:
            mod.register_sidebar(add_section)
        except Exception as exc:
            print(f'sidebar plugin register failed: {module_name}: {exc}')

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
