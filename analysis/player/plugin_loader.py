"""Discovery and dispatch for replay-player draw plugins.

Draw plugins and sidebar sections are loaded from bundles (see
``analysis.plugins`` for the bundle layout). A bundle's ``replay/`` folder
contributes lane-space draw plugins; its ``sidebar/`` folder contributes
HUD sections. Legacy single-file plugin paths are still honored for
backwards compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass

from analysis.player.plugin_api import Stage, normalize_stage
from analysis.player.sidebar_api import SidebarSectionRegistry, _escape_key


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
    """Owns replay-draw plugins + the sidebar registry for one window.

    Enabled/disabled state (and future plugin-owned settings) lives in
    the process-wide :class:`analysis.config.ConfigStore`; this manager
    subscribes to ``plugins`` so toggles from another window's Plugins
    dialog propagate to the renderer without restart."""

    def __init__(self, config=None):
        from analysis.config import get_config
        self._config = config if config is not None else get_config()
        self._plugins: list[DrawPlugin] = []
        self.sidebar = SidebarSectionRegistry(config=self._config)
        self.bundles = []
        self._config_sub = self._config.subscribe(
            'plugins', self._on_config_change)
        # Runtime failures (an exception inside a draw callable) force
        # the plugin off regardless of config — these keys stay off
        # even if the user flips the dialog checkbox back on. Cleared
        # on rediscovery.
        self._runtime_disabled: set[str] = set()

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
            enabled=bool(enabled) and not self._is_disabled(key),
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
                self._runtime_disabled.add(plugin.key)
                src = f' ({plugin.module})' if plugin.module else ''
                print(f'player plugin disabled: {plugin.name}{src}: {exc}')

    def all_plugins(self):
        return list(self._plugins)

    def enabled_count(self):
        return sum(1 for p in self._plugins if p.enabled)

    def set_enabled(self, key, enabled):
        """Flip a plugin on/off through the config store. Returns True
        if the key matches a known plugin."""
        key = str(key)
        if not any(p.key == key for p in self._plugins):
            return False
        cleared_latch = False
        if enabled and key in self._runtime_disabled:
            # User explicitly re-enabled — forget the runtime-failure
            # latch so the plugin gets another chance.
            self._runtime_disabled.discard(key)
            cleared_latch = True
        changed = self._config.set(
            f'plugins.{_escape_key(key)}.replay_disabled',
            not bool(enabled))
        if not changed and cleared_latch:
            # Config value already matched the target; the fanout
            # handler wouldn't fire, but we still need to honor the
            # latch clear on this manager's plugins.
            for p in self._plugins:
                if p.key == key:
                    p.enabled = bool(enabled)
        return True

    def toggle_enabled(self, key):
        for plugin in self._plugins:
            if plugin.key == key:
                return self.set_enabled(key, not plugin.enabled)
        return False

    def close(self):
        """Drop the config subscription + close the sidebar registry."""
        if self._config_sub is not None:
            self._config.unsubscribe(self._config_sub)
            self._config_sub = None
        try:
            self.sidebar.close()
        except Exception:
            pass

    def _is_disabled(self, key: str) -> bool:
        return bool(self._config.get(
            f'plugins.{_escape_key(key)}.replay_disabled', False))

    def _on_config_change(self, path, old, new):
        if len(path) < 3 or path[-1] != 'replay_disabled':
            return
        escaped = path[1]
        disabled = bool(new) if new is not None else False
        for p in self._plugins:
            if _escape_key(p.key) != escaped:
                continue
            if disabled:
                p.enabled = False
            elif p.key in self._runtime_disabled:
                # Config says "on" but the plugin crashed earlier — keep
                # it off until discover() resets.
                pass
            else:
                p.enabled = True

    @classmethod
    def discover(cls, extra_paths=None, active_theme_key=None, config=None):
        from analysis.plugins import discover_bundles
        from analysis.player import theme as theme_mod
        mgr = cls(config=config)
        mgr.bundles = discover_bundles(extra_paths)
        for bundle in mgr.bundles:
            for mod in bundle.replay_modules:
                mgr._register_replay(mod, bundle)
            for mod in bundle.sidebar_modules:
                mgr._register_sidebar(mod, bundle)
            # Library-toolbar actions can live in any of the bundle's
            # module roles (viz is the natural home — a viz bundle may
            # want a "go live" button — but we don't force a shape).
            for mod in (list(bundle.replay_modules)
                        + list(bundle.sidebar_modules)
                        + list(getattr(bundle, 'viz_modules', []) or [])):
                mgr._register_library_actions(mod, bundle)
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

    def _register_library_actions(self, mod, bundle):
        """Let a bundle module contribute toolbar buttons to the library
        tab. Modules opt-in by exposing ``register_library_actions(add)``
        at the top level; the registry is process-wide so every window's
        library tab shows the same set."""
        if not hasattr(mod, 'register_library_actions'):
            return
        from analysis.gui.library_actions import get_registry
        module_name = f'{bundle.key}/{getattr(mod, "__name__", "")}'
        registry = get_registry()

        def add_action(label, callback, *, key=None):
            registry.add(label, callback, key=key, module=module_name)
        try:
            mod.register_library_actions(add_action)
        except Exception as exc:
            print(f'library-action register failed: {module_name}: {exc}')

