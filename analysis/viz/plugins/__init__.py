"""Visualization registry. Modules that expose a top-level
``register(add)`` function in a bundle's ``viz/`` folder are discovered
through ``analysis.plugins.discover_bundles``.

``builder(replay, game, **kw)`` must return a matplotlib Figure (static) or
a QWidget (interactive). Everything discovered here shows up in the GUI's
"Visualize" menu and works for any number of keys ; builders should derive
keycount from the replay.
"""
from __future__ import annotations


_REGISTRY = []  # [(name, builder, category)]


def add(name, builder, category='chart'):
    _REGISTRY.append((name, builder, category))


def discover():
    """Import viz modules from every discovered bundle."""
    from analysis.plugins import discover_bundles
    from analysis.components.registry import discover_from_bundles
    from analysis.components.viz_backend import bridge_into_viz_registry

    _REGISTRY.clear()
    bundles = discover_bundles()
    for bundle in bundles:
        for mod in bundle.viz_modules:
            if hasattr(mod, 'register'):
                try:
                    mod.register(add)
                except Exception as exc:
                    src = getattr(mod, '__name__', '?')
                    print(f'viz plugin register failed: {bundle.key}/{src}: {exc}')
    components = discover_from_bundles(bundles)
    for name, builder, category in bridge_into_viz_registry(components):
        add(name, builder, category=category)
    return list(_REGISTRY)


def all_visualizations():
    if not _REGISTRY:
        discover()
    return list(_REGISTRY)
