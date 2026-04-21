"""Visualization plugin registry. Drop a .py file in this folder with a
top-level `register(add)` function that calls `add(name, builder)` for each
visualization you want to expose. `builder(replay, game, **kw)` must return a
matplotlib Figure (for static plots) or a QWidget (for interactive ones).

Everything discovered here shows up in the GUI's "Visualize" menu and works
for any number of keys — builders should derive keycount from the replay.
"""
import importlib
import pkgutil
from pathlib import Path


_REGISTRY = []  # [(name, builder, category)]


def add(name, builder, category='chart'):
    _REGISTRY.append((name, builder, category))


def discover():
    """Import every sibling module; modules with a `register(add)` add themselves."""
    _REGISTRY.clear()
    pkg_dir = Path(__file__).parent
    for info in pkgutil.iter_modules([str(pkg_dir)]):
        if info.name.startswith('_'):
            continue
        mod = importlib.import_module(f'{__name__}.{info.name}')
        if hasattr(mod, 'register'):
            mod.register(add)
    return list(_REGISTRY)


def all_visualizations():
    if not _REGISTRY:
        discover()
    return list(_REGISTRY)
