"""Live wrap of ``plugins/builtin/viz/scatter_timeline.py``.

Per-hit offset scatter with hit index on the x-axis. The static viz
converts noterows to seconds; for live data our noterows are a
synthetic ``arange``, so x here is effectively "hit number" rather
than real session time. The shape of the scatter is the same
information either way.
"""
from __future__ import annotations


def build(replay=None, game='osu', **_):
    from plugins.unsafe.osu_live.live_viz import LiveFigureWidget
    from plugins.builtin.viz.scatter_timeline import build as build_static

    def _rebuild(rep):
        return build_static(rep, game='osu')

    return LiveFigureWidget(_rebuild)


def register(add):
    add('Live: scatter timeline', build, category='chart')
