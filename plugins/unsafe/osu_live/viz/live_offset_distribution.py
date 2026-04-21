"""Live wrap of ``plugins/builtin/viz/offset_distribution.py``.

Histogram of per-hit timing offsets, rebuilt each tick from the
native/tosu feed. Uses only ``offsets`` and ``columns``; columns are
the synthetic round-robin lanes, which means the per-hand split is
not real osu! hand data — but the overall distribution shape is
still meaningful and tracks the live session accurately.
"""
from __future__ import annotations


def build(replay=None, game='osu', **_):
    from plugins.unsafe.osu_live.live_viz import LiveFigureWidget
    from plugins.builtin.viz.offset_distribution import build as build_static

    def _rebuild(rep):
        return build_static(rep, game='osu')

    return LiveFigureWidget(_rebuild)


def register(add):
    add('Live: offset distribution', build, category='chart')
