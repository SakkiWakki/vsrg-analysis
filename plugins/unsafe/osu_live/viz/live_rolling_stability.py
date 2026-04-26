"""Live wrap of ``plugins/builtin/viz/rolling_stability.py``.

Rolling standard-deviation of offsets ; shows whether timing is
settling or drifting mid-session. The per-hand split uses synthetic
round-robin columns, so the overall curve is the one to trust.
"""
from __future__ import annotations


def build(replay=None, game='osu', **_):
    from analysis.viz.live_figure import build_live_figure
    from plugins.builtin.viz.rolling_stability import build as build_static
    return build_live_figure(build_static, game='osu')


def register(add):
    add('Live: rolling stability', build, category='chart')
