"""Live wrap of ``plugins/builtin/viz/per_column.py``.

Per-column mean-offset bars. **Caveat:** we don't receive per-hit
column from osu!, so the client round-robins hits across the keycount
for continuity. The bar heights here are therefore *not* real lane
timing ; they converge to the session mean as hits accumulate. Kept
wrapped so the viz is discoverable in the live slot, but treat it as
a sanity check rather than lane analysis until a real column feed
lands.
"""
from __future__ import annotations


def build(replay=None, game='osu', **_):
    from analysis.viz.live_figure import build_live_figure
    from plugins.builtin.viz.per_column import build as build_static
    return build_live_figure(build_static, game='osu')


def register(add):
    add('Live: per-column (synthetic lanes)', build, category='chart')
