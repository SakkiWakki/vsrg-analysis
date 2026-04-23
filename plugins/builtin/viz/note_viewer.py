"""Scrollable Quaver-style note and press visualizer."""
from analysis.components import Manifest, SURFACE_VIZ, VizFields
from analysis.components.viz_backend import VIZ_CATEGORY_WIDGET


MANIFEST = Manifest(
    key='builtin:viz:note_viewer',
    name='Note visualizer (scrollable)',
    supported_surfaces={SURFACE_VIZ},
    plugin_fields={'viz': VizFields(category=VIZ_CATEGORY_WIDGET)},
)


def _draw(ctx):
    od = None
    judge = None
    if ctx.entry is not None:
        od = ctx.entry.get('od')
        judge = ctx.entry.get('judge')
    ctx.widget(ctx.build_note_visualizer(od=od, judge=judge))


def register_components(add):
    add(MANIFEST, _draw)
