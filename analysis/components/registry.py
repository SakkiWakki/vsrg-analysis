"""Unified component registry.

A plugin module exposes::

    from analysis.components import ComponentManifest, SURFACE_GUI
    from plugins.builtin.sidepanel import SidebarFields

    MANIFEST = ComponentManifest(
        key='builtin:judgments',
        name='Judgments',
        supported_surfaces={SURFACE_GUI},
        requires_data={'judgment_counts'},
        plugin_fields={
            'sidebar': SidebarFields(priority=200, draggable=True),
        },
    )

    def draw(ctx):
        ...

    def register_components(add):
        add(MANIFEST, draw)

The registry discovers these modules across bundles the same way
``PluginManager.discover`` handles replay/sidebar modules. It *does
not* replace the existing sidebar/overlay registries — it feeds them
by registering an adapter section/spec so the rest of the renderer
continues to work without surgery. Plugins opt in to the new API one
at a time.
"""
from __future__ import annotations

from dataclasses import dataclass

from analysis.components.api import (
    Component,
    ComponentManifest,
    SURFACE_GUI,
    SURFACE_OVERLAY,
)


@dataclass
class _MountReport:
    """Per-surface decision about one component: did it mount, and if
    not, why? Surfaced so the plugin-panel UI can explain why a
    component didn't appear where the user expected."""
    surface: str
    mounted: bool
    reason: str = ''


class ComponentRegistry:
    """Owns all components that opted into the unified API.

    The registry's job is:
      1. Collect ``(manifest, draw)`` pairs as plugins register.
      2. For each declared surface, verify the surface's data source
         exposes every field in ``manifest.requires_data``; if not,
         skip that surface with a recorded reason.
      3. Expose ``components_for(surface)`` so each surface backend
         can iterate just its share.
    """

    def __init__(self):
        self._components: list[Component] = []
        # component.key → {surface: MountReport}. Populated lazily the
        # first time a surface asks about its components.
        self._mount_reports: dict[str, dict[str, _MountReport]] = {}

    # ── Registration ─────────────────────────────────────────────

    def add(self, manifest: ComponentManifest, draw) -> None:
        if not manifest.supported_surfaces:
            print(f'component {manifest.key}: no surfaces declared, skipping')
            return
        self._components.append(Component(manifest=manifest, draw=draw))

    def all_components(self) -> list[Component]:
        return list(self._components)

    def get(self, key: str) -> Component | None:
        for c in self._components:
            if c.manifest.key == str(key):
                return c
        return None

    # ── Mount decisions ──────────────────────────────────────────

    def components_for(self, surface: str,
                       *, data_source_fields: frozenset[str]
                       ) -> list[Component]:
        """Return components allowed on ``surface`` given the surface's
        live data-source capabilities.

        ``data_source_fields`` is the set of field names the caller's
        data source will answer (e.g. ``PlayerDataSource._FIELDS``).
        Components whose ``requires_data`` is not a subset are dropped.
        """
        out: list[Component] = []
        reports = self._mount_reports
        for c in self._components:
            m = c.manifest
            per = reports.setdefault(m.key, {})
            if surface not in m.supported_surfaces:
                per[surface] = _MountReport(
                    surface, False, 'surface not in manifest')
                continue
            missing = m.requires_data - data_source_fields
            if missing:
                per[surface] = _MountReport(
                    surface, False,
                    f'data source missing: {sorted(missing)}')
                continue
            per[surface] = _MountReport(surface, True)
            out.append(c)
        return out

    def report(self, key: str) -> dict[str, _MountReport]:
        """Return the last mount-decision per surface for a component,
        or ``{}`` if no surface has asked about it yet."""
        return dict(self._mount_reports.get(str(key), {}))


# ── Discovery ───────────────────────────────────────────────────


def discover_from_bundles(bundles) -> ComponentRegistry:
    """Walk bundle modules and collect components.

    Looks for ``register_components(add)`` on any module in
    ``bundle.replay_modules``, ``bundle.sidebar_modules``,
    ``bundle.overlay_modules``. This mirrors how the existing
    sidebar/overlay registries search — a plugin author can put a
    unified component anywhere, and the right surface picks it up.
    """
    registry = ComponentRegistry()
    for bundle in bundles:
        sidebar_mods = list(getattr(bundle, 'sidebar_modules', []) or [])
        overlay_mods = list(getattr(bundle, 'overlay_modules', []) or [])
        replay_mods = list(getattr(bundle, 'replay_modules', []) or [])
        for mod in sidebar_mods + overlay_mods + replay_mods:
            if not hasattr(mod, 'register_components'):
                continue
            module_name = f'{bundle.key}/{getattr(mod, "__name__", "")}'

            def _add(manifest: ComponentManifest, draw,
                     _module_name=module_name):
                # Re-emit with module attribution. The manifest is
                # frozen, so we build a replacement dataclass with the
                # module filled in.
                from dataclasses import replace
                registry.add(replace(manifest, module=_module_name), draw)

            try:
                mod.register_components(_add)
            except Exception as exc:
                print(f'component register failed: {module_name}: {exc}')
    return registry


# ── Bridge to existing gui + overlay runtimes ────────────────

def bridge_into_gui_registry(components: ComponentRegistry,
                              sidebar_registry) -> list[str]:
    """For every component that lists ``gui`` as a surface, add a
    section to the existing :class:`SidebarSectionRegistry` whose draw
    callable runs the component through the GUI backend.

    Returns the list of keys added. Components whose
    ``requires_data`` isn't satisfied by :class:`PlayerDataSource`
    are skipped with a recorded mount report.

    This is the glue that lets the existing Qt renderer host unified
    components without knowing anything about the new API — the section
    behaves like any other sidebar section.
    """
    from analysis.components.gui_backend import (
        PlayerDataSource,
        draw_component_in_sidebar,
    )
    from plugins.builtin.sidepanel import SidebarFields

    _sidebar_defaults = SidebarFields()
    fields = PlayerDataSource._FIELDS
    mounted: list[str] = []
    for comp in components.components_for(SURFACE_GUI,
                                          data_source_fields=fields):
        m = comp.manifest
        sf = m.plugin_fields.get('sidebar', _sidebar_defaults)

        def _draw_section(sctx, _comp=comp):
            draw_component_in_sidebar(_comp, sctx, player=sctx.player)

        sidebar_registry.add(
            m.name, _draw_section,
            key=m.key, module=m.module,
            priority=sf.priority,
            pin_bottom=sf.pin_bottom,
            draggable=sf.draggable,
            default_region=sf.region,
            default_free_xy=sf.default_free_xy,
            default_size=sf.default_size,
        )
        mounted.append(m.key)
    return mounted


from analysis.components.overlay_backend import (
        OverlayGameStateDataSource,
        OverlayComponentContext,
    )
from analysis.components.pal.base import OverlayFrame

from analysis import diag
def bridge_into_overlay_registry(components: ComponentRegistry,
                                 overlay_registry) -> list[str]:
    """Same idea for the overlay: register each eligible component as
    an overlay spec whose draw callable pumps the component's emitters
    through the overlay backend.

    The overlay backend's data source is :class:`OverlayGameStateDataSource`,
    so components that need e.g. ``judgment_windows`` (replay-only) are
    silently skipped.
    """

    fields = OverlayGameStateDataSource._FIELDS
    eligible = components.components_for(SURFACE_OVERLAY,
                                         data_source_fields=fields)
    diag.log('bridge.overlay',
             f'eligible: {[c.manifest.key for c in eligible]} '
             f'(of {len(components.all_components())} total)')
    mounted: list[str] = []
    for comp in eligible:
        m = comp.manifest

        # Per-component diagnostic latches. The draw fires at 30 Hz on
        # the overlay thread, so we only print on transitions ("first
        # frame after state appeared", "first frame after state
        # disappeared") to avoid flooding the log. The dict can't be
        # named ``diag`` because the closure also references the
        # ``analysis.diag`` module — Python's name resolution would
        # shadow the import.
        latches = {'last_state_present': None, 'last_records': -1}

        # Closure captures the component so the overlay-side draw_fn
        # can pull live game state at frame time. The game state lookup
        # is deliberately deferred — the overlay registry invokes draw_fn
        # on its thread; whichever module owns the state provides it
        # via a registered callback.
        def _draw(frame, _comp=comp, _m=m, _state=latches):
            # Overlay components that target the new API must accept a
            # _neutral_ OverlayFrame (PAL type), not the publisher's
            # _FrameBuilder. The overlay registry's draw path today
            # hands us a _FrameBuilder; we buffer into a PAL frame and
            # replay. This keeps the new backend independent of the
            # publisher's internals.
            from analysis.components.pal.gamescope import (
                GamescopeOverlayPlatform,
            )
            neutral = OverlayFrame(width=frame._pub.width,
                                   height=frame._pub.height)
            # Resolve game state via a shared hook (see provider.py).
            from analysis.components.provider import current_game_state
            state = current_game_state()
            present = state is not None
            if present != _state['last_state_present']:
                _state['last_state_present'] = present
                diag.log(
                    f'bridge.overlay:{_m.key}',
                    f'game state {"available" if present else "missing"}'
                    + (f' game={state.game} keycount={state.keycount}'
                       f' judgments={state.judgments}'
                       if present else ''))
            if state is None:
                return
            from analysis.components.overlay_backend import OverlayFields
            _of_defaults = OverlayFields()
            _of = _m.plugin_fields.get('overlay', _of_defaults)
            ox = int(_of.default_xy[0] * neutral.width)
            oy = int(_of.default_xy[1] * neutral.height)
            sw = int(_of.default_size[0] * neutral.width)
            sh = int(_of.default_size[1] * neutral.height)
            cctx = OverlayComponentContext(
                neutral, component_key=_m.key,
                origin_px=(ox, oy), size_px=(sw, sh),
                fb_w=neutral.width, fb_h=neutral.height,
                data_source=OverlayGameStateDataSource(state),
                supports_input=False)
            neutral.begin_group(_m.key)
            try:
                _comp.draw(cctx)
            finally:
                neutral.end_group()
            n = len(neutral.records)
            if n != _state['last_records']:
                _state['last_records'] = n
                diag.log(
                    f'bridge.overlay:{_m.key}',
                    f'emitted {n} records (origin_px={ox,oy} '
                    f'size_px={sw,sh} fb={neutral.width}x{neutral.height})')
            GamescopeOverlayPlatform._replay(neutral, frame)

        from analysis.components.overlay_backend import OverlayFields
        of = comp.manifest.plugin_fields.get('overlay', OverlayFields())
        overlay_registry.add(
            m.name, _draw, key=m.key, module=m.module,
            hz=of.hz)
        mounted.append(m.key)
    diag.log('bridge.overlay', f'mounted: {mounted}')
    return mounted
