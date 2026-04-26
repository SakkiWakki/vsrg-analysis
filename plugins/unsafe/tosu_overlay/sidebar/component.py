"""Unified-API components, one per discovered tosu overlay.

Each overlay gets its own Manifest + draw closure so the sidebar
plugin panel shows 'tosu: <overlay name>' as a togglable row. Every
overlay is **disabled by default** on first run; the user enables
whichever ones they want from the plugins panel (Shift-Tab, then the
Plugins tab).

Backed by the web-texture PAL:
  * SURFACE_GUI -> KIND_QPIXMAP (QPixmapBackend).
  * SURFACE_OVERLAY -> KIND_DMABUF_FD (DmabufBackend) when the
    injected gl_layer is listening; otherwise silently no-op.

The overlay's HTML/CSS/JS is unmodified -- our shim.js replaces
``window.WebSocket`` before any page JS runs and forwards state
through a QWebChannel bridge. Translation between our GameState
API and the tosu v1+v2 protocol lives in :mod:`translation`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from analysis.components.api import (
    LayerDeclaration,
    LayerPlacement,
    LAYER_LEAF,
    Manifest,
    SURFACE_GUI,
    SURFACE_OVERLAY,
)
from analysis.components.pal.web import (
    SURFACE_CROSSPROC_GL,
    SURFACE_LOCAL_CPU,
    WebTexturePAL,
)
from plugins.builtin.sidepanel import REGION_FREE, SidebarFields
from plugins.unsafe.tosu_overlay.bridge import OverlayBridge
from plugins.unsafe.tosu_overlay.discovery import discovery_roots, find_overlays
from plugins.unsafe.tosu_overlay.translation import (
    build_precise_state,
    build_tosu_state,
    prune_to_filters,
)


# Per-mount state. Keyed by the component key + surface so each overlay
# on each surface owns its own WebTexture. Populated lazily on first
# draw; a disabled section never draws, so its mount is never created.
_mounts: dict[str, '_Mount'] = {}


# ── Mount (owns the WebTexture + bridge) ──────────────────────────

class _Mount:
    """Live state for one tosu overlay component instance.

    The PAL surface choice drives which backend produces frames:
      - SURFACE_LOCAL_CPU     -> QPixmapBackend (GUI / sidebar).
      - SURFACE_CROSSPROC_GL  -> DmabufBackend  (injected gl_layer).
    The mount is otherwise identical between the two: same shim,
    same bridge, same state-push cadence. Only ``latest_frame()``'s
    kind changes.
    """

    def __init__(self, width: int, height: int, overlay_path: Path,
                 *, pal_surface: str = SURFACE_LOCAL_CPU):
        self.width = width
        self.height = height
        self.overlay_path = overlay_path
        self.last_generation = -1
        self.cached_frame = None

        pal = WebTexturePAL.default()
        self.texture = pal.create(
            surface=pal_surface, width=width, height=height)

        self._install_shim()

        self.bridge = OverlayBridge(self.texture.view)
        from PySide6.QtWebChannel import QWebChannel
        channel = QWebChannel(self.texture.view)
        channel.registerObject('bridge', self.bridge)
        self.texture.view.page().setWebChannel(channel)
        self.bridge.pushToJs.connect(self._deliver_to_js)
        self.texture.attach_bridge(self.bridge)

        from PySide6.QtCore import QUrl
        self.texture.view.load(
            QUrl.fromLocalFile(str(overlay_path.resolve())))

    def _install_shim(self) -> None:
        from PySide6.QtWebEngineCore import QWebEngineScript

        page = self.texture.view.page()
        scripts = page.scripts()

        candidates = [
            Path(__file__).parent.parent.parent.parent
                / 'third_party' / 'qwebchannel.js',
            Path('/usr/share/qt6/webchannel/qwebchannel.js'),
            Path('/usr/lib/qt6/qml/QtWebChannel/qwebchannel.js'),
        ]
        qwc = next((p for p in candidates if p.exists()), None)
        if qwc is not None:
            s1 = QWebEngineScript()
            s1.setName('tosu:qwebchannel')
            s1.setSourceCode(qwc.read_text(encoding='utf-8'))
            s1.setInjectionPoint(
                QWebEngineScript.InjectionPoint.DocumentCreation)
            s1.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
            scripts.insert(s1)

        shim_path = Path(__file__).parent.parent / 'shim.js'
        s2 = QWebEngineScript()
        s2.setName('tosu:shim')
        s2.setSourceCode(shim_path.read_text(encoding='utf-8'))
        s2.setInjectionPoint(
            QWebEngineScript.InjectionPoint.DocumentCreation)
        s2.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        scripts.insert(s2)

    def _deliver_to_js(self, json_str: str) -> None:
        safe = json_str.replace('\\', '\\\\').replace('`', '\\`')
        self.texture.view.page().runJavaScript(
            f'window._tosuPush && window._tosuPush(`{safe}`);')

    def push_state(self, game_state) -> None:
        try:
            state = build_tosu_state(game_state)
            pruned = prune_to_filters(state, self.texture.active_filters())
            self.bridge.push(json.dumps(pruned))
            precise = build_precise_state(game_state)
            self.texture.push_precise_state(json.dumps(precise))
        except Exception as exc:
            print(f'tosu_overlay.component: push_state failed: {exc}')

    def sync_frame(self):
        frame = self.texture.latest_frame()
        if frame is None:
            return self.cached_frame
        if frame.generation != self.last_generation:
            self.last_generation = frame.generation
            self.cached_frame = frame
        return self.cached_frame

    def close(self) -> None:
        self.texture.close()


# ── Per-overlay registration ──────────────────────────────────────

_DEFAULT_W, _DEFAULT_H = 640, 360

# Cap how many overlays we register. The tosuapp/counters repo has ~60
# entries; each Manifest + disabled row in the plugins panel is cheap,
# but we don't want a broken discovery source dumping 10k entries.
_MAX_OVERLAYS = 128


def _slug(name: str) -> str:
    """Normalize an overlay name to an ASCII slug usable in config
    paths + component keys. The config store escapes dots via
    ``_escape_key`` (dots -> underscores), but we keep it simple and
    avoid dots entirely."""
    s = re.sub(r'[^A-Za-z0-9]+', '_', name.strip()).strip('_').lower()
    return s or 'overlay'


def _surface_to_pal(ctx_surface: str) -> str:
    if ctx_surface == SURFACE_OVERLAY:
        return SURFACE_CROSSPROC_GL
    return SURFACE_LOCAL_CPU


def _make_draw(overlay_path: Path, slug: str):
    """Build a per-overlay draw closure. Captures the overlay path so
    the mount instantiates the right page."""
    def draw(ctx) -> None:
        key = f'{slug}:{ctx.surface}'
        mount = _mounts.get(key)
        if mount is None:
            try:
                mount = _Mount(_DEFAULT_W, _DEFAULT_H, overlay_path,
                               pal_surface=_surface_to_pal(ctx.surface))
            except Exception as exc:
                # PAL couldn't produce a texture (e.g. dmabuf backend
                # offline on SURFACE_OVERLAY). Show a one-line hint on
                # GUI; silent elsewhere.
                if ctx.surface == SURFACE_GUI:
                    ctx.draw_heading(f'tosu: {overlay_path.parent.name}')
                    ctx.draw_hint(f'backend unavailable: {exc}')
                return
            _mounts[key] = mount

        mount.push_state(ctx.data)

        width = ctx.w
        height = int(width * _DEFAULT_H / _DEFAULT_W)
        if (width, height) != (mount.width, mount.height):
            mount.width, mount.height = width, height
            mount.texture.resize(width, height)

        frame = mount.sync_frame()
        if frame is not None:
            ctx.image((0, ctx.y, width, height), frame)
        ctx.y += height
    return draw


def _make_manifest(overlay_name: str, slug: str) -> Manifest:
    """Build the Manifest for one overlay. The layer declaration is
    keyed by slug so the layers panel shows each overlay as its own
    togglable layer (useful for per-overlay show/hide without
    disabling the whole plugin)."""
    key = f'tosu_overlay:{slug}'
    return Manifest(
        key=key,
        name=f'tosu: {overlay_name}',
        supported_surfaces=frozenset({SURFACE_GUI, SURFACE_OVERLAY}),
        requires_data=frozenset(),
        optional_data=frozenset({
            'chart_metadata', 'chart_stats', 'chart_paths',
            'judgment_counts', 'hit_errors_ms', 'unstable_rate',
            'combo', 'max_combo', 'score', 'current_grade',
            'mods_short', 'mods_raw', 'play_rate_effective',
            'player_name', 'paused', 't_now', 'game',
        }),
        layers=(
            LayerDeclaration(
                key=key,
                name=overlay_name,
                placement=LayerPlacement('inside', 'root'),
                kind=LAYER_LEAF,
                default_visible=True,
            ),
        ),
        # Each overlay spawns in the free region so the user can drag
        # it wherever; default size matches the 16:9 mount aspect.
        plugin_fields={
            'sidebar': SidebarFields(
                priority=2000,                    # appear after built-ins
                draggable=True,
                region=REGION_FREE,
                default_size=(420, 236),
            ),
        },
    )


# ── Disable-by-default config ─────────────────────────────────────
#
# We mark the "sidebar_disabled" flag True in the shared config store
# for every tosu overlay key that has never been seen before. First
# run = all disabled. After the user toggles one on, their state is
# authoritative -- we never overwrite.

_CONFIG_SENTINEL = object()


def _disable_if_unseen(key: str) -> None:
    """Write ``sidebar_disabled=True`` for ``key`` iff no prior value
    was ever set. Keeps user-driven toggles stable across restarts."""
    try:
        from analysis.config import get_config
    except ImportError:
        return
    try:
        store = get_config()
    except Exception:
        return
    # Mirror SidebarSectionRegistry._escape_key: dots -> underscores.
    escaped = key.replace('.', '_')
    path = f'plugins.{escaped}.sidebar_disabled'
    existing = store.get(path, _CONFIG_SENTINEL)
    if existing is _CONFIG_SENTINEL:
        store.set(path, True)


# ── Entry point ──────────────────────────────────────────────────

def register_components(add) -> None:
    """Enumerate every discovered overlay and register a component for
    each. All start disabled; the plugins panel toggles them on.

    If discovery returns nothing, register nothing. The user sees an
    empty plugins-panel section under 'tosu:...' and can fix the
    discovery path (clone tosuapp/counters, or drop index.html files
    into ``plugins/overlays/<name>/``).
    """
    overlays = find_overlays()[:_MAX_OVERLAYS]
    if not overlays:
        roots = ', '.join(str(p) for p in discovery_roots())
        print('tosu_overlay: no overlays discovered. '
              f'checked roots: {roots}. '
              'If using TOSU_OVERLAYS_DIRS, export it before launching app.')
        return
    seen_slugs: set[str] = set()
    for name, path in overlays:
        slug = _slug(name)
        # Disambiguate slug collisions by appending a numeric tag.
        # Real data rarely trips this (overlay names are distinct
        # directory names) but if two normalize to the same slug the
        # later one gets a suffix so config paths stay unique.
        base = slug
        n = 2
        while slug in seen_slugs:
            slug = f'{base}_{n}'
            n += 1
        seen_slugs.add(slug)

        manifest = _make_manifest(name, slug)
        _disable_if_unseen(manifest.key)
        add(manifest, _make_draw(path, slug))
