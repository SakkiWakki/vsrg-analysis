"""Unified-API component that embeds a tosu overlay as a draggable widget.

Backed by the web-texture PAL: on SURFACE_GUI we get a :data:`KIND_QPIXMAP`
frame (via :class:`QPixmapBackend`), blit it with :meth:`Context.image`,
and ship the pruned v1+v2 state through the shim each frame.

Registration is deferred: the module exposes :func:`register_components`
so the component registry discovery path picks us up. Each active mount
of this component owns its own :class:`WebTexture` via a side-channel
keyed by the component instance; destroying the sidebar section tears
the texture down too.

The overlay to load is resolved at draw time from a small config
dict (persisted via the component config store): which overlay name,
its index.html path, and its desired aspect. A future UI panel picks
the overlay; today a default (``--tosu-overlay`` cli flag or env var)
is sufficient.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from analysis.components.api import (
    LayerDeclaration,
    LayerPlacement,
    LAYER_LEAF,
    Manifest,
    REGION_FREE,
    SURFACE_GUI,
)
from analysis.components.pal.web import (
    SURFACE_LOCAL_CPU,
    WebTexturePAL,
)
from plugins.unsafe.tosu_overlay.bridge import OverlayBridge
from plugins.unsafe.tosu_overlay.discovery import find_overlays
from plugins.unsafe.tosu_overlay.translation import (
    build_precise_state,
    build_tosu_state,
    prune_to_filters,
)


MANIFEST = Manifest(
    key='tosu_overlay:web',
    name='tosu overlay',
    supported_surfaces=frozenset({SURFACE_GUI}),
    requires_data=frozenset(),     # degrades gracefully across fields
    optional_data=frozenset({
        'chart_metadata', 'chart_stats', 'chart_paths',
        'judgment_counts', 'hit_errors_ms', 'unstable_rate',
        'combo', 'max_combo', 'score', 'current_grade',
        'mods_short', 'mods_raw', 'play_rate_effective',
        'player_name', 'paused', 't_now', 'game',
    }),
    layers=(
        LayerDeclaration(
            key='tosu_overlay:web',
            name='tosu overlay',
            placement=LayerPlacement('inside', 'root'),
            kind=LAYER_LEAF,
            default_visible=True,
        ),
    ),
)


# Per-mount state. Keyed by the id() of the component's key string so
# multiple mounts of the same component (multi-window) don't share a
# WebTexture. Bounded lifecycle: entries are added on first draw and
# removed when the owning sidebar section is torn down (see the
# WebTexture's deleteLater on close).
_mounts: dict[str, '_Mount'] = {}


class _Mount:
    """Live state for one TosuOverlayComponent instance."""

    def __init__(self, width: int, height: int, overlay_path: Path):
        self.width = width
        self.height = height
        self.overlay_path = overlay_path
        self.last_generation = -1
        self.cached_frame = None

        pal = WebTexturePAL.default()
        self.texture = pal.create(
            surface=SURFACE_LOCAL_CPU, width=width, height=height)

        # Wire shim + web channel on the underlying QWebEngineView. The
        # view is created by QPixmapBackend; we install the standard
        # tosu scripts + bridge before the first navigation so any
        # page JS sees window.WebSocket already replaced.
        self._install_shim()

        self.bridge = OverlayBridge(self.texture.view)
        from PySide6.QtWebChannel import QWebChannel
        channel = QWebChannel(self.texture.view)
        channel.registerObject('bridge', self.bridge)
        self.texture.view.page().setWebChannel(channel)
        self.bridge.pushToJs.connect(self._deliver_to_js)
        self.texture.attach_bridge(self.bridge)

        # Kick the load after scripts are in place.
        from PySide6.QtCore import QUrl
        self.texture.view.load(
            QUrl.fromLocalFile(str(overlay_path.resolve())))

    def _install_shim(self) -> None:
        """Install qwebchannel.js + shim.js at DocumentCreation time
        on the view's page-scoped script collection."""
        from PySide6.QtWebEngineCore import QWebEngineScript

        page = self.texture.view.page()
        scripts = page.scripts()

        # qwebchannel.js path search
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

        shim_path = Path(__file__).parent / 'shim.js'
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
        """Pull the latest texture frame and cache it. Returns the
        cached frame whether or not it changed; the blit is cheap so we
        don't need to guard on generation at the consumer side."""
        frame = self.texture.latest_frame()
        if frame is None:
            return self.cached_frame
        if frame.generation != self.last_generation:
            self.last_generation = frame.generation
            self.cached_frame = frame
        return self.cached_frame

    def close(self) -> None:
        self.texture.close()


# ── Overlay selection ──────────────────────────────────────────────

def _pick_default_overlay() -> Path | None:
    """Resolve the overlay to mount. Priority:

      1. $TOSU_OVERLAY env var (absolute path or name in discovery dirs).
      2. First overlay found in discovery dirs.
      3. None -- component renders an empty placeholder.
    """
    want = os.environ.get('TOSU_OVERLAY', '').strip()
    if want:
        p = Path(want)
        if p.is_file():
            return p
        # Name match against discovery.
        for name, path in find_overlays():
            if name == want:
                return path
    found = find_overlays()
    if found:
        return found[0][1]
    return None


# ── Draw ───────────────────────────────────────────────────────────

_DEFAULT_W, _DEFAULT_H = 640, 360


def _draw(ctx) -> None:
    key = ctx.config.get('mount_key', None)
    if key is None:
        # Fresh component: mint a unique key so repeat mounts don't
        # share state. Persisted so the same section keeps its
        # WebTexture across repaints.
        key = f'mount-{id(ctx)}'
        ctx.config.set('mount_key', key)

    mount = _mounts.get(key)
    if mount is None:
        overlay = _pick_default_overlay()
        if overlay is None:
            ctx.draw_heading('tosu overlay')
            ctx.draw_hint('No overlays found.')
            ctx.draw_hint('Clone tosuapp/counters into /tmp/tosu-counters')
            ctx.draw_hint('or drop one in plugins/overlays/<name>/index.html.')
            return
        mount = _Mount(_DEFAULT_W, _DEFAULT_H, overlay)
        _mounts[key] = mount

    # Push state every frame; the overlay has its own internal cadence.
    mount.push_state(ctx.data)

    # Reserve the drawing area. The component grows to its full width
    # and a fixed aspect-ratio height (16:9 default for wide overlays).
    width = ctx.w
    height = int(width * _DEFAULT_H / _DEFAULT_W)
    if (width, height) != (mount.width, mount.height):
        mount.width, mount.height = width, height
        mount.texture.resize(width, height)

    frame = mount.sync_frame()
    if frame is not None:
        ctx.image((0, ctx.y, width, height), frame)
    ctx.y += height


def register_components(add) -> None:
    add(MANIFEST, _draw)
