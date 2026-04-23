"""QWebEngineView that hosts a community tosu overlay HTML file.

Lifecycle:
    view = TosuOverlayView(player)
    view.load_overlay(Path('/path/to/overlay/index.html'))
    # embed view in a Qt layout; call push_state() each tick

The shim + qwebchannel.js are injected as QWebEngineScripts at
DocumentCreation injection point -- this fires before any page JS,
so ReconnectingWebSocket and socket.js see window.WebSocket already
replaced.
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineScript
from PySide6.QtWebEngineWidgets import QWebEngineView

from plugins.unsafe.tosu_overlay.bridge import OverlayBridge
from plugins.unsafe.tosu_overlay.translation import (
    build_precise_state,
    build_tosu_state,
    prune_to_filters,
)


_SHIM_PATH = Path(__file__).parent / 'shim.js'

_QWEBCHANNEL_CANDIDATES = [
    Path(__file__).parent.parent.parent.parent / 'third_party' / 'qwebchannel.js',
    Path('/usr/share/qt6/webchannel/qwebchannel.js'),
    Path('/usr/lib/qt6/qml/QtWebChannel/qwebchannel.js'),
    Path('/usr/share/qt/qwebchannel.js'),
]


def _read_shim() -> str:
    return _SHIM_PATH.read_text(encoding='utf-8')


def _read_qwebchannel() -> str | None:
    for p in _QWEBCHANNEL_CANDIDATES:
        if p.exists():
            return p.read_text(encoding='utf-8')
    return None


def _make_script(name: str, src: str,
                 injection_point=QWebEngineScript.InjectionPoint.DocumentCreation,
                 world=QWebEngineScript.ScriptWorldId.MainWorld) -> QWebEngineScript:
    s = QWebEngineScript()
    s.setName(name)
    s.setSourceCode(src)
    s.setInjectionPoint(injection_point)
    s.setWorldId(world)
    s.setRunsOnSubFrames(False)
    return s


class TosuOverlayView(QWebEngineView):
    """QWebEngineView wired to a FakeWebSocket shim + QWebChannel bridge.

    After construction call ``load_overlay(path)`` to load an overlay.
    Call ``push_state()`` on each player tick to forward state.
    """

    def __init__(self, player_or_state, parent=None):
        super().__init__(parent)
        # Accept either a Player (convenience path: we wrap in PlayerDataSource)
        # or a GameState directly. The translation layer only touches
        # GameState methods, so the plugin never sees Player internals.
        self._game_state = self._coerce_state(player_or_state)

        self._bridge = OverlayBridge(self)
        self._channel = QWebChannel(self)
        self._channel.registerObject('bridge', self._bridge)
        self.page().setWebChannel(self._channel)

        self._bridge.pushToJs.connect(self._deliver_to_js)
        self._overlay_loaded = False
        self.page().loadFinished.connect(self._on_load_finished)

        self._install_scripts()

    @staticmethod
    def _coerce_state(obj):
        # Duck-type check: a GameState has callable ``game()``. If it does,
        # pass through. Otherwise wrap in PlayerDataSource.
        if callable(getattr(obj, 'game', None)) and callable(
                getattr(obj, 'chart_metadata', None)):
            return obj
        from analysis.components.gui_backend import PlayerDataSource
        return PlayerDataSource(obj)

    def _install_scripts(self) -> None:
        scripts = self.page().scripts()

        qwc = _read_qwebchannel()
        if qwc:
            scripts.insert(_make_script('tosu:qwebchannel', qwc))

        shim = _read_shim()
        # Shim must run after qwebchannel.js (it references QWebChannel).
        # Both are at DocumentCreation; within that point they execute in
        # insertion order, so insert shim second.
        scripts.insert(_make_script('tosu:shim', shim))

    def load_overlay(self, html_path: Path) -> None:
        self._overlay_loaded = False
        url = QUrl.fromLocalFile(str(html_path.resolve()))
        self.load(url)

    def push_state(self) -> None:
        if not self._overlay_loaded:
            return
        try:
            state = build_tosu_state(self._game_state)
            pruned = prune_to_filters(state, self._bridge.active_filters)
            self._bridge.push(json.dumps(pruned))
            self._push_precise()
        except Exception as exc:
            print(f'tosu overlay: push_state failed: {exc}')

    def _push_precise(self) -> None:
        try:
            precise = build_precise_state(self._game_state)
            safe = json.dumps(precise).replace('\\', '\\\\').replace('`', '\\`')
            self.page().runJavaScript(
                f'window._tosuPushPrecise && window._tosuPushPrecise(`{safe}`);')
        except Exception as exc:
            print(f'tosu overlay: precise push failed: {exc}')

    def _on_load_finished(self, ok: bool) -> None:
        # Scripts injected via QWebEngineScript run automatically; we just
        # need to mark the overlay as ready to receive pushes.
        if ok:
            self._overlay_loaded = True

    def _deliver_to_js(self, json_str: str) -> None:
        safe = json_str.replace('\\', '\\\\').replace('`', '\\`')
        self.page().runJavaScript(
            f'window._tosuPush && window._tosuPush(`{safe}`);')
