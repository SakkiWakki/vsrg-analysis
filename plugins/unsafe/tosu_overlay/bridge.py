"""QWebChannel bridge: exposes a QObject on the JS side.

The overlay's FakeWebSocket shim calls ``bridge.receiveFromJs(data)``
(a Qt slot); Python calls ``bridge.push(json_str)`` which emits
``pushToJs`` (a Qt signal) that the shim connects to on init.

Separation of concerns:
  - This module is pure bridge wiring (Qt signals/slots only).
  - Translation (parse filters, build tosu JSON) lives in translation.py.
  - The view (QWebEngineView + page setup) lives in view.py.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from plugins.unsafe.tosu_overlay.translation import parse_filter_message


class OverlayBridge(QObject):
    """Registered as ``bridge`` on the QWebChannel.

    JS calls:   bridge.receiveFromJs(string)
    Python calls: bridge.push(string)  -> emits pushToJs -> JS _tosuPush
    """

    pushToJs = Signal(str)
    filters_changed = Signal(object)  # emits frozenset[str]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filters: frozenset[str] = frozenset()

    @Slot(str)
    def receiveFromJs(self, data: str) -> None:
        filters = parse_filter_message(data)
        if filters is not None and filters != self._filters:
            self._filters = filters
            self.filters_changed.emit(filters)

    def push(self, json_str: str) -> None:
        self.pushToJs.emit(json_str)

    @property
    def active_filters(self) -> frozenset[str]:
        return self._filters
