"""tosu overlay integration: host community HTML/JS overlays in a
QWebEngineView with a FakeWebSocket shim so they receive data directly
from the player without a running tosu server.

Public API:
    from plugins.builtin.tosu_overlay.view import TosuOverlayView
    from plugins.builtin.tosu_overlay.discovery import find_overlays
"""
