from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from analysis.gui.library.actions import get_registry

class PluginActionsController:
    def __init__(self, tab):
        self.tab = tab
        self._unsubscribe = None
        self._discover_and_subscribe()

    def _discover_and_subscribe(self):
        from analysis.player.plugin_loader import PluginManager

        PluginManager.discover()
        self._unsubscribe = get_registry().subscribe(self.rebuild)

    def rebuild(self):
        menu = self.tab._plugin_actions_menu
        button = self.tab._plugin_actions_btn

        menu.clear()

        actions = get_registry().actions()
        for action in actions:
            callback = action.callback
            item = menu.addAction(action.label)
            item.triggered.connect(
                lambda _checked=False, fn=callback: self.invoke(fn)
            )

        button.setVisible(bool(actions))

    def invoke(self, fn):
        try:
            fn()
        except Exception as exc:
            QMessageBox.warning(
                self.tab,
                'Plugin action failed',
                f'A plugin-contributed toolbar action raised an error:\n{exc}',
            )

    def close(self):
        if self._unsubscribe is None:
            return
        get_registry().unsubscribe(self._unsubscribe)
        self._unsubscribe = None