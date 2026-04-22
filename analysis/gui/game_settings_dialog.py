"""Dialog for per-game library settings.

One row per registered game adapter. The checkbox controls
``library.enabled_games`` in the shared :class:`ConfigStore`; disabling
a game skips its adapter during ``build_library`` so its entries drop
out of the tree on the next scan. A per-row "Rebuild" button triggers
that game's ``adapter.rebuild()`` in a background worker and asks the
library tab to reload from all adapters' caches when it finishes.

Like the plugins dialog, toggles persist immediately — there's no
Apply step.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QTreeWidget, QTreeWidgetItem, QDialogButtonBox,
                               QPushButton)

from analysis.core import game as game_mod
from analysis.core.search import enabled_games, set_game_enabled


_ROLE_NAME = Qt.UserRole + 1


class GameSettingsDialog(QDialog):
    def __init__(self, parent=None, on_rebuild=None):
        """``on_rebuild(name)`` is invoked on the GUI thread when the user
        clicks Rebuild for a game. The library tab wires this to its
        existing ``_rebuild_game`` so the status bar + worker lifecycle
        stay in one place."""
        super().__init__(parent)
        self.setWindowTitle('Game Settings')
        self.resize(520, 320)
        self._on_rebuild = on_rebuild

        v = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel(
            'Enable/disable games and rebuild their library caches.'))
        header.addStretch(1)
        v.addLayout(header)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Game', 'Cached entries', ''])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        h = self.tree.header()
        h.setStretchLastSection(False)
        h.resizeSection(0, 220)
        h.resizeSection(1, 140)
        h.resizeSection(2, 120)
        self.tree.itemChanged.connect(self._on_item_changed)
        v.addWidget(self.tree, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        v.addWidget(btns)

        self._suspend_signals = False
        self._populate()

    def _populate(self):
        self._suspend_signals = True
        try:
            self.tree.clear()
            enabled = enabled_games()
            for name, adapter in game_mod.all_games().items():
                try:
                    cached = adapter.load_cached()
                except Exception:
                    cached = None
                count = len(cached) if cached else 0
                label = getattr(adapter, 'label', None) or name

                item = QTreeWidgetItem([label, f'{count} entries', ''])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(
                    0, Qt.Checked if name in enabled else Qt.Unchecked)
                item.setData(0, _ROLE_NAME, name)
                self.tree.addTopLevelItem(item)

                # QTreeWidget doesn't host child widgets by default; use
                # setItemWidget so each row gets its own Rebuild button.
                btn = QPushButton('Rebuild')
                btn.clicked.connect(
                    lambda _checked=False, n=name: self._rebuild(n))
                self.tree.setItemWidget(item, 2, btn)

            if not game_mod.all_games():
                self.tree.addTopLevelItem(
                    QTreeWidgetItem(['No games registered.', '', '']))
        finally:
            self._suspend_signals = False

    def _on_item_changed(self, item, column):
        if self._suspend_signals or column != 0:
            return
        name = item.data(0, _ROLE_NAME)
        if not name:
            return
        set_game_enabled(name, item.checkState(0) == Qt.Checked)

    def _rebuild(self, name):
        if self._on_rebuild is not None:
            self._on_rebuild(name)
