"""Dialog listing installed plugins, grouped by bundle.

Invoked from the library toolbar's "Plugins" button. Shows each bundle
with its trust tag (trusted/sandboxed/refused), and under each bundle
the registered replay plugins and sidebar sections. Each leaf row has
a checkbox; toggling it writes through the shared
:class:`analysis.config.ConfigStore`, which both persists the change
to ``~/.config/vsrg-analysis/config.json`` and fans it out to every
running window's plugin registry via subscription — a disabled plugin
vanishes from a replay in progress on the next frame.

The dialog does not have an Apply step — toggles take effect
immediately and survive app restart.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QTreeWidget, QTreeWidgetItem, QDialogButtonBox,
                               QPushButton)


_TRUST_LABELS = {
    'trusted': 'trusted',
    'sandboxed': 'sandboxed',
    'refused': 'refused',
}

# QTreeWidgetItem user-data roles for dispatching checkbox changes.
_ROLE_KIND = Qt.UserRole + 1    # 'replay' | 'sidebar'
_ROLE_KEY = Qt.UserRole + 2     # plugin / section key


def _bundle_trust(bundle):
    if bundle.load_errors:
        return 'refused'
    return 'trusted' if bundle.trusted else 'sandboxed'


def _bundle_key_of(module_path: str) -> str:
    """Replay plugins and sidebar sections tag ``module`` with the
    dotted bundle loader path (``_ea_bundle.<key>.<role>.<file>``) or a
    ``<key>/<module>`` shorthand — handle both."""
    s = str(module_path)
    if s.startswith('_ea_bundle.'):
        parts = s.split('.')
        return parts[1] if len(parts) >= 2 else ''
    if '/' in s:
        return s.split('/', 1)[0]
    return ''


class PluginsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Plugins')
        self.resize(560, 440)

        v = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel(
            'Installed plugins — toggle checkboxes to enable/disable.'))
        header.addStretch(1)
        reload_btn = QPushButton('Rediscover')
        reload_btn.clicked.connect(self._rediscover)
        header.addWidget(reload_btn)
        v.addLayout(header)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Plugin', 'Kind'])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        h = self.tree.header()
        h.setStretchLastSection(False)
        h.resizeSection(0, 340)
        h.resizeSection(1, 140)
        self.tree.itemChanged.connect(self._on_item_changed)
        v.addWidget(self.tree, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        v.addWidget(btns)

        self._mgr = None
        self._suspend_signals = False
        self._populate()

    def _rediscover(self):
        from analysis.player.plugin_loader import PluginManager
        self._mgr = PluginManager.discover()
        self._populate()

    def _populate(self):
        from analysis.player.plugin_loader import PluginManager
        mgr = self._mgr or PluginManager.discover()
        self._mgr = mgr

        self._suspend_signals = True
        try:
            self.tree.clear()

            replay_by_bundle: dict[str, list] = {}
            for p in mgr.all_plugins():
                replay_by_bundle.setdefault(
                    _bundle_key_of(p.module), []).append(p)

            sidebar_by_bundle: dict[str, list] = {}
            for s in mgr.sidebar.all_sections():
                sidebar_by_bundle.setdefault(
                    _bundle_key_of(s.module), []).append(s)

            for bundle in mgr.bundles:
                trust = _bundle_trust(bundle)
                label = f'{bundle.name}  [{_TRUST_LABELS[trust]}]'
                root = QTreeWidgetItem([label, 'bundle'])
                # Bundles themselves aren't toggle-able — they're just
                # grouping rows. Block the checkbox by not setting the
                # Checkable flag.
                self.tree.addTopLevelItem(root)
                root.setExpanded(True)

                for p in replay_by_bundle.get(bundle.key, []):
                    self._add_leaf(root, p.name, 'replay', p.key, p.enabled)

                for s in sidebar_by_bundle.get(bundle.key, []):
                    self._add_leaf(root, s.name, 'sidebar', s.key, s.enabled)

                if bundle.load_errors:
                    for role, fname, exc in bundle.load_errors:
                        reason = f'refused: {type(exc).__name__}'
                        child = QTreeWidgetItem(
                            root, [f'{role}/{fname}', reason])
                        child.setForeground(0, Qt.gray)
                        child.setForeground(1, Qt.gray)

                if (not replay_by_bundle.get(bundle.key)
                        and not sidebar_by_bundle.get(bundle.key)
                        and not bundle.load_errors):
                    QTreeWidgetItem(root, ['(empty)', ''])

            if not mgr.bundles:
                self.tree.addTopLevelItem(
                    QTreeWidgetItem(['No bundles discovered.', '']))
        finally:
            self._suspend_signals = False

    def _add_leaf(self, parent, name, kind, key, enabled):
        item = QTreeWidgetItem(parent, [name, kind])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Checked if enabled else Qt.Unchecked)
        item.setData(0, _ROLE_KIND, kind)
        item.setData(0, _ROLE_KEY, key)

    def _on_item_changed(self, item, column):
        if self._suspend_signals or column != 0:
            return
        kind = item.data(0, _ROLE_KIND)
        key = item.data(0, _ROLE_KEY)
        if not kind or not key:
            return
        enabled = item.checkState(0) == Qt.Checked
        if kind == 'replay':
            self._mgr.set_enabled(key, enabled)
        elif kind == 'sidebar':
            self._mgr.sidebar.set_enabled(key, enabled)
