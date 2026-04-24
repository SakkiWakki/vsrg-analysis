"""Dialog listing installed plugins, grouped by bundle.

Invoked from the library toolbar's "Plugins" button. Shows each bundle
with its trust tag (trusted/sandboxed/refused), and under each bundle
the registered replay plugins, sidebar sections, and overlay feeds.

Each leaf row has a checkbox; toggling it writes through the shared
ConfigStore-backed registries immediately. There is no Apply step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QDialogButtonBox,
    QPushButton,
)

from analysis.player.plugin.plugin_loader import PluginManager
from analysis.overlay.publisher import discover_overlays


PluginKind = Literal['replay', 'sidebar', 'overlay']

_TRUST_LABELS = {
    'trusted': 'trusted',
    'sandboxed': 'sandboxed',
    'refused': 'refused',
}

# QTreeWidgetItem user-data roles for dispatching checkbox changes.
_ROLE_KIND = Qt.UserRole + 1    # 'replay' | 'sidebar' | 'overlay'
_ROLE_KEY = Qt.UserRole + 2     # plugin / section key


@dataclass(frozen=True)
class PluginLeaf:
    name: str
    kind: PluginKind
    key: str
    enabled: bool


@dataclass(frozen=True)
class LoadErrorRow:
    label: str
    reason: str


@dataclass(frozen=True)
class BundleRow:
    key: str
    label: str
    leaves: list[PluginLeaf]
    errors: list[LoadErrorRow]


def _bundle_trust(bundle) -> str:
    if bundle.load_errors:
        return 'refused'
    return 'trusted' if bundle.trusted else 'sandboxed'


def _bundle_key_of(module_path: str) -> str:
    """Extract bundle key from a plugin module path.

    Handles both:
      - _ea_bundle.<key>.<role>.<file>
      - <key>/<module>
    """
    s = str(module_path)
    if s.startswith('_ea_bundle.'):
        parts = s.split('.')
        return parts[1] if len(parts) >= 2 else ''
    if '/' in s:
        return s.split('/', 1)[0]
    return ''


def _group_by_bundle(items) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for item in items:
        grouped.setdefault(_bundle_key_of(item.module), []).append(item)
    return grouped


def _leaf_rows(kind: PluginKind, items) -> list[PluginLeaf]:
    return [
        PluginLeaf(
            name=item.name,
            kind=kind,
            key=item.key,
            enabled=item.enabled,
        )
        for item in items
    ]


def _load_error_rows(bundle) -> list[LoadErrorRow]:
    return [
        LoadErrorRow(
            label=f'{role}/{fname}',
            reason=f'refused: {type(exc).__name__}',
        )
        for role, fname, exc in bundle.load_errors
    ]


def build_bundle_rows(mgr, overlay_registry) -> list[BundleRow]:
    """Build a Qt-free view model for the plugin tree."""
    groups: dict[PluginKind, dict[str, list]] = {
        'replay': _group_by_bundle(mgr.all_plugins()),
        'sidebar': _group_by_bundle(mgr.sidebar.all_sections()),
        'overlay': _group_by_bundle(overlay_registry.all_overlays()),
    }

    rows: list[BundleRow] = []
    for bundle in mgr.bundles:
        trust = _bundle_trust(bundle)
        leaves = [
            leaf
            for kind, by_bundle in groups.items()
            for leaf in _leaf_rows(kind, by_bundle.get(bundle.key, ()))
        ]

        rows.append(
            BundleRow(
                key=bundle.key,
                label=f'{bundle.name}  [{_TRUST_LABELS[trust]}]',
                leaves=leaves,
                errors=_load_error_rows(bundle),
            )
        )

    return rows


class PluginsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Plugins')
        self.resize(560, 440)

        self._mgr = None
        self._overlay_registry = None
        self._suspend_signals = False

        self._build_ui()
        self._populate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel(
            'Installed plugins — toggle checkboxes to enable/disable.'
        ))
        header.addStretch(1)

        reload_btn = QPushButton('Rediscover')
        reload_btn.clicked.connect(self._rediscover)
        header.addWidget(reload_btn)

        layout.addLayout(header)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Plugin', 'Kind'])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)

        header_view = self.tree.header()
        header_view.setStretchLastSection(False)
        header_view.resizeSection(0, 340)
        header_view.resizeSection(1, 140)

        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _ensure_registries(self, *, rediscover: bool = False) -> None:
        if rediscover or self._mgr is None:
            self._mgr = PluginManager.discover()

        if rediscover or self._overlay_registry is None:
            self._overlay_registry = discover_overlays(
                config=getattr(self._mgr, '_config', None)
            )

    def _rediscover(self) -> None:
        self._ensure_registries(rediscover=True)
        self._populate()

    def _populate(self) -> None:
        self._ensure_registries()
        rows = build_bundle_rows(self._mgr, self._overlay_registry)

        self._suspend_signals = True
        try:
            self._render_rows(rows)
        finally:
            self._suspend_signals = False

    def _render_rows(self, rows: list[BundleRow]) -> None:
        self.tree.clear()

        if not rows:
            self.tree.addTopLevelItem(
                QTreeWidgetItem(['No bundles discovered.', ''])
            )
            return

        for row in rows:
            root = self._add_bundle_root(row)

            for leaf in row.leaves:
                self._add_leaf(root, leaf)

            for error in row.errors:
                self._add_error(root, error)

            if not row.leaves and not row.errors:
                QTreeWidgetItem(root, ['(empty)', ''])

    def _add_bundle_root(self, row: BundleRow) -> QTreeWidgetItem:
        root = QTreeWidgetItem([row.label, 'bundle'])
        self.tree.addTopLevelItem(root)
        root.setExpanded(True)
        return root

    def _add_leaf(self, parent: QTreeWidgetItem, leaf: PluginLeaf) -> None:
        item = QTreeWidgetItem(parent, [leaf.name, leaf.kind])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Checked if leaf.enabled else Qt.Unchecked)
        item.setData(0, _ROLE_KIND, leaf.kind)
        item.setData(0, _ROLE_KEY, leaf.key)

    def _add_error(self, parent: QTreeWidgetItem, error: LoadErrorRow) -> None:
        item = QTreeWidgetItem(parent, [error.label, error.reason])
        item.setForeground(0, Qt.gray)
        item.setForeground(1, Qt.gray)

    def _toggle_handlers(self):
        return {
            'replay': self._mgr.set_enabled,
            'sidebar': self._mgr.sidebar.set_enabled,
            'overlay': self._overlay_registry.set_enabled,
        }

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._suspend_signals or column != 0:
            return

        kind = item.data(0, _ROLE_KIND)
        key = item.data(0, _ROLE_KEY)
        if not kind or not key:
            return

        handler = self._toggle_handlers().get(kind)
        if handler is None:
            return

        enabled = item.checkState(0) == Qt.Checked
        handler(key, enabled)