from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QMessageBox

from analysis.gui.library.model import EntryRow, GroupRow


class LibraryTreeController:
    def __init__(self, tree: QTreeWidget, tab):
        self.tree = tree
        self.tab = tab

    def render(self, rows: list[EntryRow | GroupRow]) -> None:
        self.tree.clear()

        for row in rows:
            if isinstance(row, EntryRow):
                self._add_entry(row)
            else:
                self._add_group(row)

    def selected_entry(self, *, parent=None):
        items = self.tree.selectedItems()
        if not items:
            QMessageBox.warning(parent, 'no selection', 'pick a score')
            return None
        return items[0].data(0, Qt.UserRole)

    def _add_entry(self, row: EntryRow) -> QTreeWidgetItem:
        item = QTreeWidgetItem(row.values)
        item.setData(0, Qt.UserRole, row.entry)
        self.tree.addTopLevelItem(item)
        return item

    def _add_group(self, row: GroupRow) -> QTreeWidgetItem:
        parent = QTreeWidgetItem(row.parent.values)
        parent.setData(0, Qt.UserRole, row.parent.entry)
        self.tree.addTopLevelItem(parent)

        for child in row.children:
            item = QTreeWidgetItem(child.values)
            item.setData(0, Qt.UserRole, child.entry)
            parent.addChild(item)

        return parent