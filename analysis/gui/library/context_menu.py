from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMenu


class LibraryContextMenu:
    def __init__(self, tab):
        self.tab = tab

    def open(self, pos):
        item = self.tab.tree.itemAt(pos)
        if item is None:
            return

        entry = item.data(0, Qt.UserRole)
        if not entry:
            return

        menu = QMenu(self.tab.tree)
        a_play = menu.addAction('▶ Watch replay')
        a_viz = menu.addAction('Analyze (open visualization)')
        menu.addSeparator()
        a_html = menu.addAction('HTML report')
        menu.addSeparator()
        a_copy = menu.addAction('Copy replay path')
        a_copy_chart = (
            menu.addAction('Copy chart path') if entry.get('chart_path') else None
        )
        a_open_folder = menu.addAction('Open containing folder')

        chosen = menu.exec(self.tab.tree.viewport().mapToGlobal(pos))

        if chosen is a_play:
            self.tab.openers.open_player_for(entry)
        elif chosen is a_viz:
            self.tab.openers.open_viz(entry)
        elif chosen is a_html:
            self.tab.openers.html_selected()
        elif chosen is a_copy:
            QApplication.clipboard().setText(entry.get('replay_path', ''))
        elif a_copy_chart is not None and chosen is a_copy_chart:
            QApplication.clipboard().setText(entry.get('chart_path', ''))
        elif chosen is a_open_folder:
            self._open_containing_folder(entry)

    def _open_containing_folder(self, entry):
        target = entry.get('chart_path') or entry.get('replay_path', '')
        if not target:
            return

        folder = str(Path(target).parent)
        if sys.platform.startswith('linux'):
            subprocess.Popen(['xdg-open', folder])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', folder])
        elif sys.platform.startswith('win'):
            subprocess.Popen(['explorer', folder])