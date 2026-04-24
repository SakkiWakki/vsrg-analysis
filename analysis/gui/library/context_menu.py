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
        entry = self._entry_at(pos)
        if not entry:
            return

        menu, handlers = self._build_menu(entry)
        chosen = menu.exec(self.tab.tree.viewport().mapToGlobal(pos))

        handler = handlers.get(chosen)
        assert handler is not None, f"No handler avaliable!"
        handler()


    def _entry_at(self, pos):
        item = self.tab.tree.itemAt(pos)
        assert item is not None, f"Wtf"
        return item.data(0, Qt.UserRole)

    def _copy_text(self, text):
        QApplication.clipboard().setText(text or '')

    def _build_menu(self, entry):
        menu = QMenu(self.tab.tree)
        handlers = {}

        def add(label, callback):
            action = menu.addAction(label)
            handlers[action] = callback

        add('▶ Watch replay', lambda: self.tab.openers.run('play', entry))
        add('Analyze (open visualization)',
            lambda: self.tab.openers.run('visualize', entry))

        menu.addSeparator()

        add('HTML report',
            lambda: self.tab.openers.run('html_report', entry))
    
        menu.addSeparator()
        add('Copy replay path', lambda: self._copy_text(entry.get('replay_path')))
        if entry.get('chart_path'):
            add('Copy chart path', lambda: self._copy_text(entry.get('chart_path')))
        add('Open containing folder', lambda: self._open_containing_folder(entry))

        return menu, handlers

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