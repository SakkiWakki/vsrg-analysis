"""Install-path setup dialog.

Shown automatically on first launch (no prior saved paths) and re-openable
later from the Library tab. Iterates over every registered GUI adapter and
builds one row per game ; no game-literal names live in this module."""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QFileDialog,
                               QDialogButtonBox, QMessageBox, QComboBox)

from analysis.core import gui_adapter as gui_mod


class _GameRow:
    """Bundles the widgets + adapter for one game's path-config row."""
    def __init__(self, parent, adapter, autodetect=None):
        self.adapter = adapter
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(adapter.placeholder)
        initial = adapter.get_root_override() or (autodetect or '')
        if initial:
            self.edit.setText(str(initial))
        self.status = QLabel('')
        self.profile_label = QLabel('Profile:')
        self.profile_combo = QComboBox()
        self.profile_label.hide()
        self.profile_combo.hide()
        self._parent = parent

    def build(self, layout):
        layout.addWidget(_section_header(self.adapter.label))
        layout.addWidget(_hint(self.adapter.hint))
        row = QHBoxLayout()
        row.addWidget(self.edit, 1)
        browse = QPushButton('Browse…')
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        layout.addLayout(row)
        layout.addWidget(self.status)
        prow = QHBoxLayout()
        prow.addWidget(self.profile_label)
        prow.addWidget(self.profile_combo, 1)
        layout.addLayout(prow)

    def _browse(self):
        start = self.edit.text().strip() or str(Path.home())
        p = QFileDialog.getExistingDirectory(
            self._parent, f'Select {self.adapter.label.lower()}', start)
        if p:
            self.edit.setText(p)

    def refresh_status(self):
        txt = self.edit.text().strip()
        if not txt:
            self.status.setText('')
        elif self.adapter.validate_root(txt):
            self.status.setText('<span style="color:#6c6;">✓ looks good</span>')
        else:
            self.status.setText(
                f'<span style="color:#c66;">{self.adapter.error_hint}</span>')

    def refresh_profiles(self):
        txt = self.edit.text().strip()
        profiles = self.adapter.list_profiles(txt) if txt else []
        if len(profiles) > 1:
            current = self.adapter.get_profile_override()
            self.profile_combo.blockSignals(True)
            self.profile_combo.clear()
            self.profile_combo.addItems(profiles)
            if current and current in profiles:
                self.profile_combo.setCurrentText(current)
            self.profile_combo.blockSignals(False)
            self.profile_label.show()
            self.profile_combo.show()
        else:
            self.profile_label.hide()
            self.profile_combo.hide()

    def validate_on_accept(self):
        txt = self.edit.text().strip()
        if txt and not self.adapter.validate_root(txt):
            return f'{self.adapter.label}: {self.adapter.error_hint}'
        return None

    def commit(self):
        txt = self.edit.text().strip() or None
        self.adapter.set_root_override(txt)
        if self.profile_combo.isVisible():
            self.adapter.set_profile_override(
                self.profile_combo.currentText() or None)


def _section_header(text):
    lbl = QLabel(f'<b>{text}</b>')
    lbl.setTextFormat(Qt.RichText)
    return lbl


def _hint(text):
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet('color: #888;')
    return lbl


class PathsDialog(QDialog):
    """Modal dialog for configuring install paths. Persists via QSettings
    on accept. On first-run we also autofill from detection so the user
    can just hit OK if the defaults look right."""

    def __init__(self, parent=None, *, first_run=False):
        super().__init__(parent)
        self.setWindowTitle('Set install paths' if not first_run
                            else 'Welcome ; set your install paths')
        self.setMinimumWidth(560)

        v = QVBoxLayout(self)
        if first_run:
            intro = QLabel(
                "Looks like this is your first run. Point the app at your "
                "game install folders ; you can change these later from the "
                "Library tab.")
            intro.setWordWrap(True)
            v.addWidget(intro)

        self.rows = []
        for _name, adapter in gui_mod.all_games().items():
            row = _GameRow(self, adapter,
                           autodetect=adapter.default_install_hint())
            row.build(v)
            row.edit.textChanged.connect(row.refresh_status)
            row.edit.textChanged.connect(row.refresh_profiles)
            row.refresh_status()
            row.refresh_profiles()
            self.rows.append(row)
            v.addSpacing(8)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

    def _accept(self):
        bad = [msg for msg in (r.validate_on_accept() for r in self.rows)
               if msg]
        if bad:
            ok = QMessageBox.question(
                self, 'Paths look off',
                '\n'.join(bad) + '\n\nSave anyway?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ok != QMessageBox.Yes:
                return
        for row in self.rows:
            row.commit()
        self.accept()


def prompt_if_first_run(parent=None):
    """If no paths are saved yet and we haven't marked first-run done, show
    the dialog once. Returns True if the dialog was shown."""
    from analysis.gui.settings import is_first_run_done, mark_first_run_done
    if is_first_run_done():
        return False
    if any(a.get_root_override() for a in gui_mod.all_games().values()):
        mark_first_run_done()
        return False
    dlg = PathsDialog(parent, first_run=True)
    dlg.exec()
    mark_first_run_done()
    return True
