"""Install-path setup dialog.

Shown automatically on first launch (no prior saved paths) and re-openable
later from the Library tab. Iterates over every registered `GameManifest`
and renders one row per declared `PathField` ; no game-literal names
live in this module, and adding a new game is purely additive."""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QFileDialog,
                               QDialogButtonBox, QMessageBox, QComboBox)

from analysis.core import manifest as manifest_mod
from analysis.core import path_overrides
from analysis.core.manifest import resolve_placeholder


class _FieldRow:
    """Renders one `PathField` for one game. Holds a text-edit (always)
    and a combo (only when the field declares `list_choices`)."""

    def __init__(self, parent, field, sibling_root_text=None):
        self.parent = parent
        self.field = field
        # When this field needs a sibling root path to populate its
        # choices (osu profile depends on the chosen root), this is a
        # zero-arg callable returning that text live.
        self._sibling_root_text = sibling_root_text

        self.edit = QLineEdit()
        self.edit.setPlaceholderText(resolve_placeholder(field.placeholder))
        initial = path_overrides.get(field.settings_key)
        if not initial and field.autodetect is not None:
            initial = field.autodetect()
        if initial:
            self.edit.setText(str(initial))
        self.status = QLabel('')
        self.combo_label = QLabel(f'{field.label}:')
        self.combo = QComboBox()
        self.combo_label.hide()
        self.combo.hide()

    def build(self, layout):
        layout.addWidget(_section_header(self.field.label))
        if self.field.hint:
            layout.addWidget(_hint(self.field.hint))
        if self.field.list_choices is None:
            row = QHBoxLayout()
            row.addWidget(self.edit, 1)
            browse = QPushButton('Browse…')
            browse.clicked.connect(self._browse)
            row.addWidget(browse)
            layout.addLayout(row)
            layout.addWidget(self.status)
        else:
            crow = QHBoxLayout()
            crow.addWidget(self.combo_label)
            crow.addWidget(self.combo, 1)
            layout.addLayout(crow)

    def _browse(self):
        start = self.edit.text().strip() or str(Path.home())
        p = QFileDialog.getExistingDirectory(
            self.parent, f'Select {self.field.label.lower()}', start)
        if p:
            self.edit.setText(p)

    def refresh_status(self):
        if self.field.validate is None:
            self.status.setText('')
            return
        txt = self.edit.text().strip()
        if not txt:
            self.status.setText('')
        elif self.field.validate(txt):
            self.status.setText('<span style="color:#6c6;">✓ looks good</span>')
        else:
            self.status.setText(
                f'<span style="color:#c66;">{self.field.error_hint}</span>')

    def refresh_choices(self):
        """Populate the combo (when this is a `list_choices` field).
        Resolved against the sibling root's current text so the picker
        updates as the user edits the root path live."""
        if self.field.list_choices is None:
            return
        root = (self._sibling_root_text() if self._sibling_root_text
                else '').strip()
        choices = self.field.list_choices(root) if root else []
        if len(choices) > 1:
            current = path_overrides.get(self.field.settings_key)
            self.combo.blockSignals(True)
            self.combo.clear()
            self.combo.addItems(choices)
            if current and current in choices:
                self.combo.setCurrentText(current)
            self.combo.blockSignals(False)
            self.combo_label.show()
            self.combo.show()
        else:
            self.combo_label.hide()
            self.combo.hide()

    def validate_on_accept(self):
        if self.field.validate is None:
            return None
        txt = self.edit.text().strip()
        if txt and not self.field.validate(txt):
            return f'{self.field.label}: {self.field.error_hint}'
        return None

    def commit(self):
        if self.field.list_choices is None:
            value = self.edit.text().strip() or None
        elif self.combo.isVisible():
            value = self.combo.currentText() or None
        else:
            return
        path_overrides.set(self.field.settings_key, value)


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
    """Modal dialog for configuring install paths. Persists via the
    path_overrides shopkeeper on accept ; on first-run we autofill from
    each field's autodetect so the user can just hit OK if defaults look
    right."""

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

        self.rows: list[_FieldRow] = []
        for _name, manifest in manifest_mod.all_manifests().items():
            # First field is the "primary" path; later fields (e.g. osu
            # profile) can read its live text via the sibling closure.
            primary_row = None
            for field in manifest.path_fields:
                sibling = (lambda r=primary_row: r.edit.text()
                           if r is not None else '')
                row = _FieldRow(self, field, sibling_root_text=sibling
                                 if primary_row is not None else None)
                row.build(v)
                if primary_row is None:
                    primary_row = row
                row.edit.textChanged.connect(row.refresh_status)
                # When the primary edit changes, refresh every dependent
                # row's choices so the profile combo follows the root.
                if primary_row is row:
                    row.edit.textChanged.connect(self._refresh_dependents)
                row.refresh_status()
                row.refresh_choices()
                self.rows.append(row)
            v.addSpacing(8)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

    def _refresh_dependents(self, *_):
        for row in self.rows:
            if row.field.list_choices is not None:
                row.refresh_choices()

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
    has_any = any(
        path_overrides.get(field.settings_key)
        for manifest in manifest_mod.all_manifests().values()
        for field in manifest.path_fields)
    if has_any:
        mark_first_run_done()
        return False
    dlg = PathsDialog(parent, first_run=True)
    dlg.exec()
    mark_first_run_done()
    return True
