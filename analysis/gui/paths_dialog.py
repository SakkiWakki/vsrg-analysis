"""Install-path setup dialog.

Shown automatically on first launch (no prior saved paths) and re-openable
later from the Library tab. Lets the user point the app at their Etterna
Save folder and osu! Songs folder. Both are optional — leaving one blank
falls back to autodetection the next time path-lookup runs.
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QFileDialog,
                               QDialogButtonBox, QMessageBox)

from analysis.gui.settings import (get_etterna_save_override,
                                   set_etterna_save_override,
                                   get_osu_songs_override,
                                   set_osu_songs_override,
                                   validate_etterna_save, validate_osu_songs)


class PathsDialog(QDialog):
    """Modal dialog for configuring install paths. Persists via QSettings
    on accept. On first-run we also autofill from detection so the user
    can just hit OK if the defaults look right."""
    ETTERNA_HINT = ('Point at your Etterna `Save/` folder — the one that '
                    'contains `LocalProfiles/` and/or `Etterna.xml`.')
    OSU_HINT = ('Point at your osu! `Songs/` folder. Replays are picked up '
                'from the sibling `Data/r/` folder automatically.')

    def __init__(self, parent=None, *, first_run=False,
                 autodetect_etterna=None, autodetect_osu=None):
        super().__init__(parent)
        self.setWindowTitle('Set install paths' if not first_run
                            else 'Welcome — set your install paths')
        self.setMinimumWidth(560)

        v = QVBoxLayout(self)

        if first_run:
            intro = QLabel(
                "Looks like this is your first run. Point the app at your "
                "Etterna and/or osu! install folders — you can change these "
                "later from the Library tab.")
            intro.setWordWrap(True)
            v.addWidget(intro)

        v.addWidget(self._section_header('Etterna save folder'))
        v.addWidget(self._hint(self.ETTERNA_HINT))
        self.ett_edit = QLineEdit()
        self.ett_edit.setPlaceholderText('e.g. ~/.etterna/Save')
        initial_ett = get_etterna_save_override() or (autodetect_etterna or '')
        if initial_ett:
            self.ett_edit.setText(str(initial_ett))
        row = QHBoxLayout()
        row.addWidget(self.ett_edit, 1)
        b = QPushButton('Browse…'); b.clicked.connect(self._browse_ett)
        row.addWidget(b)
        v.addLayout(row)
        self.ett_status = QLabel('')
        v.addWidget(self.ett_status)

        v.addSpacing(8)
        v.addWidget(self._section_header('osu! Songs folder'))
        v.addWidget(self._hint(self.OSU_HINT))
        self.osu_edit = QLineEdit()
        self.osu_edit.setPlaceholderText('e.g. ~/.local/share/osu-wine/osu!/Songs')
        initial_osu = get_osu_songs_override() or (autodetect_osu or '')
        if initial_osu:
            self.osu_edit.setText(str(initial_osu))
        row = QHBoxLayout()
        row.addWidget(self.osu_edit, 1)
        b = QPushButton('Browse…'); b.clicked.connect(self._browse_osu)
        row.addWidget(b)
        v.addLayout(row)
        self.osu_status = QLabel('')
        v.addWidget(self.osu_status)

        self.ett_edit.textChanged.connect(self._refresh_status)
        self.osu_edit.textChanged.connect(self._refresh_status)
        self._refresh_status()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

    @staticmethod
    def _section_header(text):
        lbl = QLabel(f'<b>{text}</b>')
        lbl.setTextFormat(Qt.RichText)
        return lbl

    @staticmethod
    def _hint(text):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet('color: #888;')
        return lbl

    def _browse_ett(self):
        start = self.ett_edit.text().strip() or str(Path.home())
        p = QFileDialog.getExistingDirectory(self, 'Select Etterna Save folder',
                                              start)
        if p:
            self.ett_edit.setText(p)

    def _browse_osu(self):
        start = self.osu_edit.text().strip() or str(Path.home())
        p = QFileDialog.getExistingDirectory(self, 'Select osu! Songs folder',
                                              start)
        if p:
            self.osu_edit.setText(p)

    def _refresh_status(self):
        e = self.ett_edit.text().strip()
        if not e:
            self.ett_status.setText('')
        elif validate_etterna_save(e):
            self.ett_status.setText('<span style="color:#6c6;">✓ looks good</span>')
        else:
            self.ett_status.setText(
                '<span style="color:#c66;">folder missing LocalProfiles/ '
                'and Etterna.xml</span>')
        o = self.osu_edit.text().strip()
        if not o:
            self.osu_status.setText('')
        elif validate_osu_songs(o):
            self.osu_status.setText('<span style="color:#6c6;">✓ folder exists</span>')
        else:
            self.osu_status.setText('<span style="color:#c66;">folder does not exist</span>')

    def _accept(self):
        e = self.ett_edit.text().strip() or None
        o = self.osu_edit.text().strip() or None
        bad = []
        if e and not validate_etterna_save(e):
            bad.append(f'Etterna path "{e}" has no LocalProfiles/ or Etterna.xml.')
        if o and not validate_osu_songs(o):
            bad.append(f'osu! path "{o}" does not exist.')
        if bad:
            ok = QMessageBox.question(
                self, 'Paths look off',
                '\n'.join(bad) + '\n\nSave anyway?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ok != QMessageBox.Yes:
                return
        set_etterna_save_override(e)
        set_osu_songs_override(o)
        self.accept()


def prompt_if_first_run(parent=None):
    """If neither path is saved yet and we haven't marked first-run done,
    show the dialog once. Returns True if the dialog was shown."""
    from analysis.gui.settings import is_first_run_done, mark_first_run_done
    if is_first_run_done():
        return False
    if get_etterna_save_override() or get_osu_songs_override():
        mark_first_run_done()
        return False
    # Preload autodetect results so the user gets pre-filled suggestions.
    from analysis.etterna.replay import find_etterna_dirs
    from analysis.osu.replay import find_osu_dirs
    ett = find_etterna_dirs().get('save_dir')
    osu = find_osu_dirs().get('songs_dir')
    dlg = PathsDialog(parent, first_run=True,
                      autodetect_etterna=ett, autodetect_osu=osu)
    dlg.exec()
    mark_first_run_done()
    return True
