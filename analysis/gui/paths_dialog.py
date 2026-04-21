"""Install-path setup dialog.

Shown automatically on first launch (no prior saved paths) and re-openable
later from the Library tab. Lets the user point the app at their Etterna
install folder and osu! install folder. Both are optional — leaving one
blank falls back to autodetection the next time path-lookup runs.

The stored overrides are **install roots** (e.g. `~/etterna/` or
`~/.local/share/osu-wine/osu!/`). Resolution into Save/Songs/Data uses
`Preferences.ini` (Etterna's AdditionalSongFolders) and
`osu!.<user>.cfg` (BeatmapDirectory) so user-configured subpaths are
respected without needing separate fields."""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QFileDialog,
                               QDialogButtonBox, QMessageBox, QComboBox)

from analysis.gui.settings import (get_etterna_root_override,
                                   set_etterna_root_override,
                                   get_osu_root_override,
                                   set_osu_root_override,
                                   get_osu_profile_override,
                                   set_osu_profile_override,
                                   validate_etterna_root, validate_osu_root)


class PathsDialog(QDialog):
    """Modal dialog for configuring install paths. Persists via QSettings
    on accept. On first-run we also autofill from detection so the user
    can just hit OK if the defaults look right."""
    ETTERNA_HINT = ('Point at your Etterna install folder — the one that '
                    'contains `Save/` and `Songs/`. Additional song folders '
                    'listed in `Preferences.ini` are picked up automatically.')
    OSU_HINT = ('Point at your osu! install folder — the one that contains '
                '`osu!.<username>.cfg`. The Songs folder is resolved from '
                'the config\'s BeatmapDirectory setting.')

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

        v.addWidget(self._section_header('Etterna install folder'))
        v.addWidget(self._hint(self.ETTERNA_HINT))
        self.ett_edit = QLineEdit()
        self.ett_edit.setPlaceholderText('e.g. ~/.etterna or ~/etterna')
        initial_ett = get_etterna_root_override() or (autodetect_etterna or '')
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
        v.addWidget(self._section_header('osu! install folder'))
        v.addWidget(self._hint(self.OSU_HINT))
        self.osu_edit = QLineEdit()
        self.osu_edit.setPlaceholderText('e.g. ~/.local/share/osu-wine/osu!')
        initial_osu = get_osu_root_override() or (autodetect_osu or '')
        if initial_osu:
            self.osu_edit.setText(str(initial_osu))
        row = QHBoxLayout()
        row.addWidget(self.osu_edit, 1)
        b = QPushButton('Browse…'); b.clicked.connect(self._browse_osu)
        row.addWidget(b)
        v.addLayout(row)
        self.osu_status = QLabel('')
        v.addWidget(self.osu_status)

        # Profile combo: only visible when >1 osu!.<user>.cfg is found.
        self.osu_profile_label = QLabel('osu! profile:')
        self.osu_profile_combo = QComboBox()
        self.osu_profile_row = QHBoxLayout()
        self.osu_profile_row.addWidget(self.osu_profile_label)
        self.osu_profile_row.addWidget(self.osu_profile_combo, 1)
        self._profile_row_host = QLabel()  # wrapper, hidden when not needed
        v.addLayout(self.osu_profile_row)
        self.osu_profile_label.hide()
        self.osu_profile_combo.hide()

        self.ett_edit.textChanged.connect(self._refresh_status)
        self.osu_edit.textChanged.connect(self._refresh_status)
        self.osu_edit.textChanged.connect(self._refresh_profiles)
        self._refresh_status()
        self._refresh_profiles()

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
        p = QFileDialog.getExistingDirectory(self, 'Select Etterna install folder',
                                              start)
        if p:
            self.ett_edit.setText(p)

    def _browse_osu(self):
        start = self.osu_edit.text().strip() or str(Path.home())
        p = QFileDialog.getExistingDirectory(self, 'Select osu! install folder',
                                              start)
        if p:
            self.osu_edit.setText(p)

    def _refresh_status(self):
        e = self.ett_edit.text().strip()
        if not e:
            self.ett_status.setText('')
        elif validate_etterna_root(e):
            self.ett_status.setText('<span style="color:#6c6;">✓ looks good</span>')
        else:
            self.ett_status.setText(
                '<span style="color:#c66;">folder missing Save/ '
                '(or LocalProfiles/Etterna.xml)</span>')
        o = self.osu_edit.text().strip()
        if not o:
            self.osu_status.setText('')
        elif validate_osu_root(o):
            self.osu_status.setText(
                '<span style="color:#6c6;">✓ osu! config found</span>')
        else:
            self.osu_status.setText(
                '<span style="color:#c66;">no osu!.&lt;user&gt;.cfg in folder</span>')

    def _refresh_profiles(self):
        from analysis.games.osu.replay import list_osu_profiles
        o = self.osu_edit.text().strip()
        profiles = list_osu_profiles(o) if o else []
        # Only ask the user to pick when there's ambiguity.
        if len(profiles) > 1:
            current = get_osu_profile_override()
            self.osu_profile_combo.blockSignals(True)
            self.osu_profile_combo.clear()
            self.osu_profile_combo.addItems(profiles)
            if current and current in profiles:
                self.osu_profile_combo.setCurrentText(current)
            self.osu_profile_combo.blockSignals(False)
            self.osu_profile_label.show()
            self.osu_profile_combo.show()
        else:
            self.osu_profile_label.hide()
            self.osu_profile_combo.hide()

    def _accept(self):
        e = self.ett_edit.text().strip() or None
        o = self.osu_edit.text().strip() or None
        bad = []
        if e and not validate_etterna_root(e):
            bad.append(f'Etterna path "{e}" has no Save/ folder with '
                       f'LocalProfiles/ or Etterna.xml.')
        if o and not validate_osu_root(o):
            bad.append(f'osu! path "{o}" has no osu!.<user>.cfg file.')
        if bad:
            ok = QMessageBox.question(
                self, 'Paths look off',
                '\n'.join(bad) + '\n\nSave anyway?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ok != QMessageBox.Yes:
                return
        set_etterna_root_override(e)
        set_osu_root_override(o)
        # Persist the profile choice iff we actually showed the picker.
        if self.osu_profile_combo.isVisible():
            set_osu_profile_override(self.osu_profile_combo.currentText() or None)
        self.accept()


def prompt_if_first_run(parent=None):
    """If neither path is saved yet and we haven't marked first-run done,
    show the dialog once. Returns True if the dialog was shown."""
    from analysis.gui.settings import is_first_run_done, mark_first_run_done
    if is_first_run_done():
        return False
    if get_etterna_root_override() or get_osu_root_override():
        mark_first_run_done()
        return False
    # Preload autodetect results so the user gets pre-filled suggestions.
    from analysis.games.etterna.replay import find_etterna_dirs
    from analysis.games.osu.replay import find_osu_dirs
    ett_save = find_etterna_dirs().get('save_dir')
    # Autodetect returns the Save dir; offer its parent (install root) in the
    # dialog so the default matches the new field semantics.
    ett_root = str(Path(ett_save).parent) if ett_save else None
    osu_root = find_osu_dirs().get('root')
    dlg = PathsDialog(parent, first_run=True,
                      autodetect_etterna=ett_root, autodetect_osu=osu_root)
    dlg.exec()
    mark_first_run_done()
    return True
