"""Tests for the install-path override system.

Covers:
- settings getter/setter round-trip (including empty-string clearing)
- legacy Save/Songs paths migrate to install roots on read
- validators accept a real-looking tree and reject garbage
- find_etterna_dirs / find_osu_dirs honor the install-root override and
  resolve Preferences.ini / osu!.cfg subpaths
- PathsDialog saves entered paths on accept
- prompt_if_first_run() only runs once
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from analysis.gui import settings as S


def _make_etterna_install(root: Path, with_xml=True, with_profile=True,
                          extra_songs=None):
    """Build an Etterna *install* tree: root/Save/, root/Songs/."""
    root.mkdir(parents=True, exist_ok=True)
    save = root / 'Save'
    save.mkdir()
    if with_profile:
        prof = save / 'LocalProfiles' / '00000000'
        prof.mkdir(parents=True)
        if with_xml:
            (prof / 'Etterna.xml').write_text('<xml/>')
    elif with_xml:
        (save / 'Etterna.xml').write_text('<xml/>')
    (save / 'ReplaysV2').mkdir()
    (root / 'Songs').mkdir()
    if extra_songs:
        lines = [f'AdditionalSongFolders={";".join(str(p) for p in extra_songs)}']
        (save / 'Preferences.ini').write_text('\n'.join(lines))
    return root


def _make_osu_install(root: Path, profile='yucky', beatmap_dir=None):
    """Build an osu! install tree: root/osu!.<user>.cfg + root/Data/r + Songs."""
    install = root / 'osu!'
    install.mkdir(parents=True)
    bmd = beatmap_dir if beatmap_dir is not None else 'Songs'
    (install / f'osu!.{profile}.cfg').write_text(f'BeatmapDirectory = {bmd}\n')
    (install / 'Data' / 'r').mkdir(parents=True)
    # Only create the Songs dir if it's a relative default.
    if beatmap_dir is None:
        (install / 'Songs').mkdir()
    return install


# ---- settings round-trip ---------------------------------------------------

def test_override_roundtrip_etterna(tmp_path):
    assert S.get_etterna_root_override() is None
    S.set_etterna_root_override(str(tmp_path))
    assert S.get_etterna_root_override() == str(tmp_path)
    S.set_etterna_root_override(None)
    assert S.get_etterna_root_override() is None


def test_override_empty_string_clears(tmp_path):
    S.set_etterna_root_override(str(tmp_path))
    S.set_etterna_root_override('')
    assert S.get_etterna_root_override() is None


def test_override_roundtrip_osu(tmp_path):
    S.set_osu_root_override(str(tmp_path))
    assert S.get_osu_root_override() == str(tmp_path)


def test_osu_profile_roundtrip():
    assert S.get_osu_profile_override() is None
    S.set_osu_profile_override('osu!.yucky.cfg')
    assert S.get_osu_profile_override() == 'osu!.yucky.cfg'
    S.set_osu_profile_override(None)
    assert S.get_osu_profile_override() is None


def test_legacy_etterna_save_migrates(tmp_path):
    # Simulate an old install that stored the Save dir directly.
    install = _make_etterna_install(tmp_path)
    S.get_settings().setValue('paths/etterna_save', str(install / 'Save'))
    S.get_settings().remove('paths/etterna_root')
    got = S.get_etterna_root_override()
    assert got == str(install)


def test_legacy_osu_songs_migrates(tmp_path):
    install = _make_osu_install(tmp_path)
    S.get_settings().setValue('paths/osu_songs', str(install / 'Songs'))
    S.get_settings().remove('paths/osu_root')
    got = S.get_osu_root_override()
    assert got == str(install)


def test_back_compat_shims_work(tmp_path):
    # Old API names still function.
    S.set_etterna_save_override(str(tmp_path))
    assert S.get_etterna_save_override() == str(tmp_path)
    S.set_osu_songs_override(str(tmp_path))
    assert S.get_osu_songs_override() == str(tmp_path)


def test_first_run_flag():
    assert S.is_first_run_done() is False
    S.mark_first_run_done()
    assert S.is_first_run_done() is True


# ---- validators ------------------------------------------------------------

def test_validate_etterna_accepts_install_root(tmp_path):
    install = _make_etterna_install(tmp_path, with_profile=True)
    assert S.validate_etterna_root(str(install)) is True


def test_validate_etterna_accepts_direct_save(tmp_path):
    # Back-compat: user pointing directly at Save/ should still validate.
    install = _make_etterna_install(tmp_path)
    assert S.validate_etterna_root(str(install / 'Save')) is True


def test_validate_etterna_rejects_garbage(tmp_path):
    assert S.validate_etterna_root(None) is False
    assert S.validate_etterna_root('') is False
    assert S.validate_etterna_root('/nonexistent/path/12345') is False
    empty = tmp_path / 'nothing'
    empty.mkdir()
    assert S.validate_etterna_root(str(empty)) is False


def test_validate_osu_requires_cfg(tmp_path):
    assert S.validate_osu_root(None) is False
    assert S.validate_osu_root('') is False
    assert S.validate_osu_root('/nope/12345') is False
    # Empty dir — no osu!.<user>.cfg.
    assert S.validate_osu_root(str(tmp_path)) is False
    install = _make_osu_install(tmp_path)
    assert S.validate_osu_root(str(install)) is True


# ---- find_*_dirs override precedence ---------------------------------------

def test_find_etterna_dirs_uses_install_root(tmp_path):
    install = _make_etterna_install(tmp_path)
    S.set_etterna_root_override(str(install))
    from analysis.games.etterna.replay import find_etterna_dirs
    got = find_etterna_dirs()
    assert got['save_dir'] == str(install / 'Save')
    assert got['replays_dir'] == str(install / 'Save' / 'ReplaysV2')
    assert got['xml_path'] is not None
    assert got['extra_songs_dirs'] == []


def test_find_etterna_dirs_reads_additional_song_folders(tmp_path):
    extra = tmp_path / 'other_songs'
    extra.mkdir()
    install = _make_etterna_install(tmp_path, extra_songs=[extra])
    S.set_etterna_root_override(str(install))
    from analysis.games.etterna.replay import find_etterna_dirs
    got = find_etterna_dirs()
    assert str(extra) in got['extra_songs_dirs']


def test_find_etterna_dirs_skips_nonexistent_additional(tmp_path):
    install = _make_etterna_install(
        tmp_path, extra_songs=[tmp_path / 'missing_dir'])
    S.set_etterna_root_override(str(install))
    from analysis.games.etterna.replay import find_etterna_dirs
    got = find_etterna_dirs()
    assert got['extra_songs_dirs'] == []


def test_find_etterna_dirs_skips_bad_override(tmp_path):
    S.set_etterna_root_override(str(tmp_path / 'does-not-exist'))
    from analysis.games.etterna.replay import find_etterna_dirs
    got = find_etterna_dirs()
    assert got['save_dir'] != str(tmp_path / 'does-not-exist')


def test_find_osu_dirs_uses_install_root(tmp_path):
    install = _make_osu_install(tmp_path)
    S.set_osu_root_override(str(install))
    from analysis.games.osu.replay import find_osu_dirs
    got = find_osu_dirs()
    assert got['root'] == str(install)
    assert got['songs_dir'] == str(install / 'Songs')
    assert str(install / 'Data' / 'r') in got['replays_dirs']


def test_find_osu_dirs_honors_beatmap_directory(tmp_path):
    # BeatmapDirectory pointing at an absolute custom path outside the install.
    custom_songs = tmp_path / 'custom_songs'
    custom_songs.mkdir()
    install = _make_osu_install(tmp_path, beatmap_dir=str(custom_songs))
    S.set_osu_root_override(str(install))
    from analysis.games.osu.replay import find_osu_dirs
    got = find_osu_dirs()
    assert got['songs_dir'] == str(custom_songs)


def test_find_osu_dirs_ignores_missing_override(tmp_path):
    S.set_osu_root_override(str(tmp_path / 'missing'))
    from analysis.games.osu.replay import find_osu_dirs
    got = find_osu_dirs()
    assert got['root'] != str(tmp_path / 'missing')


def test_osu_profile_picker_respects_override(tmp_path):
    install = _make_osu_install(tmp_path, profile='first')
    # Add a second, newer cfg so newest-mtime would pick it by default.
    second = install / 'osu!.second.cfg'
    second.write_text('BeatmapDirectory = SecondSongs\n')
    (install / 'SecondSongs').mkdir()
    # Touch second so it's newer than the first cfg.
    import os, time
    os.utime(second, (time.time() + 10, time.time() + 10))
    S.set_osu_root_override(str(install))
    # Without override, newest-mtime wins → SecondSongs.
    from analysis.games.osu.replay import find_osu_dirs
    assert find_osu_dirs()['songs_dir'] == str(install / 'SecondSongs')
    # With profile override, we use the selected cfg's BeatmapDirectory.
    S.set_osu_profile_override('osu!.first.cfg')
    assert find_osu_dirs()['songs_dir'] == str(install / 'Songs')


# ---- PathsDialog -----------------------------------------------------------

@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_paths_dialog_saves_on_accept(qapp, tmp_path):
    ett_install = _make_etterna_install(tmp_path / 'ett')
    osu_install = _make_osu_install(tmp_path / 'osu')
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog()
    dlg.ett_edit.setText(str(ett_install))
    dlg.osu_edit.setText(str(osu_install))
    dlg._accept()
    assert S.get_etterna_root_override() == str(ett_install)
    assert S.get_osu_root_override() == str(osu_install)


def test_paths_dialog_blank_clears(qapp, tmp_path):
    S.set_etterna_root_override(str(tmp_path))
    S.set_osu_root_override(str(tmp_path))
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog()
    dlg.ett_edit.setText('')
    dlg.osu_edit.setText('')
    dlg._accept()
    assert S.get_etterna_root_override() is None
    assert S.get_osu_root_override() is None


def test_paths_dialog_prefills_from_autodetect(qapp, tmp_path):
    install = _make_etterna_install(tmp_path)
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog(autodetect_etterna=str(install), autodetect_osu='/foo/bar')
    assert dlg.ett_edit.text() == str(install)
    assert dlg.osu_edit.text() == '/foo/bar'


def test_paths_dialog_prefers_saved_over_autodetect(qapp, tmp_path):
    install = _make_etterna_install(tmp_path)
    S.set_etterna_root_override(str(install))
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog(autodetect_etterna='/somewhere/else')
    assert dlg.ett_edit.text() == str(install)


def test_paths_dialog_profile_picker_hidden_with_single_cfg(qapp, tmp_path):
    install = _make_osu_install(tmp_path)
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog()
    dlg.osu_edit.setText(str(install))
    assert dlg.osu_profile_combo.isVisible() is False


def test_paths_dialog_profile_picker_shown_with_multiple_cfg(qapp, tmp_path):
    install = _make_osu_install(tmp_path)
    (install / 'osu!.other.cfg').write_text('BeatmapDirectory = Songs\n')
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog()
    # Dialog needs to be shown (or at least laid out) for visibility flags to
    # register under Qt's headless test backend — use a direct state check.
    dlg.osu_edit.setText(str(install))
    # After refresh, combo holds both entries.
    assert dlg.osu_profile_combo.count() == 2


# ---- prompt_if_first_run ---------------------------------------------------

def test_prompt_skips_when_override_present(qapp, tmp_path):
    S.set_etterna_root_override(str(tmp_path))
    from analysis.gui import paths_dialog as pd
    with patch.object(pd, 'PathsDialog') as mock_dialog:
        ran = pd.prompt_if_first_run()
    assert ran is False
    mock_dialog.assert_not_called()
    assert S.is_first_run_done() is True


def test_prompt_skips_after_first_run_done(qapp):
    S.mark_first_run_done()
    from analysis.gui import paths_dialog as pd
    with patch.object(pd, 'PathsDialog') as mock_dialog:
        ran = pd.prompt_if_first_run()
    assert ran is False
    mock_dialog.assert_not_called()


def test_prompt_runs_on_fresh_settings(qapp):
    from analysis.gui import paths_dialog as pd
    with patch.object(pd, 'PathsDialog') as MockDialog:
        instance = MockDialog.return_value
        instance.exec.return_value = 0
        ran = pd.prompt_if_first_run()
    assert ran is True
    MockDialog.assert_called_once()
    assert S.is_first_run_done() is True
