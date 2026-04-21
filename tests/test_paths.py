"""Tests for the install-path override system.

Covers:
- settings getter/setter round-trip (including empty-string clearing)
- validators accept a real-looking tree and reject garbage
- find_etterna_dirs / find_osu_dirs honor the override when valid and fall
  through to autodetection when not
- PathsDialog saves entered paths on accept
- prompt_if_first_run() only runs once
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from analysis.gui import settings as S


def _make_etterna_tree(root: Path, with_xml=True, with_profile=True):
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
    return save


def _make_osu_tree(root: Path):
    songs = root / 'osu!' / 'Songs'
    songs.mkdir(parents=True)
    data_r = root / 'osu!' / 'Data' / 'r'
    data_r.mkdir(parents=True)
    return songs, data_r


# ---- settings round-trip ---------------------------------------------------

def test_override_roundtrip_etterna(tmp_path):
    assert S.get_etterna_save_override() is None
    S.set_etterna_save_override(str(tmp_path))
    assert S.get_etterna_save_override() == str(tmp_path)
    S.set_etterna_save_override(None)
    assert S.get_etterna_save_override() is None


def test_override_empty_string_clears(tmp_path):
    S.set_etterna_save_override(str(tmp_path))
    S.set_etterna_save_override('')
    assert S.get_etterna_save_override() is None


def test_override_roundtrip_osu(tmp_path):
    S.set_osu_songs_override(str(tmp_path))
    assert S.get_osu_songs_override() == str(tmp_path)


def test_first_run_flag():
    assert S.is_first_run_done() is False
    S.mark_first_run_done()
    assert S.is_first_run_done() is True


# ---- validators ------------------------------------------------------------

def test_validate_etterna_accepts_localprofiles(tmp_path):
    save = _make_etterna_tree(tmp_path, with_profile=True)
    assert S.validate_etterna_save(str(save)) is True


def test_validate_etterna_accepts_direct_xml(tmp_path):
    save = tmp_path / 'Save'
    save.mkdir()
    (save / 'Etterna.xml').write_text('<xml/>')
    assert S.validate_etterna_save(str(save)) is True


def test_validate_etterna_rejects_garbage(tmp_path):
    assert S.validate_etterna_save(None) is False
    assert S.validate_etterna_save('') is False
    assert S.validate_etterna_save('/nonexistent/path/12345') is False
    empty = tmp_path / 'nothing'
    empty.mkdir()
    assert S.validate_etterna_save(str(empty)) is False


def test_validate_osu(tmp_path):
    assert S.validate_osu_songs(None) is False
    assert S.validate_osu_songs('') is False
    assert S.validate_osu_songs('/nope/12345') is False
    assert S.validate_osu_songs(str(tmp_path)) is True


# ---- find_*_dirs override precedence ---------------------------------------

def test_find_etterna_dirs_uses_override(tmp_path):
    save = _make_etterna_tree(tmp_path)
    S.set_etterna_save_override(str(save))
    from analysis.etterna.replay import find_etterna_dirs
    got = find_etterna_dirs()
    assert got['save_dir'] == str(save)
    assert got['replays_dir'] == str(save / 'ReplaysV2')
    assert got['xml_path'] is not None


def test_find_etterna_dirs_skips_bad_override(tmp_path):
    # A nonexistent override should *not* short-circuit — we must fall through
    # to autodetection. We can't easily assert what autodetect returns on the
    # dev box, but we can at least check we didn't echo the bogus override.
    S.set_etterna_save_override(str(tmp_path / 'does-not-exist'))
    from analysis.etterna.replay import find_etterna_dirs
    got = find_etterna_dirs()
    assert got['save_dir'] != str(tmp_path / 'does-not-exist')


def test_find_osu_dirs_uses_override(tmp_path):
    songs, data_r = _make_osu_tree(tmp_path)
    S.set_osu_songs_override(str(songs))
    from analysis.osu.replay import find_osu_dirs
    got = find_osu_dirs()
    assert got['songs_dir'] == str(songs)
    # Replay dir adjacent to Songs/ should get picked up automatically.
    assert str(data_r) in got['replays_dirs']


def test_find_osu_dirs_ignores_missing_override(tmp_path):
    S.set_osu_songs_override(str(tmp_path / 'missing'))
    from analysis.osu.replay import find_osu_dirs
    got = find_osu_dirs()
    assert got['songs_dir'] != str(tmp_path / 'missing')


# ---- PathsDialog -----------------------------------------------------------

@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_paths_dialog_saves_on_accept(qapp, tmp_path):
    save = _make_etterna_tree(tmp_path)
    songs, _ = _make_osu_tree(tmp_path)
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog()
    dlg.ett_edit.setText(str(save))
    dlg.osu_edit.setText(str(songs))
    dlg._accept()
    assert S.get_etterna_save_override() == str(save)
    assert S.get_osu_songs_override() == str(songs)


def test_paths_dialog_blank_clears(qapp, tmp_path):
    S.set_etterna_save_override(str(tmp_path))
    S.set_osu_songs_override(str(tmp_path))
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog()
    dlg.ett_edit.setText('')
    dlg.osu_edit.setText('')
    dlg._accept()
    assert S.get_etterna_save_override() is None
    assert S.get_osu_songs_override() is None


def test_paths_dialog_prefills_from_autodetect(qapp, tmp_path):
    save = _make_etterna_tree(tmp_path)
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog(autodetect_etterna=str(save), autodetect_osu='/foo/bar')
    assert dlg.ett_edit.text() == str(save)
    assert dlg.osu_edit.text() == '/foo/bar'


def test_paths_dialog_prefers_saved_over_autodetect(qapp, tmp_path):
    save = _make_etterna_tree(tmp_path)
    S.set_etterna_save_override(str(save))
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog(autodetect_etterna='/somewhere/else')
    assert dlg.ett_edit.text() == str(save)


# ---- prompt_if_first_run ---------------------------------------------------

def test_prompt_skips_when_override_present(qapp, tmp_path):
    S.set_etterna_save_override(str(tmp_path))
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
