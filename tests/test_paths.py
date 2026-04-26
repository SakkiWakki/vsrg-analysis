"""Tests for the install-path system.

Covers:
- path-override round-trip via the shopkeeper (Qt-backed in test runs)
- per-game manifest validators accept real-looking trees and reject garbage
- find_etterna_dirs / find_osu_dirs honor the install-root override and
  resolve Preferences.ini / osu!.cfg subpaths
- PathsDialog saves entered paths on accept
- prompt_if_first_run() only runs once
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from analysis.core import manifest as manifest_mod
from analysis.core import path_overrides
from analysis.gui import settings as S


# Manifest discovery is module-cached ; the conftest doesn't reset it
# because it's a one-time scan. Tests that monkeypatch a manifest method
# should restore it themselves.
manifest_mod.discover_manifests()


def _field(game, key):
    """Lookup helper: return the named PathField on the given manifest."""
    for f in manifest_mod.get(game).path_fields:
        if f.key == key:
            return f
    raise AssertionError(f'no field {key!r} on {game!r} manifest')


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


# ---- shopkeeper round-trip via the QSettings backend -----------------------

def test_override_roundtrip_etterna(tmp_path):
    key = _field('etterna', 'root').settings_key
    assert path_overrides.get(key) is None
    path_overrides.set(key, str(tmp_path))
    assert path_overrides.get(key) == str(tmp_path)
    path_overrides.set(key, None)
    assert path_overrides.get(key) is None


def test_override_empty_string_clears(tmp_path):
    key = _field('etterna', 'root').settings_key
    path_overrides.set(key, str(tmp_path))
    path_overrides.set(key, '')
    assert path_overrides.get(key) is None


def test_override_roundtrip_osu(tmp_path):
    key = _field('osu', 'root').settings_key
    path_overrides.set(key, str(tmp_path))
    assert path_overrides.get(key) == str(tmp_path)


def test_osu_profile_roundtrip():
    key = _field('osu', 'profile').settings_key
    assert path_overrides.get(key) is None
    path_overrides.set(key, 'osu!.yucky.cfg')
    assert path_overrides.get(key) == 'osu!.yucky.cfg'
    path_overrides.set(key, None)
    assert path_overrides.get(key) is None


def test_first_run_flag():
    assert S.is_first_run_done() is False
    S.mark_first_run_done()
    assert S.is_first_run_done() is True


# ---- manifest validators ---------------------------------------------------

def test_validate_etterna_accepts_install_root(tmp_path):
    install = _make_etterna_install(tmp_path, with_profile=True)
    assert _field('etterna', 'root').validate(str(install)) is True


def test_validate_etterna_accepts_direct_save(tmp_path):
    # Back-compat: user pointing directly at Save/ should still validate.
    install = _make_etterna_install(tmp_path)
    assert _field('etterna', 'root').validate(str(install / 'Save')) is True


def test_validate_etterna_rejects_garbage(tmp_path):
    validate = _field('etterna', 'root').validate
    assert validate(None) is False
    assert validate('') is False
    assert validate('/nonexistent/path/12345') is False
    empty = tmp_path / 'nothing'
    empty.mkdir()
    assert validate(str(empty)) is False


def test_validate_osu_requires_cfg(tmp_path):
    validate = _field('osu', 'root').validate
    assert validate(None) is False
    assert validate('') is False
    assert validate('/nope/12345') is False
    # Empty dir ; no osu!.<user>.cfg.
    assert validate(str(tmp_path)) is False
    install = _make_osu_install(tmp_path)
    assert validate(str(install)) is True


# ---- find_*_dirs override precedence ---------------------------------------

def test_find_etterna_dirs_uses_install_root(tmp_path):
    install = _make_etterna_install(tmp_path)
    path_overrides.set('paths/etterna_root', str(install))
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
    path_overrides.set('paths/etterna_root', str(install))
    from analysis.games.etterna.replay import find_etterna_dirs
    got = find_etterna_dirs()
    assert str(extra) in got['extra_songs_dirs']


def test_find_etterna_dirs_skips_nonexistent_additional(tmp_path):
    install = _make_etterna_install(
        tmp_path, extra_songs=[tmp_path / 'missing_dir'])
    path_overrides.set('paths/etterna_root', str(install))
    from analysis.games.etterna.replay import find_etterna_dirs
    got = find_etterna_dirs()
    assert got['extra_songs_dirs'] == []


def test_find_etterna_dirs_skips_bad_override(tmp_path):
    path_overrides.set('paths/etterna_root', str(tmp_path / 'does-not-exist'))
    from analysis.games.etterna.replay import find_etterna_dirs
    got = find_etterna_dirs()
    assert got['save_dir'] != str(tmp_path / 'does-not-exist')


def test_find_osu_dirs_uses_install_root(tmp_path):
    install = _make_osu_install(tmp_path)
    path_overrides.set('paths/osu_root', str(install))
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
    path_overrides.set('paths/osu_root', str(install))
    from analysis.games.osu.replay import find_osu_dirs
    got = find_osu_dirs()
    assert got['songs_dir'] == str(custom_songs)


def test_find_osu_dirs_ignores_missing_override(tmp_path):
    path_overrides.set('paths/osu_root', str(tmp_path / 'missing'))
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
    path_overrides.set('paths/osu_root', str(install))
    # Without override, newest-mtime wins → SecondSongs.
    from analysis.games.osu.replay import find_osu_dirs
    assert find_osu_dirs()['songs_dir'] == str(install / 'SecondSongs')
    # With profile override, we use the selected cfg's BeatmapDirectory.
    path_overrides.set('paths/osu_profile', 'osu!.first.cfg')
    assert find_osu_dirs()['songs_dir'] == str(install / 'Songs')


# ---- PathsDialog -----------------------------------------------------------

@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _row_for(dlg, game, field_key='root'):
    """Find the FieldRow for a given (game, field_key) pair on the
    dialog. Each game contributes one row per declared path field."""
    for row in dlg.rows:
        if row.field.settings_key == _field(game, field_key).settings_key:
            return row
    raise AssertionError(f'no row for ({game!r}, {field_key!r})')


def test_paths_dialog_saves_on_accept(qapp, tmp_path):
    ett_install = _make_etterna_install(tmp_path / 'ett')
    osu_install = _make_osu_install(tmp_path / 'osu')
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog()
    _row_for(dlg, 'etterna').edit.setText(str(ett_install))
    _row_for(dlg, 'osu').edit.setText(str(osu_install))
    dlg._accept()
    assert path_overrides.get('paths/etterna_root') == str(ett_install)
    assert path_overrides.get('paths/osu_root') == str(osu_install)


def test_paths_dialog_blank_clears(qapp, tmp_path):
    path_overrides.set('paths/etterna_root', str(tmp_path))
    path_overrides.set('paths/osu_root', str(tmp_path))
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog()
    _row_for(dlg, 'etterna').edit.setText('')
    _row_for(dlg, 'osu').edit.setText('')
    dlg._accept()
    assert path_overrides.get('paths/etterna_root') is None
    assert path_overrides.get('paths/osu_root') is None


def test_paths_dialog_prefers_saved_over_autodetect(qapp, tmp_path):
    install = _make_etterna_install(tmp_path)
    path_overrides.set('paths/etterna_root', str(install))
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog()
    assert _row_for(dlg, 'etterna').edit.text() == str(install)


def test_paths_dialog_prefills_from_autodetect(qapp, tmp_path, monkeypatch):
    install = _make_etterna_install(tmp_path)
    # PathField + GameManifest are frozen ; rebuild the registry entries
    # with patched fields for this test, restored on teardown.
    import dataclasses

    def _swap_autodetect(game_name, fake):
        man = manifest_mod.get(game_name)
        new_fields = [
            dataclasses.replace(f, autodetect=fake)
            if f.key == 'root' else f
            for f in man.path_fields
        ]
        new_man = dataclasses.replace(man, path_fields=new_fields)
        monkeypatch.setitem(manifest_mod._REGISTRY, game_name, new_man)

    _swap_autodetect('etterna', lambda: str(install))
    _swap_autodetect('osu', lambda: '/foo/bar')

    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog()
    assert _row_for(dlg, 'etterna').edit.text() == str(install)
    assert _row_for(dlg, 'osu').edit.text() == '/foo/bar'


def test_paths_dialog_profile_picker_hidden_with_single_cfg(qapp, tmp_path):
    install = _make_osu_install(tmp_path)
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog()
    root_row = _row_for(dlg, 'osu', 'root')
    profile_row = _row_for(dlg, 'osu', 'profile')
    root_row.edit.setText(str(install))
    profile_row.refresh_choices()
    assert profile_row.combo.isVisible() is False


def test_paths_dialog_profile_picker_shown_with_multiple_cfg(qapp, tmp_path):
    install = _make_osu_install(tmp_path)
    (install / 'osu!.other.cfg').write_text('BeatmapDirectory = Songs\n')
    from analysis.gui.paths_dialog import PathsDialog
    dlg = PathsDialog()
    root_row = _row_for(dlg, 'osu', 'root')
    profile_row = _row_for(dlg, 'osu', 'profile')
    root_row.edit.setText(str(install))
    profile_row.refresh_choices()
    # After refresh, combo holds both entries.
    assert profile_row.combo.count() == 2


# ---- prompt_if_first_run ---------------------------------------------------

def test_prompt_skips_when_override_present(qapp, tmp_path):
    path_overrides.set('paths/etterna_root', str(tmp_path))
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
