from analysis.games.etterna.adapter import EtternaAdapter


def test_resolve_song_asset_matches_filename_case_insensitively(tmp_path):
    song_dir = tmp_path / 'World of Corruption' / '[Zeta] Hall Of Kings'
    song_dir.mkdir(parents=True)
    chart = song_dir / 'Hall of Kings.ssc'
    chart.write_text('#MUSIC:Hall Of Kings.ogg;\n', encoding='utf-8')
    audio = song_dir / 'Hall of Kings.ogg'
    audio.write_bytes(b'not real audio')

    assert EtternaAdapter._resolve_song_asset(chart, 'Hall Of Kings.ogg') == str(audio)


def test_resolve_song_asset_handles_nested_case_drift(tmp_path):
    song_dir = tmp_path / 'Song'
    media_dir = song_dir / 'Audio'
    media_dir.mkdir(parents=True)
    chart = song_dir / 'chart.ssc'
    chart.write_text('#MUSIC:audio/Track.ogg;\n', encoding='utf-8')
    audio = media_dir / 'track.ogg'
    audio.write_bytes(b'not real audio')

    assert EtternaAdapter._resolve_song_asset(chart, 'audio/Track.ogg') == str(audio)


def test_resolve_music_asset_falls_back_to_first_song_folder_audio(tmp_path):
    song_dir = tmp_path / 'Song'
    song_dir.mkdir()
    chart = song_dir / 'chart.ssc'
    chart.write_text('#MUSIC:missing.ogg;\n', encoding='utf-8')
    first = song_dir / 'alpha.ogg'
    second = song_dir / 'zeta.mp3'
    second.write_bytes(b'not real audio')
    first.write_bytes(b'not real audio')

    assert EtternaAdapter._resolve_music_asset(chart, 'missing.ogg') == str(first)


def test_resolve_song_asset_path_with_slash_is_relative_to_install_root(tmp_path):
    chart = tmp_path / 'Songs' / 'Pack' / 'Song' / 'chart.ssc'
    chart.parent.mkdir(parents=True)
    audio = tmp_path / 'Shared' / 'Music.ogg'
    audio.parent.mkdir()
    audio.write_bytes(b'not real audio')

    assert EtternaAdapter._resolve_song_asset(chart, 'shared/music.ogg') == str(audio)
