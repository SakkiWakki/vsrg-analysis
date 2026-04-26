from __future__ import annotations

import pytest

from analysis.games.quaver.adapter import (
    QuaverAdapter,
    _quaver_global_audio_offset_s,
    _quaver_root_for_replay,
)
from analysis.gui.player_tab import PlayerTab


class _DummyPlayer:
    def __init__(self, rate=1.0):
        self.play_rate = rate


def _tab_with_offset(offset_s, *, scales=True, rate=1.0):
    tab = PlayerTab.__new__(PlayerTab)
    tab.player = _DummyPlayer(rate)
    tab._audio_chart_offset_s = float(offset_s)
    tab._audio_chart_offset_scales_with_rate = bool(scales)
    tab._audio_chart_offset_rate = float(rate)
    return tab


def test_quaver_global_audio_offset_reads_cfg(tmp_path, monkeypatch):
    root = tmp_path / 'Quaver'
    (root / 'Songs').mkdir(parents=True)
    (root / 'quaver.cfg').write_text(
        '[Config]\nGlobalAudioOffset = 75\n',
        encoding='utf-8',
    )
    monkeypatch.setenv('QUAVER_ROOT', str(root))

    assert _quaver_global_audio_offset_s() == pytest.approx(0.075)


def test_quaver_global_audio_offset_reads_explicit_root(tmp_path):
    root = tmp_path / 'Quaver'
    root.mkdir()
    (root / 'quaver.cfg').write_text(
        '[Config]\nGlobalAudioOffset = -42\n',
        encoding='utf-8',
    )

    assert _quaver_global_audio_offset_s(root=str(root)) == pytest.approx(-0.042)


def test_quaver_root_for_replay_uses_chart_path(tmp_path):
    chart = tmp_path / 'Quaver' / 'Songs' / 'Set' / 'chart.qua'
    chart.parent.mkdir(parents=True)
    chart.write_text('AudioFile: audio.mp3\n', encoding='utf-8')

    assert _quaver_root_for_replay({'chart_path': str(chart)}) == str(
        tmp_path / 'Quaver')


def test_quaver_resolve_audio_uses_replay_audio_without_reparsing(tmp_path, monkeypatch):
    chart = tmp_path / 'chart.qua'
    audio = tmp_path / 'audio.mp3'
    chart.write_text('not a real qua', encoding='utf-8')
    audio.write_bytes(b'not real audio')

    def fail_parse(_path):
        raise AssertionError('resolve_audio should not reparse chart')

    monkeypatch.setattr('analysis.games.quaver.qua_chart.parse_qua_file',
                        fail_parse)

    got = QuaverAdapter().resolve_audio({
        'chart_path': str(chart),
        '_quaver_audio_file': 'audio.mp3',
    })
    assert got == str(audio)


def test_player_tab_maps_chart_time_to_quaver_source_time_at_rate():
    tab = _tab_with_offset(0.075, scales=True, rate=1.5)

    assert tab._chart_to_audio_time(10.0) == pytest.approx(9.8875)
    assert tab._audio_to_chart_time(9.8875) == pytest.approx(10.0)


def test_player_tab_audio_playing_state_uses_source_time():
    tab = _tab_with_offset(0.050, scales=True, rate=2.0)
    tab.player.t_intended = 4.0
    tab.player.paused = False

    audio_t, rate, playing = tab._audio_playing_state()
    assert audio_t == pytest.approx(3.9)
    assert rate == pytest.approx(2.0)
    assert playing is True


def test_player_tab_audio_getter_does_not_read_player_rate():
    class PlayerWithLockedRate:
        @property
        def play_rate(self):
            raise AssertionError('audio getter must not read player.play_rate')

    class FakeAudio:
        def current_chart_time(self):
            return 1.0

    tab = _tab_with_offset(0.050, scales=True, rate=2.0)
    tab.player = PlayerWithLockedRate()
    tab._audio = FakeAudio()

    assert tab._audio_current_chart_time() == pytest.approx(1.1)


def test_player_tab_initial_audio_seek_uses_source_time():
    class FakeAudio:
        def __init__(self):
            self.ready = True
            self._base_duration = 60.0
            self.seeks = []
            self._pitch_correct = True

        def prewarm_rates(self, _rates):
            pass

        def seek(self, t):
            self.seeks.append(float(t))

        def current_chart_time(self):
            return 0.0

        def callback_status_snapshot(self):
            return 0, ''

    fake = FakeAudio()

    tab = _tab_with_offset(0.050, scales=True, rate=2.0)
    tab.player.t_intended = 0.0
    tab.player.t_max = 10.0
    tab.player.attach_audio_clock = lambda _getter: None
    tab.player.attach_audio_status = lambda _getter: None
    tab.player.game = 'quaver'
    tab.audio_state = type('AudioState', (), {'ready': False})()
    tab.audio_state.last_sync_state = None
    tab._sync_audio = lambda force=False: None
    tab._audio_worker = object()

    tab._on_audio_built(fake)

    assert fake.seeks == pytest.approx([-0.1])
    assert tab.player.t_max == pytest.approx(60.1)
