"""Embedded replay player tab: pygame Surface streamed into a QLabel."""
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QLineEdit, QSizePolicy)

from analysis.gui.settings import get_settings
from analysis.gui.widgets import JumpSlider


class PlayerTab(QWidget):
    """Embedded replay player: pygame Surface streamed into a QLabel."""
    def __init__(self, replay, game='etterna', od=None, judge=None, bpms=None,
                 sm_offset=0.0, audio_path=None, scroll_ms=400.0,
                 scroll_mode=None, play_rate=1.0):
        super().__init__()
        from analysis.player.player import Player
        from analysis.core import game as game_mod
        adapter = game_mod.get(game)
        player_kwargs = adapter.player_kwargs(replay, od=od, judge=judge)
        skin = str(get_settings().value('player/skin', 'bar'))
        press_hide = bool(get_settings().value('player/press_hide', False, type=bool))
        self.player = Player(replay, game=game, bpms=bpms,
                             sm_offset=sm_offset, audio_path=None,
                             window_w=900, window_h=800, headless=True,
                             scroll_ms=scroll_ms, scroll_mode=scroll_mode,
                             skin=skin, press_hide=press_hide, **player_kwargs)
        # Replay's original playback rate (Etterna "Rate" from XML, osu!
        # "ModRate"). The chart + offsets are in chart-time, but the audio
        # file is unrated — playing at 1.0 leaves the music slower/faster
        # than the score was actually played. Set the rate up-front so the
        # audio engine resamples on first sync.
        try:
            pr = float(play_rate or 1.0)
        except (TypeError, ValueError):
            pr = 1.0
        self.player.play_rate = max(0.1, min(4.0, pr))
        self._audio_ready = False
        self._audio_path = audio_path
        self._music_started_at = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setMinimumSize(400, 400)
        self.view.setFocusPolicy(Qt.StrongFocus)
        self.view.installEventFilter(self)
        layout.addWidget(self.view, 1)

        bar = QHBoxLayout()
        self.play_btn = QPushButton('▶')
        self.play_btn.setMaximumWidth(36)
        self.play_btn.setFocusPolicy(Qt.NoFocus)
        self.play_btn.clicked.connect(lambda _checked=False: self._toggle())
        bar.addWidget(self.play_btn)

        self.time_lbl = QLabel('0:00')
        self.time_lbl.setMinimumWidth(44)
        bar.addWidget(self.time_lbl)

        self.playbar = JumpSlider(Qt.Horizontal)
        self.playbar.setFocusPolicy(Qt.NoFocus)
        self.playbar.setRange(0, 1000)
        self.playbar.setValue(0)
        self._scrubbing = False
        self._suppress_playbar = False
        self._resume_after_scrub = False
        self.playbar.valueChanged.connect(self._on_playbar_changed)
        self.playbar.sliderPressed.connect(self._on_playbar_pressed)
        self.playbar.sliderReleased.connect(self._on_playbar_released)
        bar.addWidget(self.playbar, 1)

        self.dur_lbl = QLabel('0:00')
        self.dur_lbl.setMinimumWidth(44)
        bar.addWidget(self.dur_lbl)
        layout.addLayout(bar)

        ctl = QHBoxLayout()
        for label, fn in [
            ('scroll −', lambda: self.player.nudge_scroll(1 / 1.15)),
            ('scroll +', lambda: self.player.nudge_scroll(1.15)),
            ('rate −', lambda: self._nudge_rate(-0.1)),
            ('rate +', lambda: self._nudge_rate(0.1)),
            ('restart', self._restart),
        ]:
            b = QPushButton(label)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _checked=False, f=fn: f())
            ctl.addWidget(b)

        self.sv_btn = QPushButton()
        self.sv_btn.setFocusPolicy(Qt.NoFocus)
        self._refresh_sv_btn()
        self.sv_btn.clicked.connect(lambda _checked=False: self._toggle_sv())
        ctl.addWidget(self.sv_btn)

        self.mode_btn = QPushButton()
        self.mode_btn.setFocusPolicy(Qt.NoFocus)
        self.mode_btn.setToolTip(
            'Linear: constant ms from top to judgment (osu!mania-style).\n'
            'CMOD: scroll locked to a BPM (Etterna CMOD-style) — higher-BPM '
            'sections scroll faster on screen.')
        self.mode_btn.clicked.connect(lambda _checked=False: self._toggle_mode())
        ctl.addWidget(self.mode_btn)
        self._refresh_mode_btn()

        self.skin_btn = QPushButton()
        self.skin_btn.setFocusPolicy(Qt.NoFocus)
        self.skin_btn.setToolTip('Cycle note skin')
        self.skin_btn.clicked.connect(lambda _checked=False: self._cycle_skin())
        ctl.addWidget(self.skin_btn)
        self._refresh_skin_btn()

        self.press_btn = QPushButton()
        self.press_btn.setFocusPolicy(Qt.NoFocus)
        self.press_btn.setToolTip(
            'When off, notes vanish once the player actually presses them '
            '(LNs stick their head to the judgment line while held, then '
            'vanish on release). Misses stay visible, dimmed.')
        self.press_btn.clicked.connect(lambda _checked=False: self._toggle_press_hide())
        ctl.addWidget(self.press_btn)
        self._refresh_press_btn()

        self.pitch_btn = QPushButton()
        self.pitch_btn.setFocusPolicy(Qt.NoFocus)
        self.pitch_btn.setToolTip(
            'Pitch correction: when on, rate changes speed but not pitch '
            '(like osu! DoubleTime). When off, pitch shifts with rate '
            '(like Nightcore / Etterna stock rate-mod).')
        self.pitch_btn.clicked.connect(lambda _checked=False: self._toggle_pitch())
        ctl.addWidget(self.pitch_btn)

        ctl.addWidget(QLabel('scroll:'))
        self.scroll_edit = QLineEdit()
        self.scroll_edit.setMaximumWidth(70)
        self.scroll_edit.setPlaceholderText('ms')
        self.scroll_edit.returnPressed.connect(self._apply_scroll_edit)
        ctl.addWidget(self.scroll_edit)

        ctl.addStretch(1)
        self.hud = QLabel('')
        ctl.addWidget(self.hud)
        layout.addLayout(ctl)

        self._last_ms = None
        self.timer = QTimer(self)
        self.timer.setInterval(1000 // 120)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        from analysis.player.audio import AudioEngine
        pitch_correct = bool(get_settings().value(
            'player/pitch_correct', True, type=bool))
        self._audio = AudioEngine(audio_path, pitch_correct=pitch_correct)
        self._audio_ready = self._audio.ready
        self._refresh_pitch_btn()
        if self._audio_ready:
            self._audio.prewarm_rates([0.8, 0.9, 1.1, 1.2, 1.3, 1.5])

        if self._audio_ready and self._audio._base_duration:
            self.player.t_max = max(self.player.t_max,
                                    float(self._audio._base_duration))

        self.player.paused = False
        self.play_btn.setText('⏸')
        self._sync_audio()

    def _sync_audio(self):
        if self._audio_ready:
            self._audio.set_state(self.player.t, self.player.play_rate,
                                  not self.player.paused)

    def _toggle(self):
        self.player.toggle_pause()
        self.play_btn.setText('▶' if self.player.paused else '⏸')
        self._sync_audio()

    def _seek(self, ds):
        self.player.seek_rel(ds)
        self._sync_audio()

    def _nudge_rate(self, d):
        self.player.nudge_rate(d)
        self._sync_audio()

    def _refresh_sv_btn(self):
        if not self.player.sv_sections:
            self.sv_btn.setText('SV: n/a')
            self.sv_btn.setEnabled(False)
        else:
            self.sv_btn.setEnabled(True)
            self.sv_btn.setText('SV: on' if self.player.sv_enabled else 'SV: off')

    def _toggle_sv(self):
        self.player.toggle_sv()
        self._refresh_sv_btn()

    def _refresh_mode_btn(self):
        if self.player.scroll_mode == self.player.SCROLL_MODE_CMOD:
            self.mode_btn.setText('Mode: CMOD')
            if hasattr(self, 'scroll_edit'):
                self.scroll_edit.setPlaceholderText('CMOD BPM')
        else:
            self.mode_btn.setText('Mode: Linear')
            if hasattr(self, 'scroll_edit'):
                self.scroll_edit.setPlaceholderText('ms')

    def _refresh_skin_btn(self):
        self.skin_btn.setText(f'Skin: {self.player.skin}')

    def _cycle_skin(self):
        self.player.toggle_skin()
        self._refresh_skin_btn()
        get_settings().setValue('player/skin', self.player.skin)

    def _refresh_press_btn(self):
        self.press_btn.setText(
            f'Display hits: {"off" if self.player.press_hide else "on"}')

    def _toggle_press_hide(self):
        self.player.toggle_press_hide()
        self._refresh_press_btn()
        get_settings().setValue('player/press_hide', self.player.press_hide)

    def _refresh_pitch_btn(self):
        if not hasattr(self, 'pitch_btn'):
            return
        on = self._audio_ready and getattr(self._audio, '_pitch_correct', True)
        self.pitch_btn.setText(f'Pitch-correct: {"on" if on else "off"}')
        self.pitch_btn.setEnabled(self._audio_ready)

    def _toggle_pitch(self):
        if not self._audio_ready:
            return
        new = not self._audio._pitch_correct
        self._audio.set_pitch_correct(new)
        get_settings().setValue('player/pitch_correct', new)
        self._refresh_pitch_btn()
        self._sync_audio()

    def _toggle_mode(self):
        new_mode = (self.player.SCROLL_MODE_LINEAR
                    if self.player.scroll_mode == self.player.SCROLL_MODE_CMOD
                    else self.player.SCROLL_MODE_CMOD)
        self.player.set_scroll_mode(new_mode)
        self._refresh_mode_btn()
        get_settings().setValue('player/scroll_mode', new_mode)

    def _apply_scroll_edit(self):
        raw = self.scroll_edit.text().strip()
        if not raw:
            return
        try:
            val = float(raw)
        except ValueError:
            self.scroll_edit.clear()
            return
        if self.player.scroll_mode == self.player.SCROLL_MODE_CMOD:
            self.player.cmod_bpm = max(60.0, min(5000.0, val))
        else:
            self.player.set_scroll_ms(val)
        self.scroll_edit.clear()

    def _restart(self):
        self.player.restart()
        self._sync_audio()

    def _t_to_slider(self, t):
        tmin, tmax = self.player.t_min, self.player.t_max
        if tmax <= tmin:
            return 0
        return int(round((t - tmin) / (tmax - tmin) * 1000))

    def _slider_to_t(self, v):
        tmin, tmax = self.player.t_min, self.player.t_max
        return tmin + (v / 1000.0) * (tmax - tmin)

    def _on_playbar_pressed(self):
        # Pause audio + stop chart advancement while the user scrubs. Remember
        # whether we need to resume on release; don't flip the paused UI state
        # since this is a transient drag, not a user-visible pause.
        self._scrubbing = True
        self._resume_after_scrub = not self.player.paused
        if self._audio_ready:
            self._audio.set_state(self.player.t, self.player.play_rate, False)

    def _on_playbar_released(self):
        self._scrubbing = False
        self.player.t = self._slider_to_t(self.playbar.value())
        # Resume only if we were playing before the drag started.
        if self._audio_ready:
            self._audio.set_state(self.player.t, self.player.play_rate,
                                  self._resume_after_scrub)

    def _on_playbar_changed(self, v):
        if self._suppress_playbar:
            return
        self.player.t = self._slider_to_t(v)
        if not self.playbar.isSliderDown():
            self._sync_audio()

    @staticmethod
    def _fmt_time(t):
        sign = '-' if t < 0 else ''
        t = abs(t)
        return f'{sign}{int(t // 60)}:{int(t % 60):02d}'

    def resizeEvent(self, ev):
        w, h = self.view.width(), self.view.height()
        if w > 0 and h > 0:
            self.player.resize(w, h)
        super().resizeEvent(ev)

    def _tick(self):
        now = time.monotonic()
        if self._last_ms is None:
            self._last_ms = now
        dt = now - self._last_ms
        self._last_ms = now
        if self._scrubbing:
            # Freeze chart clock while scrubbing; the playbar drives t directly
            # via _on_playbar_changed.
            dt = 0
        buf, (w, h) = self.player.tick(dt)
        if (not self.player.paused and self._audio_ready
                and getattr(self._audio, '_ended', False)):
            self.player.paused = True
            self.play_btn.setText('▶')
        img = QImage(buf, w, h, w * 3, QImage.Format_RGB888)
        self.view.setPixmap(QPixmap.fromImage(img))

        if not self._scrubbing:
            self._suppress_playbar = True
            self.playbar.setValue(self._t_to_slider(self.player.t))
            self._suppress_playbar = False

        self.time_lbl.setText(self._fmt_time(self.player.t))
        self.dur_lbl.setText(self._fmt_time(self.player.t_max))
        self.hud.setText(
            f'rate={self.player.play_rate:.2f}x  '
            f'scroll={int(self.player.scroll_ms)}ms')
        if not self._scrubbing:
            self._sync_audio()

    def eventFilter(self, obj, ev):
        if obj is self.view:
            t = ev.type()
            if t == ev.Type.KeyPress:
                k = ev.key()
                if k in (Qt.Key_Space, Qt.Key_P):
                    self._toggle(); return True
                if k == Qt.Key_Left:
                    self._seek(-10 if ev.modifiers() & Qt.ShiftModifier else -2)
                    return True
                if k == Qt.Key_Right:
                    self._seek(10 if ev.modifiers() & Qt.ShiftModifier else 2)
                    return True
                if k == Qt.Key_Up:
                    self.player.nudge_scroll(1.15); return True
                if k == Qt.Key_Down:
                    self.player.nudge_scroll(1 / 1.15); return True
                if k in (Qt.Key_Plus, Qt.Key_Equal):
                    self.player.nudge_rate(0.1); return True
                if k == Qt.Key_Minus:
                    self.player.nudge_rate(-0.1); return True
                if k == Qt.Key_R:
                    self._restart(); return True
            elif t == ev.Type.Wheel:
                step = ev.angleDelta().y() / 120.0 * 0.5
                if ev.modifiers() & Qt.ShiftModifier:
                    step *= 10
                self._seek(step)
                return True
        return super().eventFilter(obj, ev)

    def cleanup(self):
        self.timer.stop()
        if self._audio_ready:
            try:
                self._audio.stop()
            except Exception:
                pass
