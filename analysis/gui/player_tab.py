"""Embedded replay player tab: native Qt canvas + transport controls."""
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QLineEdit)

from analysis.gui.player_canvas import PlayerCanvas
from analysis.gui.settings import get_settings
from analysis.gui.widgets import JumpSlider


class PlayerTab(QWidget):
    """Embedded replay player with a native QPainter chart canvas."""
    def __init__(self, replay, game='etterna', od=None, judge=None, bpms=None,
                 sm_offset=0.0, audio_path=None, scroll_ms=400.0,
                 scroll_mode=None, play_rate=1.0, cmod_bpm=600.0,
                 osu_speed=20):
        super().__init__()
        from analysis.player.player import Player
        from analysis.core import game as game_mod
        adapter = game_mod.get(game)
        player_kwargs = adapter.player_kwargs(replay, od=od, judge=judge)
        skin = str(get_settings().value('player/skin', 'bar'))
        press_hide = bool(get_settings().value('player/press_hide', False, type=bool))
        self.player = Player(replay, game=game, bpms=bpms,
                             sm_offset=sm_offset, audio_path=None,
                             window_w=900, window_h=800,
                             scroll_ms=scroll_ms, scroll_mode=scroll_mode,
                             cmod_bpm=cmod_bpm, osu_speed=osu_speed,
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
        self._muted = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = PlayerCanvas(self.player)
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

        # Toggles (SV / Skin / Display hits / Pitch-correct) and the scroll
        # value editor all live in the sidebar HUD now — clicking the painted
        # rects dispatches through `_on_hud_action`. A transient QLineEdit
        # overlay (created on-demand) handles click-to-edit for the scroll
        # readout.
        self.scroll_edit: QLineEdit | None = None

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
        if self._audio_ready:
            self._audio.prewarm_rates([0.8, 0.9, 1.1, 1.2, 1.3, 1.5])

        if self._audio_ready and self._audio._base_duration:
            self.player.t_max = max(self.player.t_max,
                                    float(self._audio._base_duration))

        self.player.paused = False
        self.play_btn.setText('⏸')
        self._sync_audio()
        self.player.add_scroll_change_listener(self._on_scroll_change)
        self.player.add_hud_action_listener(self._on_hud_action)

    def _sync_audio(self):
        # Mirror audio-engine status onto the player so the painted HUD
        # (which has no handle to the audio engine) can render a correct
        # Pitch-correct label and distinguish "unavailable" from "off".
        self.player._ui_status = {
            'audio_ready': bool(self._audio_ready),
            'pitch_correct': bool(
                getattr(self._audio, '_pitch_correct', True)),
        }
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

    def _sync_settings_toggles(self):
        """Pull per-tab-shared toggles from QSettings. Called each tick so
        that toggling in another PlayerTab propagates here instead of the
        two tabs showing contradictory states."""
        s = get_settings()
        stored_ph = bool(s.value('player/press_hide', False, type=bool))
        if stored_ph != self.player.press_hide:
            self.player.set_press_hide(stored_ph)
        stored_skin = str(s.value('player/skin', 'bar'))
        if stored_skin != self.player.skin:
            self.player.set_skin(stored_skin)

    def _toggle_mute(self):
        if not self._audio_ready:
            return
        self._muted = not self._muted
        self._audio.set_volume(0.0 if self._muted else 0.5)

    def _on_scroll_change(self):
        """Called by the player when the painted-HUD scroll controls cycle
        the mode, nudge the speed, switch the game, or nudge play rate.
        Re-syncs audio so rate nudges from the HUD actually retime the music,
        and persists the mode."""
        self._sync_audio()
        get_settings().setValue('player/scroll_mode', self.player.scroll_mode)

    def _on_hud_action(self, action, payload):
        """Dispatcher for sidebar-HUD hitbox clicks that need Qt-side work
        (audio sync, QSettings persistence, transient overlays). The player's
        built-in handler already processed the logical side of `action`."""
        if action == 'toggle_sv':
            self.player.toggle_sv()
        elif action == 'cycle_skin':
            self.player.toggle_skin()
            get_settings().setValue('player/skin', self.player.skin)
        elif action == 'toggle_press_hide':
            self.player.toggle_press_hide()
            get_settings().setValue('player/press_hide', self.player.press_hide)
        elif action == 'toggle_pitch':
            if self._audio_ready:
                new = not self._audio._pitch_correct
                self._audio.set_pitch_correct(new)
                get_settings().setValue('player/pitch_correct', new)
                self._sync_audio()
        elif action == 'edit_scroll_value':
            self._open_scroll_edit(payload)

    def _open_scroll_edit(self, rect):
        """Drop a transient QLineEdit overlay on top of the scroll readout
        so the user can type a new mode-native value. Enter applies, Escape
        (or losing focus) cancels."""
        from analysis.player import scroll as scroll_registry
        if self.scroll_edit is not None:
            self.scroll_edit.deleteLater()
            self.scroll_edit = None
        rx, ry, rw, rh = rect
        # Sidebar rect lives in canvas coords — translate to tab coords.
        origin = self.view.mapTo(self, self.view.rect().topLeft())
        edit = QLineEdit(self)
        mode = scroll_registry.get(self.player.scroll_mode)
        cur = self.player._current_mode_value()
        if mode and mode.format_value:
            edit.setText(mode.format_value(cur).split(' ', 1)[0])
        else:
            edit.setText(f'{cur:.2f}')
        edit.setGeometry(origin.x() + rx, origin.y() + ry, rw, rh)
        edit.selectAll()
        edit.returnPressed.connect(self._apply_scroll_edit)
        edit.editingFinished.connect(self._close_scroll_edit)
        edit.show()
        edit.setFocus(Qt.MouseFocusReason)
        self.scroll_edit = edit

    def _apply_scroll_edit(self):
        if self.scroll_edit is None:
            return
        raw = self.scroll_edit.text().strip()
        try:
            val = float(raw)
        except ValueError:
            self._close_scroll_edit()
            return
        self.player._set_current_mode_value(val)
        self._close_scroll_edit()

    def _close_scroll_edit(self):
        if self.scroll_edit is None:
            return
        w = self.scroll_edit
        self.scroll_edit = None
        w.deleteLater()

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
        # Keep per-tab view of globally-stored toggles in sync. Each
        # PlayerTab has its own Player instance, but QSettings are app-wide;
        # if the user toggles display-hits (or other player toggles) in a
        # different tab, mirror it here so the label and rendering don't
        # drift. Cheap: just a dict lookup + attr compare per frame.
        self._sync_settings_toggles()
        if self._scrubbing:
            # Freeze chart clock while scrubbing; the playbar drives t directly
            # via _on_playbar_changed.
            dt = 0
        self.player.advance(dt)
        if (not self.player.paused and self._audio_ready
                and getattr(self._audio, '_ended', False)):
            self.player.paused = True
            self.play_btn.setText('▶')
        self.view.update()

        if not self._scrubbing:
            self._suppress_playbar = True
            self.playbar.setValue(self._t_to_slider(self.player.t))
            self._suppress_playbar = False

        self.time_lbl.setText(self._fmt_time(self.player.t))
        self.dur_lbl.setText(self._fmt_time(self.player.t_max))
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
                    self._nudge_rate(0.1); return True
                if k == Qt.Key_Minus:
                    self._nudge_rate(-0.1); return True
                if k == Qt.Key_M:
                    self._toggle_mute(); return True
                if k == Qt.Key_R:
                    self.player.restart(); self._sync_audio(); return True
                if k in (Qt.Key_Q, Qt.Key_Escape):
                    self.window().close(); return True
            elif t == ev.Type.Wheel:
                step = ev.angleDelta().y() / 120.0 * 0.5
                if ev.modifiers() & Qt.ShiftModifier:
                    step *= 10
                self._seek(step)
                return True
            elif t == ev.Type.MouseButtonPress and ev.button() == Qt.LeftButton:
                self.view.setFocus(Qt.MouseFocusReason)
                pos = ev.position() if hasattr(ev, 'position') else ev.pos()
                if self.player.handle_mouse_down(int(pos.x()), int(pos.y())):
                    return True
        return super().eventFilter(obj, ev)

    def cleanup(self):
        self.timer.stop()
        if self._audio_ready:
            try:
                self._audio.stop()
            except Exception:
                pass

    def closeEvent(self, ev):
        self.cleanup()
        super().closeEvent(ev)
