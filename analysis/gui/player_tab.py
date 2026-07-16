"""Embedded replay player tab: native Qt canvas + transport controls."""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
)

from analysis.gui.player_canvas import PlayerCanvas
from analysis.gui.settings import (
    load_player_settings,
    save_player_setting,
    signals as settings_signals,
)
from analysis.gui.widgets import JumpSlider


@dataclass
class AudioState:
    ready: bool = False
    path: str | None = None
    muted: bool = False
    last_sync_state: tuple[float, float, bool] | None = None


@dataclass
class ScrubState:
    active: bool = False
    suppress_slider: bool = False
    resume_after_release: bool = False


def _sync_gc_to_playback(playing: bool) -> None:
    """Keep the cycle collector's full passes out of playback frames.

    Parsed charts, numpy caches, and Qt wrappers make an old generation
    large enough that an automatic gen-2 collection lands as a visible
    frame hitch. While playing, freeze the current heap so collections
    only traverse allocations made since; on pause, unfreeze and run the
    deferred full collection while nothing is animating.
    """
    if playing:
        gc.collect()
        gc.freeze()
    else:
        gc.unfreeze()
        gc.collect()


class PlayerTab(QWidget):
    """Embedded replay player with a native QPainter chart canvas."""

    SLIDER_MAX = 1000

    def __init__(
        self,
        replay,
        game='etterna',
        od=None,
        judge=None,
        bpms=None,
        sm_offset=0.0,
        audio_path=None,
        scroll_ms=400.0,
        scroll_mode=None,
        play_rate=1.0,
        audio_chart_offset_s=0.0,
        audio_chart_offset_scales_with_rate=False,
        cmod_bpm=600.0,
        xmod_value=1.0,
        osu_speed=20,
        xml_judgments=None,
        keycount=None,
    ):
        super().__init__()

        self.player = self._create_player(
            replay,
            game=game,
            od=od,
            judge=judge,
            bpms=bpms,
            sm_offset=sm_offset,
            audio_path=audio_path,
            scroll_ms=scroll_ms,
            scroll_mode=scroll_mode,
            cmod_bpm=cmod_bpm,
            xmod_value=xmod_value,
            osu_speed=osu_speed,
            xml_judgments=xml_judgments,
            keycount=keycount,
        )

        self._set_initial_play_rate(play_rate)
        self._audio_chart_offset_s = float(audio_chart_offset_s or 0.0)
        self._audio_chart_offset_scales_with_rate = bool(
            audio_chart_offset_scales_with_rate
        )
        self._audio_chart_offset_rate = float(self.player.play_rate)

        self.audio_state = AudioState(path=audio_path)
        self.scrub = ScrubState()
        self.scroll_edit: QLineEdit | None = None
        self._audio = None
        self._audio_worker = None
        self._time_text = None
        self._dur_text = None
        self._legacy_hz = self._legacy_render_hz()

        self._build_ui()
        self._build_frame_loop()
        self._build_audio(audio_path)
        self._connect_player_events()
        self._build_input_router()

        self._start_playing()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _create_player(
        self,
        replay,
        *,
        game,
        od,
        judge,
        bpms,
        sm_offset,
        audio_path,
        scroll_ms,
        scroll_mode,
        cmod_bpm,
        xmod_value,
        osu_speed,
        xml_judgments,
        keycount,
    ):
        from analysis.player.player_api import Player
        from analysis.core import game as game_mod

        adapter = game_mod.get(game)
        player_kwargs = adapter.player_kwargs(replay, od=od, judge=judge)
        prefs = load_player_settings(game)

        effective_mode = (
            scroll_mode if scroll_mode is not None else prefs['scroll_mode']
        )

        return Player(
            replay,
            game=game,
            bpms=bpms,
            sm_offset=sm_offset,
            audio_path=None,
            window_w=900,
            window_h=800,
            scroll_ms=scroll_ms,
            scroll_mode=effective_mode,
            cmod_bpm=cmod_bpm,
            xmod_value=xmod_value,
            osu_speed=osu_speed,
            skin=prefs['skin'],
            press_hide=prefs['press_hide'],
            xml_judgments=xml_judgments,
            keycount=keycount,
            **player_kwargs,
        )

    def _set_initial_play_rate(self, play_rate) -> None:
        try:
            rate = float(play_rate or 1.0)
        except (TypeError, ValueError):
            rate = 1.0

        self.player.play_rate = max(0.1, min(4.0, rate))

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = PlayerCanvas(self.player,
                                 swap_paced=self._legacy_hz is None)
        self.view.installEventFilter(self)
        layout.addWidget(self.view, 1)

        layout.addLayout(self._build_transport_bar())

    def _build_transport_bar(self) -> QHBoxLayout:
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
        self.playbar.setRange(0, self.SLIDER_MAX)
        self.playbar.setValue(0)
        self.playbar.valueChanged.connect(self._on_playbar_changed)
        self.playbar.sliderPressed.connect(self._on_playbar_pressed)
        self.playbar.sliderReleased.connect(self._on_playbar_released)
        bar.addWidget(self.playbar, 1)

        self.dur_lbl = QLabel('0:00')
        self.dur_lbl.setMinimumWidth(44)
        bar.addWidget(self.dur_lbl)

        return bar

    def _build_frame_loop(self) -> None:
        """Wire the per-displayed-frame chores and, in legacy mode only,
        a wall-clock repaint timer.

        Default mode has no timer: the canvas self-schedules paints from
        `frameSwapped` while playing, and every paint's swap drives
        `_on_frame` (playbar, labels, end-of-chart, audio drift sync).
        While paused nothing animates, so repaints come from input
        handlers on demand and `_on_frame` still follows each one."""
        self.view.frameSwapped.connect(self._on_frame)
        settings_signals.player_setting_changed.connect(
            self._on_setting_changed)

        self.timer = None
        if self._legacy_hz is None:
            return
        self.timer = QTimer(self)
        interval = (0 if self._legacy_hz == 0
                    else max(1, int(round(1000.0 / self._legacy_hz))))
        self.timer.setInterval(interval)
        try:
            self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        except Exception:
            pass
        self.timer.timeout.connect(self.view.update)
        self.timer.start()

    def _build_audio(self, audio_path) -> None:
        if not audio_path:
            return

        prefs = load_player_settings(self.player.game)
        pitch_correct = prefs['pitch_correct']

        from analysis.gui.loaders import Worker

        def job(_progress):
            from analysis.player.audio import AudioEngine
            return AudioEngine(audio_path, pitch_correct=pitch_correct)

        worker = Worker(job)
        self._audio_worker = worker
        worker.done.connect(self._on_audio_built)
        worker.failed.connect(self._on_audio_failed)
        worker.start()

    def _on_audio_built(self, engine) -> None:
        self._audio_worker = None
        self._audio = engine
        self.audio_state.ready = bool(self._audio.ready)

        if self.audio_state.ready:
            self._audio.prewarm_rates([0.8, 0.9, 1.1, 1.2, 1.3, 1.5])

        if self.audio_state.ready and self._audio._base_duration:
            self._refresh_audio_chart_offset_rate()
            self.player.t_max = max(
                self.player.t_max,
                self._audio_to_chart_time(float(self._audio._base_duration)),
            )
            self._audio.seek(self._chart_to_audio_time(self.player.t_intended))
            self.player.attach_audio_clock(self._audio_current_chart_time)
            self.player.attach_audio_status(self._audio.callback_status_snapshot)
            self._sync_audio(force=True)
            self.view.update()

    def _on_audio_failed(self, tb) -> None:
        self._audio_worker = None
        self.audio_state.ready = False
        self._audio = None
        print(f'audio: failed to initialize\n{tb}')

    def _connect_player_events(self) -> None:
        self.player.events.on('scroll_changed', self._on_scroll_change)
        self.player.events.on('hud_action', self._on_hud_action)

    def _build_input_router(self) -> None:
        from analysis.gui.region import InputRouter, SidebarRegion, LanesRegion

        self.input_router = InputRouter()
        self.input_router.add(SidebarRegion(self.player))
        self.input_router.add(LanesRegion(self.player, self._seek))

    def _start_playing(self) -> None:
        self.player.paused = False
        self.play_btn.setText('⏸')
        self._sync_audio(force=True)
        _sync_gc_to_playback(True)
        self.view.update()

    # ------------------------------------------------------------------
    # Render timer
    # ------------------------------------------------------------------

    @staticmethod
    def _env_flag(name: str) -> bool | None:
        raw = os.environ.get(name)
        if raw is None:
            return None

        value = raw.strip().lower()
        if value in ('1', 'true', 'yes', 'on'):
            return True
        if value in ('0', 'false', 'no', 'off'):
            return False
        return None

    def _legacy_render_hz(self) -> int | None:
        """Opt-in escape hatch for compositors where `frameSwapped`
        pacing misbehaves (broken vsync throttling, VMs). Returns None
        for the default swap-paced loop, 0 for an uncapped timer, or a
        timer rate in Hz."""
        prefs = load_player_settings(self.player.game)
        env_uncapped = self._env_flag('VSRG_RENDER_UNCAPPED')
        uncapped = (bool(prefs.get('render_uncapped', False))
                    if env_uncapped is None else env_uncapped)
        if uncapped:
            return 0

        raw = os.environ.get('VSRG_RENDER_HZ')
        if raw is None:
            return None
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Audio / transport
    # ------------------------------------------------------------------

    def _audio_is_ready(self) -> bool:
        return bool(self.audio_state.ready)

    def _current_audio_chart_offset(self) -> float:
        offset = self._audio_chart_offset_s
        if self._audio_chart_offset_scales_with_rate:
            offset *= self._audio_chart_offset_rate
        return offset

    def _refresh_audio_chart_offset_rate(self) -> float:
        self._audio_chart_offset_rate = float(self.player.play_rate)
        return self._audio_chart_offset_rate

    def _chart_to_audio_time(self, chart_t: float) -> float:
        return float(chart_t) - self._current_audio_chart_offset()

    def _audio_to_chart_time(self, audio_t: float) -> float:
        return float(audio_t) + self._current_audio_chart_offset()

    def _audio_current_chart_time(self) -> float:
        return self._audio_to_chart_time(self._audio.current_chart_time())

    def _audio_playing_state(self) -> tuple[float, float, bool]:
        rate = self._refresh_audio_chart_offset_rate()
        return (
            self._chart_to_audio_time(self.player.t_intended),
            rate,
            bool(not self.player.paused),
        )

    def _sync_audio(self, *, force: bool = False) -> None:
        self.player._ui_status = {
            'audio_ready': self._audio_is_ready(),
            'pitch_correct': bool(getattr(self._audio, '_pitch_correct', True)),
        }

        if not self._audio_is_ready():
            return

        state = self._audio_playing_state()
        if force or state != self.audio_state.last_sync_state:
            self._audio.set_state(*state)
            self.audio_state.last_sync_state = state

    def _toggle(self) -> None:
        seeked_to_start = self._restart_if_resuming_after_end()

        self.player.toggle_pause()
        self.play_btn.setText('▶' if self.player.paused else '⏸')

        # Seek on every resume, not just restart-from-end: blocks that
        # were already in flight when the pause landed advanced the
        # audio position slightly past the frozen visual playhead, and
        # the seek re-aligns the two so resume is snap-free.
        resumed = not self.player.paused
        if self._audio_is_ready() and (seeked_to_start or resumed):
            self._audio.seek(self._chart_to_audio_time(self.player.t_intended))

        self._sync_audio(force=True)
        _sync_gc_to_playback(not self.player.paused)
        self.view.update()

    def _restart_if_resuming_after_end(self) -> bool:
        if not self.player.paused:
            return False

        audio_done = self._audio_is_ready() and getattr(
            self._audio, '_ended', False
        )
        clock_done = self.player.t >= self.player.t_max - 1e-3

        if audio_done or clock_done:
            self.player.restart()
            return True

        return False

    def _seek(self, ds) -> None:
        self.player.seek_rel(ds)

        if self._audio_is_ready():
            self._audio.seek(self._chart_to_audio_time(self.player.t_intended))

        self._sync_audio(force=True)
        self.view.update()

    def _nudge_rate(self, delta) -> None:
        self.player.nudge_rate(delta)
        self._sync_audio(force=True)
        self.view.update()

    def _toggle_mute(self) -> None:
        if not self._audio_is_ready():
            return

        self.audio_state.muted = not self.audio_state.muted
        self._audio.set_volume(0.0 if self.audio_state.muted else 0.5)

    # ------------------------------------------------------------------
    # Settings / HUD actions
    # ------------------------------------------------------------------

    def _sync_settings_toggles(self) -> None:
        prefs = load_player_settings(self.player.game)

        if prefs['press_hide'] != self.player.press_hide:
            self.player.set_press_hide(prefs['press_hide'])

        if prefs['skin'] != self.player.skin:
            self.player.set_skin(prefs['skin'])

    def _on_scroll_change(self) -> None:
        self._sync_audio(force=True)
        save_player_setting('scroll_mode', self.player.scroll_mode)

    def _on_hud_action(self, action, payload) -> None:
        handlers = {
            'toggle_sv': self._handle_toggle_sv,
            'cycle_skin': self._handle_cycle_skin,
            'toggle_press_hide': self._handle_toggle_press_hide,
            'toggle_pitch': self._handle_toggle_pitch,
            'edit_scroll_value': self._open_scroll_edit,
        }

        handler = handlers.get(action)
        if handler is not None:
            handler(payload)

    def _handle_toggle_sv(self, _payload=None) -> None:
        self.player.toggle_sv()

    def _handle_cycle_skin(self, _payload=None) -> None:
        self.player.toggle_skin()
        save_player_setting('skin', self.player.skin)

    def _handle_toggle_press_hide(self, _payload=None) -> None:
        self.player.toggle_press_hide()
        save_player_setting('press_hide', self.player.press_hide)

    def _handle_toggle_pitch(self, _payload=None) -> None:
        if not self._audio_is_ready():
            return

        enabled = not self._audio._pitch_correct
        self._audio.set_pitch_correct(enabled)
        save_player_setting('pitch_correct', enabled)
        self._sync_audio()

    # ------------------------------------------------------------------
    # Scroll-value editor overlay
    # ------------------------------------------------------------------

    def _open_scroll_edit(self, rect) -> None:
        from analysis.player import scroll as scroll_registry

        self._close_scroll_edit()

        rx, ry, rw, rh = rect
        origin = self.view.mapTo(self, self.view.rect().topLeft())

        edit = QLineEdit(self)
        edit.setText(self._current_scroll_value_text(scroll_registry))
        edit.setGeometry(origin.x() + rx, origin.y() + ry, rw, rh)
        edit.selectAll()
        edit.returnPressed.connect(self._apply_scroll_edit)
        edit.editingFinished.connect(self._close_scroll_edit)
        edit.show()
        edit.setFocus(Qt.MouseFocusReason)

        self.scroll_edit = edit

    def _current_scroll_value_text(self, scroll_registry) -> str:
        mode = scroll_registry.get(self.player.scroll_mode)
        value = self.player._current_mode_value()

        if mode and mode.format_value:
            return mode.format_value(value).split(' ', 1)[0]

        return f'{value:.2f}'

    def _apply_scroll_edit(self) -> None:
        if self.scroll_edit is None:
            return

        raw = self.scroll_edit.text().strip()
        try:
            value = float(raw)
        except ValueError:
            self._close_scroll_edit()
            return

        self.player._set_current_mode_value(value)
        self._close_scroll_edit()
        self.view.update()

    def _close_scroll_edit(self) -> None:
        if self.scroll_edit is None:
            return

        widget = self.scroll_edit
        self.scroll_edit = None
        widget.deleteLater()

    # ------------------------------------------------------------------
    # Playbar
    # ------------------------------------------------------------------

    def _t_to_slider(self, t) -> int:
        tmin, tmax = self.player.t_min, self.player.t_max
        if tmax <= tmin:
            return 0

        frac = (t - tmin) / (tmax - tmin)
        return int(round(frac * self.SLIDER_MAX))

    def _slider_to_t(self, value) -> float:
        tmin, tmax = self.player.t_min, self.player.t_max
        return tmin + (value / float(self.SLIDER_MAX)) * (tmax - tmin)

    def _on_playbar_pressed(self) -> None:
        self.scrub.active = True
        self.scrub.resume_after_release = not self.player.paused

        if not self._audio_is_ready():
            return

        rate = self._refresh_audio_chart_offset_rate()
        audio_t = self._chart_to_audio_time(self.player.t_intended)
        self.player.attach_audio_clock(None)
        self._audio.set_state(
            audio_t,
            rate,
            False,
        )
        self.audio_state.last_sync_state = (
            audio_t,
            rate,
            False,
        )

    def _on_playbar_released(self) -> None:
        self.scrub.active = False
        self.player.t = self._slider_to_t(self.playbar.value())

        if not self._audio_is_ready():
            return

        rate = self._refresh_audio_chart_offset_rate()
        audio_t = self._chart_to_audio_time(self.player.t_intended)
        self._audio.seek(audio_t)
        self._audio.set_state(
            audio_t,
            rate,
            self.scrub.resume_after_release,
        )
        self.audio_state.last_sync_state = (
            audio_t,
            rate,
            bool(self.scrub.resume_after_release),
        )
        self.player.attach_audio_clock(self._audio_current_chart_time)
        self.view.update()

    def _on_playbar_changed(self, value) -> None:
        if self.scrub.suppress_slider:
            return

        self.player.t = self._slider_to_t(value)

        if not self.playbar.isSliderDown():
            self._sync_audio(force=True)
        self.view.update()

    def _update_playbar(self) -> None:
        if self.scrub.active:
            return

        self.scrub.suppress_slider = True
        try:
            self.playbar.setValue(self._t_to_slider(self.player.t))
        finally:
            self.scrub.suppress_slider = False

    # ------------------------------------------------------------------
    # Tick/update
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_time(t) -> str:
        sign = '-' if t < 0 else ''
        t = abs(t)
        return f'{sign}{int(t // 60)}:{int(t % 60):02d}'

    def _finish_playback_if_needed(self) -> None:
        if self.player.paused:
            return

        audio_done = self._audio_is_ready() and getattr(
            self._audio, '_ended', False
        )
        clock_done = self.player.t >= self.player.t_max

        if audio_done or clock_done:
            self.player.paused = True
            self.play_btn.setText('▶')
            _sync_gc_to_playback(False)

    def _on_frame(self) -> None:
        """Runs once per presented frame (connected to the canvas's
        `frameSwapped`). Everything here must early-out cheaply; at
        144 Hz this is hot-path."""
        self._finish_playback_if_needed()
        self._update_playbar()
        self._update_time_labels()

        if not self.scrub.active:
            self._sync_audio()

    def _update_time_labels(self) -> None:
        # QLabel.setText re-layouts even for identical text, so compare
        # against the last-set string ourselves.
        now_text = self._fmt_time(self.player.t)
        if now_text != self._time_text:
            self._time_text = now_text
            self.time_lbl.setText(now_text)

        dur_text = self._fmt_time(self.player.t_max)
        if dur_text != self._dur_text:
            self._dur_text = dur_text
            self.dur_lbl.setText(dur_text)

    def _on_setting_changed(self, _name: str) -> None:
        self._sync_settings_toggles()
        self.view.update()

    # ------------------------------------------------------------------
    # Input events
    # ------------------------------------------------------------------

    def resizeEvent(self, ev) -> None:
        width, height = self.view.width(), self.view.height()
        if width > 0 and height > 0:
            self.player.resize(width, height)
        super().resizeEvent(ev)

    def eventFilter(self, obj, ev) -> bool:
        if obj is not self.view:
            return super().eventFilter(obj, ev)

        event_type = ev.type()

        if event_type == ev.Type.KeyPress:
            return self._handle_key_press(ev)

        if event_type == ev.Type.Wheel:
            return self._handle_wheel(ev)

        if event_type == ev.Type.MouseButtonPress:
            return self._handle_mouse_press(ev)

        if event_type == ev.Type.MouseMove:
            return self._handle_mouse_move(ev)

        if event_type == ev.Type.MouseButtonRelease:
            return self._handle_mouse_release(ev)

        return super().eventFilter(obj, ev)

    @staticmethod
    def _event_xy(ev) -> tuple[int, int]:
        pos = ev.position() if hasattr(ev, 'position') else ev.pos()
        return int(pos.x()), int(pos.y())

    def _handle_key_press(self, ev) -> bool:
        key = ev.key()
        shift = bool(ev.modifiers() & Qt.ShiftModifier)

        handlers = {
            Qt.Key_Space: self._toggle,
            Qt.Key_P: self._toggle,
            Qt.Key_Left: lambda: self._seek(-10 if shift else -2),
            Qt.Key_Right: lambda: self._seek(10 if shift else 2),
            Qt.Key_Up: lambda: self.player.nudge_scroll(1.15),
            Qt.Key_Down: lambda: self.player.nudge_scroll(1 / 1.15),
            Qt.Key_Plus: lambda: self._nudge_rate(0.1),
            Qt.Key_Equal: lambda: self._nudge_rate(0.1),
            Qt.Key_Minus: lambda: self._nudge_rate(-0.1),
            Qt.Key_M: self._toggle_mute,
            Qt.Key_R: self._restart_from_keyboard,
            Qt.Key_Q: self._close_window,
            Qt.Key_Escape: self._close_window,
        }

        if key == Qt.Key_Tab and shift:
            self.player.toggle_edit_mode()
            self.view.update()
            return True

        handler = handlers.get(key)
        if handler is None:
            return False

        handler()
        self.view.update()
        return True

    def _restart_from_keyboard(self) -> None:
        self.player.restart()
        self._sync_audio(force=True)

    def _close_window(self) -> None:
        self.window().close()

    def _handle_wheel(self, ev) -> bool:
        x, y = self._event_xy(ev)
        handled = self.input_router.dispatch_wheel(
            x,
            y,
            ev.angleDelta().y(),
            ev.modifiers(),
        )
        if handled:
            self.view.update()
        return handled

    def _handle_mouse_press(self, ev) -> bool:
        if ev.button() != Qt.LeftButton:
            return False

        self.view.setFocus(Qt.MouseFocusReason)
        x, y = self._event_xy(ev)

        handled = self.input_router.dispatch_mouse_down(
            x,
            y,
            ev.button(),
            ev.modifiers(),
        )
        if handled:
            self.view.update()

        return handled

    def _handle_mouse_move(self, ev) -> bool:
        x, y = self._event_xy(ev)

        handled = self.input_router.dispatch_mouse_move(
            x,
            y,
            ev.buttons(),
            ev.modifiers(),
        )
        if handled:
            self.view.update()

        return handled

    def _handle_mouse_release(self, ev) -> bool:
        if ev.button() != Qt.LeftButton:
            return False

        x, y = self._event_xy(ev)

        handled = self.input_router.dispatch_mouse_up(
            x,
            y,
            ev.button(),
            ev.modifiers(),
        )
        if handled:
            self.view.update()

        return handled

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        if self.timer is not None:
            self.timer.stop()
        _sync_gc_to_playback(False)

        worker = getattr(self, '_audio_worker', None)
        if worker is not None and worker.isRunning():
            worker.quit()
            worker.wait(2000)

        if not self._audio_is_ready():
            return

        try:
            self._audio.stop()
        except Exception:
            pass

    def closeEvent(self, ev) -> None:
        self.cleanup()
        super().closeEvent(ev)
