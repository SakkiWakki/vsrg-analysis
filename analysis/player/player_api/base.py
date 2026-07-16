from __future__ import annotations


class Player:
    SCROLL_MODE_MS = 'ms'
    SCROLL_MODE_CMOD = 'cmod'
    SCROLL_MODE_OSU = 'osu'
    SCROLL_MODE_XMOD = 'xmod'
    SCROLL_MODE_LINEAR = SCROLL_MODE_MS

    REFERENCE_FIELD_H = 480.0
    SKINS = ('bar', 'circle')

    def __init__(
        self,
        replay,
        game='etterna',
        od=8,
        ett_judge='J4',
        bpms=None,
        sm_offset=0.0,
        audio_path=None,
        window_w=900,
        window_h=900,
        headless=False,
        scroll_ms=400.0,
        scroll_mode=None,
        cmod_bpm=600.0,
        xmod_value=1.0,
        osu_speed=20,
        skin='bar',
        press_hide=False,
        xml_judgments=None,
        keycount=None,
    ):
        self.headless = bool(headless)
        self.W = int(window_w)
        self.H = int(window_h)
        self.replay = replay
        self.game = game
        self.audio_path = audio_path
        self.xml_judgments = dict(xml_judgments or {})

        self._install_controllers()

        self._load_replay_arrays(
            replay,
            game,
            bpms=bpms,
            sm_offset=sm_offset,
            keycount=keycount,
        )
        self._init_judge(od, ett_judge)
        self.init_state.init_palette()

        self._init_scroll_state(
            scroll_ms=scroll_ms,
            cmod_bpm=cmod_bpm,
            xmod_value=xmod_value,
            osu_speed=osu_speed,
            bpms=bpms,
            scroll_mode=scroll_mode,
        )
        self._init_playback_state(skin=skin, press_hide=press_hide)

        self.init_state.init_notes_model(replay)
        self.max_draw_pad_sec = self._compute_max_draw_pad()

        self._init_sv(replay)
        self._build_cumulative_sv()
        self._build_ghost_sv_caches()

        self._init_side_systems()

    def _install_controllers(self) -> None:
        from analysis.player.init.init_state import PlayerInitState
        from analysis.player.playback.playback import PlaybackController
        from analysis.player.scroll.scroll_state import ScrollStateController
        from analysis.player.sv.render import SvRenderController
        from analysis.player.input.layout_edit import LayoutEditController
        from analysis.player.input.hud_actions import HudActionController

        self.init_state = PlayerInitState(self)
        self.playback = PlaybackController(self)
        self.scroll_state = ScrollStateController(self)
        self.sv_render = SvRenderController(self)
        self.layout_edit = LayoutEditController(self)
        self.hud_actions = HudActionController(self)

    # ---------- init / judge / side systems ----------

    def _load_replay_arrays(self, replay, game, *, bpms, sm_offset, keycount=None):
        return self.init_state.load_replay_arrays(
            replay,
            game,
            bpms=bpms,
            sm_offset=sm_offset,
            keycount=keycount,
        )

    def _init_judge(self, od, ett_judge):
        return self.init_state.init_judge(od, ett_judge)

    def _apply_judge(self):
        return self.init_state.apply_judge()

    def _compute_max_draw_pad(self):
        return self.init_state.compute_max_draw_pad()

    def _init_side_systems(self):
        return self.init_state.init_side_systems()

    def nudge_judge(self, delta):
        return self.init_state.nudge_judge(delta)

    # ---------- playback ----------

    def _init_playback_state(self, *, skin, press_hide):
        return self.playback.init(skin=skin, press_hide=press_hide)

    @property
    def t(self) -> float:
        return self.playback.t

    @t.setter
    def t(self, value: float) -> None:
        self.playback.t = value

    @property
    def paused(self) -> bool:
        return self.playback.paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self.playback.paused = value

    @property
    def play_rate(self) -> float:
        return self.playback.play_rate

    @play_rate.setter
    def play_rate(self, value: float) -> None:
        self.playback.play_rate = value

    @property
    def t_min(self) -> float:
        return self.playback.t_min

    @property
    def t_max(self) -> float:
        return self.playback.t_max

    @t_max.setter
    def t_max(self, value: float) -> None:
        self.playback.t_max = value

    def attach_audio_clock(self, getter) -> None:
        return self.playback.attach_audio_clock(getter)

    def attach_audio_status(self, getter) -> None:
        """Register a `() -> (count, last_status_str)` callable so the
        Frame Analyzer panel (and anything else with audio diagnostics)
        can read PortAudio's underflow / overflow signals without
        needing a direct handle to the audio engine. `count > 0` means
        we've been missing the audio-callback deadline."""
        self._audio_status_getter = getter

    def audio_status_snapshot(self) -> tuple[int, str]:
        getter = getattr(self, '_audio_status_getter', None)
        if getter is None:
            return 0, ''
        try:
            count, last = getter()
            return int(count), str(last)
        except Exception:
            return 0, ''

    @property
    def t_intended(self) -> float:
        return self.playback.t_intended

    def advance(self, dt_s):
        return self.playback.advance(dt_s)

    def tick(self, dt_s):
        return self.playback.tick(dt_s)

    def restart(self):
        return self.playback.restart()

    def seek_rel(self, dt):
        return self.playback.seek_rel(dt)

    def toggle_pause(self):
        return self.playback.toggle_pause()

    def _seek(self, dt):
        return self.playback.seek(dt)

    def _toggle_pause(self):
        return self.playback.toggle_pause()

    def nudge_rate(self, d):
        return self.playback.nudge_rate(d)

    def set_skin(self, skin):
        return self.playback.set_skin(skin)

    def toggle_skin(self):
        return self.playback.toggle_skin()

    def set_press_hide(self, on):
        return self.playback.set_press_hide(on)

    def toggle_press_hide(self):
        return self.playback.toggle_press_hide()

    # ---------- scroll ----------

    def _init_scroll_state(self, *, scroll_ms, cmod_bpm, xmod_value,
                           osu_speed, bpms, scroll_mode):
        return self.scroll_state.init(
            scroll_ms=scroll_ms,
            cmod_bpm=cmod_bpm,
            xmod_value=xmod_value,
            osu_speed=osu_speed,
            bpms=bpms,
            scroll_mode=scroll_mode,
        )

    def _mode(self, key=None):
        return self.scroll_state.mode(key)

    def _state(self, key=None):
        return self.scroll_state.state(key)

    def _pxps_from_unit(self, mode_key, value, options=None):
        return self.scroll_state.pxps_from_unit(mode_key, value, options)

    def _unit_from_pxps(self, mode_key, pxps, options=None):
        return self.scroll_state.unit_from_pxps(mode_key, pxps, options)

    def _current_mode_value(self):
        return self.scroll_state.current_mode_value()

    def _set_current_mode_value(self, value):
        return self.scroll_state.set_current_mode_value(value)

    def get_mode_option(self, mode_key, option_key, default=None):
        return self.scroll_state.get_mode_option(mode_key, option_key, default)

    def set_mode_option(self, mode_key, option_key, value):
        return self.scroll_state.set_mode_option(mode_key, option_key, value)

    @property
    def scroll_speed(self):
        return self.scroll_state.scroll_speed

    @property
    def effective_scroll_ms(self):
        return self.scroll_state.effective_scroll_ms

    def set_scroll_ms(self, ms):
        return self.scroll_state.set_scroll_ms(ms)

    def set_scroll_mode(self, mode):
        return self.scroll_state.set_scroll_mode(mode)

    def nudge_scroll(self, factor):
        return self.scroll_state.nudge_scroll(factor)

    def _available_mode_keys(self):
        return self.scroll_state.available_mode_keys()

    def set_game(self, game: str) -> None:
        return self.scroll_state.set_game(game)

    def cycle_game(self) -> None:
        return self.scroll_state.cycle_game()

    # ---------- SV / render projection ----------

    def _init_sv(self, replay):
        return self.sv_render.init(replay)

    @property
    def sv_sections(self):
        return self.sv_render.sv_sections

    def _build_cumulative_sv(self):
        return self.sv_render.build_cumulative_sv()

    def _times_to_sv(self, times):
        return self.sv_render.times_to_sv(times)

    def _build_ghost_sv_caches(self):
        return self.sv_render.build_ghost_sv_caches()

    def _cumulative_sv_at(self, t):
        return self.sv_render.cumulative_sv_at(t)

    def render_frame_state(self, raw_t):
        return self.sv_render.render_frame_state(raw_t)

    def render_at(self, raw_t):
        return self.sv_render.render_at(raw_t)

    def debug_log_sv_frame(self, ctx) -> None:
        return self.sv_render.debug_log_sv_frame(ctx)

    def _sv_distance(self, t_from, t_to):
        return self.sv_render.sv_distance(t_from, t_to)

    def _visual_sv_distance_from_frame(self, frame, t_to):
        return self.sv_render.visual_sv_distance_from_frame(frame, t_to)

    def _time_to_y(self, t, t_now, frame=None):
        return self.sv_render.time_to_y(t, t_now, frame)

    def batch_time_to_y(self, times, frame, groups=None):
        return self.sv_render.batch_time_to_y(times, frame, groups=groups)

    def _reset_render_timeline(self):
        return self.sv_render.reset_render_timeline()

    def _reset_render_playhead(self, raw_t=None):
        return self.sv_render.reset_render_playhead(raw_t)

    def toggle_sv(self):
        return self.sv_render.toggle_sv()

    def sv_suspended(self) -> bool:
        return self.sv_render.sv_suspended()

    # ---------- HUD actions / layout edit ----------

    def handle_mouse_down(self, x, y):
        return self.hud_actions.handle_mouse_down(x, y)

    def _dispatch_hud_action(self, action, payload):
        return self.hud_actions.dispatch(action, payload)

    def _cycle_scroll_mode(self):
        return self.hud_actions.cycle_scroll_mode()

    def _notify_scroll_change(self):
        return self.hud_actions.notify_scroll_change()

    def _notify_hud_action(self, action, payload):
        return self.hud_actions.notify_hud_action(action, payload)

    def handle_mouse_move(self, x, y):
        return self.layout_edit.handle_mouse_move(x, y)

    def handle_mouse_up(self, x, y):
        return self.layout_edit.handle_mouse_up(x, y)

    def _begin_drag_section(self, key, x, y, grab_rect):
        return self.layout_edit.begin_drag_section(key, x, y, grab_rect)

    def _begin_resize_section(self, key, x, y):
        return self.layout_edit.begin_resize_section(key, x, y)

    def _apply_resize(self, x, y):
        return self.layout_edit.apply_resize(x, y)

    def _finish_drag(self, x, y):
        return self.layout_edit.finish_drag(x, y)

    def _place_in_panel(self, key, y, region):
        return self.layout_edit.place_in_panel(key, y, region)

    def _place_in_free_region(self, key, x, y):
        return self.layout_edit.place_in_free_region(key, x, y)

    def toggle_edit_mode(self):
        return self.layout_edit.toggle_edit_mode()

    # ---------- misc ----------

    @property
    def chart_rect(self):
        """`(x, y, w, h)` of the replay viewport: the window minus the
        sidebar. Everything replay-visual -- lane geometry, effects
        (transforms, backgrounds, clips) -- is positioned and clipped
        against this rect, never the raw window."""
        from analysis.player.render import theme
        return (0, 0, max(0, self.W - theme.SIDEBAR_WIDTH), self.H)

    def _lane_geom(self):
        margin = 60
        _cx, _cy, chart_w, _chart_h = self.chart_rect
        avail = chart_w - 2 * margin
        lane_w = min(90, max(50, avail / self.keycount))
        total = lane_w * self.keycount
        x0 = margin + (avail - total) / 2
        return x0, lane_w

    def resize(self, w, h):
        self.W, self.H = max(200, int(w)), max(200, int(h))

    def run(self):
        raise RuntimeError('Player.run() was replaced by the Qt player UI.')
