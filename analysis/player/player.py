"""Real-time replay player. Scrollable chart with note + press marks, hit window
shading, pause/seek/speed, audio sync. Works for Etterna and osu!mania.

Controls:
  Space / P       pause/resume
  Left / Right    seek by 2s (hold Shift: 10s)
  Up / Down       scroll speed +/-
  + / -           playback speed +/-
  [ / ]           scroll position when paused
  R               reset to start
  M               toggle music
  Q / Esc         quit
"""
import sys
import math
import bisect
import os
import numpy as np
from pathlib import Path

from analysis.viz.plots import col_colors
from analysis.player.plugin_loader import PluginManager
from analysis.player import scroll as scroll_registry
from analysis.player.render import theme
from analysis.player.sv_debug import LOGGER as _SV_DEBUG_LOGGER
from analysis.core import game as game_mod
from analysis.config import get_config
from analysis.components.api import REGION_FREE


from analysis.player.judgment import JCLR, judge

# Back-compat re-exports. Old imports (analyze, tests) reach into
# `analysis.player.player` for these; the real definitions now live on
# each game's adapter. Delete this shim once nothing outside imports
# them.
def etterna_windows_for(judge_name='J4'):
    from analysis.games.etterna.judgment import windows_for
    return windows_for(judge_name)


def osu_mania_windows(od):
    from analysis.games.osu.judgment import windows_for
    return windows_for(od)


def prepare_replay_times(replay, bpms=None, sm_offset=0.0):
    """Back-compat shim. The canonical path is
    `GameAdapter.prepare_replay_times`; this wrapper picks the adapter
    from `replay['chart_path']` (osu) or falls back to Etterna."""
    name = 'osu' if replay.get('chart_path') else 'etterna'
    return game_mod.get(name).prepare_replay_times(
        replay, bpms=bpms, sm_offset=sm_offset)


def _rect_contains(rect, x, y):
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


# Synthetic slot size around the end of the sidepanel section order,
# used when a drop lands before-all or after-all existing sections.
_ORDER_GAP = 10.0


def _compute_drop_order(y, targets, window_h):
    """Return the order value for a drop at cursor Y among a list of
    (y_mid_or_None, order) target pairs produced by
    SidebarSectionRegistry.reorder_targets(). Returns None when targets
    is empty. Pure function: no sidebar or HUD state."""
    if not targets:
        return None

    mids = [m for m, _ in targets]
    orders = [o for _, o in targets]

    have_rects = all(m is not None for m in mids)
    if have_rects:
        slot = next((i for i, m in enumerate(mids) if y < m), len(mids))
    else:
        frac = max(0.0, min(1.0, y / max(1, window_h)))
        slot = max(0, min(len(mids), int(round(frac * len(mids)))))

    bounds = [orders[0] - _ORDER_GAP, *orders, orders[-1] + _ORDER_GAP]
    return (bounds[slot] + bounds[slot + 1]) / 2.0


def _osu_hold_release_offsets(replay, game):
    """Per-LN release offsets (seconds, signed) for osu replays.
    Etterna has no release timing in its .bin format so returns empty."""
    if game != 'osu':
        return {}
    offsets = {}
    for entry in replay.get('hold_releases') or []:
        start_t, col, _end_t, off_ms = entry
        if off_ms is not None:
            offsets[(start_t, col)] = off_ms / 1000.0
    return offsets


def _normalize_miss_pressed(raw, n_misses):
    """Normalize the optional miss_pressed array to a bool array of
    length n_misses. Only osu populates this field; all other games
    get an all-False array so downstream code has a uniform shape."""
    if raw is not None and len(raw) == n_misses:
        return np.asarray(raw, dtype=bool)
    return np.zeros(n_misses, dtype=bool)


def _first_bpm_or_default(bpms, *, default):
    """Return the first BPM value from a chart's BPM list, or `default`
    if bpms is empty or malformed. Used to seed XMOD's reference BPM."""
    if not bpms:
        return default
    try:
        return float(bpms[0][1])
    except (IndexError, TypeError, ValueError):
        return default


class Player:
    # Scroll modes live in analysis.player.scroll; each game's adapter.py
    # registers its own (CMOD/XMOD for Etterna, osu! for osu!mania, etc.).
    # The core 'ms' mode (time-from-top-to-judgment) is always registered.
    SCROLL_MODE_MS = 'ms'
    SCROLL_MODE_CMOD = 'cmod'
    SCROLL_MODE_OSU = 'osu'
    SCROLL_MODE_XMOD = 'xmod'
    # Back-compat alias: old persisted settings may still say 'linear'.
    SCROLL_MODE_LINEAR = SCROLL_MODE_MS
    # Reference field height shared by Etterna (Til Death / fallback theme
    # ScreenHeight=480) and osu!mania stable (480-tall logical playfield).
    # Scroll modes express their formulas in this coordinate system and the
    # Player scales to window H so the fraction-of-screen-per-second matches
    # the source game — making cross-game comparisons like Etterna C952 ≈
    # osu SS 30 hold (empirically matching cmodcalc.com's SS×31.75 mapping).
    REFERENCE_FIELD_H = 480.0
    SKINS = ('bar', 'circle')


    def __init__(self, replay, game='etterna', od=8, ett_judge='J4',
                 bpms=None, sm_offset=0.0, audio_path=None,
                 window_w=900, window_h=900, headless=False,
                 sv_sections=None, scroll_ms=400.0, scroll_mode=None,
                 cmod_bpm=600.0, osu_speed=20, skin='bar', press_hide=False,
                 xml_judgments=None, keycount=None):
        # XML-sourced aggregate judgments from Etterna.xml's TapNoteScores:
        # includes HitMine / AvoidMine and per-window W1..W5 counts. The
        # .bin replay can't tell us which mines were hit, so this dict is
        # the only place the sidebar can surface mine-hit info from.
        self.headless = headless
        self.W, self.H = window_w, window_h
        self.replay = replay
        self.game = game
        self.audio_path = audio_path
        self.xml_judgments = dict(xml_judgments or {})

        self._load_replay_arrays(replay, game, bpms=bpms, sm_offset=sm_offset,
                                 keycount=keycount)
        self._init_judge(od, ett_judge)
        self.palette = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
                        for c in col_colors(self.keycount)]

        self._init_scroll_state(scroll_ms=scroll_ms, cmod_bpm=cmod_bpm,
                                osu_speed=osu_speed, bpms=bpms,
                                scroll_mode=scroll_mode)
        self._init_playback_state(skin=skin, press_hide=press_hide)

        from analysis.player.notes_model import (build_notes_model,
                                                 link_miss_holds)
        self.notes = build_notes_model(replay, self.times,
                                       self.hold_tails, game)
        link_miss_holds(self.notes, self.offsets, self.misses,
                        self.miss_pressed)
        self.max_draw_pad_sec = self._compute_max_draw_pad()

        self._init_sv(sv_sections, replay)
        self._build_cumulative_sv()
        self._build_ghost_sv_caches()

        self._init_side_systems()

    def _load_replay_arrays(self, replay, game, *, bpms, sm_offset,
                            keycount=None):
        """Delegate to the adapter for noterow-to-time, bind the per-note
        arrays, and normalize the osu-only hold-release and miss-pressed
        streams so downstream code sees a uniform shape."""
        self._adapter = game_mod.get(game)
        self.times, self.hold_tails, self.keycount = (
            self._adapter.prepare_replay_times(
                replay, bpms=bpms, sm_offset=sm_offset, keycount=keycount))
        self.columns = replay['columns']
        self.offsets = replay['offsets']
        self.misses = replay['misses']
        self.notetypes = replay['notetypes']
        self.hold_release_offsets = _osu_hold_release_offsets(replay, game)
        self.miss_pressed = _normalize_miss_pressed(
            replay.get('miss_pressed'), len(self.misses))

    def _init_judge(self, od, ett_judge):
        """Bind the initial judge value from whichever kwarg the adapter
        expects (od for osu, judge-level for Etterna) and compute the
        starting windows + per-note judgments."""
        judge_kw = self._adapter.judge_kwarg_name()
        self._active_judge = (od if judge_kw == 'od' else ett_judge)
        self._apply_judge()
        self.judge_colors = JCLR

    def _init_scroll_state(self, *, scroll_ms, cmod_bpm, osu_speed,
                           bpms, scroll_mode):
        """Seed self._mode_state from the scroll registry, override the
        defaults from legacy kwargs, pick the XMOD reference BPM, and
        resolve an active mode that's compatible with the current game."""
        scroll_registry.ensure_loaded()
        self._mode_state: dict[str, dict] = {
            m.key: {'value': m.default_value, 'options': dict(m.options)}
            for m in scroll_registry.all_modes()
        }
        self._mode_state[self.SCROLL_MODE_MS]['value'] = float(scroll_ms)
        if self.SCROLL_MODE_CMOD in self._mode_state:
            self._mode_state[self.SCROLL_MODE_CMOD]['value'] = float(cmod_bpm)
        if self.SCROLL_MODE_OSU in self._mode_state:
            self._mode_state[self.SCROLL_MODE_OSU]['value'] = max(
                0.1, min(60.0, float(osu_speed)))

        self._xmod_reference_bpm = _first_bpm_or_default(bpms, default=120.0)

        if scroll_mode == 'linear':
            scroll_mode = self.SCROLL_MODE_MS
        if not scroll_mode or not scroll_registry.is_compatible(
                scroll_mode, self.game):
            scroll_mode = scroll_registry.default_for_game(self.game)
        self.scroll_mode = scroll_mode

    def _init_playback_state(self, *, skin, press_hide):
        """Bind timeline bounds, skin, press-hide toggle, and the
        transient play-head state (rate, paused, t). Chart time lives in a
        ChartClock singleton (self._clock) — self.t/self.paused/self.rate
        are properties backed by it. When an audio engine is attached
        later, the clock switches to audio-master so there's no
        render-vs-audio drift to resync."""
        from analysis.player.chart_clock import ChartClock
        self._last_tick = None
        t_max = (float(self.times[-1]) + 5.0
                 if len(self.times) else 10.0)
        t_min = -2.0
        self._clock = ChartClock(initial=0.0, t_min=t_min, t_max=t_max)
        self._render_timeline = None
        self.hit_line_y_frac = 0.80
        self.skin = skin if skin in self.SKINS else 'bar'
        # Press-hide: once t_now >= press_t stop drawing that note (and
        # for LNs, everything past release_t). Missed notes stay visible
        # so the red X is still informative.
        self.press_hide = bool(press_hide)

    # -- playhead proxies over self._clock -----------------------------
    # Kept as properties so existing callers (renderer, plugins, tests,
    # components backend) keep reading/writing the same attribute names.

    @property
    def t(self) -> float:
        return self._clock.now()

    @t.setter
    def t(self, value: float) -> None:
        t = float(value)
        self._clock.seek(t)
        self._reset_render_timeline()

    @property
    def paused(self) -> bool:
        return self._clock.paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self._clock.set_paused(bool(value))
        self._reset_render_timeline()

    @property
    def play_rate(self) -> float:
        return self._clock.rate

    @play_rate.setter
    def play_rate(self, value: float) -> None:
        self._clock.set_rate(float(value))
        self._reset_render_timeline()

    @property
    def t_min(self) -> float:
        return self._clock.t_min

    @property
    def t_max(self) -> float:
        return self._clock.t_max

    @t_max.setter
    def t_max(self, value: float) -> None:
        self._clock.set_bounds(self._clock.t_min, float(value))

    def attach_audio_clock(self, getter) -> None:
        """Register an audio-time getter with the chart clock so the
        playhead follows the audio engine's actual source position. Pass
        `None` to detach (stop, or fall back to wall-clock during scrubbing)."""
        self._clock.set_audio_source(getter)
        self._reset_render_timeline()

    @property
    def t_intended(self) -> float:
        """Chart time the Player has asked for (wall-anchor), independent
        of what the audio engine has actually reached. The GUI sends this
        to `AudioEngine.set_state` so seeks written via `self.t = X` take
        effect even though `self.t` reads the PV's actual position back."""
        return self._clock.intended()

    def _compute_max_draw_pad(self):
        """Culling pad: how far beyond a note's head/tail its drawn
        strokes (press mark, release guide) can reach. Without this a
        note scrolls off the window while the tail of its line is still
        on-screen, and the line pops out."""
        off_abs = (float(np.max(np.abs(self.offsets)))
                   if self.offsets.size else 0.0)
        rel_abs = max((abs(v) for v in self.hold_release_offsets.values()),
                      default=0.0)
        return max(off_abs, rel_abs)

    def _init_sv(self, sv_sections, replay):
        """Ask the adapter for an SVEngine; fall back to IdentitySVEngine when
        the chart has no SV. `sv_sections` is retained as a constructor
        override for callers (mostly tests) that still supply explicit
        time-space sections — they get a TimeSpaceSVEngine directly. The
        active scroll mode's on_enter fires here too so CMOD's SV-suspend
        applies on initial load, not just on later mode switches."""
        from analysis.player.sv_engine import (IdentitySVEngine,
                                                TimeSpaceSVEngine)
        from analysis.player.render_playhead import RenderTimeline
        if sv_sections is not None:
            self._sv_engine = (TimeSpaceSVEngine(list(sv_sections))
                               if sv_sections else IdentitySVEngine())
        else:
            engine = self._adapter.build_sv_engine(replay)
            self._sv_engine = engine or IdentitySVEngine()
        self.sv_enabled = bool(getattr(self._sv_engine, 'enabled', False))
        # ChartClock stays a raw chart-time source. Rendering is a pure sample
        # of chart time through the SV engine's projection.
        self._clock.set_sv_engine(self._sv_engine)
        self._render_timeline = RenderTimeline(self._sv_engine)

        mode_desc = scroll_registry.get(self.scroll_mode)
        if mode_desc and mode_desc.on_enter:
            mode_desc.on_enter(self, self._mode_state[self.scroll_mode])

    @property
    def sv_sections(self):
        """Back-compat view: `[(time_sec, multiplier)]` projection of the
        active SV engine. Consumed by sidebar/components/plugin APIs that
        only need to display or count SV — not for positioning, which
        goes through `_sv_distance`/`_time_to_y` and hits the engine."""
        return self._sv_engine.as_sections()

    def _init_side_systems(self):
        """Bring up the plugin manager, HUD state, event bus, and the
        ui-status flags the painted HUD reads."""
        from analysis.player.hud.hud_state import HudState
        from analysis.player.events import EventBus
        self.plugins = PluginManager.discover()
        self.hud = HudState()
        self.events = EventBus()
        # Tab-owned runtime flags the painted HUD needs to render toggle
        # labels without holding a reference to the AudioEngine.
        # Keys: 'audio_ready' (bool), 'pitch_correct' (bool). Populated
        # by the Qt tab each time it syncs audio.
        self._ui_status: dict = {'audio_ready': False, 'pitch_correct': True}

    # --- Scroll abstraction ------------------------------------------------
    # All scroll logic is delegated to analysis.player.scroll; this Player
    # only knows about the registry key and a state dict per mode. Adding a
    # new mode means adding a scroll.register(...) call in a game's
    # adapter.py — no branches here.

    # --- NotesModel compat properties -------------------------------------
    # The renderer and culling layer read per-note streams by these old
    # names. Keep them as thin read-through properties so the refactor
    # didn't have to touch every call site. Add new code against
    # `self.notes.*` directly.
    # TODO: Depreciate 
    @property
    def _noterows_list(self): return self.notes.noterows_list
    @property
    def _columns_list(self): return self.notes.columns_list
    @property
    def _ln_tail_times(self): return self.notes.ln_tail_times
    @property
    def _ln_indices(self): return self.notes.ln_indices
    @property
    def _ghost_times(self): return self.notes.ghost_times
    @property
    def _ghost_cols(self): return self.notes.ghost_cols
    @property
    def _miss_hold_ln_heads_ms(self): return self.notes.miss_hold_ln_heads_ms
    @property
    def _miss_hold_press(self): return self.notes.miss_hold_press
    @property
    def _miss_hold_release(self): return self.notes.miss_hold_release
    @property
    def _miss_hold_cols(self): return self.notes.miss_hold_cols
    @property
    def _miss_hold_max_dur(self): return self.notes.miss_hold_max_dur
    @property
    def _ghost_sv_times(self): return self.notes.ghost_sv_times
    @property
    def _miss_hold_press_sv(self): return self.notes.miss_hold_press_sv
    @property
    def _miss_hold_release_sv(self): return self.notes.miss_hold_release_sv
    @property
    def _miss_hold_max_sv_dur(self): return self.notes.miss_hold_max_sv_dur
    @property
    def _miss_first_hold(self): return self.notes.miss_first_hold
    @property
    def _miss_head_suppressed(self): return self.notes.miss_head_suppressed
    @property
    def _mine_times(self): return self.notes.mine_times
    @property
    def _mine_cols(self): return self.notes.mine_cols
    @property
    def _lift_times(self): return self.notes.lift_times
    @property
    def _lift_cols(self): return self.notes.lift_cols
    @property
    def _fake_times(self): return self.notes.fake_times
    @property
    def _fake_cols(self): return self.notes.fake_cols
    @property
    def _roll_head_keys(self): return self.notes.roll_head_keys

    def _mode(self, key=None):
        return scroll_registry.get(key or self.scroll_mode)

    def _state(self, key=None):
        return self._mode_state[key or self.scroll_mode]

    def _pxps_from_unit(self, mode_key, value, options=None):
        m = scroll_registry.get(mode_key)
        if m is None:
            return 0.0
        opts = options if options is not None else self._mode_state.get(
            mode_key, {}).get('options', m.options)
        return m.to_pxps(float(value), opts, self)

    def _unit_from_pxps(self, mode_key, pxps, options=None):
        m = scroll_registry.get(mode_key)
        if m is None:
            return 0.0
        opts = options if options is not None else self._mode_state.get(
            mode_key, {}).get('options', m.options)
        return m.from_pxps(float(pxps), opts, self)

    def _current_mode_value(self):
        return self._state()['value']

    def _set_current_mode_value(self, value):
        m = self._mode()
        if m is None:
            return
        lo, hi = m.value_bounds
        self._state()['value'] = max(lo, min(hi, float(value)))

    def get_mode_option(self, mode_key, option_key, default=None):
        return self._mode_state.get(mode_key, {}).get('options', {}).get(
            option_key, default)

    def set_mode_option(self, mode_key, option_key, value):
        st = self._mode_state.get(mode_key)
        if st is None or option_key not in st['options']:
            return
        st['options'][option_key] = value

    @property
    def scroll_speed(self):
        """Effective px/sec for converting *chart-time* deltas to on-screen
        distance.

        Rate is normalized here, not in the per-mode to_pxps functions. A mode
        defines its value as visual (wall-clock) scroll speed at rate=1, and
        that visual must stay invariant across play_rate. Dividing by play_rate
        shrinks the chart-time pxps so wall-clock pxps is rate-independent:
        the same pixels flow past in the same wall-clock second regardless of
        rate.

        When writing a new game's scroll mode, return pxps at rate=1 and do
        not divide by rate yourself; the Player applies it for every mode.

        Examples: "osu 30" and "Quaver 15" describe a fixed on-screen look
        that holds at any rate. Etterna CMOD normalizes the same way
        (ArrowEffects.cpp fBPS = BPM/60/musicRate)."""
        pxps = self._pxps_from_unit(self.scroll_mode, self._current_mode_value())
        return pxps / max(0.01, self.play_rate)

    @property
    def effective_scroll_ms(self):
        """ms from the top of the screen to the judgment line, computed from
        whichever mode is active. Useful for HUD display and for a unified UI."""
        sps = max(0.001, self.scroll_speed)
        return (self.H * self.hit_line_y_frac) / sps * 1000.0

    def set_scroll_ms(self, ms):
        """Set scroll speed expressed as ms-to-judgment. Routes through the
        abstraction so e.g. switching mode while keeping visual speed works
        uniformly."""
        ms = max(50.0, min(3000.0, float(ms)))
        pxps = self._pxps_from_unit(self.SCROLL_MODE_MS, ms)
        self._set_current_mode_value(
            self._unit_from_pxps(self.scroll_mode, pxps))

    def set_skin(self, skin):
        self.skin = skin if skin in self.SKINS else 'bar'

    def toggle_skin(self):
        names = list(self.SKINS)
        idx = names.index(self.skin) if self.skin in names else 0
        self.set_skin(names[(idx + 1) % len(names)])

    def set_press_hide(self, on):
        self.press_hide = bool(on)

    def toggle_press_hide(self):
        self.press_hide = not self.press_hide
        return self.press_hide

    def set_scroll_mode(self, mode):
        """Switch scroll modes while preserving visual px/sec. Lifecycle
        callbacks on the registered mode handle side effects - e.g. CMOD's
        on_enter suspends SV and stashes the prior state in the mode's
        `_mode_state` entry; on_exit restores it."""
        if mode == 'linear':
            mode = self.SCROLL_MODE_MS
        if mode not in self._mode_state or mode == self.scroll_mode:
            return
        # Per-game modes only apply when the current game matches. Rejecting
        # here (rather than silently reinterpreting) catches plugin bugs that
        # would otherwise leave the HUD showing e.g. CMOD while game=osu.
        if not scroll_registry.is_compatible(mode, self.game):
            return

        pxps = self._pxps_from_unit(self.scroll_mode, self._current_mode_value())

        prev_mode = self._mode(self.scroll_mode)
        if prev_mode and prev_mode.on_exit:
            prev_mode.on_exit(self, self._mode_state[self.scroll_mode])

        self.scroll_mode = mode
        self._set_current_mode_value(self._unit_from_pxps(mode, pxps))

        new_mode = self._mode(mode)
        if new_mode and new_mode.on_enter:
            new_mode.on_enter(self, self._mode_state[mode])
        self._reset_render_timeline()

    def _lane_geom(self):
        margin_l = 60
        margin_r = 220  # right panel for stats/legend
        avail = self.W - margin_l - margin_r
        lane_w = min(90, max(50, avail / self.keycount))
        total = lane_w * self.keycount
        x0 = margin_l + (avail - total) / 2
        return x0, lane_w

    def _build_cumulative_sv(self):
        """Build the per-note SV-space cache the culling bisects against."""
        self._note_sv_cum = self._sv_engine.project_times(self.times)

    def _times_to_sv(self, times):
        """Project an array of chart times through the active SV engine into
        the cumulative-SV space used for note culling."""
        return self._sv_engine.project_times(np.asarray(times))

    def _build_ghost_sv_caches(self):
        """Cache ghost overlay times in the same SV-space used for note
        culling. Writes into self.notes so every per-note stream stays
        colocated."""
        m = self.notes
        m.ghost_sv_times = self._times_to_sv(m.ghost_times)

        m.miss_hold_press_sv = self._times_to_sv(m.miss_hold_press)
        m.miss_hold_release_sv = self._times_to_sv(m.miss_hold_release)
        if m.miss_hold_press.size:
            m.miss_hold_max_sv_dur = float(
                np.max(m.miss_hold_release_sv - m.miss_hold_press_sv))
        else:
            m.miss_hold_max_sv_dur = 0.0

        m.mine_sv = self._times_to_sv(m.mine_times)
        m.lift_sv = self._times_to_sv(m.lift_times)
        m.fake_sv = self._times_to_sv(m.fake_times)

    def _cumulative_sv_at(self, t):
        """Cull-space cumulative at chart time `t`. Consumed by culling.py
        to anchor the visible-note window. Matches project_times exactly."""
        return self._sv_engine.cumulative_at(float(t))

    def render_frame_state(self, raw_t):
        timeline = getattr(self, '_render_timeline', None)
        if timeline is None:
            from analysis.player.render_playhead import RenderTimeline
            timeline = self._render_timeline = RenderTimeline(self._sv_engine)
        return timeline.render_at(
            raw_t=float(raw_t),
            scroll_speed=float(self.scroll_speed),
            use_sv=bool(self.sv_enabled and self._sv_engine.enabled),
        )

    def render_at(self, raw_t):
        """Pure sampled render projection for chart time `raw_t`."""
        return self.render_frame_state(raw_t)

    def debug_log_sv_frame(self, ctx) -> None:
        if not _SV_DEBUG_LOGGER.enabled:
            return
        note_limit = max(1, int(os.environ.get('VSRG_SV_DEBUG_NOTES', '6')))
        note_idxs = list(ctx.candidates[:note_limit])
        frame = ctx.frame
        notes = []
        for i in note_idxs:
            note_t = float(self.times[i])
            cum_to = self._cumulative_sv_at(note_t)
            visual_dist = self._visual_sv_distance_from_frame(frame, note_t)
            notes.append({
                'i': int(i),
                't': note_t,
                'col': int(self._columns_list[i]),
                'cum_to': cum_to,
                'visual_dist': visual_dist,
                'y': self._time_to_y(note_t, ctx.t_now, frame),
                'sv': self._sv_engine.debug_snapshot_at(note_t),
            })
        _SV_DEBUG_LOGGER.log({
            'type': 'frame',
            'game': self.game,
            'scroll_mode': self.scroll_mode,
            'sv_enabled': bool(self.sv_enabled),
            't_now': float(ctx.t_now),
            'scroll_speed': float(self.scroll_speed),
            'frame': {
                'raw_t': float(frame.raw_t),
                'target_cum': float(frame.target_cum),
                'visual_cum_now': float(frame.visual_cum_now),
                'render_multiplier': float(frame.render_multiplier),
                'px_per_cum': float(frame.px_per_cum),
                'use_sv': bool(frame.use_sv),
            },
            'window': {
                'target_lo': float(ctx.target_lo),
                'target_hi': float(ctx.target_hi),
                'candidate_count': int(len(ctx.candidates)),
            },
            'now_sv': self._sv_engine.debug_snapshot_at(float(ctx.t_now)),
            'notes': notes,
        })

    def _sv_distance(self, t_from, t_to):
        """SV-weighted chart-time delta. Returns the plain delta when SV is
        off for any reason: engine is identity, chart has no SV, or the
        active scroll mode suspended SV (CMOD's on_enter). Single source of
        truth is `sv_enabled`, matching Etterna's ArrowEffects.cpp CMOD
        branch (which never calls GetDisplayedSpeedPercent)."""
        if not self.sv_enabled:
            return t_to - t_from
        return self._sv_engine.distance(t_from, t_to)

    def _visual_sv_distance_from_frame(self, frame, t_to):
        if not getattr(frame, 'use_sv', False):
            return t_to - frame.raw_t
        cum_to = self._cumulative_sv_at(t_to)
        return ((cum_to - frame.visual_cum_now)
                * frame.render_multiplier)

    def _time_to_y(self, t, t_now, frame=None):
        judge_y = self.H * self.hit_line_y_frac
        if frame is None:
            frame = self.render_frame_state(t_now)
        return (judge_y
                - self._visual_sv_distance_from_frame(frame, t)
                * self.scroll_speed)

    def _reset_render_timeline(self):
        # Kept so transport/UI call sites do not care whether rendering is
        # stateful. The current timeline is pure and needs no reset.
        return

    def _reset_render_playhead(self, raw_t=None):
        del raw_t
        self._reset_render_timeline()

    def toggle_sv(self):
        """Toggle SV. If the current mode has suspended SV via on_enter (its
        state dict has a sv_enabled_saved slot), flip that saved value so
        the user's intent survives leaving the mode — but don't actually
        enable SV while still in a mode that ignores it."""
        if not self._sv_engine.enabled:
            return False
        st = self._state()
        if 'sv_enabled_saved' in st:
            st['sv_enabled_saved'] = not st['sv_enabled_saved']
            self._reset_render_timeline()
            return False
        self.sv_enabled = not self.sv_enabled
        self._reset_render_timeline()
        return self.sv_enabled

    def sv_suspended(self) -> bool:
        """True when the active mode has force-disabled SV (e.g. CMOD)."""
        return 'sv_enabled_saved' in self._state()

    def handle_mouse_down(self, x, y):
        for rect, action, payload in reversed(self.hud.hitboxes):
            if not _rect_contains(rect, x, y):
                continue

            edit_mode = self.hud.edit_mode
            is_drag_grab = edit_mode and action == 'begin_drag_section'
            is_resize_grab = edit_mode and action == 'begin_resize_section'
            is_dispatchable = not edit_mode or action == 'toggle_edit_mode'

            if is_drag_grab:
                self._begin_drag_section(payload, x, y, rect)
            elif is_resize_grab:
                self._begin_resize_section(payload, x, y)
            elif is_dispatchable:
                self._dispatch_hud_action(action, payload)
            # else: edit mode swallows non-edit actions so drags don't
            # accidentally fire click handlers underneath the cursor.
            return True
        return False

    def _dispatch_hud_action(self, action, payload):
        """Run the side effect for a HUD action. Assumes the rect-hit and
        edit-mode gates in handle_mouse_down have already passed."""
        match action:
            case 'toggle_plugin_panel':
                self.hud.plugin_panel_open = not self.hud.plugin_panel_open
            case 'toggle_plugin':
                self.plugins.toggle_enabled(payload)
            case 'scroll_nudge':
                self.nudge_scroll(payload)
                self._notify_scroll_change()
            case 'cycle_scroll_mode':
                self._cycle_scroll_mode()
            case 'rate_nudge':
                self.nudge_rate(payload)
                self._notify_scroll_change()
            case 'judge_nudge':
                self.nudge_judge(payload)
                self._notify_scroll_change()
            case 'cycle_game':
                self.cycle_game()
                self._notify_scroll_change()
            case 'toggle_layer':
                self.plugins.layers.toggle(payload)
            case 'toggle_layers_panel':
                self.hud.layers_panel_open = not getattr(
                    self.hud, 'layers_panel_open', False)
            case 'toggle_flyout':
                # Clicking the open flyout's header closes it; clicking
                # a different header swaps which flyout is open.
                self.hud.open_flyout = (
                    None if self.hud.open_flyout == payload else payload)
            case 'toggle_edit_mode':
                self.toggle_edit_mode()
            case ('toggle_sv' | 'cycle_skin' | 'toggle_press_hide'
                  | 'toggle_pitch' | 'edit_scroll_value'):
                # Logical state lives elsewhere (Qt tab owns audio +
                # QSettings); this just notifies subscribers to re-read.
                self._notify_hud_action(action, payload)

    def _cycle_scroll_mode(self):
        keys = self._available_mode_keys()
        if not keys:
            return
        cur = self.scroll_mode if self.scroll_mode in keys else keys[0]
        self.set_scroll_mode(keys[(keys.index(cur) + 1) % len(keys)])
        self._notify_scroll_change()

    def _available_mode_keys(self) -> list[str]:
        """Modes visible in the scroll-type cycle for the current game.
        Core (game=None) modes like ms are always included; per-game modes
        are included when their game matches `self.game`."""
        out = []
        for m in scroll_registry.all_modes():
            if m.game is None or m.game == self.game:
                out.append(m.key)
        return out

    def set_game(self, game: str) -> None:
        """Switch the active game. If the current scroll mode is per-game
        and doesn't belong to the new game (e.g. cmod while flipping to
        osu), swap to the new game's default. Cross-game modes like ms
        stay put — the user was deliberate about choosing them."""
        if game == self.game:
            return
        self.game = game
        if not scroll_registry.is_compatible(self.scroll_mode, game):
            new_mode = scroll_registry.default_for_game(game)
            if new_mode in self._mode_state:
                self.set_scroll_mode(new_mode)

    def cycle_game(self) -> None:
        """Walk through all discovered games in registration order."""
        try:
            names = list(game_mod.all_games().keys())
        except Exception:
            return
        if not names:
            return
        cur = self.game if self.game in names else names[0]
        self.set_game(names[(names.index(cur) + 1) % len(names)])

    def _begin_drag_section(self, key, x, y, grab_rect):
        """Start a drag on the section with ``key``. ``grab_rect`` is
        the section's painted rect — we remember the offset from its
        top-left to the cursor so the ghost follows the cursor in the
        same spot the user grabbed it."""
        grab_x, grab_y, _gw, _gh = grab_rect
        self.hud.drag_key = key
        self.hud.drag_pointer = (int(x), int(y))
        self.hud.drag_origin = (int(x), int(y))
        self.hud.drag_offset = (int(x) - int(grab_x), int(y) - int(grab_y))
        # Record where this drag started so the drop handler can tell
        # "moved within sidepanel to reorder" from "moved from free to
        # sidepanel" (and vice-versa).
        self.hud.drag_origin_region = (
            self.plugins.sidebar.section_region(key))

    def _begin_resize_section(self, key, x, y):
        """Start a resize on a free-region section. The current rect
        is read off config so drag deltas translate to size changes."""
        section = self.plugins.sidebar.find_section(key)
        if section is None:
            return
        _, _, w, h = self.plugins.sidebar.section_free_rect(
            section, self.W, self.H)
        self.hud.resize_key = key
        self.hud.resize_origin = (int(x), int(y))
        self.hud.resize_origin_size = (int(w), int(h))

    def handle_mouse_move(self, x, y):
        """While in edit mode with an active drag/resize, track the
        cursor so the renderer can draw the ghost / update the rect.
        Returns True if the event was consumed, i.e. the canvas should
        schedule a repaint."""
        dragging = self.hud.drag_key is not None
        resizing = self.hud.resize_key is not None

        if dragging:
            self.hud.drag_pointer = (int(x), int(y))
        elif resizing:
            self._apply_resize(int(x), int(y))
        return dragging or resizing

    def _apply_resize(self, x, y):
        """Resize the in-flight section so its bottom-right follows the
        cursor; top-left is preserved so resize only grows outward."""
        ox, oy = self.hud.resize_origin
        ow, oh = self.hud.resize_origin_size
        new_w = max(theme.FREE_MIN_W, ow + (x - ox))
        new_h = max(theme.FREE_MIN_H, oh + (y - oy))
        section = self.plugins.sidebar.find_section(self.hud.resize_key)
        if section is None:
            return
        rx, ry, _, _ = self.plugins.sidebar.section_free_rect(
            section, self.W, self.H)
        self.plugins.sidebar.set_section_free_rect(
            section.key, rx, ry, new_w, new_h)

    def handle_mouse_up(self, x, y):
        """End an active drag/resize, routing the drop to either the
        sidepanel or the free region. Returns True when consumed."""
        dragging = self.hud.drag_key is not None
        resizing = self.hud.resize_key is not None

        if dragging:
            self._finish_drag(int(x), int(y))
        elif resizing:
            self.hud.resize_key = None
        return dragging or resizing

    def _finish_drag(self, x, y):
        key = self.hud.drag_key
        target_region = self.plugins.sidebar.region_for_x(x, self.W)

        if target_region != REGION_FREE:
            self._place_in_panel(key, y, target_region)
        else:
            self._place_in_free_region(key, x, y)

        self.hud.drag_key = None
        self.hud.drag_origin_region = None

    def _place_in_panel(self, key, y, region):
        """Dock the section to `region`, inserting its order between
        whichever neighbors bracket the cursor. When the cursor has no
        neighbors above or below, the declared priority order is preserved."""
        reg = self.plugins.sidebar
        reg.set_section_region(key, region)
        targets = reg.reorder_targets(key, region,
                                      self.hud.frame_sidepanel_rects)
        new_order = _compute_drop_order(y, targets, self.H)
        if new_order is not None:
            reg.set_section_order(key, new_order)

    def _place_in_free_region(self, key, x, y):
        """Move the section to the free region, positioning its rect's
        top-left where the drag ghost was (cursor minus grab offset),
        clamped to screen bounds. Size is preserved from the section's
        saved rect, falling back to the plugin default."""
        self.plugins.sidebar.set_section_region(key, 'free')
        section = self.plugins.sidebar.find_section(key)
        if section is None:
            return
        _, _, w, h = self.plugins.sidebar.section_free_rect(
            section, self.W, self.H)
        dx, dy = self.hud.drag_offset
        new_x = max(0, min(self.W - w, x - dx))
        new_y = max(0, min(self.H - h, y - dy))
        self.plugins.sidebar.set_section_free_rect(key, new_x, new_y, w, h)


    def toggle_edit_mode(self):
        """Flip layout-edit mode. While on, draggable sidebar
        components can be moved between the sidepanel and the free
        region; normal button actions are suppressed so drags don't
        fire clicks."""
        self.hud.edit_mode = not self.hud.edit_mode
        if not self.hud.edit_mode:
            # Clear any half-finished drag/resize so we don't leave
            # stale state when the user exits edit mode mid-gesture.
            self.hud.drag_key = None
            self.hud.drag_origin_region = None
            self.hud.resize_key = None

    def _notify_scroll_change(self):
        self.events.emit('scroll_changed')

    def _notify_hud_action(self, action, payload):
        self.events.emit('hud_action', (action, payload))

    def advance(self, dt_s):
        """No-op: chart time is driven by `self._clock`. Kept as a method
        because `_tick` and a few tests call it; the clock advances itself
        (audio-master when audio attached, wall-clock fallback otherwise)
        so explicit `dt`-based stepping is unnecessary."""
        del dt_s

    def tick(self, dt_s):
        """Advance playback state. Rendering is owned by Qt widgets."""
        self.advance(dt_s)
        return None, (self.W, self.H)

    def resize(self, w, h):
        self.W, self.H = max(200, int(w)), max(200, int(h))

    def seek_rel(self, dt):
        self._seek(dt)

    def toggle_pause(self):
        self._toggle_pause()

    def nudge_scroll(self, factor):
        """factor > 1 means faster. The active mode's nudge callback owns
        the behavior (multiplicative for CMOD/XMOD/ms, integer-step for
        osu). Values are clamped to the mode's declared bounds."""
        m = self._mode()
        if m is None or m.nudge is None:
            return
        st = self._state()
        new_val = m.nudge(st['value'], factor, st['options'])
        lo, hi = m.value_bounds
        st['value'] = max(lo, min(hi, float(new_val)))

    def nudge_rate(self, d):
        self.play_rate = max(0.1, min(4.0, self.play_rate + d))

    def nudge_judge(self, delta):
        """Ask the adapter for the next valid judge value, then rebuild
        the windows list + per-note judgments that depend on it. No-op
        if the adapter doesn't support switching."""
        new_val = self._adapter.nudge_judge(self._active_judge, delta)
        if new_val == self._active_judge:
            return
        self._active_judge = new_val
        self._apply_judge()

    def _apply_judge(self):
        """Recompute windows, label, and per-note judgments from the
        adapter using self._active_judge. Called at construction and
        whenever nudge_judge changes the value."""
        kw = {self._adapter.judge_kwarg_name(): self._active_judge}
        self.windows = self._adapter.judgement_windows(self.replay, **kw)
        self.judge_label = self._adapter.judge_label(self.replay, **kw)
        self.note_judges = [judge(off, self.windows, mi)
                            for off, mi in zip(self.offsets, self.misses)]

    def restart(self):
        self._clock.seek(self.t_min)
        self._reset_render_timeline()

    def _seek(self, dt):
        t = self.t + float(dt)
        self._clock.seek(t)
        self._reset_render_timeline()

    def _toggle_pause(self):
        self._clock.set_paused(not self._clock.paused)
        self._reset_render_timeline()

    def run(self):
        raise RuntimeError('Player.run() was replaced by the Qt player UI.')


def launch_from_replay(replay, game='etterna', od=8, bpms=None, sm_offset=0,
                       audio_path=None, sv_sections=None, scroll_ms=400.0,
                       scroll_mode=None, cmod_bpm=600.0, osu_speed=20):
    from PySide6.QtWidgets import QApplication
    from analysis.gui.player_tab import PlayerTab

    app = QApplication.instance() or QApplication(sys.argv[:1])
    tab = PlayerTab(replay, game=game, od=od, bpms=bpms,
                    sm_offset=sm_offset, audio_path=audio_path,
                    scroll_ms=scroll_ms, scroll_mode=scroll_mode,
                    cmod_bpm=cmod_bpm, osu_speed=osu_speed)
    tab.resize(1200, 900)
    tab.setWindowTitle('Replay Player')
    tab.show()
    return app.exec()


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print("usage: player.py <replay> [--osu chart.osu] [--audio file] [--od N]")
        sys.exit(1)
    path = args[0]
    od = float(args[args.index('--od') + 1]) if '--od' in args else 8

    from analysis.core.game import resolve_standalone_replay
    game, rep, bpms, sm_off, audio, _extra = resolve_standalone_replay(
        path, args=args)
    launch_from_replay(rep, game=game, od=od, bpms=bpms, sm_offset=sm_off,
                       audio_path=audio)
