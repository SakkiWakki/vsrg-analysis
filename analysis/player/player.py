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
import numpy as np
from pathlib import Path

from analysis.viz.plots import col_colors
from analysis.player.plugin_loader import PluginManager
from analysis.player import scroll as scroll_registry


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
    from analysis.core import game as game_mod
    name = 'osu' if replay.get('chart_path') else 'etterna'
    return game_mod.get(name).prepare_replay_times(
        replay, bpms=bpms, sm_offset=sm_offset)


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
                 xml_judgments=None):
        self.headless = headless
        self.W, self.H = window_w, window_h

        # XML-sourced aggregate judgments from Etterna.xml's TapNoteScores:
        # includes HitMine / AvoidMine and per-window W1..W5 counts. The
        # .bin replay can't tell us which mines were hit, so this dict is
        # the only place the sidebar can surface mine-hit info from.
        self.xml_judgments = dict(xml_judgments or {})

        self.replay = replay
        from analysis.core import game as game_mod
        self._adapter = game_mod.get(game)
        self.times, self.hold_tails, self.keycount = (
            self._adapter.prepare_replay_times(
                replay, bpms=bpms, sm_offset=sm_offset))
        # Per-LN release offset (seconds, signed). osu!mania only — Etterna
        # .bin has no release timing.
        self.hold_release_offsets = {}
        if game == 'osu':
            for entry in replay.get('hold_releases') or []:
                start_t, col, _end_t, off_ms = entry
                if off_ms is not None:
                    self.hold_release_offsets[(start_t, col)] = off_ms / 1000.0
        self.columns = replay['columns']
        self.offsets = replay['offsets']
        self.misses = replay['misses']
        self.notetypes = replay['notetypes']
        # Parallel to self.misses — True when the miss had an actual press
        # (head_off is the real offset, not the 1.0s sentinel). Lets the
        # renderer draw a red hit-line at the press time so the user can
        # see WHERE the player mispressed. Only osu populates this.
        mp = replay.get('miss_pressed')
        if mp is not None and len(mp) == len(self.misses):
            self.miss_pressed = np.asarray(mp, dtype=bool)
        else:
            self.miss_pressed = np.zeros(len(self.misses), dtype=bool)

        # Adapters own the judge-system. The adapter declares which
        # kwarg it expects ('od' for osu, 'judge' for Etterna); we stash
        # the current value in self._active_judge so nudge_judge can
        # round-trip it back to the adapter without branching here.
        judge_kw = self._adapter.judge_kwarg_name()
        self._active_judge = (od if judge_kw == 'od' else ett_judge)
        self._apply_judge(rebuild_counts=False)
        self.judge_colors = JCLR
        self.game = game
        self.palette = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
                        for c in col_colors(self.keycount)]

        # Scroll state: every mode the registry knows about gets its own
        # sub-dict in `self._mode_state`, holding the native scalar, a copy
        # of the mode's default options, plus whatever lifecycle hooks want
        # to stash there (e.g. CMOD's SV-suspend bookkeeping). All scroll
        # logic (speed lookup, nudge, mode switch) routes through the
        # registry so adding a mode is a registry write, not a branch here.
        scroll_registry.ensure_loaded()
        self._mode_state: dict[str, dict] = {}
        for m in scroll_registry.all_modes():
            self._mode_state[m.key] = {
                'value': m.default_value,
                'options': dict(m.options),
            }
        # Seed defaults from legacy kwargs so existing call-sites (library
        # tab, tests) don't need to change: scroll_ms → ms, cmod_bpm →
        # cmod, osu_speed → osu.
        self._mode_state[self.SCROLL_MODE_MS]['value'] = float(scroll_ms)
        if self.SCROLL_MODE_CMOD in self._mode_state:
            self._mode_state[self.SCROLL_MODE_CMOD]['value'] = float(cmod_bpm)
        if self.SCROLL_MODE_OSU in self._mode_state:
            self._mode_state[self.SCROLL_MODE_OSU]['value'] = max(
                0.1, min(60.0, float(osu_speed)))
        # XMOD reference BPM: picked from the first bpm in the chart if any,
        # else 120. XMOD's to_pxps reads this off the player.
        if bpms:
            try:
                self._xmod_reference_bpm = float(bpms[0][1])
            except (IndexError, TypeError, ValueError):
                self._xmod_reference_bpm = 120.0
        else:
            self._xmod_reference_bpm = 120.0

        # Back-compat: accept the old 'linear' spelling from saved settings.
        if scroll_mode == 'linear':
            scroll_mode = self.SCROLL_MODE_MS
        # Mode must be compatible with the current game. A saved 'cmod' from
        # an Etterna session is not a valid choice when opening an osu replay,
        # even though both modes are globally registered.
        if not scroll_mode or not scroll_registry.is_compatible(scroll_mode,
                                                                 game):
            scroll_mode = scroll_registry.default_for_game(game)
        self.scroll_mode = scroll_mode
        self.play_rate = 1.0
        self.paused = True
        self.t = 0.0                 # current playback time (s)
        self._last_tick = None
        self.audio_path = audio_path

        # Precompute judgment colors
        self.note_judges = []
        for off, mi in zip(self.offsets, self.misses):
            self.note_judges.append(judge(off, self.windows, mi))

        self.t_max = float(self.times[-1]) + 5.0 if len(self.times) else 10.0
        self.t_min = -2.0
        self.hit_line_y_frac = 0.80  # judgment line position
        self.skin = skin if skin in self.SKINS else 'bar'
        # Press-hide mode: once a note is actually pressed (t_now >= press_t)
        # stop drawing it. For LNs, hide everything once past release_t.
        # Missed notes stay visible so the red X is still informative.
        self.press_hide = bool(press_hide)

        # All per-note streams (LN tails, ghost taps/holds) live on
        # self.notes now. The renderer/culling code still reads them via
        # the compatibility properties below so callers aren't scattered.
        from analysis.player.notes_model import (build_notes_model,
                                                  link_miss_holds)
        self.notes = build_notes_model(replay, self.times,
                                        self.hold_tails, game)
        link_miss_holds(self.notes, self.offsets, self.misses,
                        self.miss_pressed)

        # Culling pad: how far beyond a note's head/tail its drawn strokes
        # (press mark, release guide) can reach. Without this, a note
        # scrolls off the window while the tail of its line is still
        # on-screen, and the line pops out.
        off_abs = float(np.max(np.abs(self.offsets))) if self.offsets.size else 0.0
        rel_abs = max((abs(v) for v in self.hold_release_offsets.values()),
                      default=0.0)
        self.max_draw_pad_sec = max(off_abs, rel_abs)

        # SV (scroll velocity) — list of (time_sec, sv_multiplier). When enabled,
        # note positions use the piecewise-constant integral of SV over time
        # (see _cumulative_sv_at). Falls back to constant scroll if empty.
        if sv_sections is None:
            sv_sections = replay.get('sv_sections') or []
        self.sv_sections = list(sv_sections)
        self.sv_enabled = bool(self.sv_sections)
        # Fire the active mode's on_enter so e.g. CMOD's SV-suspend runs on
        # the initial load too, not just on subsequent mode switches.
        mode_desc = scroll_registry.get(self.scroll_mode)
        if mode_desc and mode_desc.on_enter:
            mode_desc.on_enter(self, self._mode_state[self.scroll_mode])
        self._build_cumulative_sv()
        self._build_ghost_sv_caches()
        self.plugins = PluginManager.discover()
        # HUD-overlay state (sidebar scroll, panel toggles, hitboxes).
        # Separated from replay state so the player core doesn't grow
        # every time the HUD does; see analysis/player/hud_state.py.
        from analysis.player.hud.hud_state import HudState
        self.hud = HudState()
        # Event bus for host/plugin notifications. Kinds documented in
        # analysis/player/events.py; the Qt tab subscribes to
        # ``scroll_changed`` and ``hud_action``.
        from analysis.player.events import EventBus
        self.events = EventBus()
        # Tab-owned runtime flags the painted HUD needs to render toggle
        # labels without holding a reference to the AudioEngine.
        # Keys: 'audio_ready' (bool), 'pitch_correct' (bool). Populated by
        # the Qt tab each time it syncs audio.
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
        distance. Dividing by play_rate matches Etterna CMOD (ArrowEffects.cpp
        fBPS = BPM/60/musicRate): at higher rate, the same chart-second
        covers fewer pixels."""
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
        callbacks on the registered mode handle side effects — e.g. CMOD's
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

    def _lane_geom(self):
        margin_l = 60
        margin_r = 220  # right panel for stats/legend
        avail = self.W - margin_l - margin_r
        lane_w = min(90, max(50, avail / self.keycount))
        total = lane_w * self.keycount
        x0 = margin_l + (avail - total) / 2
        return x0, lane_w

    def _build_cumulative_sv(self):
        """Precompute cumulative SV integral at each section boundary for O(log n) lookup."""
        self._sv_times = [s[0] for s in self.sv_sections]
        self._sv_values = [s[1] for s in self.sv_sections]
        n = len(self.sv_sections)
        self._sv_cumulative = [0.0] * n
        for i in range(1, n):
            dt = self._sv_times[i] - self._sv_times[i - 1]
            self._sv_cumulative[i] = self._sv_cumulative[i - 1] + dt * self._sv_values[i - 1]
        # Cache sv-distance for each note time — lets us bisect directly by
        # on-screen position regardless of how wildly SV stretches the chart.
        self._note_sv_cum = np.array(
            [self._cumulative_sv_at(float(t)) for t in self.times],
            dtype=np.float64)

    def _build_ghost_sv_caches(self):
        """Cache ghost overlay times in the same SV-space used for note
        culling. Writes into self.notes so every per-note stream stays
        colocated."""
        m = self.notes
        if m.ghost_times.size:
            m.ghost_sv_times = np.array(
                [self._cumulative_sv_at(float(t)) for t in m.ghost_times],
                dtype=np.float64)
        else:
            m.ghost_sv_times = np.empty(0, dtype=np.float64)

        if m.miss_hold_press.size:
            m.miss_hold_press_sv = np.array(
                [self._cumulative_sv_at(float(t))
                 for t in m.miss_hold_press],
                dtype=np.float64)
            m.miss_hold_release_sv = np.array(
                [self._cumulative_sv_at(float(t))
                 for t in m.miss_hold_release],
                dtype=np.float64)
            m.miss_hold_max_sv_dur = float(
                np.max(m.miss_hold_release_sv - m.miss_hold_press_sv))
        else:
            m.miss_hold_press_sv = np.empty(0, dtype=np.float64)
            m.miss_hold_release_sv = np.empty(0, dtype=np.float64)
            m.miss_hold_max_sv_dur = 0.0

    def _cumulative_sv_at(self, t):
        """Integral of SV(t') dt' from the first timing point to t.
        For t before the first section, extrapolate with the first SV value."""
        if not self._sv_times:
            return t
        idx = bisect.bisect_right(self._sv_times, t) - 1
        if idx < 0:
            return (t - self._sv_times[0]) * self._sv_values[0]
        return self._sv_cumulative[idx] + (t - self._sv_times[idx]) * self._sv_values[idx]

    def _sv_distance(self, t_from, t_to):
        """SV-weighted time delta (returns plain delta if SV is off/empty).
        `sv_enabled` is the single source of truth; CMOD's on_enter forces
        it off (matching Etterna's ArrowEffects.cpp CMOD branch, which
        never calls GetDisplayedSpeedPercent). ms/xmod/osu modes layer SV
        on top when enabled."""
        if not self.sv_enabled or not self.sv_sections:
            return t_to - t_from
        return self._cumulative_sv_at(t_to) - self._cumulative_sv_at(t_from)

    def _time_to_y(self, t, t_now):
        judge_y = self.H * self.hit_line_y_frac
        return judge_y - self._sv_distance(t_now, t) * self.scroll_speed

    def toggle_sv(self):
        """Toggle SV. If the current mode has suspended SV via on_enter (its
        state dict has a sv_enabled_saved slot), flip that saved value so
        the user's intent survives leaving the mode — but don't actually
        enable SV while still in a mode that ignores it."""
        if not self.sv_sections:
            return False
        st = self._state()
        if 'sv_enabled_saved' in st:
            st['sv_enabled_saved'] = not st['sv_enabled_saved']
            return False
        self.sv_enabled = not self.sv_enabled
        return self.sv_enabled

    def sv_suspended(self) -> bool:
        """True when the active mode has force-disabled SV (e.g. CMOD)."""
        return 'sv_enabled_saved' in self._state()

    def handle_mouse_down(self, x, y):
        for rect, action, payload in reversed(self.hud.hitboxes):
            rx, ry, rw, rh = rect
            if not (rx <= x <= rx + rw and ry <= y <= ry + rh):
                continue
            # Edit-mode drag/resize grabs are consumed here regardless
            # of the general "edit swallows clicks" rule below.
            if action == 'begin_drag_section' and self.hud.edit_mode:
                self._begin_drag_section(payload, x, y, rect)
                return True
            if action == 'begin_resize_section' and self.hud.edit_mode:
                self._begin_resize_section(payload, x, y)
                return True
            # Edit mode swallows every non-edit action so drags don't
            # accidentally fire click handlers underneath the cursor.
            # Only the edit-mode toggle itself still works, so the
            # header button can turn editing back off.
            if self.hud.edit_mode and action != 'toggle_edit_mode':
                return True
            if action == 'toggle_plugin_panel':
                self.hud.plugin_panel_open = not self.hud.plugin_panel_open
                return True
            if action == 'toggle_plugin':
                self.plugins.toggle_enabled(payload)
                return True
            if action == 'scroll_nudge':
                self.nudge_scroll(payload)
                self._notify_scroll_change()
                return True
            if action == 'cycle_scroll_mode':
                keys = self._available_mode_keys()
                if not keys:
                    return True
                cur = self.scroll_mode if self.scroll_mode in keys else keys[0]
                self.set_scroll_mode(keys[(keys.index(cur) + 1) % len(keys)])
                self._notify_scroll_change()
                return True
            if action == 'rate_nudge':
                self.nudge_rate(payload)
                self._notify_scroll_change()
                return True
            if action == 'judge_nudge':
                self.nudge_judge(payload)
                self._notify_scroll_change()
                return True
            if action == 'cycle_game':
                self.cycle_game()
                self._notify_scroll_change()
                return True
            if action == 'toggle_layer':
                from analysis.config import get_config
                cfg = get_config()
                path = f'player.layer_visibility.{payload}'
                cfg.set(path, not bool(cfg.get(path, True)))
                return True
            if action == 'toggle_layers_panel':
                self.hud.layers_panel_open = not getattr(
                    self.hud, 'layers_panel_open', False)
                return True
            if action == 'toggle_flyout':
                # One-at-a-time: clicking the open flyout's header closes
                # it; clicking a different header swaps.
                self.hud.open_flyout = (
                    None if self.hud.open_flyout == payload else payload)
                return True
            if action == 'toggle_edit_mode':
                self.toggle_edit_mode()
                return True
            # Toggles / click-to-edit rects: logical state lives elsewhere
            # (Qt tab owns audio + QSettings), so just notify subscribers.
            if action in ('toggle_sv', 'cycle_skin', 'toggle_press_hide',
                          'toggle_pitch', 'edit_scroll_value'):
                self._notify_hud_action(action, payload)
                return True
        return False

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
            from analysis.core import game as game_mod
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
        from analysis.player.render import theme
        section = None
        for s in self.plugins.sidebar.all_sections():
            if s.key == key:
                section = s
                break
        if section is None:
            return
        rect = self.plugins.sidebar.section_free_rect(
            section, self.W, self.H)
        _x, _y, w, h = rect
        self.hud.resize_key = key
        self.hud.resize_origin = (int(x), int(y))
        self.hud.resize_origin_size = (int(w), int(h))
        # Store min sizes in theme so free-region resize can clamp.
        _ = theme.FREE_MIN_W  # touch to ensure imported on first use

    def handle_mouse_move(self, x, y):
        """While in edit mode with an active drag/resize, track the
        cursor so the renderer can draw the ghost / update the rect.
        Returns True if the event was consumed, i.e. the canvas should
        schedule a repaint."""
        if self.hud.drag_key is not None:
            self.hud.drag_pointer = (int(x), int(y))
            return True
        if self.hud.resize_key is not None:
            ox, oy = self.hud.resize_origin
            ow, oh = self.hud.resize_origin_size
            from analysis.player.render import theme
            new_w = max(theme.FREE_MIN_W, ow + (int(x) - ox))
            new_h = max(theme.FREE_MIN_H, oh + (int(y) - oy))
            # Find current rect's (x, y) to preserve top-left — resize
            # grows toward bottom-right only.
            for s in self.plugins.sidebar.all_sections():
                if s.key == self.hud.resize_key:
                    rx, ry, _w, _h = self.plugins.sidebar.section_free_rect(
                        s, self.W, self.H)
                    self.plugins.sidebar.set_section_free_rect(
                        s.key, rx, ry, new_w, new_h)
                    break
            return True
        return False

    def handle_mouse_up(self, x, y):
        """End an active drag/resize, routing the drop to either the
        sidepanel or the free region. Returns True when consumed."""
        if self.hud.drag_key is not None:
            self._finish_drag(int(x), int(y))
            return True
        if self.hud.resize_key is not None:
            self.hud.resize_key = None
            return True
        return False

    def _finish_drag(self, x, y):
        key = self.hud.drag_key
        from analysis.player.render import theme
        # Sidepanel drop zone = anywhere at or right of the sidebar's
        # left edge. Everything else = free region.
        sidebar_x = self.W - theme.SIDEBAR_WIDTH
        dropped_in_sidepanel = x >= sidebar_x
        if dropped_in_sidepanel:
            self.plugins.sidebar.set_section_region(key, 'sidepanel')
            # Recompute order: find the two adjacent sidepanel sections
            # at cursor Y and set order to midpoint. When there are no
            # neighbors we leave the declared priority intact.
            new_order = self._compute_drop_order(y)
            if new_order is not None:
                self.plugins.sidebar.set_section_order(key, new_order)
        else:
            self.plugins.sidebar.set_section_region(key, 'free')
            # Translate the cursor back to the rect's top-left using
            # the grab offset so the drop lands where the user sees
            # the ghost.
            dx, dy = self.hud.drag_offset
            # Look up the section's size (from saved rect if any, else
            # the plugin default).
            section = None
            for s in self.plugins.sidebar.all_sections():
                if s.key == key:
                    section = s
                    break
            if section is not None:
                _rx, _ry, w, h = self.plugins.sidebar.section_free_rect(
                    section, self.W, self.H)
                new_x = max(0, min(self.W - w, x - dx))
                new_y = max(0, min(self.H - h, y - dy))
                self.plugins.sidebar.set_section_free_rect(
                    key, new_x, new_y, w, h)
        self.hud.drag_key = None
        self.hud.drag_origin_region = None

    def _compute_drop_order(self, y):
        """Given a drop cursor Y, return the new ``order`` value for
        the dragged section so it lands between the two sidepanel
        neighbors nearest ``y``. Uses the midpoint between their
        ``section_order`` values and the *previous frame's* painted
        rects so the insertion point lines up with what the user saw.
        Returns None when there are no other sidepanel sections."""
        reg = self.plugins.sidebar
        # Non-pinned sidepanel neighbors — pinned-bottom sections live
        # in a fixed band and don't participate in reordering.
        others = [s for s in reg.all_sections()
                  if s.enabled
                  and reg.section_region(s.key) == 'sidepanel'
                  and s.key != self.hud.drag_key
                  and not s.pin_bottom]
        if not others:
            return None
        # Pair each neighbor with its last-frame Y-midpoint; fall back
        # to section_order when the rect isn't available (first frame
        # after enabling edit mode).
        rects = self.hud.frame_sidepanel_rects or {}
        pairs = []
        for s in others:
            r = rects.get(s.key)
            y_mid = (r[1] + r[3] / 2) if r else None
            pairs.append((s, y_mid, reg.section_order(s)))
        pairs.sort(key=lambda p: p[2])
        ordered_vals = [p[2] for p in pairs]
        first = ordered_vals[0] - 10.0
        last = ordered_vals[-1] + 10.0
        # Use the pixel-accurate split when we have rects; otherwise
        # fall back to the cursor-fraction heuristic.
        have_rects = all(p[1] is not None for p in pairs)
        if have_rects:
            for i, p in enumerate(pairs):
                if y < p[1]:
                    if i == 0:
                        return first
                    return (ordered_vals[i - 1] + ordered_vals[i]) / 2.0
            return last
        frac = max(0.0, min(1.0, y / max(1, self.H)))
        idx = int(round(frac * len(pairs)))
        if idx <= 0:
            return first
        if idx >= len(pairs):
            return last
        return (ordered_vals[idx - 1] + ordered_vals[idx]) / 2.0

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
        if not self.paused:
            self.t = max(self.t_min, self.t + dt_s * self.play_rate)

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
        self._apply_judge(rebuild_counts=True)

    def _apply_judge(self, *, rebuild_counts):
        """Recompute windows + label + per-note judgments from the
        adapter, using self._active_judge. Called once during
        construction (rebuild_counts=False: note_judges built right
        after) and every time nudge_judge changes the value."""
        kw = {self._adapter.judge_kwarg_name(): self._active_judge}
        self.windows = self._adapter.judgement_windows(self.replay, **kw)
        self.judge_label = self._adapter.judge_label(self.replay, **kw)
        if rebuild_counts:
            self.note_judges = [judge(off, self.windows, mi)
                                for off, mi in zip(self.offsets, self.misses)]

    def restart(self):
        self.t = self.t_min

    def _seek(self, dt):
        self.t = max(self.t_min, min(self.t_max, self.t + dt))

    def _toggle_pause(self):
        self.paused = not self.paused

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
