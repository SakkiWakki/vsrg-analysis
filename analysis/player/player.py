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


# Judgment colors
JCLR = {
    'marv': (255, 255, 255),
    'perf': (255, 213, 79),
    'great': (129, 199, 132),
    'good': (79, 195, 247),
    'bad': (186, 104, 200),
    'miss': (229, 57, 53),
}

JUDGE_WINDOWS_ETT_J4 = [
    ('marv', 0.0225),
    ('perf', 0.045),
    ('great', 0.090),
    ('good', 0.135),
    ('bad', 0.180),
]

# Scale factor relative to J4. Matches Etterna's builtin judge table.
ETT_JUDGE_SCALES = {
    'J1': 1.50, 'J2': 1.33, 'J3': 1.16, 'J4': 1.00,
    'J5': 0.84, 'J6': 0.66, 'J7': 0.50, 'J8': 0.33,
    'JUSTICE': 0.20,
}


def etterna_windows_for(judge='J4'):
    scale = ETT_JUDGE_SCALES.get(str(judge).upper(), 1.0)
    return [(n, w * scale) for (n, w) in JUDGE_WINDOWS_ETT_J4]


def osu_mania_windows(od):
    return [
        ('marv', 16.5 / 1000.0),
        ('perf', (64 - 3 * od) / 1000.0),
        ('great', (97 - 3 * od) / 1000.0),
        ('good', (127 - 3 * od) / 1000.0),
        ('bad', (151 - 3 * od) / 1000.0),
    ]


def judge(off_s, windows, is_miss):
    if is_miss:
        return 'miss'
    a = abs(off_s)
    for name, w in windows:
        if a <= w:
            return name
    return 'miss'


def prepare_replay_times(replay, bpms=None, sm_offset=0.0):
    """Convert replay noterows to absolute times (seconds). For osu, noterows are already ms."""
    keycount = replay.get('keycount') or (int(replay['columns'].max()) + 1
                                           if len(replay['columns']) else 4)
    if replay.get('chart_path'):  # osu
        times = replay['noterows'].astype(np.float64) / 1000.0
        hold_tails = {}
        for h in replay.get('holds', []):
            if len(h) == 3 and h[2] is not None:
                hold_tails[(h[0], h[1])] = h[2] / 1000.0
        return times, hold_tails, keycount
    # Etterna: use BPM map if available
    if bpms is not None:
        from analysis.games.etterna.sm_chart import row_to_time
        times = np.array([row_to_time(int(r), bpms, sm_offset)
                          for r in replay['noterows']])
    else:
        # fallback: assume 120bpm, 48 rows/beat => 96 rows/second
        times = replay['noterows'].astype(np.float64) / 96.0
    hold_tails = {}
    for h in replay.get('holds', []):
        if len(h) == 3 and h[2] is not None:
            if bpms is not None:
                from analysis.games.etterna.sm_chart import row_to_time
                hold_tails[(h[0], h[1])] = row_to_time(int(h[2]), bpms, sm_offset)
            else:
                hold_tails[(h[0], h[1])] = h[2] / 96.0
    return times, hold_tails, keycount


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
                 cmod_bpm=600.0, osu_speed=20, skin='bar', press_hide=False):
        self.headless = headless
        self.W, self.H = window_w, window_h

        self.replay = replay
        self.times, self.hold_tails, self.keycount = prepare_replay_times(
            replay, bpms=bpms, sm_offset=sm_offset)
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

        self.windows = osu_mania_windows(od) if game == 'osu' else etterna_windows_for(ett_judge)
        self.judge_colors = JCLR
        self.judge_label = f'OD {od}' if game == 'osu' else str(ett_judge)
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

        if scroll_mode is None:
            try:
                from analysis.core import game as game_mod
                scroll_mode = game_mod.get(game).default_scroll_mode()
            except Exception:
                scroll_mode = self.SCROLL_MODE_MS
        # Back-compat: accept the old 'linear' spelling from saved settings.
        if scroll_mode == 'linear':
            scroll_mode = self.SCROLL_MODE_MS
        if scroll_mode not in self._mode_state:
            scroll_mode = self.SCROLL_MODE_MS
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

        # Precomputed per-note arrays + LN lookup. The draw loop does a lot of
        # `int(self.columns[i])` / `int(self.replay['noterows'][i])` per frame,
        # which is surprisingly expensive (numpy scalar → Python int) on dense
        # charts. Materializing plain Python lists once lets the inner loop
        # stay in fast-path bytecode. `_ln_tail_times` is parallel to
        # `self.times` and holds the tail time-in-seconds per LN head (or NaN
        # for non-LNs), replacing the (row, col) dict lookup entirely.
        self._noterows_list = [int(r) for r in self.replay['noterows']]
        self._columns_list = [int(c) for c in self.columns]
        self._ln_tail_times = np.full(len(self.times), np.nan, dtype=np.float64)
        self._ln_indices = []
        for i, (row_val, col_val) in enumerate(zip(self._noterows_list,
                                                    self._columns_list)):
            end_t = self.hold_tails.get((row_val, col_val))
            if end_t is not None:
                self._ln_tail_times[i] = end_t
                self._ln_indices.append(i)

        # Ghost taps: (time_sec, column) for presses that didn't land on any
        # note. Only osu replays provide these — Etterna .bin has no raw key
        # event stream, so the list stays empty and _draw's sweep no-ops.
        # Stored as parallel numpy arrays so the per-frame windowing can use
        # bisect like the regular notes do.
        raw_ghosts = replay.get('ghost_taps') or []
        if raw_ghosts and game == 'osu':
            ghost_ts = np.array([t / 1000.0 for (t, _c) in raw_ghosts],
                                dtype=np.float64)
            ghost_cs = np.array([c for (_t, c) in raw_ghosts],
                                dtype=np.int32)
            # Sort by time for bisect windowing.
            order = np.argsort(ghost_ts, kind='stable')
            self._ghost_times = ghost_ts[order]
            self._ghost_cols = ghost_cs[order]
        else:
            self._ghost_times = np.empty(0, dtype=np.float64)
            self._ghost_cols = np.empty(0, dtype=np.int32)

        # Ghost holds: (press_t_sec, release_t_sec, column) spans where the
        # player held a key inside a missed LN. Rendered as a red hit line
        # over the LN body, showing the actual held duration — the release
        # time is NOT clipped to the LN tail, so overholds are visible.
        raw_holds = replay.get('ghost_holds') or []
        if raw_holds and game == 'osu':
            gh_heads = np.array([lh for (lh, _c, _pt, _rt) in raw_holds],
                                dtype=np.int64)
            gh_press = np.array([pt / 1000.0 for (_lh, _c, pt, _rt) in raw_holds],
                                dtype=np.float64)
            gh_rel = np.array([rt / 1000.0 for (_lh, _c, _pt, rt) in raw_holds],
                              dtype=np.float64)
            gh_cols = np.array([c for (_lh, c, _pt, _rt) in raw_holds],
                               dtype=np.int32)
            order = np.argsort(gh_press, kind='stable')
            self._ghost_hold_ln_heads_ms = gh_heads[order]
            self._ghost_hold_press = gh_press[order]
            self._ghost_hold_release = gh_rel[order]
            self._ghost_hold_cols = gh_cols[order]
            # Longest hold duration — used as a conservative lookback when
            # bisecting on press_t so spans whose press is off-screen (far
            # above the visible window) but whose release is on-screen
            # still get picked up.
            self._ghost_hold_max_dur = float(
                np.max(gh_rel - gh_press)) if gh_press.size else 0.0
        else:
            self._ghost_hold_ln_heads_ms = np.empty(0, dtype=np.int64)
            self._ghost_hold_press = np.empty(0, dtype=np.float64)
            self._ghost_hold_release = np.empty(0, dtype=np.float64)
            self._ghost_hold_cols = np.empty(0, dtype=np.int32)
            self._ghost_hold_max_dur = 0.0
        self._build_ghost_hold_note_links()

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
        self.plugin_panel_open = False
        self._hud_hitboxes = []
        # Vertical scroll offset (px) applied to the top-pinned sidebar
        # region when its content is taller than the viewport. Bottom-
        # pinned sections ignore this and stay anchored. Clamped each
        # frame in the renderer; the surrounding Qt tab writes to it on
        # mouse-wheel events that land over the sidebar.
        self.sidebar_scroll = 0
        self.sidebar_scroll_max = 0
        # Subscribers notified when scroll mode / speed changes via HUD
        # hitboxes — used by the surrounding Qt tab to persist settings and
        # refresh Qt widget state (SV button, scroll-edit placeholder).
        self._scroll_change_listeners = []
        # Subscribers notified for generic HUD hitbox actions that the tab
        # needs to react to (audio re-sync, settings persistence, pop-up
        # overlays). Receives (action, payload).
        self._hud_action_listeners = []
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
        """Cache ghost overlay times in the same SV-space used for note culling."""
        if self._ghost_times.size:
            self._ghost_sv_times = np.array(
                [self._cumulative_sv_at(float(t)) for t in self._ghost_times],
                dtype=np.float64)
        else:
            self._ghost_sv_times = np.empty(0, dtype=np.float64)

        if self._ghost_hold_press.size:
            self._ghost_hold_press_sv = np.array(
                [self._cumulative_sv_at(float(t))
                 for t in self._ghost_hold_press],
                dtype=np.float64)
            self._ghost_hold_release_sv = np.array(
                [self._cumulative_sv_at(float(t))
                 for t in self._ghost_hold_release],
                dtype=np.float64)
            self._ghost_hold_max_sv_dur = float(
                np.max(self._ghost_hold_release_sv
                       - self._ghost_hold_press_sv))
        else:
            self._ghost_hold_press_sv = np.empty(0, dtype=np.float64)
            self._ghost_hold_release_sv = np.empty(0, dtype=np.float64)
            self._ghost_hold_max_sv_dur = 0.0

    def _build_ghost_hold_note_links(self):
        """Link a missed LN's head press to its first matching ghost-hold span."""
        self._miss_first_ghost_hold = np.full(len(self.times), -1,
                                              dtype=np.int32)
        self._ghost_hold_extends_miss = np.zeros(
            self._ghost_hold_press.size, dtype=bool)
        if not self._ghost_hold_press.size:
            return

        by_head_col = {}
        for k, (head_ms, col) in enumerate(zip(self._ghost_hold_ln_heads_ms,
                                               self._ghost_hold_cols)):
            by_head_col.setdefault((int(head_ms), int(col)), []).append(k)

        # Source replay times are integer ms; allow a tiny tolerance for the
        # seconds conversion used by offsets.
        tol_ms = 2
        for i, (head_ms, col) in enumerate(zip(self._noterows_list,
                                               self._columns_list)):
            if not (self.misses[i] and self.miss_pressed[i]):
                continue
            if math.isnan(self._ln_tail_times[i]):
                continue
            press_ms = int(head_ms + round(float(self.offsets[i]) * 1000.0))
            for k in by_head_col.get((int(head_ms), int(col)), []):
                gh_press_ms = int(round(float(self._ghost_hold_press[k])
                                        * 1000.0))
                if abs(gh_press_ms - press_ms) <= tol_ms:
                    self._miss_first_ghost_hold[i] = k
                    self._ghost_hold_extends_miss[k] = True
                    break

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
        for rect, action, payload in reversed(getattr(self, '_hud_hitboxes', [])):
            rx, ry, rw, rh = rect
            if not (rx <= x <= rx + rw and ry <= y <= ry + rh):
                continue
            if action == 'toggle_plugin_panel':
                self.plugin_panel_open = not self.plugin_panel_open
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
            if action == 'cycle_game':
                self.cycle_game()
                self._notify_scroll_change()
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
        """Switch the active game. Picks that game's default scroll mode
        (Etterna → cmod, osu → osu). Core modes like ms remain available."""
        if game == self.game:
            return
        self.game = game
        try:
            from analysis.core import game as game_mod
            new_mode = game_mod.get(game).default_scroll_mode()
        except Exception:
            new_mode = self.SCROLL_MODE_MS
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

    def add_scroll_change_listener(self, cb):
        self._scroll_change_listeners.append(cb)

    def _notify_scroll_change(self):
        for cb in list(self._scroll_change_listeners):
            try:
                cb()
            except Exception:
                pass

    def add_hud_action_listener(self, cb):
        self._hud_action_listeners.append(cb)

    def _notify_hud_action(self, action, payload):
        for cb in list(self._hud_action_listeners):
            try:
                cb(action, payload)
            except Exception:
                pass

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
    audio = args[args.index('--audio') + 1] if '--audio' in args else None

    if path.endswith('.osr') or '--osu' in args:
        from analysis.games.osu.replay import parse_replay as parse_osu, find_osu_dirs
        osu_path = args[args.index('--osu') + 1] if '--osu' in args else None
        songs = find_osu_dirs().get('songs_dir')
        rep = parse_osu(path, osu_path=osu_path, songs_dir=songs)
        if audio is None and rep.get('chart_path'):
            af = Path(rep['chart_path']).parent / rep['chart_meta'].get('version', '')
            # try to resolve audio in same folder as .osu
            osu_dir = Path(rep['chart_path']).parent
            from analysis.games.osu.replay import parse_osu_file
            chart = parse_osu_file(rep['chart_path'])
            if chart.get('audio'):
                cand = osu_dir / chart['audio']
                if cand.exists():
                    audio = str(cand)
        launch_from_replay(rep, game='osu', od=od, audio_path=audio)
    else:
        from analysis.games.etterna.replay import parse_replay as parse_ett, find_etterna_dirs
        rep = parse_ett(path)
        bpms = None
        sm_off = 0.0
        if '--bpm' in args:
            bpms = [(0.0, float(args[args.index('--bpm') + 1]))]
        if '--sm' in args:
            from analysis.games.etterna.sm_chart import parse_sm, parse_ssc
            smp = args[args.index('--sm') + 1]
            data = parse_ssc(smp) if smp.endswith('.ssc') else parse_sm(smp)
            bpms = data['bpms']
            sm_off = data['offset']
            if audio is None:
                cand = Path(smp).parent / data['music']
                if cand.exists():
                    audio = str(cand)
        launch_from_replay(rep, game='etterna', bpms=bpms, sm_offset=sm_off,
                           audio_path=audio)
