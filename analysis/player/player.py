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
import os
import sys
import math
import bisect
import pygame
import numpy as np
from pathlib import Path

from analysis.viz.plots import col_colors
from analysis.player import skin as skin_module


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
        from analysis.etterna.sm_chart import row_to_time
        times = np.array([row_to_time(int(r), bpms, sm_offset)
                          for r in replay['noterows']])
    else:
        # fallback: assume 120bpm, 48 rows/beat => 96 rows/second
        times = replay['noterows'].astype(np.float64) / 96.0
    hold_tails = {}
    for h in replay.get('holds', []):
        if len(h) == 3 and h[2] is not None:
            if bpms is not None:
                from analysis.etterna.sm_chart import row_to_time
                hold_tails[(h[0], h[1])] = row_to_time(int(h[2]), bpms, sm_offset)
            else:
                hold_tails[(h[0], h[1])] = h[2] / 96.0
    return times, hold_tails, keycount


class Player:
    # Scroll-speed modes.
    #   'linear' — our native convention, inherited from Quaver/pset6 and
    #              osu!mania's effective behavior: scroll_ms = time (ms) for a
    #              note to travel from the top of the screen to the judgment
    #              line. Constant regardless of BPM/SV (SV is layered on top).
    #   'cmod'   — Etterna's CMOD convention: pixel position =
    #              secondsUntilNote * (cmod_bpm / 60) * ARROW_SPACING.
    #              Scroll speed is tied to a target BPM; faster chart sections
    #              genuinely scroll faster on screen.
    SCROLL_MODE_LINEAR = 'linear'
    SCROLL_MODE_CMOD = 'cmod'
    ARROW_SPACING = 64.0  # px per beat at BPM 60, matches Etterna metrics.ini


    def __init__(self, replay, game='etterna', od=8, ett_judge='J4',
                 bpms=None, sm_offset=0.0, audio_path=None,
                 window_w=900, window_h=900, headless=False,
                 sv_sections=None, scroll_ms=400.0, scroll_mode=None,
                 cmod_bpm=600.0, skin='bar', press_hide=False):
        self.headless = headless
        if headless:
            # Render into an offscreen Surface; no display, no audio.
            pygame.font.init()
            self.screen = pygame.Surface((window_w, window_h))
        else:
            pygame.init()
            pygame.display.set_caption('Replay Player')
            self.screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
        self.W, self.H = window_w, window_h
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('monospace', 14)
        self.big_font = pygame.font.SysFont('monospace', 20, bold=True)

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

        self.windows = osu_mania_windows(od) if game == 'osu' else etterna_windows_for(ett_judge)
        self.judge_label = f'OD {od}' if game == 'osu' else str(ett_judge)
        self.game = game
        self.palette = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
                        for c in col_colors(self.keycount)]

        # Scroll speed: linear mode uses scroll_ms (time from top → judgment);
        # CMOD mode uses cmod_bpm to match Etterna. Default mode is CMOD for
        # Etterna replays (native convention), linear everywhere else.
        self.scroll_ms = float(scroll_ms)
        self.cmod_bpm = float(cmod_bpm)
        if scroll_mode is None:
            scroll_mode = self.SCROLL_MODE_CMOD if game == 'etterna' else self.SCROLL_MODE_LINEAR
        self.scroll_mode = scroll_mode
        self.play_rate = 1.0
        self.paused = True
        self.music_on = bool(audio_path)
        self.t = 0.0                 # current playback time (s)
        self._last_tick = None

        self.audio_path = audio_path
        if audio_path and os.path.exists(audio_path) and not headless:
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.set_volume(0.5)
            except Exception as e:
                print(f"audio load failed: {e}")
                self.audio_path = None
        elif headless:
            self.audio_path = None

        # Precompute judgment colors
        self.note_judges = []
        for off, mi in zip(self.offsets, self.misses):
            self.note_judges.append(judge(off, self.windows, mi))

        self.t_max = float(self.times[-1]) + 5.0 if len(self.times) else 10.0
        self.t_min = -2.0
        self.hit_line_y_frac = 0.80  # judgment line position
        self.skin_obj = skin_module.get(skin)
        self.skin = self.skin_obj.name
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

        # SV (scroll velocity) — list of (time_sec, sv_multiplier). When enabled,
        # note positions use the piecewise-constant integral of SV over time
        # (see _cumulative_sv_at). Falls back to constant scroll if empty.
        if sv_sections is None:
            sv_sections = replay.get('sv_sections') or []
        self.sv_sections = list(sv_sections)
        self.sv_enabled = bool(self.sv_sections)
        self._build_cumulative_sv()

    @property
    def scroll_speed(self):
        """Effective px/sec for converting *chart-time* deltas to on-screen
        distance. We divide by play_rate so that increasing rate doesn't
        visually speed up the notes — the chart scrolls at the same apparent
        speed, just with more notes per real second. This is the convention
        the user prefers for practice playback."""
        if self.scroll_mode == self.SCROLL_MODE_CMOD:
            base = (self.cmod_bpm / 60.0) * self.ARROW_SPACING
        else:
            base = (self.H * self.hit_line_y_frac) / max(0.001, self.scroll_ms / 1000.0)
        # Decouple visual scroll speed from playback rate: dividing by rate
        # keeps pixels-per-real-second constant as rate changes, so the chart
        # doesn't visually speed up with 1.2x/1.5x rates.
        return base / max(0.01, self.play_rate)

    @property
    def effective_scroll_ms(self):
        """ms from the top of the screen to the judgment line, computed from
        whichever mode is active. Useful for HUD display and for a unified UI."""
        sps = max(0.001, self.scroll_speed)
        return (self.H * self.hit_line_y_frac) / sps * 1000.0

    def set_scroll_ms(self, ms):
        """Set scroll speed in ms-to-judgment. In CMOD mode, back-solves
        cmod_bpm from the requested ms + current window height."""
        ms = max(50.0, min(3000.0, float(ms)))
        if self.scroll_mode == self.SCROLL_MODE_CMOD:
            sps = (self.H * self.hit_line_y_frac) / (ms / 1000.0)
            self.cmod_bpm = sps * 60.0 / self.ARROW_SPACING
        else:
            self.scroll_ms = ms

    def set_skin(self, skin):
        self.skin_obj = skin_module.get(skin)
        self.skin = self.skin_obj.name

    def toggle_skin(self):
        names = skin_module.names()
        idx = names.index(self.skin) if self.skin in names else 0
        self.set_skin(names[(idx + 1) % len(names)])

    def set_press_hide(self, on):
        self.press_hide = bool(on)

    def toggle_press_hide(self):
        self.press_hide = not self.press_hide
        return self.press_hide

    def set_scroll_mode(self, mode):
        """Switch between linear (osu!mania-style) and CMOD (Etterna-style)
        while preserving the currently-visible ms-to-judgment."""
        if mode not in (self.SCROLL_MODE_LINEAR, self.SCROLL_MODE_CMOD):
            return
        if mode == self.scroll_mode:
            return
        current_ms = self.effective_scroll_ms
        self.scroll_mode = mode
        self.set_scroll_ms(current_ms)

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
        """SV-weighted time delta (returns plain delta if SV is off/empty)."""
        if not self.sv_enabled or not self.sv_sections:
            return t_to - t_from
        return self._cumulative_sv_at(t_to) - self._cumulative_sv_at(t_from)

    def _time_to_y(self, t, t_now):
        judge_y = self.H * self.hit_line_y_frac
        return judge_y - self._sv_distance(t_now, t) * self.scroll_speed

    def toggle_sv(self):
        if self.sv_sections:
            self.sv_enabled = not self.sv_enabled
        return self.sv_enabled

    def _draw(self, t_now):
        self.screen.fill((14, 14, 16))
        x0, lane_w = self._lane_geom()
        judge_y = int(self.H * self.hit_line_y_frac)

        # Lane backgrounds
        for c in range(self.keycount):
            rect = pygame.Rect(int(x0 + c * lane_w), 0, int(lane_w), self.H)
            pygame.draw.rect(self.screen, (22, 22, 24), rect)
            pygame.draw.line(self.screen, (40, 40, 44),
                             (int(x0 + c * lane_w), 0),
                             (int(x0 + c * lane_w), self.H))
        pygame.draw.line(self.screen, (40, 40, 44),
                         (int(x0 + self.keycount * lane_w), 0),
                         (int(x0 + self.keycount * lane_w), self.H))

        # Judgment line + hit window shading
        hw_largest = self.windows[-1][1]
        hw_y_up = judge_y - hw_largest * self.scroll_speed
        hw_y_dn = judge_y + hw_largest * self.scroll_speed
        for name, w in reversed(self.windows):
            top = judge_y - w * self.scroll_speed
            bot = judge_y + w * self.scroll_speed
            color = JCLR[name]
            surf = pygame.Surface((int(self.keycount * lane_w), int(bot - top)),
                                  pygame.SRCALPHA)
            surf.fill((color[0], color[1], color[2], 24))
            self.screen.blit(surf, (int(x0), int(top)))
        pygame.draw.line(self.screen, (255, 255, 255),
                         (int(x0), judge_y),
                         (int(x0 + self.keycount * lane_w), judge_y), 2)

        # Candidate range: pick notes whose sv-distance from t_now falls in
        # the visible screen band. Works correctly even after large seeks or
        # under extreme SV — no note-index walk to get "stuck".
        note_h = 14
        screen_margin = 80
        sps = max(1e-3, self.scroll_speed)
        sv_hi = (judge_y + screen_margin) / sps            # top of screen
        sv_lo = (judge_y - (self.H + screen_margin)) / sps  # bottom of screen
        if self.sv_enabled and self.sv_sections:
            cum_now = self._cumulative_sv_at(float(t_now))
            target_lo = cum_now + sv_lo
            target_hi = cum_now + sv_hi
            lo = int(np.searchsorted(self._note_sv_cum, target_lo, side='left'))
            hi = int(np.searchsorted(self._note_sv_cum, target_hi, side='right'))
        else:
            # Constant scroll: ms-to-judgment → time bounds directly.
            target_lo = t_now + sv_lo
            target_hi = t_now + sv_hi
            lo = bisect.bisect_left(self.times, target_lo)
            hi = bisect.bisect_right(self.times, target_hi)
        candidates = list(range(lo, hi))
        # A long-note is visible whenever its head OR its tail OR any part of
        # the body falls in the screen band — not only when the head does.
        # The forward candidate pass catches heads; this pass sweeps every
        # LN whose head is outside the band and adds it if its tail is still
        # on-screen OR the screen band is between head and tail. Without this
        # second pass the LN flickers off the frame its head exits.
        #
        # We bound the sweep at 60s of chart time on either side of the
        # candidate range — long enough to cover any reasonable LN including
        # boss-chart sustains, short enough to stay cheap.
        seen = set(candidates)
        # Only LNs are relevant to the out-of-band sweep — a tap whose head
        # is outside the screen band cannot contribute anything visible.
        # Iterating self._ln_indices keeps this O(#LNs near window) regardless
        # of how dense the tap stream is.
        if self._ln_indices:
            ln_idx = self._ln_indices
            ln_lo = bisect.bisect_left(ln_idx, lo)
            ln_hi = bisect.bisect_right(ln_idx, hi)
            # Look back from the in-band range, stop when heads are older
            # than 60s of chart time (no reasonable LN body reaches further).
            for k in range(ln_lo - 1, -1, -1):
                i = ln_idx[k]
                if i in seen:
                    continue
                if self.times[i] < t_now - 60.0:
                    break
                y_head = self._time_to_y(self.times[i], t_now)
                y_tail = self._time_to_y(self._ln_tail_times[i], t_now)
                top_y = y_head if y_head < y_tail else y_tail
                bot_y = y_tail if y_tail > y_head else y_head
                if bot_y >= -screen_margin and top_y <= self.H + screen_margin:
                    candidates.append(i); seen.add(i)
            # Look ahead for LN heads whose tails have already scrolled into
            # view (slow scroll / big SV jump).
            for k in range(ln_hi, len(ln_idx)):
                i = ln_idx[k]
                if i in seen:
                    continue
                if self.times[i] > t_now + 60.0:
                    break
                y_head = self._time_to_y(self.times[i], t_now)
                y_tail = self._time_to_y(self._ln_tail_times[i], t_now)
                top_y = y_head if y_head < y_tail else y_tail
                bot_y = y_tail if y_tail > y_head else y_head
                if bot_y >= -screen_margin and top_y <= self.H + screen_margin:
                    candidates.append(i); seen.add(i)

        # Hoist hot attrs — attr lookups in a tight loop over hundreds of
        # notes add up. Same deal for the precomputed lists.
        times_ = self.times
        offsets_ = self.offsets
        misses_ = self.misses
        columns_ = self._columns_list
        noterows_ = self._noterows_list
        ln_tails_ = self._ln_tail_times
        rel_offsets_ = self.hold_release_offsets
        palette_ = self.palette
        keycount_ = self.keycount
        time_to_y = self._time_to_y

        for i in candidates:
            note_t = times_[i]
            c = columns_[i]
            if c >= keycount_:
                continue
            off = offsets_[i]
            miss = misses_[i]
            y = time_to_y(note_t, t_now)
            lx = int(x0 + c * lane_w)
            note_color = palette_[c]

            # -------- Game-agnostic note state ---------------------------
            # All LN/press logic derives purely from (note_t, off, miss,
            # end_t, rel_off) — no game-specific branches. That way new game
            # adapters just need to provide these fields in their replay dict
            # and this renderer works unchanged.
            end_t = ln_tails_[i]
            is_ln = not math.isnan(end_t)
            if is_ln:
                rel_off = rel_offsets_.get((noterows_[i], c))
            else:
                rel_off = None
                end_t = None
            # Head "consumed" time: when the player actually pressed it.
            # If missed, the head stays visible as it falls through.
            press_t = note_t + off  # only meaningful when not miss
            release_t = (end_t + (rel_off or 0.0)) if is_ln else None

            if miss:
                ln_state = 'missed' if is_ln else 'missed_note'
            elif is_ln:
                if t_now < press_t:
                    ln_state = 'upcoming'
                elif t_now < release_t:
                    ln_state = 'held'
                else:
                    ln_state = 'released'
            else:
                ln_state = 'tap'  # regular note

            # Press-hide mode: once the player has pressed (and, for LNs,
            # released) the note, skip drawing entirely. Misses stay visible
            # so the red X + body still communicate what was missed.
            if self.press_hide and not miss:
                if is_ln:
                    if t_now >= release_t:
                        continue
                else:
                    if t_now >= press_t:
                        continue

            jname = self.note_judges[i]
            jcolor = JCLR[jname]
            dim_color = (note_color[0] // 2, note_color[1] // 2,
                         note_color[2] // 2)
            # Dim-red tint for missed LNs (osu!mania convention).
            miss_body_color = (110, 40, 40)

            # -------- LN body --------------------------------------------
            # Drawn first so overlays (head, press bar, tail marker) sit on
            # top. Body is clipped to the portion still "ahead" of the head
            # progress — once the head passes the judgment line (held state),
            # the body shrinks from judgment line up to tail, matching
            # osu!mania / Quaver gameplay.
            if is_ln:
                y_end = self._time_to_y(end_t, t_now)
                if ln_state == 'upcoming':
                    body_top, body_bot, body_color = y_end, y, note_color
                elif ln_state == 'held':
                    body_top, body_bot, body_color = y_end, y, note_color
                elif ln_state == 'released':
                    # Head gone, but the body+tail keep scrolling past the
                    # judgment line so they don't pop out mid-screen. The
                    # body spans tail → judgment line while the tail is still
                    # on screen; once it crosses, nothing draws.
                    body_top, body_bot, body_color = y_end, judge_y, dim_color
                elif ln_state == 'missed':
                    # Missed head — body continues scrolling but dim red.
                    body_top, body_bot, body_color = y_end, y, miss_body_color
                else:
                    body_top = body_bot = None
                    body_color = None
                if body_color is not None and body_bot > body_top:
                    self.skin_obj.draw_ln_body(self.screen, lx, body_top,
                                               body_bot, lane_w, note_h,
                                               body_color)

                # LN tail marker — drawn whenever the tail is still on screen,
                # regardless of hold state, so the LN doesn't pop out the
                # instant the head passes.
                tail_on_screen = (-screen_margin <= y_end <= self.H +
                                  screen_margin)
                if tail_on_screen:
                    self.skin_obj.draw_ln_tail(self.screen, lx, y_end,
                                               lane_w, note_h, dim_color)
                # Release-offset indicator: only relevant until the player
                # has actually released (osu-mania tail judgment).
                if rel_off is not None and ln_state != 'released':
                    rel_y = y_end + rel_off * self.scroll_speed
                    pygame.draw.line(self.screen, (220, 220, 220),
                                     (int(lx + lane_w / 2), int(y_end)),
                                     (int(lx + lane_w / 2), int(rel_y)), 1)
                    pygame.draw.rect(self.screen, (220, 220, 220),
                                     (lx + 8, int(rel_y) - 2,
                                      int(lane_w - 16), 4))

            head_visible = ln_state in ('upcoming', 'tap', 'missed',
                                        'missed_note')

            # -------- Note head ------------------------------------------
            if head_visible:
                self.skin_obj.draw_note_head(self.screen, lx, y, lane_w,
                                             note_h, note_color)

            # -------- Press marker (head hit indicator) ------------------
            # Drawn AFTER the head so the indicator always shows on top.
            if not miss and head_visible:
                press_y = y + off * self.scroll_speed
                pygame.draw.line(self.screen, jcolor,
                                 (int(lx + lane_w / 2), int(y)),
                                 (int(lx + lane_w / 2), int(press_y)), 1)
                pygame.draw.rect(self.screen, jcolor,
                                 (lx + 8, int(press_y) - 2,
                                  int(lane_w - 16), 4))

            if miss and head_visible:
                # Translucent red outline + X marker. Applied to both taps
                # and LN heads — the LN body is already tinted dim-red
                # above, so together they read as "missed".
                pad = 4
                ow = int(lane_w - 8) + pad * 2
                oh = note_h + pad * 2
                halo = pygame.Surface((ow, oh), pygame.SRCALPHA)
                pygame.draw.rect(halo, (255, 60, 60, 110), halo.get_rect(),
                                 width=3)
                self.screen.blit(halo, (lx + 4 - pad, int(y) - note_h // 2 - pad))
                cx = lx + lane_w / 2
                pygame.draw.line(self.screen, jcolor,
                                 (cx - 10, y - 10), (cx + 10, y + 10), 2)
                pygame.draw.line(self.screen, jcolor,
                                 (cx - 10, y + 10), (cx + 10, y - 10), 2)

        # HUD / sidebar
        sidebar_x = self.W - 210
        pygame.draw.rect(self.screen, (20, 20, 22),
                         (sidebar_x, 0, 210, self.H))
        y = 14
        sv_line = ('SV: on' if self.sv_enabled else 'SV: off') \
            if self.sv_sections else 'SV: n/a'
        for line in [
            f't = {t_now:+7.3f}s',
            f'speed = {self.play_rate:.2f}x',
            (f'scroll = C{int(self.cmod_bpm)} ({int(self.effective_scroll_ms)}ms)'
             if self.scroll_mode == self.SCROLL_MODE_CMOD
             else f'scroll = {int(self.effective_scroll_ms)} ms'),
            f'notes = {len(self.times)}',
            f'keycount = {self.keycount}',
            sv_line,
            f'{"PAUSED" if self.paused else "PLAYING"}',
        ]:
            surf = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(surf, (sidebar_x + 8, y))
            y += 18

        y += 12
        title = self.big_font.render('Judgments', True, (255, 171, 145))
        self.screen.blit(title, (sidebar_x + 8, y))
        y += 26
        counts = {n: 0 for n, _ in self.windows}
        counts['miss'] = 0
        for j in self.note_judges:
            counts[j] = counts.get(j, 0) + 1
        for name, w in self.windows:
            line = f'{name:<6}  ±{w*1000:5.1f}ms  n={counts[name]}'
            surf = self.font.render(line, True, JCLR[name])
            self.screen.blit(surf, (sidebar_x + 8, y))
            y += 18
        miss_line = f'miss             n={counts["miss"]}'
        surf = self.font.render(miss_line, True, JCLR['miss'])
        self.screen.blit(surf, (sidebar_x + 8, y))
        y += 30

        # Help
        help_lines = [
            'Space: pause', 'L/R: seek', 'Sh+L/R: seek10',
            'Up/Dn: scrollspd', '+/-: playspd',
            'M: mute', 'R: restart', 'Q: quit',
        ]
        for h in help_lines:
            surf = self.font.render(h, True, (120, 120, 130))
            self.screen.blit(surf, (sidebar_x + 8, y))
            y += 16

        if not self.headless:
            pygame.display.flip()

    def tick(self, dt_s):
        """Headless: advance time and redraw. Returns raw RGB bytes + (w,h).

        We do NOT clamp at t_max while playing — t_max is just "a bit past the
        last note"; the underlying audio file often continues (outros, silence).
        The audio engine stops on its own when the sound ends, and the GUI
        auto-pauses once the audio has finished and we're past the chart end."""
        if not self.paused:
            self.t = max(self.t_min, self.t + dt_s * self.play_rate)
        self._draw(self.t)
        return pygame.image.tobytes(self.screen, 'RGB'), (self.W, self.H)

    def resize(self, w, h):
        self.W, self.H = max(200, int(w)), max(200, int(h))
        if self.headless:
            self.screen = pygame.Surface((self.W, self.H))
        else:
            self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)

    def seek_rel(self, dt):
        self._seek(dt)

    def toggle_pause(self):
        self._toggle_pause()

    def nudge_scroll(self, factor):
        # factor > 1 means "faster" (bigger px/sec).
        if self.scroll_mode == self.SCROLL_MODE_CMOD:
            self.cmod_bpm = max(60.0, min(5000.0, self.cmod_bpm * factor))
        else:
            # In ms-to-judgment units, faster = smaller ms, so invert.
            self.scroll_ms = max(50.0, min(3000.0, self.scroll_ms / factor))

    def nudge_rate(self, d):
        self.play_rate = max(0.1, min(4.0, self.play_rate + d))

    def restart(self):
        self.t = self.t_min

    def _seek(self, dt):
        self.t = max(self.t_min, min(self.t_max, self.t + dt))
        if self.music_on and self.audio_path and not self.paused:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.play(start=max(0, self.t))
            except pygame.error:
                pass

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.music_on and self.audio_path:
            try:
                if self.paused:
                    pygame.mixer.music.pause()
                else:
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play(start=max(0, self.t))
                    else:
                        pygame.mixer.music.unpause()
            except pygame.error:
                pass

    def run(self):
        running = True
        while running:
            now_ms = pygame.time.get_ticks()
            if self._last_tick is None:
                self._last_tick = now_ms
            dt_ms = now_ms - self._last_tick
            self._last_tick = now_ms
            if not self.paused:
                self.t += (dt_ms / 1000.0) * self.play_rate

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    self.W, self.H = ev.w, ev.h
                    self.screen = pygame.display.set_mode((self.W, self.H),
                                                           pygame.RESIZABLE)
                elif ev.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()
                    shift = mods & pygame.KMOD_SHIFT
                    if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif ev.key in (pygame.K_SPACE, pygame.K_p):
                        self._toggle_pause()
                    elif ev.key == pygame.K_LEFT:
                        self._seek(-10 if shift else -2)
                    elif ev.key == pygame.K_RIGHT:
                        self._seek(10 if shift else 2)
                    elif ev.key == pygame.K_UP:
                        self.nudge_scroll(1.15)
                    elif ev.key == pygame.K_DOWN:
                        self.nudge_scroll(1 / 1.15)
                    elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.play_rate = min(4.0, self.play_rate + 0.1)
                    elif ev.key == pygame.K_MINUS:
                        self.play_rate = max(0.1, self.play_rate - 0.1)
                    elif ev.key == pygame.K_r:
                        self.t = self.t_min
                        if self.music_on and self.audio_path:
                            try:
                                pygame.mixer.music.play(start=0)
                                if self.paused:
                                    pygame.mixer.music.pause()
                            except pygame.error:
                                pass
                    elif ev.key == pygame.K_m:
                        self.music_on = not self.music_on
                        if self.audio_path:
                            try:
                                if self.music_on and not self.paused:
                                    pygame.mixer.music.play(start=max(0, self.t))
                                else:
                                    pygame.mixer.music.stop()
                            except pygame.error:
                                pass
                    elif ev.key == pygame.K_LEFTBRACKET and self.paused:
                        self.t = max(self.t_min, self.t - 0.25)
                    elif ev.key == pygame.K_RIGHTBRACKET and self.paused:
                        self.t = min(self.t_max, self.t + 0.25)
                elif ev.type == pygame.MOUSEWHEEL:
                    self.t = max(self.t_min, min(self.t_max, self.t - ev.y * 0.5))
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    x, y = ev.pos
                    pb_y = self.H - 18
                    if pb_y <= y <= pb_y + 8 and 10 <= x <= self.W - 220:
                        frac = (x - 10) / max(1, (self.W - 230))
                        self.t = self.t_min + frac * (self.t_max - self.t_min)

            self._draw(self.t)
            self.clock.tick(60)

        pygame.quit()


def launch_from_replay(replay, game='etterna', od=8, bpms=None, sm_offset=0,
                       audio_path=None, sv_sections=None, scroll_ms=400.0,
                       scroll_mode=None, cmod_bpm=600.0):
    p = Player(replay, game=game, od=od, bpms=bpms, sm_offset=sm_offset,
               audio_path=audio_path, sv_sections=sv_sections,
               scroll_ms=scroll_ms, scroll_mode=scroll_mode, cmod_bpm=cmod_bpm)
    p.run()


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print("usage: player.py <replay> [--osu chart.osu] [--audio file] [--od N]")
        sys.exit(1)
    path = args[0]
    od = float(args[args.index('--od') + 1]) if '--od' in args else 8
    audio = args[args.index('--audio') + 1] if '--audio' in args else None

    if path.endswith('.osr') or '--osu' in args:
        from analysis.osu.replay import parse_replay as parse_osu, find_osu_dirs
        osu_path = args[args.index('--osu') + 1] if '--osu' in args else None
        songs = find_osu_dirs().get('songs_dir')
        rep = parse_osu(path, osu_path=osu_path, songs_dir=songs)
        if audio is None and rep.get('chart_path'):
            af = Path(rep['chart_path']).parent / rep['chart_meta'].get('version', '')
            # try to resolve audio in same folder as .osu
            osu_dir = Path(rep['chart_path']).parent
            from analysis.osu.replay import parse_osu_file
            chart = parse_osu_file(rep['chart_path'])
            if chart.get('audio'):
                cand = osu_dir / chart['audio']
                if cand.exists():
                    audio = str(cand)
        launch_from_replay(rep, game='osu', od=od, audio_path=audio)
    else:
        from analysis.etterna.replay import parse_replay as parse_ett, find_etterna_dirs
        rep = parse_ett(path)
        bpms = None
        sm_off = 0.0
        if '--bpm' in args:
            bpms = [(0.0, float(args[args.index('--bpm') + 1]))]
        if '--sm' in args:
            from analysis.etterna.sm_chart import parse_sm, parse_ssc
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
