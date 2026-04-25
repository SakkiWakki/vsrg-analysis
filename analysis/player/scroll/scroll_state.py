from __future__ import annotations

from analysis.core import game as game_mod
from analysis.player import scroll as scroll_registry


def first_bpm_or_default(bpms, *, default):
    if not bpms:
        return default

    try:
        return float(bpms[0][1])
    except (IndexError, TypeError, ValueError):
        return default


class ScrollStateController:
    def __init__(self, player):
        self.p = player

    def init(self, *, scroll_ms, cmod_bpm, xmod_value=1.0,
             osu_speed, bpms, scroll_mode):
        p = self.p

        scroll_registry.ensure_loaded()
        p._mode_state = {
            mode.key: {'value': mode.default_value, 'options': dict(mode.options)}
            for mode in scroll_registry.all_modes()
        }

        p._mode_state[p.SCROLL_MODE_MS]['value'] = float(scroll_ms)

        if p.SCROLL_MODE_CMOD in p._mode_state:
            p._mode_state[p.SCROLL_MODE_CMOD]['value'] = float(cmod_bpm)

        if p.SCROLL_MODE_XMOD in p._mode_state:
            p._mode_state[p.SCROLL_MODE_XMOD]['value'] = float(xmod_value)

        if p.SCROLL_MODE_OSU in p._mode_state:
            p._mode_state[p.SCROLL_MODE_OSU]['value'] = max(
                0.1,
                min(60.0, float(osu_speed)),
            )

        p._xmod_reference_bpm = first_bpm_or_default(bpms, default=120.0)

        if scroll_mode == 'linear':
            scroll_mode = p.SCROLL_MODE_MS

        if not scroll_mode or not scroll_registry.is_compatible(scroll_mode, p.game):
            scroll_mode = scroll_registry.default_for_game(p.game)

        p.scroll_mode = scroll_mode

    def mode(self, key=None):
        return scroll_registry.get(key or self.p.scroll_mode)

    def state(self, key=None):
        return self.p._mode_state[key or self.p.scroll_mode]

    def pxps_from_unit(self, mode_key, value, options=None):
        mode = scroll_registry.get(mode_key)
        if mode is None:
            return 0.0

        opts = options
        if opts is None:
            opts = self.p._mode_state.get(mode_key, {}).get('options', mode.options)

        return mode.to_pxps(float(value), opts, self.p)

    def unit_from_pxps(self, mode_key, pxps, options=None):
        mode = scroll_registry.get(mode_key)
        if mode is None:
            return 0.0

        opts = options
        if opts is None:
            opts = self.p._mode_state.get(mode_key, {}).get('options', mode.options)

        return mode.from_pxps(float(pxps), opts, self.p)

    def current_mode_value(self):
        return self.state()['value']

    def set_current_mode_value(self, value):
        mode = self.mode()
        if mode is None:
            return

        lo, hi = mode.value_bounds
        self.state()['value'] = max(lo, min(hi, float(value)))

    def get_mode_option(self, mode_key, option_key, default=None):
        return self.p._mode_state.get(mode_key, {}).get('options', {}).get(
            option_key,
            default,
        )

    def set_mode_option(self, mode_key, option_key, value):
        state = self.p._mode_state.get(mode_key)
        if state is None or option_key not in state['options']:
            return
        state['options'][option_key] = value

    @property
    def scroll_speed(self):
        pxps = self.p._pxps_from_unit(
            self.p.scroll_mode,
            self.p._current_mode_value(),
        )
        return pxps / max(0.01, self.p.play_rate)

    @property
    def effective_scroll_ms(self):
        sps = max(0.001, self.scroll_speed)
        return (self.p.H * self.p.hit_line_y_frac) / sps * 1000.0

    def set_scroll_ms(self, ms):
        p = self.p
        ms = max(50.0, min(3000.0, float(ms)))
        pxps = p._pxps_from_unit(p.SCROLL_MODE_MS, ms)
        p._set_current_mode_value(p._unit_from_pxps(p.scroll_mode, pxps))

    def set_scroll_mode(self, mode):
        p = self.p

        if mode == 'linear':
            mode = p.SCROLL_MODE_MS

        if mode not in p._mode_state or mode == p.scroll_mode:
            return

        if not scroll_registry.is_compatible(mode, p.game):
            return

        pxps = p._pxps_from_unit(p.scroll_mode, p._current_mode_value())

        prev_mode = self.mode(p.scroll_mode)
        if prev_mode and prev_mode.on_exit:
            prev_mode.on_exit(p, p._mode_state[p.scroll_mode])

        p.scroll_mode = mode
        p._set_current_mode_value(p._unit_from_pxps(mode, pxps))

        new_mode = self.mode(mode)
        if new_mode and new_mode.on_enter:
            new_mode.on_enter(p, p._mode_state[mode])

        p._reset_render_timeline()

    def nudge_scroll(self, factor):
        mode = self.mode()
        if mode is None or mode.nudge is None:
            return

        state = self.state()
        new_val = mode.nudge(state['value'], factor, state['options'])
        lo, hi = mode.value_bounds
        state['value'] = max(lo, min(hi, float(new_val)))

    def available_mode_keys(self):
        return [
            mode.key
            for mode in scroll_registry.all_modes()
            if mode.game is None or mode.game == self.p.game
        ]

    def set_game(self, game: str):
        p = self.p
        if game == p.game:
            return

        p.game = game
        if not scroll_registry.is_compatible(p.scroll_mode, game):
            new_mode = scroll_registry.default_for_game(game)
            if new_mode in p._mode_state:
                self.set_scroll_mode(new_mode)

    def cycle_game(self):
        try:
            names = list(game_mod.all_games().keys())
        except Exception:
            return

        if not names:
            return

        cur = self.p.game if self.p.game in names else names[0]
        self.set_game(names[(names.index(cur) + 1) % len(names)])
