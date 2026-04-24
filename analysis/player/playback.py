from __future__ import annotations


class PlaybackController:
    def __init__(self, player):
        self.p = player

    def init(self, *, skin, press_hide):
        from analysis.player.chart_clock import ChartClock

        p = self.p
        p._last_tick = None

        t_max = float(p.times[-1]) + 5.0 if len(p.times) else 10.0
        p._clock = ChartClock(initial=0.0, t_min=-2.0, t_max=t_max)

        p.hit_line_y_frac = 0.80
        self.set_skin(skin)
        self.set_press_hide(press_hide)

    @property
    def t(self):
        return self.p._clock.now()

    @t.setter
    def t(self, value):
        self.p._clock.seek(float(value))
        self.p._reset_render_timeline()

    @property
    def paused(self):
        return self.p._clock.paused

    @paused.setter
    def paused(self, value):
        self.p._clock.set_paused(bool(value))
        self.p._reset_render_timeline()

    @property
    def play_rate(self):
        return self.p._clock.rate

    @play_rate.setter
    def play_rate(self, value):
        self.p._clock.set_rate(float(value))
        self.p._reset_render_timeline()

    @property
    def t_min(self):
        return self.p._clock.t_min

    @property
    def t_max(self):
        return self.p._clock.t_max

    @t_max.setter
    def t_max(self, value):
        self.p._clock.set_bounds(self.p._clock.t_min, float(value))

    def attach_audio_clock(self, getter):
        self.p._clock.set_audio_source(getter)
        self.p._reset_render_timeline()

    @property
    def t_intended(self):
        return self.p._clock.intended()

    def set_skin(self, skin):
        self.p.skin = skin if skin in self.p.SKINS else 'bar'

    def toggle_skin(self):
        names = list(self.p.SKINS)
        idx = names.index(self.p.skin) if self.p.skin in names else 0
        self.set_skin(names[(idx + 1) % len(names)])

    def set_press_hide(self, on):
        self.p.press_hide = bool(on)

    def toggle_press_hide(self):
        self.p.press_hide = not self.p.press_hide
        return self.p.press_hide

    def nudge_rate(self, delta):
        self.play_rate = max(0.1, min(4.0, self.play_rate + delta))

    def advance(self, dt_s):
        del dt_s

    def tick(self, dt_s):
        self.advance(dt_s)
        return None, (self.p.W, self.p.H)

    def restart(self):
        self.p._clock.seek(self.t_min)
        self.p._reset_render_timeline()

    def seek_rel(self, dt):
        return self.seek(dt)

    def seek(self, dt):
        self.p._clock.seek(self.t + float(dt))
        self.p._reset_render_timeline()

    def toggle_pause(self):
        self.p._clock.set_paused(not self.p._clock.paused)
        self.p._reset_render_timeline()
