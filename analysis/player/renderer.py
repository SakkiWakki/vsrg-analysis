"""Ordered replay-player render pipeline."""
from __future__ import annotations

import pygame

from analysis.player import culling
from analysis.player.layers import ghosts, hud, judgment, lanes, notes
from analysis.player.plugin_api import Stage
from analysis.player.plugin_loader import PluginManager
from analysis.player.render_context import RenderContext


class PlayerRenderer:
    def __init__(self, plugin_manager=None):
        self.plugins = plugin_manager or PluginManager.discover()

    def build_context(self, player, t_now):
        x0, lane_w = player._lane_geom()
        ctx = RenderContext(
            player=player,
            screen=player.screen,
            pygame=pygame,
            colors=player.judge_colors,
            t_now=float(t_now),
            x0=x0,
            lane_w=lane_w,
            judge_y=int(player.H * player.hit_line_y_frac),
        )
        culling.prepare_time_window(ctx)
        ctx.candidates = culling.select_note_candidates(ctx)
        return ctx

    def draw(self, player, t_now):
        player.screen.fill((14, 14, 16))
        ctx = self.build_context(player, t_now)

        lanes.draw(ctx)
        self.plugins.draw(Stage.AFTER_LANES, ctx)

        judgment.draw(ctx)
        self.plugins.draw(Stage.AFTER_JUDGMENT, ctx)

        notes.draw(ctx)
        self.plugins.draw(Stage.AFTER_NOTES, ctx)

        ghosts.draw_ghost_holds(ctx)
        ghosts.draw_ghost_taps(ctx)
        self.plugins.draw(Stage.AFTER_GHOSTS, ctx)

        hud.draw(ctx)
        self.plugins.draw(Stage.HUD, ctx)
        self.plugins.draw(Stage.POST_FRAME, ctx)

        if not player.headless:
            pygame.display.flip()
