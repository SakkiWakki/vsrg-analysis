"""Judgment-line and hit-window render layer."""


def draw(ctx):
    p = ctx.player
    pygame = ctx.pygame
    for name, w in reversed(p.windows):
        top = ctx.judge_y - w * p.scroll_speed
        bot = ctx.judge_y + w * p.scroll_speed
        color = ctx.colors[name]
        surf = pygame.Surface((int(p.keycount * ctx.lane_w), int(bot - top)),
                              pygame.SRCALPHA)
        surf.fill((color[0], color[1], color[2], 24))
        ctx.screen.blit(surf, (int(ctx.x0), int(top)))
    pygame.draw.line(ctx.screen, (255, 255, 255),
                     (int(ctx.x0), ctx.judge_y),
                     (int(ctx.x0 + p.keycount * ctx.lane_w), ctx.judge_y), 2)
