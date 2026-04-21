"""Lane-background render layer."""


def draw(ctx):
    p = ctx.player
    pygame = ctx.pygame
    for c in range(p.keycount):
        rect = pygame.Rect(int(ctx.x0 + c * ctx.lane_w), 0,
                           int(ctx.lane_w), p.H)
        pygame.draw.rect(ctx.screen, (22, 22, 24), rect)
        pygame.draw.line(ctx.screen, (40, 40, 44),
                         (int(ctx.x0 + c * ctx.lane_w), 0),
                         (int(ctx.x0 + c * ctx.lane_w), p.H))
    pygame.draw.line(ctx.screen, (40, 40, 44),
                     (int(ctx.x0 + p.keycount * ctx.lane_w), 0),
                     (int(ctx.x0 + p.keycount * ctx.lane_w), p.H))
