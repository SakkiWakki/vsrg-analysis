"""Sidebar HUD render layer."""


def draw(ctx):
    p = ctx.player
    pygame = ctx.pygame
    sidebar_x = p.W - 210
    p._hud_hitboxes = []
    pygame.draw.rect(ctx.screen, (20, 20, 22), (sidebar_x, 0, 210, p.H))
    y = 14
    sv_line = ('SV: on' if p.sv_enabled else 'SV: off') \
        if p.sv_sections else 'SV: n/a'
    for line in [
        f't = {ctx.t_now:+7.3f}s',
        f'speed = {p.play_rate:.2f}x',
        (f'scroll = C{int(p.cmod_bpm)} ({int(p.effective_scroll_ms)}ms)'
         if p.scroll_mode == p.SCROLL_MODE_CMOD
         else f'scroll = {int(p.effective_scroll_ms)} ms'),
        f'notes = {len(p.times)}',
        f'keycount = {p.keycount}',
        sv_line,
        f'{"PAUSED" if p.paused else "PLAYING"}',
    ]:
        surf = p.font.render(line, True, (220, 220, 220))
        ctx.screen.blit(surf, (sidebar_x + 8, y))
        y += 18

    y += 12
    title = p.big_font.render('Judgments', True, (255, 171, 145))
    ctx.screen.blit(title, (sidebar_x + 8, y))
    y += 26
    counts = {n: 0 for n, _ in p.windows}
    counts['miss'] = 0
    for j in p.note_judges:
        counts[j] = counts.get(j, 0) + 1
    for name, w in p.windows:
        line = f'{name:<6}  ±{w*1000:5.1f}ms  n={counts[name]}'
        surf = p.font.render(line, True, ctx.colors[name])
        ctx.screen.blit(surf, (sidebar_x + 8, y))
        y += 18
    miss_line = f'miss             n={counts["miss"]}'
    surf = p.font.render(miss_line, True, ctx.colors['miss'])
    ctx.screen.blit(surf, (sidebar_x + 8, y))
    y += 30

    help_lines = [
        'Space: pause', 'L/R: seek', 'Sh+L/R: seek10',
        'Up/Dn: scrollspd', '+/-: playspd',
        'M: mute', 'R: restart', 'Q: quit',
    ]
    for h in help_lines:
        surf = p.font.render(h, True, (120, 120, 130))
        ctx.screen.blit(surf, (sidebar_x + 8, y))
        y += 16

    y += 12
    y = _draw_plugin_controls(ctx, sidebar_x, y)
    ctx.plugin_data['hud_y'] = y
    ctx.plugin_data['sidebar_x'] = sidebar_x


def _draw_plugin_controls(ctx, sidebar_x, y):
    p = ctx.player
    pygame = ctx.pygame
    plugins = p.renderer.plugins.all_plugins()
    enabled = p.renderer.plugins.enabled_count()
    total = len(plugins)

    header_rect = (sidebar_x + 8, y, 190, 20)
    pygame.draw.rect(ctx.screen, (32, 32, 36), header_rect)
    pygame.draw.rect(ctx.screen, (68, 68, 76), header_rect, 1)
    p._hud_hitboxes.append((header_rect, 'toggle_plugin_panel', None))

    marker = '[-]' if p.plugin_panel_open else '[+]'
    label = f'{marker} Plugins {enabled}/{total}'
    surf = p.font.render(label, True, (220, 220, 220))
    ctx.screen.blit(surf, (sidebar_x + 14, y + 3))
    y += 24

    if not p.plugin_panel_open:
        return y

    if not plugins:
        surf = p.font.render('no plugins found', True, (120, 120, 130))
        ctx.screen.blit(surf, (sidebar_x + 18, y))
        return y + 18

    row_h = 18
    max_rows = max(0, (p.H - y - 8) // row_h)
    for idx, plugin in enumerate(plugins[:max_rows]):
        row_rect = (sidebar_x + 10, y, 188, row_h)
        p._hud_hitboxes.append((row_rect, 'toggle_plugin', plugin.key))
        box_rect = (sidebar_x + 14, y + 3, 10, 10)
        pygame.draw.rect(ctx.screen, (16, 16, 18), box_rect)
        pygame.draw.rect(ctx.screen, (110, 110, 120), box_rect, 1)
        if plugin.enabled:
            pygame.draw.line(ctx.screen, (160, 230, 160),
                             (box_rect[0] + 2, box_rect[1] + 5),
                             (box_rect[0] + 4, box_rect[1] + 8), 2)
            pygame.draw.line(ctx.screen, (160, 230, 160),
                             (box_rect[0] + 4, box_rect[1] + 8),
                             (box_rect[0] + 9, box_rect[1] + 2), 2)

        color = (210, 210, 215) if plugin.enabled else (110, 110, 116)
        name = _shorten(plugin.name, 22)
        surf = p.font.render(name, True, color)
        ctx.screen.blit(surf, (sidebar_x + 30, y + 1))
        y += row_h

    if len(plugins) > max_rows:
        more = len(plugins) - max_rows
        surf = p.font.render(f'+{more} more', True, (120, 120, 130))
        ctx.screen.blit(surf, (sidebar_x + 18, y))
        y += row_h
    return y


def _shorten(text, max_chars):
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max(0, max_chars - 1)] + '~'
