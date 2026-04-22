"""Built-in sidebar section: per-judgment window + miss counts, with
a judge switcher row. The actual judge→windows mapping is owned by the
game's adapter; this plugin is game-agnostic — it just nudges through
whatever the adapter's `nudge_judge` accepts."""
from __future__ import annotations

from analysis.player.render import theme


_NUDGE_BTN_W = 28
# Etterna judges are integer steps — sign is all that matters. osu OD is
# float; ±0.1 per click mirrors the rate slider's feel.
_OSU_OD_STEP = 0.1


def _draw_judgments(sctx):
    p = sctx.player
    sctx.spacer()
    sctx.draw_heading('Judgments')

    # Switcher row: [−] {label} [+], same layout as the rate slider.
    step = 1.0 if p.game == 'etterna' else _OSU_OD_STEP
    row_y = sctx.y
    sctx.button_at((sctx.col_x, row_y, _NUDGE_BTN_W, theme.ROW_BUTTON_H),
                   '−', 'judge_nudge', -step, center=True)
    sctx.button_at((sctx.col_x + sctx.col_w - _NUDGE_BTN_W, row_y,
                    _NUDGE_BTN_W, theme.ROW_BUTTON_H),
                   '+', 'judge_nudge', step, center=True)
    # Centered label between the nudge buttons (no hitbox).
    readout_x = sctx.col_x + _NUDGE_BTN_W
    readout_w = sctx.col_w - 2 * _NUDGE_BTN_W
    label = str(p.judge_label)
    sctx.text(label,
              readout_x + max(0, (readout_w - len(label) * 6) // 2),
              row_y + theme.TEXT_BASELINE_BUTTON,
              theme.BTN_FG)
    sctx.y += theme.ROW_TALL_H

    counts = {n: 0 for n, _ in p.windows}
    counts['miss'] = 0
    for j in p.note_judges:
        counts[j] = counts.get(j, 0) + 1
    for name, w in p.windows:
        line = f'{name:<6}  ±{w*1000:5.1f}ms  n={counts[name]}'
        sctx.draw_text(line, color=p.judge_colors[name])
    sctx.draw_text(f'miss             n={counts["miss"]}',
                   color=p.judge_colors['miss'])

    # Mines: hit count is XML-only (.bin replay doesn't record which
    # mines were triggered), total is chart-derived. We fold both into
    # the same list so the player sees mines alongside the regular
    # judgments instead of a separate section. "n=hit/total" mirrors
    # the in-game results screen ("Mines 012/1337").
    xml_j = getattr(p, 'xml_judgments', None) or {}
    hit = xml_j.get('HitMine')
    total_mines = len(p.replay.get('chart_mines') or [])
    if hit is not None or total_mines:
        if total_mines:
            line = f'mines hit        n={int(hit or 0)}/{total_mines}'
        else:
            line = f'mines hit        n={int(hit or 0)}'
        sctx.draw_text(line, color=p.judge_colors.get('miss', (220, 60, 60)))


def register_sidebar(add):
    add('Judgments', _draw_judgments, priority=200, key='builtin:judgments',
        draggable=True, default_free_xy=(0.02, 0.04),
        default_size=(210, 200))
