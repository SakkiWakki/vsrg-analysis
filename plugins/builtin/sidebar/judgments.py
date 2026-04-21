"""Built-in sidebar section: per-judgment window + miss counts."""
from __future__ import annotations

from analysis.player import theme


def _draw_judgments(sctx):
    p = sctx.player
    sctx.spacer()
    sctx.draw_heading('Judgments')
    counts = {n: 0 for n, _ in p.windows}
    counts['miss'] = 0
    for j in p.note_judges:
        counts[j] = counts.get(j, 0) + 1
    for name, w in p.windows:
        line = f'{name:<6}  ±{w*1000:5.1f}ms  n={counts[name]}'
        sctx.draw_text(line, color=p.judge_colors[name])
    sctx.draw_text(f'miss             n={counts["miss"]}',
                   color=p.judge_colors['miss'])


def register_sidebar(add):
    add('Judgments', _draw_judgments, priority=200, key='builtin:judgments')
