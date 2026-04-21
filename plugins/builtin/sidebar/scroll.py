"""Built-in sidebar section: scroll/rate/game controls, pinned to the bottom."""
from __future__ import annotations

from analysis.player import theme


_NUDGE_BTN_W = 28
_SCROLL_NUDGE_FACTOR = 1.15
_RATE_STEP = 0.1


def _draw_scroll(sctx):
    from analysis.player import scroll as scroll_registry
    p = sctx.player
    mode = scroll_registry.get(p.scroll_mode)

    sctx.draw_heading('Scroll')

    # Scroll value readout — click to open an edit overlay at the hitbox.
    val_str = (mode.format_value(p._current_mode_value())
               if mode and mode.format_value
               else f'{p._current_mode_value():.2f}')
    value_rect = (sctx.col_x, sctx.y, sctx.col_w, theme.ROW_BUTTON_H)
    sctx.rect(value_rect, theme.BTN_FILL, outline=theme.BTN_BORDER)
    sctx.text(f'{val_str} ({int(p.effective_scroll_ms)}ms)',
              sctx.col_x + 4, sctx.y + theme.TEXT_BASELINE_BUTTON,
              theme.BTN_FG)
    sctx.add_hitbox(value_rect, 'edit_scroll_value', value_rect)
    sctx.y += theme.ROW_BUTTON_H

    # Scroll ± row: two half-width buttons side by side.
    left, right = sctx.split_row(2, gap=4)
    for (rx, rw), label, factor in (
        (left, 'scroll −', 1 / _SCROLL_NUDGE_FACTOR),
        (right, 'scroll +', _SCROLL_NUDGE_FACTOR),
    ):
        sctx.button_at((rx, sctx.y, rw, theme.ROW_BUTTON_H),
                       label, 'scroll_nudge', factor)
    sctx.y += theme.ROW_TALL_H

    # Scroll-mode cycler.
    sctx.draw_button(
        f'scroll type: {mode.label if mode else p.scroll_mode}',
        'cycle_scroll_mode',
    )
    sctx.y += theme.ROW_TALL_H - theme.ROW_BUTTON_H

    # Rate row: [−] {rate}x [+]  — minus/plus on the outside, rate centered.
    rate_y = sctx.y
    sctx.button_at((sctx.col_x, rate_y, _NUDGE_BTN_W, theme.ROW_BUTTON_H),
                   '−', 'rate_nudge', -_RATE_STEP, center=True)
    sctx.button_at((sctx.col_x + sctx.col_w - _NUDGE_BTN_W, rate_y,
                    _NUDGE_BTN_W, theme.ROW_BUTTON_H),
                   '+', 'rate_nudge', _RATE_STEP, center=True)
    # Rate readout between the nudge buttons (no hitbox).
    readout_x = sctx.col_x + _NUDGE_BTN_W
    readout_w = sctx.col_w - 2 * _NUDGE_BTN_W
    rate_txt = f'{p.play_rate:.2f}x'
    sctx.text(rate_txt,
              readout_x + max(0, (readout_w - len(rate_txt) * 6) // 2),
              rate_y + theme.TEXT_BASELINE_BUTTON,
              theme.BTN_FG)
    sctx.y += theme.ROW_TALL_H

    # Game cycle.
    try:
        from analysis.core import game as game_mod
        games = list(game_mod.all_games().keys())
    except Exception:
        games = []
    sctx.draw_button(f'game: {p.game}', 'cycle_game',
                     enabled=len(games) > 1)
    sctx.y += theme.ROW_TALL_H - theme.ROW_BUTTON_H


def register_sidebar(add):
    add('Scroll', _draw_scroll, priority=800, key='builtin:scroll',
        pin_bottom=True)
