"""Built-in sidebar section: scroll/rate/game controls, rendered as a
flyout. The sidebar shows a one-line ``Scroll ▸`` button; clicking opens
a flyout panel to the left of the sidebar that holds the full control
set."""
from __future__ import annotations

from analysis.player.render import theme


_KEY = 'builtin:scroll'
_NUDGE_BTN_W = 28
_SCROLL_NUDGE_FACTOR = 1.15
_RATE_STEP = 0.1


def _collapsed_header(sctx):
    p = sctx.player
    open_ = p.hud.open_flyout == _KEY
    sctx.draw_button(f'Scroll {"▾" if open_ else "▸"}',
                     'toggle_flyout', _KEY)


def _draw_flyout(sctx):
    from analysis.player import scroll as scroll_registry
    p = sctx.player
    mode = scroll_registry.get(p.scroll_mode)

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

    left, right = sctx.split_row(2, gap=4)
    for (rx, rw), label, factor in (
        (left, 'scroll −', 1 / _SCROLL_NUDGE_FACTOR),
        (right, 'scroll +', _SCROLL_NUDGE_FACTOR),
    ):
        sctx.button_at((rx, sctx.y, rw, theme.ROW_BUTTON_H),
                       label, 'scroll_nudge', factor)
    sctx.y += theme.ROW_TALL_H

    sctx.draw_button(
        f'scroll type: {mode.label if mode else p.scroll_mode}',
        'cycle_scroll_mode',
    )
    sctx.y += theme.ROW_TALL_H - theme.ROW_BUTTON_H

    rate_y = sctx.y
    sctx.button_at((sctx.col_x, rate_y, _NUDGE_BTN_W, theme.ROW_BUTTON_H),
                   '−', 'rate_nudge', -_RATE_STEP, center=True)
    sctx.button_at((sctx.col_x + sctx.col_w - _NUDGE_BTN_W, rate_y,
                    _NUDGE_BTN_W, theme.ROW_BUTTON_H),
                   '+', 'rate_nudge', _RATE_STEP, center=True)
    readout_x = sctx.col_x + _NUDGE_BTN_W
    readout_w = sctx.col_w - 2 * _NUDGE_BTN_W
    rate_txt = f'{p.play_rate:.2f}x'
    sctx.text(rate_txt,
              readout_x + max(0, (readout_w - len(rate_txt) * 6) // 2),
              rate_y + theme.TEXT_BASELINE_BUTTON,
              theme.BTN_FG)
    sctx.y += theme.ROW_TALL_H

    try:
        from analysis.core import game as game_mod
        games = list(game_mod.all_games().keys())
    except Exception:
        games = []
    sctx.draw_button(f'game: {p.game}', 'cycle_game',
                     enabled=len(games) > 1)
    sctx.y += theme.ROW_TALL_H - theme.ROW_BUTTON_H

    sv = getattr(p, 'sv_render', None)
    if sv is not None and hasattr(sv, 'available_engine_keys'):
        engine_keys = sv.available_engine_keys()
        if len(engine_keys) > 1:
            sctx.draw_button(f'engine: {sv.active_engine_label()}',
                             'cycle_sv_engine')
            sctx.y += theme.ROW_TALL_H - theme.ROW_BUTTON_H


def register_sidebar(add):
    add('Scroll', _collapsed_header, priority=800, key=_KEY,
        pin_bottom=True, draw_expanded=_draw_flyout)
