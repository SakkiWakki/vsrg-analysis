"""Notes layer: replay-stream taps and long-notes.

Renders note heads, LN bodies/tails, release guides, press marks, and
miss-X overlays for every visible candidate. Chart-stream notes (mines,
lifts, fakes, ghost taps, miss-holds) live in chart_extras.py.

Public API:
- `prepare(ctx)` builds `ctx.note_views` once per frame from the player
  candidate list (shared by the `taps` + `lns` layers so the per-note
  state is computed once).
- `draw_taps(ctx, painter)` / `draw_lns(ctx, painter)` are the per-layer
  drawers the `NoteType` registrations hand out.
- `NoteType` is the per-adapter note-kind spec; `default_note_types()`
  returns the full set every game defaults to.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, NamedTuple

from analysis.player.render.layers import chart_extras as _extras

if TYPE_CHECKING:
    from analysis.player.render.render_context import RenderContext


# Shared vector constant for LN release-guide + press-mark strokes.
# Every other sprite color lives inside the sprite-cache rasterize
# callbacks (see `layers/note_sprites.py`).
_RELEASE_GUIDE = (220, 220, 220)


# ── note state ───────────────────────────────────────────────────

@dataclass
class _NoteView:
    i: int
    col: int
    y: int
    y_end: int
    lx: int
    off: float
    press_t: float
    release_t: float | None
    rel_off: float | None
    end_t: float | None
    is_ln: bool
    is_roll: bool
    miss: bool
    state: str       # upcoming | tap | held | released | missed | missed_note
    note_color: tuple
    jcolor: tuple


def _classify(ctx, press_t, release_t, is_ln, miss) -> str:
    if miss:
        return 'missed' if is_ln else 'missed_note'
    if not is_ln:
        return 'tap'
    if ctx.t_now < press_t:
        return 'upcoming'
    if ctx.t_now < release_t:
        return 'held'
    return 'released'


def _build(ctx, i, pos) -> _NoteView | None:
    p = ctx.player
    col = p._columns_list[i]
    if col >= p.keycount:
        return None
    if p.misses[i] and i < len(p._miss_head_suppressed) \
            and p._miss_head_suppressed[i]:
        return None

    note_t = p.times[i]
    end_t = p._ln_tail_times[i]
    is_ln = not math.isnan(end_t)
    off = p.offsets[i]
    press_t = note_t + off

    rel_off = None
    release_t = None
    y_end = 0
    if is_ln:
        rel_off = p.hold_release_offsets.get((p._noterows_list[i], col))
        release_t = end_t + (rel_off or 0.0)
        y_end = float(ctx.candidate_tail_y[pos])
    else:
        end_t = None

    return _NoteView(
        i=i, col=col,
        y=float(ctx.candidate_head_y[pos]), y_end=y_end,
        lx=int(ctx.lane_x(col)),
        off=off, press_t=press_t,
        release_t=release_t, rel_off=rel_off, end_t=end_t,
        is_ln=is_ln,
        is_roll=bool(is_ln and p._roll_head_keys
                     and (p._noterows_list[i], col) in p._roll_head_keys),
        miss=bool(p.misses[i]),
        state=_classify(ctx, press_t, release_t, is_ln, p.misses[i]),
        note_color=p.palette[col],
        jcolor=p.judge_colors[p.note_judges[i]],
    )


# ── public entry points ──────────────────────────────────────────

def prepare(ctx) -> None:
    """Build every per-candidate `_NoteView` once. The `taps` and `lns`
    layer drawers read from `ctx.note_views` — splitting the previous
    combined loop into two layer passes would otherwise rebuild each
    view twice."""
    views: list[_NoteView | None] = []
    for pos, i in enumerate(ctx.candidates):
        views.append(_build(ctx, i, pos))
    ctx.note_views = views


def draw_taps(ctx, painter) -> None:
    """Draw taps (non-LN notes) from the prebuilt views."""
    for n in ctx.note_views:
        if n is None or n.is_ln:
            continue
        _draw_replay_note(ctx, painter, n)


def draw_lns(ctx, painter) -> None:
    """Draw LN bodies, tails, release guides, and heads from the
    prebuilt views."""
    for n in ctx.note_views:
        if n is None or not n.is_ln:
            continue
        _draw_ln(ctx, painter, n)
        _draw_replay_note(ctx, painter, n)


def _draw_replay_note(ctx, painter, n) -> None:
    if _draw_head(ctx, painter, n):
        _draw_press_mark(ctx, painter, n)
        if n.miss:
            _draw_miss_x(ctx, painter, n)


# ── LN drawing ───────────────────────────────────────────────────

def _draw_ln(ctx, painter, n):
    p = ctx.player
    hide = p.press_hide

    # ── body fill ──
    span = _ln_body_span(ctx, n, hide)
    if span is not None:
        top, bot, body_state = span
        if bot > top:
            _draw_ln_body_tile(ctx, painter, n, top, bot, body_state)

    # ── tail sprite ──
    on_screen = -ctx.screen_margin <= n.y_end <= p.H + ctx.screen_margin
    hidden = hide and n.state == 'released'
    if on_screen and not hidden:
        _draw_ln_tail_sprite(ctx, painter, n)

    # ── release guide ──
    if n.rel_off is not None and n.state not in ('released', 'missed') and not hide:
        rel_y = ctx.time_to_y(float(n.release_t))
        _draw_stroke_with_tick(ctx, painter, _RELEASE_GUIDE,
                                n.lx, n.y_end, rel_y)


def _ln_body_span(ctx, n, hide) -> tuple | None:
    """Return `(top_y, bot_y, sprite_state)` where sprite_state drives
    the ln_body cache key. None means no body draws."""
    if n.state == 'missed':
        return n.y_end, n.y, 'miss_ln'
    match n.state:
        case 'upcoming':
            return n.y_end, n.y, 'normal'
        case 'held':
            return n.y_end, (ctx.judge_y if hide else n.y), 'normal'
        case 'released' if not hide:
            return n.y_end, ctx.judge_y, 'released'
        case _:
            return None


def _draw_ln_body_tile(ctx, painter, n, top, bot, state):
    from PySide6.QtCore import QRectF
    pm = ctx.sprite_cache.get('ln_body', ctx,
                              col=n.col, state=state, is_roll=n.is_roll)
    # Tile vertically over the body rect. Using QRectF + drawTiledPixmap
    # so fractional pixel heights (high-DPI) render cleanly.
    painter.drawTiledPixmap(QRectF(n.lx, top, ctx.lane_w, bot - top), pm)


def _draw_ln_tail_sprite(ctx, painter, n):
    from PySide6.QtCore import QPointF
    from analysis.player.render.layers.note_sprites import HEAD_PAD
    state = _tail_state(n)
    pm = ctx.sprite_cache.get('ln_tail', ctx, col=n.col, state=state)
    painter.drawPixmap(
        QPointF(float(n.lx), float(n.y_end - ctx.note_h / 2 - HEAD_PAD)), pm)


def _tail_state(n) -> str:
    if n.miss:
        return 'miss_ln'
    if n.is_roll:
        return 'roll'
    if n.state == 'released':
        return 'released'
    return 'normal'


# ── note head ────────────────────────────────────────────────────

def _draw_head(ctx, painter, n) -> bool:
    """Blit the head sprite from the cache if visible. Returns whether
    it was drawn (press-mark and miss-X both key off this)."""
    from PySide6.QtCore import QPointF
    from analysis.player.render.layers.note_sprites import HEAD_PAD
    visible, state, y = _head_vis(ctx, n)
    if not visible:
        return False
    sprite_name = 'ln_head' if n.is_ln else 'tap_head'
    pm = ctx.sprite_cache.get(sprite_name, ctx, col=n.col, state=state)
    painter.drawPixmap(
        QPointF(float(n.lx), float(y - ctx.note_h / 2 - HEAD_PAD)), pm)
    return True


def _head_vis(ctx, n) -> tuple:
    """`(visible, sprite_state, y)` — sprite_state drives the cache
    key, not a raw color. `normal` / `miss_tap` / `miss_ln`."""
    if n.miss:
        state = 'miss_ln' if n.is_ln else 'miss_tap'
        return True, state, n.y

    hide = ctx.player.press_hide
    if not hide:
        return n.state in ('upcoming', 'tap', 'held'), 'normal', n.y

    if n.is_ln:
        vis = n.state in ('upcoming', 'held')
        y = ctx.judge_y if n.state == 'held' else n.y
        return vis, 'normal', y
    return ctx.t_now < n.press_t, 'normal', n.y


# ── press mark + miss X ──────────────────────────────────────────

def _draw_press_mark(ctx, painter, n):
    p = ctx.player
    # skip: missed LNs, misses where player never pressed, held LNs in press_hide
    if n.miss and (n.is_ln or not p.miss_pressed[n.i]):
        return
    if n.is_ln and n.state == 'held' and p.press_hide:
        return

    press_y = ctx.time_to_y(float(n.press_t))
    color = p.judge_colors['miss'] if n.miss else n.jcolor
    _draw_stroke_with_tick(ctx, painter, color, n.lx, n.y, press_y)


def _draw_stroke_with_tick(ctx, painter, color, lx, y_from, y_to):
    """Vertical line from y_from to y_to + cached tick sprite at y_to.
    The line's endpoints change per note so it stays vector; the tick
    is a fixed-geometry sprite cached per color."""
    from PySide6.QtCore import QPointF
    _extras.draw_lane_line(painter, color, lx, ctx.lane_w, y_from, y_to)
    pm = ctx.sprite_cache.get('tick', ctx, color=color)
    painter.drawPixmap(QPointF(float(lx), float(y_to - 2)), pm)


def _draw_miss_x(ctx, painter, n):
    from PySide6.QtCore import QPointF
    from analysis.player.render.layers.note_sprites import MISS_X_PAD
    pm = ctx.sprite_cache.get('miss_x', ctx, jcolor=n.jcolor)
    # miss_x pixmap is `note_h + 2*pad` tall — shift the blit up by
    # `pad` so the note-head area inside the sprite lines up with the
    # actual note head beneath.
    painter.drawPixmap(
        QPointF(float(n.lx), float(n.y - ctx.note_h / 2 - MISS_X_PAD)), pm)



# ── per-adapter note-type registration ───────────────────────────

class NoteType(NamedTuple):
    """One note kind a game declares. Each maps 1:1 to a toggleable
    layer in the render plan.

    - `key`   — stable layer id (matches `LayerRegistry` key).
    - `name`  — human label for the HUD visibility tree.
    - `source` — 'player' (shares the prebuilt candidate list +
      `_NoteView` prepass) or 'chart' (owns its own cull).
    - `draw`  — `(ctx, painter) -> None`; same signature as any layer.
    - `stage` — optional plugin stage to fire *after* this layer draws.
      Preserves the `AFTER_NOTES` / `AFTER_GHOSTS` hook points now that
      the layers that used to own them are per-adapter.
    """
    key: str
    name: str
    source: str
    draw: Callable
    stage: object = None


# Keys match the existing `LayerRegistry` builtin names so tests that
# assert on layer ordering, plugin stages, and visibility config keep
# working. New keys ('taps', 'lns') replace the old combined 'notes'.
def default_note_types() -> list[NoteType]:
    # Imported locally so core modules don't depend on the plugin API.
    from analysis.player.plugin_api import Stage
    return [
        NoteType('taps',       'Taps',        'player', draw_taps),
        NoteType('lns',        'Long notes',  'player', draw_lns),
        NoteType('mines',      'Mines',       'chart',  _extras.draw_mines),
        NoteType('lifts',      'Lifts',       'chart',  _extras.draw_lifts),
        NoteType('fakes',      'Fakes',       'chart',  _extras.draw_fakes,
                 stage=Stage.AFTER_NOTES),
        NoteType('miss_holds', 'Miss holds',  'chart',  _extras.draw_miss_holds),
        NoteType('ghost_taps', 'Ghost taps',  'chart',  _extras.draw_ghost_taps,
                 stage=Stage.AFTER_GHOSTS),
    ]