"""Regression tests for renderer-level bugs.

These cover pure-logic branches of the Qt renderer (no Qt painter
needed). Draw primitives are monkey-patched to record calls so the
test can assert what was/wasn't drawn.
"""
from types import SimpleNamespace

import numpy as np

from analysis.player.render import culling
from analysis.player.render.layers import notes as _notes_layer
from analysis.player.render.qt_renderer import QtPlayerRenderer, _NoteView


def _make_note(*, miss, is_ln, miss_pressed_i=True, off=0.0, state='tap',
               press_y=200):
    """`_NoteView` with just the fields `_draw_press_mark` touches."""
    return _NoteView(
        i=0, col=0, y=100, y_end=100, press_y=press_y, lx=0, off=off,
        press_t=0.0, release_t=None, rel_off=None, end_t=None,
        is_ln=is_ln, is_roll=False, miss=miss, state=state,
        note_color=(255, 255, 255), jcolor=(255, 0, 0),
    )


def _make_press_ctx(renderer, miss_pressed=(True,)):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    empty_pm = QPixmap(1, 1)
    empty_pm.fill(Qt.transparent)

    player = SimpleNamespace(
        miss_pressed=np.array(miss_pressed, dtype=bool),
        press_hide=False,
        scroll_speed=1000.0,
        judge_colors={'miss': (255, 0, 0)},
        H=600,
    )
    sprite_cache = SimpleNamespace(get=lambda *a, **k: empty_pm)
    return SimpleNamespace(player=player, lane_w=80,
                            # drawers removed - sprite cache is the sole draw path
                            sprite_cache=sprite_cache,
                            scroll_speed=player.scroll_speed,
                            screen_margin=100,
                            time_to_y=lambda t: 100 + float(t) * 1000.0)


def _patch_draw_recorders(monkeypatch):
    """Replace the lane-line helper + track sprite-cache tick blits.

    `_draw_stroke_with_tick` calls `chart_extras.draw_lane_line` for
    the vertical stroke and `painter.drawPixmap` with a cached tick
    pixmap at the tick position. We intercept the first and inject a
    fake painter recorder for the second so tests can still assert
    stroke + tick presence without a real QPainter.

    Returns `(lines, ticks)` lists plus a fake painter ready to pass
    through to the press-mark call site. Each tick entry is
    `(y, pixmap)` ; we don't know the color at blit time because the
    cache collapses that into the pixmap identity, but count/position
    are what these tests care about."""
    lines = []
    ticks = []

    def fake_line(painter, color, lx, lane_w, y0, y1, width=1):
        lines.append((color, lx, lane_w, y0, y1, width))

    def fake_tick(painter, color, lane_x, y, lane_w):
        # Still kept for any caller that hits `_extras.draw_tick`
        # directly (none after the sprite migration but cheap to keep).
        ticks.append((color, lane_x, y, lane_w))

    def fake_draw_pixmap(point, _pm):
        ticks.append(('pixmap', float(point.x()), float(point.y())))

    from analysis.player.render.layers import chart_extras as _ce
    monkeypatch.setattr(_ce, 'draw_lane_line', fake_line)
    monkeypatch.setattr(_ce, 'draw_tick', fake_tick)

    fake_painter = SimpleNamespace(
        drawPixmap=fake_draw_pixmap,
        drawTiledPixmap=lambda *a, **k: None,
    )
    return lines, ticks, fake_painter


def test_missed_ln_skips_press_mark_when_pressed(monkeypatch):
    """Regression: a missed LN where the player DID press (miss_pressed=True)
    must not draw a press-mark ; `_draw_miss_holds` owns that stroke, so
    drawing press-mark too would double up."""
    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    lines, ticks, fake_painter = _patch_draw_recorders(monkeypatch)

    ctx = _make_press_ctx(renderer, miss_pressed=(True,))
    note = _make_note(miss=True, is_ln=True, off=-0.150, state='missed')

    _notes_layer._draw_press_mark(ctx, painter=fake_painter, n=note)

    assert lines == []
    assert ticks == []


def test_missed_ln_skips_press_mark_when_not_pressed(monkeypatch):
    """Same rule for a missed LN with no recorded press ; it's still the
    miss-hold drawer's domain (if any), never the press-mark's."""
    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    lines, ticks, fake_painter = _patch_draw_recorders(monkeypatch)

    ctx = _make_press_ctx(renderer, miss_pressed=(False,))
    note = _make_note(miss=True, is_ln=True, off=1.0, state='missed')

    _notes_layer._draw_press_mark(ctx, painter=fake_painter, n=note)

    assert lines == []
    assert ticks == []


def test_hit_tap_draws_press_mark(monkeypatch):
    """Sanity: non-miss non-LN still draws both line and tick."""
    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    lines, ticks, fake_painter = _patch_draw_recorders(monkeypatch)

    ctx = _make_press_ctx(renderer)
    note = _make_note(miss=False, is_ln=False, off=0.020, state='tap')

    _notes_layer._draw_press_mark(ctx, painter=fake_painter, n=note)

    assert len(lines) == 1
    assert len(ticks) == 1


def test_missed_tap_still_draws_press_mark(monkeypatch):
    """Missed non-LN draws a red press-mark ; the rule only excludes LNs
    and never-pressed misses."""
    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    lines, ticks, fake_painter = _patch_draw_recorders(monkeypatch)

    ctx = _make_press_ctx(renderer, miss_pressed=(True,))
    note = _make_note(miss=True, is_ln=False, off=0.080, state='missed_note')

    _notes_layer._draw_press_mark(ctx, painter=fake_painter, n=note)

    assert len(lines) == 1
    assert len(ticks) == 1
    assert lines[0][0] == (255, 0, 0)  # miss color


def test_missed_tap_without_press_skips_press_mark(monkeypatch):
    """Regression: osu replays write a 1.0s sentinel offset for misses
    the player never pressed. Drawing a press-mark for those produces a
    full-second line that crosses unrelated notes on the same column,
    visually connecting unrelated misses (TWO-TORIAL col=3, ~10.4s)."""
    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    lines, ticks, fake_painter = _patch_draw_recorders(monkeypatch)

    ctx = _make_press_ctx(renderer, miss_pressed=(False,))
    note = _make_note(miss=True, is_ln=False, off=1.0,
                      state='missed_note')

    _notes_layer._draw_press_mark(ctx, painter=fake_painter, n=note)

    assert lines == []
    assert ticks == []


def test_press_mark_uses_precomputed_press_y(monkeypatch):
    """Press marks must use the same projected time->Y mapping as notes,
    not a raw `offset * scroll_speed` shortcut. Observable as the
    `y1` endpoint passed to `chart_extras.draw_lane_line` ; that's
    whatever the precomputed `_NoteView.press_y` carried from the
    batched per-frame y projection."""
    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    lines, _ticks, fake_painter = _patch_draw_recorders(monkeypatch)

    ctx = _make_press_ctx(renderer)
    note = _make_note(miss=False, is_ln=False, off=0.125, state='tap',
                      press_y=321.0)
    note = note.__class__(**{**note.__dict__, 'press_t': 0.125})

    _notes_layer._draw_press_mark(ctx, painter=fake_painter, n=note)

    # (color, lx, lane_w, y_from, y_to, width)
    assert len(lines) == 1
    assert lines[0][0] == (255, 0, 0)   # jcolor (non-miss path)
    assert lines[0][3] == 100            # y_head
    assert lines[0][4] == 321.0          # cached y_press


def test_ln_release_guide_uses_projected_time_to_y(monkeypatch):
    """Release guides must also use the projected time mapping so they
    stay attached in SV sections. The guide draws a vector line from
    the tail's y to `ctx.time_to_y(release_t)` + a cached tick at the
    release y; we observe the projected y via the line endpoints."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    empty_pm = QPixmap(1, 1)
    empty_pm.fill(Qt.transparent)

    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    lines, _ticks, fake_painter = _patch_draw_recorders(monkeypatch)

    player = SimpleNamespace(press_hide=False, skin='bar', H=600)
    sprite_cache = SimpleNamespace(get=lambda *a, **k: empty_pm)
    ctx = SimpleNamespace(
        player=player,
        lane_w=80, note_h=20, judge_y=500, screen_margin=100,
        # drawers removed - sprite cache is the sole draw path
        sprite_cache=sprite_cache,
        time_to_y=lambda t: 456.0 if abs(t - 0.75) < 1e-9 else -999.0,
    )
    note = _NoteView(
        i=0, col=0, y=100, y_end=200, press_y=100, lx=0, off=0.0,
        press_t=0.0, release_t=0.75, rel_off=0.1, end_t=0.65,
        is_ln=True, is_roll=False, miss=False, state='held',
        note_color=(255, 255, 255), jcolor=(255, 0, 0),
    )

    _notes_layer._draw_ln(ctx, painter=fake_painter, n=note)

    # The body fill + tail blit go through drawPixmap/drawTiledPixmap;
    # only the release guide hits draw_lane_line, so `lines` should
    # contain exactly the guide stroke.
    assert len(lines) == 1
    assert lines[0][3] == 200              # y_tail
    assert lines[0][4] == 456.0            # projected y_release


# ---------------------------------------------------------------------------
# Culling pad: notes stay in the candidate set while their drawn strokes
# (press mark, release guide) could still be on-screen, even after the
# note's head time has scrolled past the window edge.
# ---------------------------------------------------------------------------

def _cull_ctx(player, *, target_lo, target_hi):
    return SimpleNamespace(
        player=player,
        target_lo=target_lo,
        target_hi=target_hi,
        use_sv_space=False,
        t_now=0.0,
        screen_margin=100,
    )


def test_culling_pad_keeps_note_whose_press_mark_extends_into_window():
    """Regression: a note with a large hit offset that scrolls off the top
    should stay in the candidate set until its press-mark (anchored at
    `head + off`) has also left the window. Widening the culling window
    by `max_draw_pad_sec` on each side covers both press and release
    strokes in one bisect."""
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    player = SimpleNamespace(
        times=times,
        _note_sv_cum=times,
        sv_enabled=False,
        sv_sections=[],
        notes=SimpleNamespace(ln_indices=[]),
        max_draw_pad_sec=0.5,
    )
    # Window is [1.2, 2.8]. Without padding, only index 2 (time=2.0) is in.
    # With pad=0.5, window grows to [0.7, 3.3], adding index 1 (time=1.0)
    # and index 3 (time=3.0).
    ctx = _cull_ctx(player, target_lo=1.2, target_hi=2.8)

    candidates = culling.select_note_candidates(ctx)
    assert candidates == [1, 2, 3]


def test_culling_pad_zero_matches_legacy_window():
    """With no offsets in the replay, `max_draw_pad_sec` is 0 and the
    candidate window is the raw target window (no regression in the
    common case)."""
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    player = SimpleNamespace(
        times=times,
        _note_sv_cum=times,
        sv_enabled=False,
        sv_sections=[],
        notes=SimpleNamespace(ln_indices=[]),
        max_draw_pad_sec=0.0,
    )
    ctx = _cull_ctx(player, target_lo=1.2, target_hi=2.8)

    assert culling.select_note_candidates(ctx) == [2]


def test_sv_culling_pad_converts_seconds_to_cumulative_units():
    """SV-space candidate expansion must convert the pad into cumulative
    units instead of adding raw seconds to a cumulative window."""
    times = np.array([0.8, 1.0, 1.3], dtype=np.float64)
    player = SimpleNamespace(
        times=times,
        _note_sv_cum=np.array([9.5, 11.0, 13.0], dtype=np.float64),
        notes=SimpleNamespace(ln_indices=[]),
        max_draw_pad_sec=0.2,
        _cumulative_sv_at=lambda t: float(t) * 10.0,
    )
    ctx = SimpleNamespace(
        player=player,
        target_lo=11.0,
        target_hi=12.0,
        use_sv_space=True,
        t_now=1.0,
        visual_cum_now=10.0,
        screen_margin=100,
        frame=None,
    )

    # Pad is 0.2s -> 2 cumulative units, so [11, 12] expands to [9, 14],
    # including all three notes. Treating "0.2" as cumulative units would
    # incorrectly exclude the first and third notes.
    assert culling.select_note_candidates(ctx) == [0, 1, 2]


def test_run_sections_always_draws_header_and_records_anchor_when_open():
    """Regression: the collapsed header must draw even when its flyout
    is open ; it stays on screen as the anchor point and re-click target.
    When open, the header's rect is recorded in
    ``plugin_data['flyout_anchors']`` so the flyout panel can align to
    it. Non-flyout sections draw normally and don't record anchors."""
    from analysis.player.hud.sidebar_api import SidebarSection

    seen = []

    def collapsed_scroll(sctx):
        seen.append('scroll-collapsed')
        sctx.y += 20  # simulate the header button's row height

    def collapsed_opts(sctx):
        seen.append('opts-collapsed')
        sctx.y += 20

    def normal_section(sctx):
        seen.append('normal')
        sctx.y += 30

    sections = [
        SidebarSection(key='a:scroll', name='Scroll', draw=collapsed_scroll,
                       draw_expanded=lambda s: None),
        SidebarSection(key='a:opts', name='Opts', draw=collapsed_opts,
                       draw_expanded=lambda s: None),
        SidebarSection(key='a:plain', name='Plain', draw=normal_section),
    ]

    plugin_data = {}
    render_ctx = SimpleNamespace(plugin_data=plugin_data)
    player = SimpleNamespace(hud=SimpleNamespace(open_flyout='a:scroll'))
    sctx = SimpleNamespace(
        player=player, render_ctx=render_ctx,
        sidebar_x=1000, sidebar_w=210, y=100,
    )

    QtPlayerRenderer._run_sections(sections, sctx)

    # All three in-place draws fired, including the open flyout's header.
    assert seen == ['scroll-collapsed', 'opts-collapsed', 'normal']
    # Only the open flyout's header recorded an anchor.
    assert set(plugin_data['flyout_anchors'].keys()) == {'a:scroll'}
    x, y, w, h = plugin_data['flyout_anchors']['a:scroll']
    assert (x, y, w, h) == (1000, 100, 210, 20)


def test_toggle_flyout_action_swaps_and_closes():
    """Regression: clicking the same flyout's header twice closes it;
    clicking a different flyout's header while one is open swaps."""
    from analysis.player.hud.hud_state import HudState

    hud = HudState()
    assert hud.open_flyout is None

    # First click: opens scroll.
    hud.open_flyout = (None if hud.open_flyout == 'scroll' else 'scroll')
    assert hud.open_flyout == 'scroll'

    # Click options: swap.
    hud.open_flyout = (None if hud.open_flyout == 'options' else 'options')
    assert hud.open_flyout == 'options'

    # Click options again: close.
    hud.open_flyout = (None if hud.open_flyout == 'options' else 'options')
    assert hud.open_flyout is None


def test_layer_visibility_gates_builtin_draws_but_not_plugin_stage():
    """Regression: a hidden layer in the tree must skip the built-in draw
    fn for that layer, but still fire the plugin stage."""
    drawn = []
    stages = []

    class FakePlugins:
        def draw(self, stage, ctx):
            stages.append(stage)

    renderer = QtPlayerRenderer(plugin_manager=FakePlugins())
    layers_called = set()

    def make_recorder(name):
        def fn(ctx, painter):
            layers_called.add(name)
            drawn.append(name)
        return fn

    # Monkeypatch _layers to a minimal set so we can assert precisely.
    from analysis.player.plugin.plugin_api import Stage
    fake_layers = (
        ('background', make_recorder('background'), None),
        ('lanes',      make_recorder('lanes'),      Stage.AFTER_LANES),
        ('hud',        make_recorder('hud'),        Stage.HUD),
    )
    type(renderer)._layers = property(lambda self: fake_layers)
    try:
        ctx = SimpleNamespace(
            plugin_data={'layer_visibility_tree': (
                SimpleNamespace(key='background', visible=True, children=()),
                SimpleNamespace(key='lanes', visible=False, children=()),
                SimpleNamespace(key='hud', visible=True, children=()),
            )},
        )
        # Stub build_context + PRE_FRAME write.
        renderer.build_context = lambda p, pa, t: ctx
        renderer.draw(player=None, painter=None, t_now=0.0)
    finally:
        del type(renderer)._layers

    assert layers_called == {'background', 'hud'}  # lanes skipped
    # But AFTER_LANES stage still fires.
    assert Stage.AFTER_LANES in stages
    assert Stage.PRE_FRAME in stages
    assert Stage.POST_FRAME in stages


def test_culling_pad_missing_attr_defaults_to_zero():
    """Defensive: old player instances without `max_draw_pad_sec` (e.g.
    external callers building their own player) still cull correctly."""
    times = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    player = SimpleNamespace(
        times=times,
        _note_sv_cum=times,
        sv_enabled=False,
        sv_sections=[],
        notes=SimpleNamespace(ln_indices=[]),
    )
    ctx = _cull_ctx(player, target_lo=0.5, target_hi=1.5)

    assert culling.select_note_candidates(ctx) == [1]
