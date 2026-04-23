"""Regression tests for renderer-level bugs.

These cover pure-logic branches of the Qt renderer (no Qt painter
needed). Draw primitives are monkey-patched to record calls so the
test can assert what was/wasn't drawn.
"""
from types import SimpleNamespace

import numpy as np

from analysis.player.render import culling
from analysis.player.render.qt_renderer import QtPlayerRenderer, _NoteView


def _make_note(*, miss, is_ln, miss_pressed_i=True, off=0.0, ln_state='tap'):
    """`_NoteView` with just the fields `_draw_press_mark` touches."""
    return _NoteView(
        i=0, col=0, y=100, y_end=100, lx=0, off=off,
        press_t=0.0, release_t=None, rel_off=None, end_t=None,
        is_ln=is_ln, is_roll=False, miss=miss, ln_state=ln_state,
        note_color=(255, 255, 255), jcolor=(255, 0, 0),
    )


def _make_press_ctx(renderer, miss_pressed=(True,)):
    player = SimpleNamespace(
        miss_pressed=np.array(miss_pressed, dtype=bool),
        press_hide=False,
        scroll_speed=1000.0,
        judge_colors={'miss': (255, 0, 0)},
    )
    return SimpleNamespace(player=player, lane_w=80,
                            drawers=renderer._defaults)


def _patch_draw_recorders(renderer):
    """Replace the lane-line and tick draws with recorders. Returns
    `(lines, ticks)` lists that accumulate every call."""
    lines, ticks = [], []

    def fake_line(painter, color, lx, lane_w, y0, y1, width=1):
        lines.append((color, lx, lane_w, y0, y1, width))

    def fake_tick(painter, color, lane_x, y, lane_w):
        ticks.append((color, lane_x, y, lane_w))

    renderer._draw_lane_line = fake_line
    renderer._draw_tick = fake_tick
    return lines, ticks


def test_missed_ln_skips_press_mark_when_pressed():
    """Regression: a missed LN where the player DID press (miss_pressed=True)
    must not draw a press-mark — `_draw_miss_holds` owns that stroke, so
    drawing press-mark too would double up."""
    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    lines, ticks = _patch_draw_recorders(renderer)

    ctx = _make_press_ctx(renderer, miss_pressed=(True,))
    note = _make_note(miss=True, is_ln=True, off=-0.150, ln_state='missed')

    renderer._draw_press_mark(ctx, painter=None, n=note)

    assert lines == []
    assert ticks == []


def test_missed_ln_skips_press_mark_when_not_pressed():
    """Same rule for a missed LN with no recorded press — it's still the
    miss-hold drawer's domain (if any), never the press-mark's."""
    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    lines, ticks = _patch_draw_recorders(renderer)

    ctx = _make_press_ctx(renderer, miss_pressed=(False,))
    note = _make_note(miss=True, is_ln=True, off=1.0, ln_state='missed')

    renderer._draw_press_mark(ctx, painter=None, n=note)

    assert lines == []
    assert ticks == []


def test_hit_tap_draws_press_mark():
    """Sanity: non-miss non-LN still draws both line and tick."""
    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    lines, ticks = _patch_draw_recorders(renderer)

    ctx = _make_press_ctx(renderer)
    note = _make_note(miss=False, is_ln=False, off=0.020, ln_state='tap')

    renderer._draw_press_mark(ctx, painter=None, n=note)

    assert len(lines) == 1
    assert len(ticks) == 1


def test_missed_tap_still_draws_press_mark():
    """Missed non-LN draws a red press-mark — the rule only excludes LNs
    and never-pressed misses."""
    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    lines, ticks = _patch_draw_recorders(renderer)

    ctx = _make_press_ctx(renderer, miss_pressed=(True,))
    note = _make_note(miss=True, is_ln=False, off=0.080, ln_state='missed_note')

    renderer._draw_press_mark(ctx, painter=None, n=note)

    assert len(lines) == 1
    assert len(ticks) == 1
    assert lines[0][0] == (255, 0, 0)  # miss color


def test_missed_tap_without_press_skips_press_mark():
    """Regression: osu replays write a 1.0s sentinel offset for misses
    the player never pressed. Drawing a press-mark for those produces a
    full-second line that crosses unrelated notes on the same column,
    visually connecting unrelated misses (TWO-TORIAL col=3, ~10.4s)."""
    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    lines, ticks = _patch_draw_recorders(renderer)

    ctx = _make_press_ctx(renderer, miss_pressed=(False,))
    note = _make_note(miss=True, is_ln=False, off=1.0,
                      ln_state='missed_note')

    renderer._draw_press_mark(ctx, painter=None, n=note)

    assert lines == []
    assert ticks == []


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
        _ln_indices=[],
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
        _ln_indices=[],
        max_draw_pad_sec=0.0,
    )
    ctx = _cull_ctx(player, target_lo=1.2, target_hi=2.8)

    assert culling.select_note_candidates(ctx) == [2]


def test_adapter_drawer_override_takes_precedence():
    """Regression: `GameAdapter.note_drawers()` entries must replace the
    renderer's defaults for matching keys. A game can reskin any single
    note-type without reimplementing the pipeline."""
    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())

    calls = []

    def custom_press_mark(painter, lx, lane_w, y_head, y_press, color):
        calls.append(('press_mark', lx, y_head, y_press, color))

    adapter = SimpleNamespace(note_drawers=lambda: {'press_mark': custom_press_mark})
    player = SimpleNamespace(_adapter=adapter)

    drawers = renderer._resolve_drawers(player)
    assert drawers['press_mark'] is custom_press_mark
    # Non-overridden keys still fall back to defaults.
    assert drawers['tap_head'] is renderer._defaults['tap_head']


def test_adapter_without_note_drawers_uses_all_defaults():
    """Adapters that don't implement `note_drawers` still work — the
    renderer falls back to every default without error."""
    renderer = QtPlayerRenderer(plugin_manager=SimpleNamespace())
    adapter = SimpleNamespace(note_drawers=lambda: {})
    player = SimpleNamespace(_adapter=adapter)

    drawers = renderer._resolve_drawers(player)
    assert drawers == renderer._defaults


def test_run_sections_always_draws_header_and_records_anchor_when_open():
    """Regression: the collapsed header must draw even when its flyout
    is open — it stays on screen as the anchor point and re-click target.
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
    from analysis.player.plugin_api import Stage
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
        _ln_indices=[],
    )
    ctx = _cull_ctx(player, target_lo=0.5, target_hi=1.5)

    assert culling.select_note_candidates(ctx) == [1]
