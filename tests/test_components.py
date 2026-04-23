"""Tests for the unified component API.

Covers:
- Manifest frozenset normalisation
- ComponentRegistry filters by supported_surfaces + requires_data
- PlayerDataSource answers the fields it claims to support
- OverlayGameStateDataSource answers overlay fields + raises
  DataNotAvailable for sidebar-only ones
- SidebarContext advances the outer sctx.y correctly and
  translates local coords into absolute
- OverlayContext produces a PAL frame with grouped records
  in normalized coords
- Null + Gamescope PAL detection + replay of frames
- Judgments component registers through the unified path
"""
from __future__ import annotations

from types import SimpleNamespace

from analysis.components import (
    Component,
    Manifest,
    ComponentRegistry,
    DataNotAvailable,
    LayerDeclaration,
    LayerPlacement,
    LAYER_AFTER,
    SURFACE_OVERLAY,
    SURFACE_GUI,
)
from analysis.components.pal import detect
from analysis.components.pal.base import (
    CMD_GROUP_BEGIN,
    CMD_GROUP_END,
    CMD_RECT,
    CMD_TEXT,
    OverlayFrame,
)
from analysis.components.pal.null import NullOverlayPlatform
from analysis.components.overlay_backend import (
    OverlayContext,
    OverlayGameStateDataSource,
    draw_component_into_frame,
)
from analysis.components.gui_backend import (
    PlayerDataSource,
    SidebarContext,
    draw_component_in_sidebar,
)
from analysis.overlay.api import OverlayGameState
from analysis.config.store import ConfigStore
from analysis.player.render.layer_registry import LayerRegistry


# ── Manifest ────────────────────────────────────────────


def test_manifest_normalises_sets():
    m = Manifest(
        key='k', name='n',
        supported_surfaces=[SURFACE_GUI, SURFACE_OVERLAY],
        requires_data=['judgment_counts'],
        optional_data=['game'])
    assert isinstance(m.supported_surfaces, frozenset)
    assert isinstance(m.requires_data, frozenset)
    assert isinstance(m.optional_data, frozenset)
    assert SURFACE_GUI in m.supported_surfaces


def test_manifest_normalises_layers():
    m = Manifest(
        key='k',
        name='n',
        supported_surfaces=[SURFACE_GUI],
        layers=[
            LayerDeclaration(
                key='k:after_notes',
                name='After notes',
                placement=LayerPlacement(LAYER_AFTER, 'notes'),
            ),
        ],
    )
    assert isinstance(m.layers, tuple)
    assert m.layers[0].placement.target == 'notes'


# ── ComponentRegistry gating ─────────────────────────────────────


def test_registry_gates_by_surface():
    reg = ComponentRegistry()
    m_sidebar = Manifest(
        key='a', name='A', supported_surfaces={SURFACE_GUI})
    m_both = Manifest(
        key='b', name='B',
        supported_surfaces={SURFACE_GUI, SURFACE_OVERLAY})
    reg.add(m_sidebar, lambda ctx: None)
    reg.add(m_both, lambda ctx: None)
    side = reg.components_for(
        SURFACE_GUI, data_source_fields=frozenset())
    over = reg.components_for(
        SURFACE_OVERLAY, data_source_fields=frozenset())
    assert {c.manifest.key for c in side} == {'a', 'b'}
    assert {c.manifest.key for c in over} == {'b'}


def test_registry_gates_by_required_data():
    reg = ComponentRegistry()
    m = Manifest(
        key='needs_windows', name='X',
        supported_surfaces={SURFACE_OVERLAY},
        requires_data={'judgment_windows'})
    reg.add(m, lambda ctx: None)
    # Overlay source doesn't expose windows → component filtered out.
    out = reg.components_for(
        SURFACE_OVERLAY,
        data_source_fields=OverlayGameStateDataSource._FIELDS)
    assert out == []
    rep = reg.report('needs_windows')
    assert rep[SURFACE_OVERLAY].mounted is False
    assert 'judgment_windows' in rep[SURFACE_OVERLAY].reason


# ── Data sources ─────────────────────────────────────────────────


def _store(tmp_path):
    store = ConfigStore(tmp_path / 'config.json', autosave=False)
    store.load()
    return store


def _fake_player(*, layer_registry=None, hud=None):
    plugins = SimpleNamespace(
        layers=(layer_registry if layer_registry is not None else
                SimpleNamespace(
                    layer_visible=lambda key: key != 'ghost_taps',
                    layer_tree=lambda: (),
                )),
    )
    return SimpleNamespace(
        game='etterna',
        keycount=4,
        windows=[('marv', 0.023), ('perf', 0.045)],
        note_judges=['marv', 'marv', 'perf', 'miss'],
        judge_colors={'marv': (0, 255, 0), 'perf': (255, 255, 0),
                      'miss': (255, 0, 0)},
        judge_label='J4',
        combo=0,
        _render_t_now=12.345,
        play_rate=1.0,
        paused=False,
        sv_enabled=False,
        sv_sections=[],
        times=[0.0, 1.0, 2.0, 3.0],
        skin='bar',
        press_hide=False,
        scroll_mode='ms',
        effective_scroll_ms=1000.0,
        plugins=plugins,
        hud=(hud if hud is not None else SimpleNamespace(
            edit_mode=False,
            layers_panel_open=False,
            plugin_panel_open=False,
            open_flyout=None,
        )),
        _current_mode_value=lambda: 1000.0,
        sv_suspended=lambda: False,
    )


def _player_with_layer_registry(tmp_path, *, layers_panel_open=False):
    hud = SimpleNamespace(
        edit_mode=False,
        layers_panel_open=layers_panel_open,
        plugin_panel_open=False,
        open_flyout=None,
    )
    registry = LayerRegistry(config=_store(tmp_path))
    return _fake_player(layer_registry=registry, hud=hud)


def _invoke_supported_field(ds, field):
    match field:
        case 'layer_visible':
            return ds.layer_visible('notes')
        case _:
            return getattr(ds, field)()


def test_player_data_source_answers_fields():
    ds = PlayerDataSource(_fake_player())
    assert ds.supports('judgment_counts')
    assert ds.supports('t_now')
    assert ds.supports('note_count')
    counts = ds.judgment_counts()
    assert counts == {'marv': 2, 'perf': 1, 'miss': 1}
    assert ds.judge_label() == 'J4'
    assert ds.judgment_windows() == [('marv', 0.023), ('perf', 0.045)]


def test_player_data_source_raises_for_accuracy():
    # accuracy is intentionally not exposed mid-replay; components
    # should branch on DataNotAvailable rather than relying on it.
    ds = PlayerDataSource(_fake_player())
    try:
        ds.accuracy()
    except DataNotAvailable:
        pass
    else:
        raise AssertionError('accuracy should raise DataNotAvailable')


def test_player_data_source_delegates_layer_visibility():
    ds = PlayerDataSource(_fake_player())
    assert ds.layer_visible('notes') is True
    assert ds.layer_visible('ghost_taps') is False


def test_player_data_source_answers_all_supported_fields(tmp_path):
    ds = PlayerDataSource(_player_with_layer_registry(tmp_path))
    results = {
        field: _invoke_supported_field(ds, field)
        for field in sorted(ds._FIELDS)
    }

    assert results['game'] == 'etterna'
    assert results['keycount'] == 4
    assert results['t_now'] == 12.345
    assert results['layer_visible'] is True
    assert results['layer_tree']


def test_overlay_data_source_answers_counts_only():
    state = OverlayGameState(
        game='osu', keycount=4, combo=12, accuracy=0.95,
        judgments=(('300', 10), ('100', 2), ('miss', 1)))
    ds = OverlayGameStateDataSource(state)
    assert ds.combo() == 12
    assert ds.accuracy() == 0.95
    assert ds.judgment_counts() == {'300': 10, '100': 2, 'miss': 1}
    for field in ('judgment_windows', 'judgment_colors', 'judge_label'):
        try:
            getattr(ds, field)()
        except DataNotAvailable:
            continue
        raise AssertionError(f'{field} should raise DataNotAvailable')


# ── Sidebar backend ──────────────────────────────────────────────


class _FakeSctx:
    """Minimal stand-in for SidebarContext.

    Records every text/rect/line/hitbox so tests can assert on the
    absolute coordinates the backend emitted.
    """
    measure_only = True
    painter = None
    renderer = None

    def __init__(self, col_x=500, col_w=200, y=100):
        self.col_x = col_x
        self.col_w = col_w
        self.y = y
        self.texts = []
        self.rects = []
        self.lines = []
        self.hitboxes = []
        # PlayerReplayState is constructed lazily; player can be a stub.
        from types import SimpleNamespace
        self.player = SimpleNamespace()

    def text(self, s, x, baseline, color=None):
        self.texts.append((s, int(x), int(baseline)))

    def rect(self, rect, color=None, outline=None, outline_w=1):
        self.rects.append(tuple(rect))

    def line(self, start, end, color, width=1):
        self.lines.append((tuple(start), tuple(end)))

    def add_hitbox(self, rect, action, payload=None):
        self.hitboxes.append((tuple(rect), action, payload))


def test_sidebar_ctx_translates_local_to_absolute():
    sctx = _FakeSctx(col_x=500, col_w=200, y=100)
    ds = PlayerDataSource(_fake_player())
    cctx = SidebarContext(
        sctx, x0=500, y0=100, w=200, h=0, data_source=ds)
    # Local (10, 20) should land at absolute (510, 120).
    cctx.rect((10, 20, 50, 30))
    assert sctx.rects == [(510, 120, 50, 30)]


def test_sidebar_ctx_advances_outer_y_on_row_helpers():
    sctx = _FakeSctx(y=100)
    ds = PlayerDataSource(_fake_player())
    cctx = SidebarContext(
        sctx, x0=0, y0=100, w=200, h=0, data_source=ds)
    cctx.draw_text('hello')
    # draw_text advances by ROW_TEXT_H (18) by default.
    assert sctx.y == 100 + 18
    assert cctx.y == 18


def test_sidebar_ctx_button_registers_hitbox_at_absolute_rect():
    sctx = _FakeSctx(y=100)
    ds = PlayerDataSource(_fake_player())
    # measure_only on sctx is True which would swallow hitboxes in a real
    # sidebar flow, but our fake records every hitbox regardless — so
    # we can assert on the absolute coords.
    cctx = SidebarContext(
        sctx, x0=500, y0=100, w=200, h=0, data_source=ds)
    cctx.draw_button('go', 'noop')
    assert sctx.hitboxes
    rect, action, _ = sctx.hitboxes[0]
    assert action == 'noop'
    assert rect[0] == 500 and rect[1] == 100


# ── Overlay backend ──────────────────────────────────────────────


def test_overlay_ctx_normalises_coords_and_groups():
    frame = OverlayFrame(width=1000, height=500)
    state = OverlayGameState(game='osu', keycount=4, combo=1)
    ds = OverlayGameStateDataSource(state)
    manifest = Manifest(
        key='x', name='X', supported_surfaces={SURFACE_OVERLAY})
    comp = Component(manifest=manifest, draw=lambda ctx: ctx.rect(
        (0, 0, 100, 50), color=(200, 200, 200)))
    draw_component_into_frame(
        comp, frame, game_state=state,
        origin_px=(250, 100), size_px=(100, 50))
    kinds = [r[0] for r in frame.records]
    assert kinds[0] == CMD_GROUP_BEGIN
    assert kinds[-1] == CMD_GROUP_END
    # Exactly one rect, normalised to (0.25, 0.20, 0.10, 0.10).
    rects = [r for r in frame.records if r[0] == CMD_RECT]
    assert len(rects) == 1
    _, (_id, x, y, w, h, _color, _anchor) = rects[0]
    assert abs(x - 0.25) < 1e-6
    assert abs(y - 0.20) < 1e-6
    assert abs(w - 0.10) < 1e-6
    assert abs(h - 0.10) < 1e-6


def test_overlay_ctx_text_emits_record():
    frame = OverlayFrame(width=1000, height=500)
    state = OverlayGameState(game='osu', keycount=4)
    ds = OverlayGameStateDataSource(state)
    cctx = OverlayContext(
        frame, component_key='x', origin_px=(0, 0),
        size_px=(200, 100), fb_w=1000, fb_h=500,
        data_source=ds)
    cctx.text('combo', 10, 30, color=(255, 255, 255))
    texts = [r for r in frame.records if r[0] == CMD_TEXT]
    assert len(texts) == 1


def test_overlay_ctx_has_no_input():
    frame = OverlayFrame(width=1000, height=500)
    state = OverlayGameState(game='osu', keycount=4)
    ds = OverlayGameStateDataSource(state)
    cctx = OverlayContext(
        frame, component_key='x', origin_px=(0, 0), size_px=(100, 100),
        fb_w=1000, fb_h=500, data_source=ds, supports_input=False)
    assert cctx.supports_input is False


# ── PAL ──────────────────────────────────────────────────────────


def test_null_platform_round_trip():
    null = NullOverlayPlatform()
    assert null.is_available() is False
    handle = null.setup('x', width=1000, height=500)
    frame = OverlayFrame()
    null.submit_frame(handle, frame)  # must not raise
    null.teardown(handle)


def test_detect_returns_a_platform():
    plat = detect()
    # Must produce *some* OverlayPlatform; on CI with no env it's Null.
    assert hasattr(plat, 'setup')
    assert hasattr(plat, 'submit_frame')


# ── End-to-end: discovery bridges judgments ─────────────────────


def test_judgments_registers_through_unified_path():
    """The ported judgments plugin should appear in both the unified
    registry and the legacy sidebar registry after discovery."""
    from analysis.player.plugin_loader import PluginManager

    mgr = PluginManager.discover()
    unified_keys = {c.manifest.key for c in mgr.components.all_components()}
    sidebar_keys = {s.key for s in mgr.sidebar.all_sections()}
    assert 'builtin:judgments' in unified_keys
    assert 'builtin:judgments' in sidebar_keys


def test_status_component_mounts_on_gui():
    from analysis.player.plugin_loader import PluginManager

    mgr = PluginManager.discover()
    report = mgr.components.report('builtin:status')
    assert report[SURFACE_GUI].mounted is True


def test_mounted_gui_component_requirements_are_callable(tmp_path):
    from analysis.player.plugin_loader import PluginManager

    mgr = PluginManager.discover(config=_store(tmp_path))
    try:
        ds = PlayerDataSource(_player_with_layer_registry(tmp_path))
        components = mgr.components.components_for(
            SURFACE_GUI,
            data_source_fields=PlayerDataSource._FIELDS,
        )
        for comp in components:
            for field in sorted(comp.manifest.requires_data):
                _invoke_supported_field(ds, field)
    finally:
        mgr.close()


def test_layers_component_draws_builtin_layers(tmp_path):
    from analysis.player.plugin_loader import PluginManager

    mgr = PluginManager.discover(config=_store(tmp_path))
    try:
        comp = mgr.components.get('builtin:layers')
        assert comp is not None
        sctx = _FakeSctx(y=100)
        sctx.player = _player_with_layer_registry(
            tmp_path,
            layers_panel_open=True,
        )

        draw_component_in_sidebar(comp, sctx, player=sctx.player)

        labels = [text for text, _x, _y in sctx.texts]
        assert 'Background' in labels
        assert 'Notes' in labels
    finally:
        mgr.close()
