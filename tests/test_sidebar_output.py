"""Backend-agnostic sidebar rendering tests.

These tests drive the same ``draw`` callable through both the QImage
(CPU) backend and the OpenGL (QOpenGLPaintDevice) backend, and assert:

1. Each backend produces non-empty output in the right rows.
2. The two backends produce visually-equivalent rasters (similarity
   near 1.0) for the primitives the sidebar uses.

The OpenGL backend is skipped when the host can't create a GL context
(headless CI, no display). Goldens live under ``tests/goldens/sidebar/``
and are regenerated with ``UPDATE_GOLDENS=1`` in the environment.

The point of these tests is to stay honest across a future migration
from QPainter-on-QWidget to QPainter-on-QOpenGLPaintDevice (Phase 2):
they assert on *output*, not on specific Qt method call sequences, so
they pass regardless of which backend the renderer plumbs through.
"""
from __future__ import annotations

import pytest

from tests._sidebar_harness import (
    ALL_BACKENDS,
    BACKEND_OPENGL,
    BACKEND_QIMAGE,
    image_contains_any_nonbackground,
    raster_similarity,
    render_sidebar_section_to_image,
)


# ---------------------------------------------------------------------------
# Scenarios: small, self-contained sidebar draws
# ---------------------------------------------------------------------------
# Each one is a ``draw(sctx)`` callable that exercises a subset of the
# SidebarContext primitives. Parametrize both backends across all of
# them.

def _draw_heading_only(sctx):
    sctx.draw_heading('Panel')


def _draw_heading_and_text(sctx):
    sctx.draw_heading('Stats')
    sctx.draw_text('total: 42')
    sctx.draw_hint('(approx)')


def _draw_button_row(sctx):
    sctx.draw_heading('Controls')
    sctx.draw_button('Apply', 'apply')


def _draw_primitives(sctx):
    sctx.rect((10, 10, 40, 20), (200, 80, 40))
    sctx.line((0, 50), (100, 80), (255, 0, 0), width=2)
    sctx.checkbox(5, 90, True)


SCENARIOS = {
    'heading_only':      _draw_heading_only,
    'heading_and_text':  _draw_heading_and_text,
    'button_row':        _draw_button_row,
    'primitives_mix':    _draw_primitives,
}


# ---------------------------------------------------------------------------
# Per-backend output validity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', ALL_BACKENDS)
@pytest.mark.parametrize('name', list(SCENARIOS))
def test_backend_produces_nonempty_output(backend, name):
    """Each draw scenario produces *something* visible through each
    backend. The qimage path always succeeds; the opengl path skips
    when GL isn't available (headless CI)."""
    image = render_sidebar_section_to_image(
        SCENARIOS[name], backend=backend, width=240, height=200)
    if image is None:
        pytest.skip(f'backend {backend!r} unavailable on this host')

    assert image.width() == 240
    assert image.height() == 200
    # The image was filled with black background; any draw call must
    # leave at least one non-background pixel in the sidebar column.
    assert image_contains_any_nonbackground(image, (0, 0, 240, 200)), \
        f'{name} produced an empty raster under {backend}'


# ---------------------------------------------------------------------------
# Cross-backend equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', list(SCENARIOS))
def test_qimage_and_opengl_produce_similar_rasters(name):
    """For the primitives the sidebar draws, swapping the paint device
    from QImage to QOpenGLPaintDevice must not change the visible
    output materially. A similarity threshold of 0.99 tolerates tiny
    antialiasing differences on diagonal lines while still catching
    genuine rendering regressions."""
    img_q = render_sidebar_section_to_image(
        SCENARIOS[name], backend=BACKEND_QIMAGE, width=240, height=200)
    img_gl = render_sidebar_section_to_image(
        SCENARIOS[name], backend=BACKEND_OPENGL, width=240, height=200)

    if img_gl is None:
        pytest.skip('opengl backend unavailable on this host')

    sim = raster_similarity(img_q, img_gl)
    assert sim >= 0.99, (
        f'{name}: qimage vs opengl similarity {sim:.4f} below 0.99')


# ---------------------------------------------------------------------------
# Raster-similarity metric self-checks
# ---------------------------------------------------------------------------

class TestRasterSimilarity:
    def test_identical_images(self):
        img = render_sidebar_section_to_image(
            _draw_heading_only, backend=BACKEND_QIMAGE,
            width=100, height=50)
        assert raster_similarity(img, img) == 1.0

    def test_size_mismatch_returns_zero(self):
        a = render_sidebar_section_to_image(
            _draw_heading_only, backend=BACKEND_QIMAGE,
            width=100, height=50)
        b = render_sidebar_section_to_image(
            _draw_heading_only, backend=BACKEND_QIMAGE,
            width=50, height=100)
        assert raster_similarity(a, b) == 0.0

    def test_different_content_reduces_similarity(self):
        a = render_sidebar_section_to_image(
            _draw_heading_only, backend=BACKEND_QIMAGE,
            width=100, height=50)
        b = render_sidebar_section_to_image(
            lambda s: None, backend=BACKEND_QIMAGE,
            width=100, height=50)  # empty draw
        assert raster_similarity(a, b) < 1.0

    def test_none_input_returns_zero(self):
        img = render_sidebar_section_to_image(
            _draw_heading_only, backend=BACKEND_QIMAGE,
            width=50, height=50)
        assert raster_similarity(None, img) == 0.0
        assert raster_similarity(img, None) == 0.0


# ---------------------------------------------------------------------------
# End-to-end: drive the real layers component through the real sidebar
# rendering path on both backends and verify equivalence.
# ---------------------------------------------------------------------------

class TestLayersComponentRasterEquivalence:
    """Complement ``test_layers_component_draws_builtin_layers`` in
    ``test_components.py``: that test asserts on the Context calls the
    component made (logical contract); this one asserts the *rendered
    output* matches across both backends (renderer contract)."""

    def _setup(self, tmp_path):
        """Return ``(draw_fn, player, mgr)`` for the real layers
        component against a minimal fake player with a populated
        ``LayerRegistry``. The harness wires ``player`` through
        ``render_ctx.player`` for us, so ``draw_fn`` just invokes the
        component against the supplied ``sctx``.
        """
        from analysis.components.gui_backend import (
            draw_component_in_sidebar)
        from analysis.config.store import ConfigStore
        from analysis.player.plugin.plugin_loader import PluginManager
        from analysis.player.render.layer_registry import LayerRegistry
        from types import SimpleNamespace

        store = ConfigStore(tmp_path / 'config.json', autosave=False)
        store.load()
        registry = LayerRegistry(config=store)
        hud = SimpleNamespace(edit_mode=False, layers_panel_open=True,
                              plugin_panel_open=False, open_flyout=None,
                              add_hitbox=lambda *a, **kw: None)
        plugins = SimpleNamespace(layers=registry)
        player = SimpleNamespace(
            game='etterna', keycount=4,
            windows=[('marv', 0.023), ('perf', 0.045)],
            note_judges=[], judge_colors={}, judge_label='J4',
            combo=0, _render_t_now=0.0, play_rate=1.0, paused=True,
            sv_enabled=False, sv_sections=[], times=[], skin='bar',
            press_hide=False, scroll_mode='ms', effective_scroll_ms=1000.0,
            plugins=plugins, hud=hud,
            _current_mode_value=lambda: 1000.0,
            sv_suspended=lambda: False,
            replay={},
        )

        mgr = PluginManager.discover(config=store)
        comp = mgr.components.get('builtin:layers')
        assert comp is not None, 'builtin:layers component missing'

        def draw(sctx):
            draw_component_in_sidebar(comp, sctx, player=player)

        return draw, player, mgr

    def test_both_backends_render_nonempty(self, tmp_path):
        draw, player, mgr = self._setup(tmp_path)
        try:
            img_q = render_sidebar_section_to_image(
                draw, backend=BACKEND_QIMAGE,
                width=240, height=300, player=player)
            img_gl = render_sidebar_section_to_image(
                draw, backend=BACKEND_OPENGL,
                width=240, height=300, player=player)

            assert image_contains_any_nonbackground(img_q, (0, 0, 240, 300))
            if img_gl is not None:
                assert image_contains_any_nonbackground(
                    img_gl, (0, 0, 240, 300))
        finally:
            mgr.close()

    def test_backends_produce_equivalent_rasters(self, tmp_path):
        draw, player, mgr = self._setup(tmp_path)
        try:
            img_q = render_sidebar_section_to_image(
                draw, backend=BACKEND_QIMAGE,
                width=240, height=300, player=player)
            img_gl = render_sidebar_section_to_image(
                draw, backend=BACKEND_OPENGL,
                width=240, height=300, player=player)
            if img_gl is None:
                pytest.skip('opengl backend unavailable on this host')
            sim = raster_similarity(img_q, img_gl)
            assert sim >= 0.99, (
                f'layers component: qimage vs opengl similarity {sim:.4f} '
                f'below 0.99')
        finally:
            mgr.close()
