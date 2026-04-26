"""Tests for the web-texture PAL: dispatcher selection policy + the
qpixmap backend, stopping short of anything that actually launches a
QWebEngineView.
"""
from __future__ import annotations

import pytest

from analysis.components.pal.web.base import (
    KIND_DMABUF_FD,
    KIND_GL_TEXTURE,
    KIND_QPIXMAP,
    SURFACE_CROSSPROC_GL,
    SURFACE_LOCAL_CPU,
    SURFACE_LOCAL_GL,
    WebTextureBackendCaps,
    WebTextureFrame,
)
from analysis.components.pal.web.dispatcher import NoBackendError, WebTexturePAL


# ---------------------------------------------------------------------------
# Fake backends for dispatcher policy
# ---------------------------------------------------------------------------

class _FakeBackend:
    def __init__(self, name, caps, available=True):
        self.name = name
        self._caps = caps
        self._available = available
        self.created = 0

    def capabilities(self):
        return self._caps

    def is_available(self):
        return self._available

    def create(self, *, width, height):
        self.created += 1
        return object()     # we don't test the texture here


def _caps(surfaces, zero_copy=False, cross_process=False):
    return WebTextureBackendCaps(
        produces=(KIND_QPIXMAP,),
        zero_copy=zero_copy,
        cross_process=cross_process,
        needs_qapplication=False,
        surfaces=frozenset(surfaces),
    )


class TestDispatcher:
    def test_raises_when_no_backends_registered(self):
        pal = WebTexturePAL()
        with pytest.raises(NoBackendError):
            pal.select(SURFACE_LOCAL_CPU)

    def test_raises_when_surface_not_covered(self):
        pal = WebTexturePAL()
        pal.register(_FakeBackend('cpu', _caps({SURFACE_LOCAL_CPU})))
        with pytest.raises(NoBackendError):
            pal.select(SURFACE_CROSSPROC_GL)

    def test_raises_when_backend_unavailable(self):
        pal = WebTexturePAL()
        pal.register(_FakeBackend(
            'cpu', _caps({SURFACE_LOCAL_CPU}), available=False))
        with pytest.raises(NoBackendError):
            pal.select(SURFACE_LOCAL_CPU)

    def test_picks_single_match(self):
        pal = WebTexturePAL()
        b = _FakeBackend('cpu', _caps({SURFACE_LOCAL_CPU}))
        pal.register(b)
        assert pal.select(SURFACE_LOCAL_CPU) is b

    def test_prefers_zero_copy(self):
        """Both backends cover the surface; the zero-copy one wins."""
        pal = WebTexturePAL()
        readback = _FakeBackend(
            'readback', _caps({SURFACE_LOCAL_GL}, zero_copy=False))
        gl = _FakeBackend(
            'gl', _caps({SURFACE_LOCAL_GL}, zero_copy=True))
        # Register readback first; zero-copy should still win.
        pal.register(readback)
        pal.register(gl)
        assert pal.select(SURFACE_LOCAL_GL) is gl

    def test_breaks_ties_by_registration_order(self):
        """Two zero-copy backends: earlier one wins."""
        pal = WebTexturePAL()
        a = _FakeBackend('a', _caps({SURFACE_LOCAL_GL}, zero_copy=True))
        b = _FakeBackend('b', _caps({SURFACE_LOCAL_GL}, zero_copy=True))
        pal.register(a)
        pal.register(b)
        assert pal.select(SURFACE_LOCAL_GL) is a

    def test_skips_unavailable_even_when_zero_copy(self):
        """An offline zero-copy backend must not shadow a live readback."""
        pal = WebTexturePAL()
        pal.register(_FakeBackend(
            'offline_gl', _caps({SURFACE_LOCAL_GL}, zero_copy=True),
            available=False))
        live = _FakeBackend(
            'live_cpu', _caps({SURFACE_LOCAL_GL}, zero_copy=False))
        pal.register(live)
        assert pal.select(SURFACE_LOCAL_GL) is live

    def test_create_routes_through_select(self):
        pal = WebTexturePAL()
        b = _FakeBackend('cpu', _caps({SURFACE_LOCAL_CPU}))
        pal.register(b)
        pal.create(surface=SURFACE_LOCAL_CPU, width=100, height=50)
        assert b.created == 1


# ---------------------------------------------------------------------------
# Default PAL factory
# ---------------------------------------------------------------------------

class TestDefaultPAL:
    def setup_method(self):
        WebTexturePAL.reset_default_for_tests()

    def teardown_method(self):
        WebTexturePAL.reset_default_for_tests()

    def test_lazy_init(self):
        """default() returns the same instance on repeat calls."""
        pal1 = WebTexturePAL.default()
        pal2 = WebTexturePAL.default()
        assert pal1 is pal2

    def test_custom_factory(self):
        called = []
        def factory(pal):
            called.append(pal)
            pal.register(_FakeBackend('custom', _caps({SURFACE_LOCAL_CPU})))
        pal = WebTexturePAL.default(factory=factory)
        assert called == [pal]
        assert pal.select(SURFACE_LOCAL_CPU).name == 'custom'

    def test_builtin_registers_qpixmap(self):
        """With no factory override, the qpixmap backend is registered."""
        pal = WebTexturePAL.default()
        names = [b.name for b in pal.backends()]
        assert 'qpixmap' in names


# ---------------------------------------------------------------------------
# QPixmapBackend capability shape
# ---------------------------------------------------------------------------

class TestQPixmapBackendCaps:
    """We don't instantiate the QWebEngineView here (needs QApplication);
    we just validate the capability object is correct so the dispatcher
    picks it for the right surfaces."""

    def test_capabilities_advertise_cpu_and_gl_surfaces(self):
        from analysis.components.pal.web.qpixmap_backend import QPixmapBackend
        caps = QPixmapBackend().capabilities()
        assert SURFACE_LOCAL_CPU in caps.surfaces
        assert SURFACE_LOCAL_GL  in caps.surfaces
        assert caps.zero_copy is False
        assert caps.cross_process is False
        assert KIND_QPIXMAP in caps.produces

    def test_capabilities_do_not_claim_crossprocess(self):
        from analysis.components.pal.web.qpixmap_backend import QPixmapBackend
        caps = QPixmapBackend().capabilities()
        assert SURFACE_CROSSPROC_GL not in caps.surfaces


# ---------------------------------------------------------------------------
# Frame object shape
# ---------------------------------------------------------------------------

class TestWebTextureFrame:
    def test_fields_populated(self):
        f = WebTextureFrame(
            width=800, height=600, kind=KIND_QPIXMAP,
            handle=object(), generation=3)
        assert f.width == 800
        assert f.generation == 3
        assert f.wait_token is None
        assert f.meta == {}

    def test_meta_is_per_instance(self):
        """Default-factory dict must not be shared across instances."""
        a = WebTextureFrame(width=1, height=1, kind=KIND_GL_TEXTURE,
                            handle=42, generation=0)
        b = WebTextureFrame(width=1, height=1, kind=KIND_DMABUF_FD,
                            handle=43, generation=0)
        a.meta['modifier'] = 'foo'
        assert 'modifier' not in b.meta


# ---------------------------------------------------------------------------
# _pixmap_from dispatch helper (GUI backend)
# ---------------------------------------------------------------------------

class TestPixmapFromHelper:
    """The ``ctx.image()`` primitive on the GUI backend dispatches to
    ``_pixmap_from`` to resolve a WebTextureFrame (or raw QPixmap) into
    something ``QPainter.drawPixmap`` can draw. Covers the dispatch
    without needing a live QApplication -- QPixmap construction is OK
    as long as we don't try to paint."""

    @pytest.fixture
    def qapp(self):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    def test_raw_qpixmap_passes_through(self, qapp):
        from PySide6.QtGui import QPixmap
        from analysis.components.gui_backend import _pixmap_from
        pix = QPixmap(10, 10)
        assert _pixmap_from(pix) is pix

    def test_qpixmap_kinded_frame(self, qapp):
        from PySide6.QtGui import QPixmap
        from analysis.components.gui_backend import _pixmap_from
        pix = QPixmap(4, 4)
        frame = WebTextureFrame(
            width=4, height=4, kind=KIND_QPIXMAP, handle=pix, generation=1)
        assert _pixmap_from(frame) is pix

    def test_unknown_kind_returns_none(self, qapp):
        from analysis.components.gui_backend import _pixmap_from
        frame = WebTextureFrame(
            width=4, height=4, kind=KIND_DMABUF_FD,
            handle={'fd': 42}, generation=1)
        assert _pixmap_from(frame) is None

    def test_meta_fallback_pixmap(self, qapp):
        """A future GL-backed frame that ships a downgrade copy in
        ``meta['qpixmap_fallback']`` is still drawable."""
        from PySide6.QtGui import QPixmap
        from analysis.components.gui_backend import _pixmap_from
        pix = QPixmap(8, 8)
        frame = WebTextureFrame(
            width=8, height=8, kind=KIND_GL_TEXTURE,
            handle=123, generation=1,
            meta={'qpixmap_fallback': pix})
        assert _pixmap_from(frame) is pix

    def test_bare_non_frame_returns_none(self, qapp):
        from analysis.components.gui_backend import _pixmap_from
        assert _pixmap_from(None) is None
        assert _pixmap_from('oops') is None
        assert _pixmap_from(42) is None


# ---------------------------------------------------------------------------
# DmabufBackend (Linux-only; may be absent on Windows CI)
# ---------------------------------------------------------------------------

class TestDmabufBackendCaps:
    """The DmabufBackend targets the cross-process gl_layer path. It
    must *only* advertise SURFACE_CROSSPROC_GL; claiming local surfaces
    would pull every local PAL caller onto the dmabuf path and then
    fail on is_available() because no overlay socket exists."""

    def _backend(self):
        pytest.importorskip('analysis.components.pal.web.dmabuf_backend')
        from analysis.components.pal.web.dmabuf_backend import DmabufBackend
        return DmabufBackend()

    def test_name_is_dmabuf(self):
        assert self._backend().name == 'dmabuf'

    def test_capabilities_exclusive_to_crossproc_gl(self):
        caps = self._backend().capabilities()
        assert caps.surfaces == frozenset({SURFACE_CROSSPROC_GL})
        assert SURFACE_LOCAL_CPU not in caps.surfaces
        assert SURFACE_LOCAL_GL not in caps.surfaces

    def test_capabilities_claim_zero_copy_and_crossprocess(self):
        caps = self._backend().capabilities()
        assert caps.zero_copy is True
        assert caps.cross_process is True
        assert KIND_DMABUF_FD in caps.produces

    def test_is_available_false_without_overlay_socket(self, tmp_path,
                                                       monkeypatch):
        """The overlay socket's existence is the liveness probe. When
        it's absent (normal case in CI / dev without running osu!), the
        backend must decline so the PAL falls through to qpixmap."""
        b = self._backend()
        # Redirect to a path that definitely doesn't exist.
        monkeypatch.setenv('VSRG_OVERLAY_WEB_SOCKET',
                           str(tmp_path / 'no-such-sock'))
        # Need a QApplication live for the other availability checks
        # to pass the first gate; after that, the missing socket must
        # flip is_available() to False.
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            QApplication([])
        assert b.is_available() is False


class TestDmabufBackendRegisteredBeforeQPixmap:
    """The default PAL factory must register the dmabuf backend before
    qpixmap so that SURFACE_CROSSPROC_GL gets dmabuf (when available)
    and never accidentally falls through to a local-only backend."""

    def setup_method(self):
        WebTexturePAL.reset_default_for_tests()

    def teardown_method(self):
        WebTexturePAL.reset_default_for_tests()

    def test_dmabuf_registered_when_crate_present(self):
        # The crate is expected to build on the dev box; if it's not
        # present, the dispatcher silently skips it and this test is
        # an xfail on that host.
        try:
            import web_texture_ipc  # noqa: F401
        except ImportError:
            pytest.skip('web_texture_ipc extension not built')

        pal = WebTexturePAL.default()
        names = [b.name for b in pal.backends()]
        assert 'dmabuf' in names
        # And: dmabuf comes first so it wins surface matches when
        # available. qpixmap is the floor.
        assert names.index('dmabuf') < names.index('qpixmap')
