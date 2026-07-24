"""Opt-in Drawable pipeline (analysis/player/render/storyboard/pipeline.py).

Exercises the end-to-end delegate offscreen on a synthetic frame: a fake
static-doc compiler builds a real ``storyboard_native`` doc, the pipeline
crosses Seam A once, samples the whole static doc per frame (``frame(t)``,
NO feeds - the static doc's Snapshots are static commands), runs the real
GL executor, and presents the composed screen image into the chart rect.

STATIC-DOC ADAPTATION (this wave): the pipeline switched from the
per-frame feed bridge (``drawable_bridge.build_doc`` + ``feed_frame``) to
the static tree-order doc, ASYNC contract (``drawable_doc.prepare_static_doc``
-> recorded ops the pipeline replays). Every fake
here therefore exposes ``prepare_static_doc(compiled) -> (ops, id_maps,
report)`` (a 3-tuple) whose doc carries its OWN items (no dynamic feed
drawable), and the pipeline monkeypatch target is ``pl._load_doc``. The
adaptations from the feed-model tests are documented at each fake:

- ``_fake_doc`` replaces ``_fake_bridge``: the doc's screen root carries a
  SRC_FILL item directly (channel-backed via ``item(...)``), so a plain
  ``frame(t)`` composes the fill - no feed_frame, no dynamic drawable.
- The bytes/coverage feed test is retired: the static doc has no feed
  return to unpack, so there is nothing to serialize. Its intent (the
  pipeline draws through the real evaluator) is covered by the compose
  test. The remaining count is preserved by adding the element-image and
  rebuild-on-growth tests this wave owns.
- ``_field_doc`` replaces ``_field_bridge``: the doc's screen root reads a
  command-less FIELD drawable via a static SRC_DRAWABLE item; the pipeline
  binds a handed capture into it (the D1 behavior, unchanged).
- ``raise_on_build`` replaces ``raise_on_feed`` for the disable test: with
  no per-frame feed, the failure that permanently disables the pipeline is
  raised from the per-frame ``frame`` sampling instead (a wrapped evaluator).

Also guards the two invariants the renderer relies on: the pipeline never
raises out of a frame (any error permanently self-disables), and a build
that cannot proceed reports unavailable rather than crashing.
"""
import os

import numpy as np
import pytest

pytest.importorskip('storyboard_native')
pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import storyboard_native as sn  # noqa: E402
from PySide6.QtCore import QRectF  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtOpenGL import (QOpenGLFramebufferObject,  # noqa: E402
                             QOpenGLPaintDevice)

from analysis.player.render.storyboard import pipeline as pl  # noqa: E402


CHART_RECT = (40, 30, 320, 240)


@pytest.fixture(scope="module")
def gl(_qapp):
    """A current offscreen GL 3+ context (the QOffscreenSurface pattern). The
    pipeline is GL-ONLY, so its drawing tests need a GL painter; this fixture
    skips the whole module when no context can be made (headless box)."""
    from PySide6.QtGui import (QOffscreenSurface, QOpenGLContext,
                               QSurfaceFormat)
    fmt = QSurfaceFormat()
    fmt.setMajorVersion(3)
    fmt.setMinorVersion(2)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    context = QOpenGLContext()
    context.setFormat(fmt)
    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    if not (context.create() and surface.isValid()
            and context.makeCurrent(surface)):
        pytest.skip("no OpenGL context on this platform")
    yield context
    context.doneCurrent()


class _GLTarget:
    """A GL render target the pipeline can present onto: an FBO + a
    QOpenGLPaintDevice-backed QPainter (so ``gl_capture.usable`` sees an
    OpenGL2 engine). Pre-paints the chart rect a backdrop color when asked, so
    a transparent composite proves the backdrop shows through. Reads back for
    assertions only (the app path never reads back; tests may)."""

    def __init__(self, rect, backdrop=None):
        w = rect[0] + rect[2] + 20
        h = rect[1] + rect[3] + 20
        self.fbo = QOpenGLFramebufferObject(
            w, h, QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
        self.fbo.bind()
        self.device = QOpenGLPaintDevice(w, h)
        self.painter = QPainter(self.device)
        self.painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 0))
        if backdrop is not None:
            x, y, rw, rh = rect
            self.painter.fillRect(QRectF(float(x), float(y), float(rw),
                                         float(rh)), backdrop)

    def present(self, pipe, ctx, **kw):
        drew = pipe.delegate(frame=object(), ctx=ctx, painter=self.painter, **kw)
        self.painter.end()
        image = self.fbo.toImage()
        self.fbo.release()
        return drew, image
_SCREEN_W = pl._SCREEN_W
_SCREEN_H = pl._SCREEN_H


class _Ctx:
    def __init__(self, t_now, player):
        self.t_now = t_now
        self.player = player
        self.chart_rect = CHART_RECT


class _Player:
    """Minimal stand-in exposing the two attributes the pipeline reads."""

    def __init__(self, compiled):
        self._adapter = _Adapter(compiled)
        self.replay = {'filepath': 'fake.sm'}


class _Adapter:
    def __init__(self, compiled):
        self._compiled = compiled

    def _compiled_modfile(self, replay):
        return self._compiled


# --------------------------------------------------------------------------
# Fake static-doc compilers (build_static_doc -> (evaluator, id_maps, report))
# --------------------------------------------------------------------------

def _fill_doc_ops():
    """Recorded ops for a doc whose screen root carries ONE static SRC_FILL
    item scaled to the full screen (a white curtain) - the async prepare
    contract: plain (method, args, kwargs) tuples the pipeline replays onto a
    real DocBuilder on the render thread."""
    return [('item', (0, sn.SRC_FILL, 0),
             {'sx_rest': float(_SCREEN_W), 'sy_rest': float(_SCREEN_H)})]


def _spin_present(pipe, ctx, rect=CHART_RECT, tries=200, backdrop=None,
                  **kw):
    """Present repeatedly until the async build (worker prepare + budgeted
    replay) completes and the pipeline draws, or tries run out. Returns the
    last (drew, target). ``backdrop`` is the _GLTarget ctor arg; the rest
    forwards to delegate."""
    import time as _t
    drew, target = False, None
    for _ in range(tries):
        gt = (_GLTarget(rect, backdrop=backdrop) if backdrop is not None
              else _GLTarget(rect))
        drew, target = gt.present(pipe, ctx, **kw)
        if drew or not pipe.healthy:
            break
        # Production waits for the topology settle window before the first
        # prepare; tests fast-forward it (the fakes' topology is static).
        pipe._settle_since = -1e9
        _t.sleep(0.005)
    return drew, target


def _report(**over):
    base = {'instances': 0, 'captures': 0, 'fills': 0, 'aft': 0, 'proxy': 0,
            'z_groups': 0, 'fields': 0, 'slots': 0, 'images': 0,
            'elements_below': 0, 'elements_above': 0, 'element_skips': {}}
    base.update(over)
    return base


class _WrapEvaluator:
    """Wraps a real Evaluator to raise from ``frame`` on demand - installed
    onto an already-built pipeline by the frame-failure test (the fakes can't
    pre-wrap: the pipeline assembles the evaluator itself now)."""

    def __init__(self, inner, raise_on_frame=False):
        self._inner = inner
        self._raise = raise_on_frame

    def frame(self, t):
        if self._raise:
            raise RuntimeError("synthetic frame failure")
        return self._inner.frame(t)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _fake_doc():
    """A fake static-doc compiler for the ASYNC contract:
    ``prepare_static_doc`` returns (ops, id_maps, report); the pipeline
    replays the ops onto a real DocBuilder on the render thread."""

    class Doc:
        @staticmethod
        def prepare_static_doc(compiled, screen_w=_SCREEN_W, screen_h=_SCREEN_H):
            id_maps = {'screen': 0, 'slots': {}, 'fields': {}, 'images': {}}
            return _fill_doc_ops(), id_maps, _report(fills=1, instances=1)

    return Doc()


def _install_doc(monkeypatch, doc):
    monkeypatch.setattr(pl, '_load_doc', lambda: doc)


def _build_pipeline(monkeypatch, doc):
    _install_doc(monkeypatch, doc)
    player = _Player(compiled={'field_instances': [{'name': 'i0'}]})
    pipe = pl.build_pipeline(player)
    assert pipe is not None
    return player, pipe


def test_delegate_composes_and_blits_into_chart_rect(gl, monkeypatch):
    player, pipe = _build_pipeline(monkeypatch, _fake_doc())

    ctx = _Ctx(t_now=1.0, player=player)
    drew, target = _spin_present(pipe, ctx)

    assert drew is True
    x, y, w, h = CHART_RECT
    # The static white fill covers the whole screen drawable, presented into
    # the chart rect: a pixel well inside the rect is opaque white, one outside
    # is clear.
    inside = target.pixelColor(x + w // 2, y + h // 2)
    outside = target.pixelColor(x + w + 5, y + h + 5)
    assert (inside.alpha() > 200 and inside.red() > 200
            and inside.green() > 200 and inside.blue() > 200)
    assert outside.alpha() == 0


def test_delegate_leaves_chart_rect_bounds(gl, monkeypatch):
    player, pipe = _build_pipeline(monkeypatch, _fake_doc())
    ctx = _Ctx(t_now=0.0, player=player)
    _drew, target = _spin_present(pipe, ctx)
    x, y, w, h = CHART_RECT
    # Nothing presented above/left of the chart rect (the present quad covers
    # only the rect region).
    assert target.pixelColor(x - 1, y - 1).alpha() == 0
    assert target.pixelColor(x + w // 2, y - 1).alpha() == 0


def test_frame_exception_disables_permanently(gl, monkeypatch):
    player, pipe = _build_pipeline(monkeypatch, _fake_doc())
    ctx = _Ctx(t_now=0.0, player=player)
    drew, _target = _spin_present(pipe, ctx)
    assert drew is True
    # Sabotage the LIVE evaluator: the next frame's sampling raises, the
    # frame reports "not drawn", and the pipeline permanently disables.
    pipe._evaluator = _WrapEvaluator(pipe._evaluator, raise_on_frame=True)
    drew, _target = _GLTarget(CHART_RECT).present(pipe, ctx)
    assert drew is False
    assert pipe.healthy is False
    # A subsequent delegate also returns False without re-raising.
    drew2, _t2 = _GLTarget(CHART_RECT).present(pipe, ctx)
    assert drew2 is False


def test_missing_doc_compiler_reports_unavailable(monkeypatch):
    monkeypatch.setattr(pl, '_load_doc', lambda: None)
    player = _Player(compiled={'field_instances': [{'name': 'i0'}]})
    assert pl.build_pipeline(player) is None


def test_no_compiled_document_reports_unavailable(monkeypatch):
    _install_doc(monkeypatch, _fake_doc())
    player = _Player(compiled={})
    assert pl.build_pipeline(player) is None


def test_pipeline_for_caches_and_marks_unavailable(monkeypatch):
    monkeypatch.setattr(pl, '_load_doc', lambda: None)
    player = _Player(compiled={'field_instances': [{'name': 'i0'}]})
    assert pl.pipeline_for(player) is None
    # Cached as unavailable: a second probe does not rebuild (still None).
    assert pl.pipeline_for(player) is None
    assert getattr(player, pl._PLAYER_ATTR) is pl._UNAVAILABLE


def test_pipeline_for_returns_healthy_then_caches(monkeypatch):
    _install_doc(monkeypatch, _fake_doc())
    player = _Player(compiled={'field_instances': [{'name': 'i0'}]})
    first = pl.pipeline_for(player)
    assert first is not None and first.healthy
    # Same instance is returned on the next frame (cached on the player).
    assert pl.pipeline_for(player) is first


# --------------------------------------------------------------------------
# Field-capture ingest (D1: bound texture in a command-less field drawable)
# --------------------------------------------------------------------------

def _field_doc():
    """A fake static-doc compiler whose screen root reads a command-less
    FIELD drawable (a static SRC_DRAWABLE item scaled to the full screen)
    with no items of its own, and whose id_maps map the 'field' scope to it.
    With no capture bound the field reads empty; the pipeline binds a handed
    capture into it. Replaces the old ``_field_bridge`` - the field blit is a
    STATIC item, not a fed one."""

    class Doc:
        @staticmethod
        def prepare_static_doc(compiled, screen_w=_SCREEN_W, screen_h=_SCREEN_H):
            # Recorded ops: drawable() mints id 1 (screen root is 0), the
            # screen's item reads it 1:1.
            ops = [('drawable', (float(screen_w), float(screen_h), False,
                                 False), {}),
                   ('item', (0, sn.SRC_DRAWABLE, 1), {})]
            id_maps = {'screen': 0, 'slots': {}, 'images': {},
                       'fields': {'field': 1}}
            return ops, id_maps, _report(proxy=1, fields=1)

    return Doc()


class _FakeGLHandle:
    """A stand-in for gl_capture._GLHandle: an FBO whose ``.texture()`` the
    pipeline binds as a field drawable's content. Fills the FBO a solid color
    so the composited field pixels are checkable."""

    def __init__(self, w, h, color):
        self.fbo = QOpenGLFramebufferObject(w, h)
        self.fbo.bind()
        dev = QOpenGLPaintDevice(w, h)
        p = QPainter(dev)
        p.fillRect(0, 0, w, h, color)
        p.end()
        self.fbo.release()


def test_fed_field_capture_appears_in_the_composite(gl, monkeypatch):
    # The core D1 behavior, GL-side: the renderer hands a live field GL
    # capture; the pipeline binds its FBO texture into the mapped field
    # drawable so the SRC_DRAWABLE field blit draws those real pixels into the
    # chart rect - no readback.
    player, pipe = _build_pipeline(monkeypatch, _field_doc())

    field_capture = _FakeGLHandle(_SCREEN_W, _SCREEN_H,
                                  QColor(0, 180, 255, 255))
    ctx = _Ctx(t_now=0.0, player=player)
    drew, target = _spin_present(pipe, ctx,
                                 field_captures={'field': field_capture})
    assert drew is True
    x, y, w, h = CHART_RECT
    inside = target.pixelColor(x + w // 2, y + h // 2)
    assert (inside.red() < 60 and inside.green() > 120
            and inside.blue() > 200)  # the bound field color composited in


def test_no_field_capture_composites_transparently_not_black(gl, monkeypatch):
    # With no field content bound (and no other opaque source), the composite
    # is TRANSPARENT over the chart rect - the black-region fix: the screen
    # root's clear override means the delegate overlays the backdrop instead
    # of covering it with opaque black.
    player, pipe = _build_pipeline(monkeypatch, _field_doc())
    ctx = _Ctx(t_now=0.0, player=player)
    drew, target = _spin_present(pipe, ctx,
                                 backdrop=QColor(120, 40, 40, 255),
                                 field_captures=None)
    assert drew is True
    # The backdrop shows through the transparent composite (not black).
    x, y, w, h = CHART_RECT
    inside = target.pixelColor(x + w // 2, y + h // 2)
    assert inside.red() > 100 and inside.green() < 70 and inside.blue() < 70


# --------------------------------------------------------------------------
# Element images (this wave): a static-doc SRC_IMAGE blit loads real art
# --------------------------------------------------------------------------

def _write_solid_png(tmp_path, name, color):
    """Write a solid-color PNG and return its absolute path (an element's
    asset). The pipeline's lazy image table loads it as a QImage."""
    path = tmp_path / name
    img = QImage(16, 16, QImage.Format.Format_ARGB32)
    img.fill(color)
    assert img.save(str(path), 'PNG')
    return str(path)


def _image_doc(image_path):
    """A fake static-doc compiler whose screen root draws ONE SRC_IMAGE item
    (image id 0) scaled to the full screen, with ``id_maps['images']`` mapping
    id 0 to ``image_path``. The pipeline must load that path as a QImage and
    hand it to the executor so the image blit draws real art."""

    class Doc:
        @staticmethod
        def prepare_static_doc(compiled, screen_w=_SCREEN_W, screen_h=_SCREEN_H):
            ops = [('item', (0, sn.SRC_IMAGE, 0),
                    {'sx_rest': float(screen_w) / 16.0,
                     'sy_rest': float(screen_h) / 16.0})]
            id_maps = {'screen': 0, 'slots': {}, 'fields': {},
                       'images': {0: image_path}}
            return ops, id_maps, _report(images=1, elements_below=1)

    return Doc()


def test_static_doc_element_image_appears_in_composite(gl, monkeypatch, tmp_path):
    # The element-image deliverable: id_maps['images'] {id -> path} is loaded
    # lazily as a QImage and handed to the GL executor, so a SRC_IMAGE element
    # blit draws the real art into the chart rect.
    green = _write_solid_png(tmp_path, 'green.png', QColor(0, 200, 40, 255).rgba())
    player, pipe = _build_pipeline(monkeypatch, _image_doc(green))
    ctx = _Ctx(t_now=0.0, player=player)
    drew, target = _spin_present(pipe, ctx)
    assert drew is True
    x, y, w, h = CHART_RECT
    inside = target.pixelColor(x + w // 2, y + h // 2)
    # The loaded green image composited across the chart rect.
    assert (inside.red() < 60 and inside.green() > 150
            and inside.blue() < 80)


def test_unreadable_element_image_skips_without_crashing(gl, monkeypatch):
    # An unreadable image path must NOT crash the frame: the lazy table logs
    # once, resolves to None, and the executor draws nothing for that id - the
    # pipeline still presents (a transparent frame), never disabling.
    player, pipe = _build_pipeline(
        monkeypatch, _image_doc('/nonexistent/does_not_exist.png'))
    ctx = _Ctx(t_now=0.0, player=player)
    drew, _target = _spin_present(pipe, ctx)
    assert drew is True
    assert pipe.healthy is True


def test_lazy_images_table_loads_and_tolerates_bad_paths(tmp_path, _qapp):
    # Unit-level: the lazy image table loads a real path as a QImage on first
    # .get (cached), and resolves an unreadable path to None (logged once),
    # never raising.
    good = _write_solid_png(tmp_path, 'ok.png', QColor(255, 0, 0, 255).rgba())
    table = pl._LazyImages({0: good, 1: '/no/such/file.png'})
    first = table.get(0)
    assert isinstance(first, QImage) and not first.isNull()
    assert table.get(0) is first  # cached, one load
    assert table.get(1) is None   # bad path -> None, no raise
    assert table.get(1) is None   # cached None (logged once)


# --------------------------------------------------------------------------
# Staleness: rebuild the static doc when the provider's topology grows
# --------------------------------------------------------------------------

class _GrowingProvider:
    """A lazy field-instance provider whose list GROWS on demand (mirrors the
    sim sweep filling in topology). ``grow`` appends a named instance; each
    call to the provider returns the current list."""

    def __init__(self, names):
        self._names = list(names)

    def grow(self, name):
        self._names.append(name)

    def __call__(self):
        return [{'name': n} for n in self._names]


def _counting_field_doc(builds):
    """A static-doc compiler that counts build_static_doc invocations and
    returns a field-reading doc. ``builds`` is a one-element list mutated per
    build, so a test can assert the pipeline rebuilt on topology growth."""

    class Doc:
        @staticmethod
        def prepare_static_doc(compiled, screen_w=_SCREEN_W, screen_h=_SCREEN_H):
            builds[0] += 1
            ops = [('drawable', (float(screen_w), float(screen_h), False,
                                 False), {}),
                   ('item', (0, sn.SRC_DRAWABLE, 1), {})]
            id_maps = {'screen': 0, 'slots': {}, 'images': {},
                       'fields': {'field': 1}}
            return ops, id_maps, _report(proxy=1, fields=1)

    return Doc()


def test_topology_growth_triggers_a_rebuild(gl, monkeypatch):
    # The static doc reflects a SNAPSHOT of the provider's instance list. When
    # the lazy sweep grows that list (count changes), the pipeline must rebuild
    # the doc so new topology renders - one build at first frame, another after
    # growth, none on a steady frame.
    provider = _GrowingProvider(['a', 'b'])
    builds = [0]
    _install_doc(monkeypatch, _counting_field_doc(builds))
    player = _Player(compiled={'field_instances': provider})
    pipe = pl.build_pipeline(player)
    assert pipe is not None

    ctx = _Ctx(t_now=0.0, player=player)
    drew, _t = _spin_present(pipe, ctx)
    assert drew and builds[0] == 1  # built once (async prepare + replay)

    # A steady frame (no growth) does NOT rebuild.
    _drew, _t = _GLTarget(CHART_RECT).present(pipe, ctx)
    assert builds[0] == 1

    # The sweep grows the instance list. The SETTLE GATE means the next
    # frame only starts tracking (no rebuild while topology churns); the
    # rebuild fires once the signature has been stable for the settle
    # window - simulated by rewinding the settle clock.
    provider.grow('c')
    _drew, _t = _GLTarget(CHART_RECT).present(pipe, ctx)
    assert builds[0] == 1        # churn frame: tracked, not rebuilt
    pipe._settle_since = -1e9    # the growth settled long ago
    import time as _time
    for _ in range(200):         # async re-prepare + replay + adopt
        _drew, _t = _GLTarget(CHART_RECT).present(pipe, ctx)
        if builds[0] == 2 and pipe._assembly is None \
                and pipe._prepared is None:
            break
        _time.sleep(0.005)
    assert builds[0] == 2
    assert pipe.healthy is True


def test_topology_signature_tracks_count_and_last_name():
    # The cheap per-frame signature is (count, last instance name): growth
    # changes the count; an equal-length in-place swap changes the last name.
    prov = _GrowingProvider(['a', 'b'])
    compiled = {'field_instances': prov}
    sig0 = pl._topology_signature(compiled)
    assert sig0 == (2, 'b')
    prov.grow('c')
    assert pl._topology_signature(compiled) == (3, 'c')
    # An empty / absent provider is None (treated as unchanged, no rebuild).
    assert pl._topology_signature({'field_instances': None}) is None


# --------------------------------------------------------------------------
# Real-doc integration (the actual drawable_doc, not a fake)
# --------------------------------------------------------------------------

def test_end_to_end_through_the_real_static_doc(gl, monkeypatch):
    # Integration with the ACTUAL drawable_doc.build_static_doc (not a fake): a
    # synthetic single-proxy chart builds a real static doc and the pipeline
    # samples + presents it without error.
    doc = pytest.importorskip('analysis.games.notitg.drawable_doc')
    fc = pytest.importorskip('analysis.games.notitg.field_compose')
    from analysis.player.render.effects.timeline import Keyframe

    def link(**pokes):
        keyframes = {p: [Keyframe(0.0, (float(v),), 0.0, 0)]
                     for p, v in pokes.items()}
        return fc.link_timelines(keyframes)

    instances = [
        fc.instance('copyA', 'proxy', 1, [link(x=100.0, y=50.0, alpha=0.5)]),
        fc.instance('copyB', 'proxy', 1, [link(x=-40.0, y=10.0, alpha=1.0)]),
    ]
    compiled = {'field_instances': list(instances), 'base_field_hidden': None}
    monkeypatch.setattr(pl, '_load_doc', lambda: doc)
    player = _Player(compiled=compiled)

    pipe = pl.build_pipeline(player)
    assert pipe is not None
    ctx = _Ctx(t_now=0.0, player=player)
    drew, _target = _spin_present(pipe, ctx)
    # The real doc compiled the proxy instances and the pipeline drew a frame
    # without disabling itself.
    assert drew is True
    assert pipe.healthy is True
