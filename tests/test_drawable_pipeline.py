"""Opt-in Drawable pipeline (analysis/player/render/storyboard/pipeline.py).

Exercises the end-to-end delegate offscreen on a synthetic frame: a fake
B3 bridge builds a real ``storyboard_native`` doc, the pipeline crosses
Seam A once, feeds a per-frame item stream across Seam B, runs the real
RasterExecutor, and blits the composed screen image into the chart rect.

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
from PySide6.QtGui import QImage, QPainter  # noqa: E402

from analysis.player.render.storyboard import pipeline as pl  # noqa: E402


CHART_RECT = (40, 30, 320, 240)


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


def _fake_bridge(fed_color=(1.0, 0.0, 0.0), raise_on_feed=False,
                 bytes_feed=False):
    """A fake B3 bridge: build_doc returns a real Evaluator whose screen
    consumes one DYNAMIC drawable; feed_frame emits one fed fill item that
    the executor rasterizes red into the screen.

    ``bytes_feed`` mirrors the real B3 contract: feed_frame returns a
    5-tuple whose buffers are already ``.tobytes()`` and whose trailing
    element is a coverage dict."""

    class Bridge:
        @staticmethod
        def build_doc(compiled, screen_w=640, screen_h=480):
            b = sn.DocBuilder(float(screen_w), float(screen_h))
            notes = b.drawable(float(screen_w), float(screen_h), False, True)
            b.item(0, sn.SRC_DRAWABLE, notes)
            evaluator = b.finish()
            id_maps = {
                'images': {},
                'drawable_sizes': [(screen_w, screen_h), (screen_w, screen_h)],
                'dynamic_id': notes,
            }
            return evaluator, id_maps

        @staticmethod
        def feed_frame(compiled, t, id_maps):
            if raise_on_feed:
                raise RuntimeError("synthetic feed failure")
            notes = id_maps['dynamic_id']
            fu = 4
            ff = 14
            u = np.zeros((1, fu), dtype=np.uint32)
            f = np.zeros((1, ff), dtype=np.float32)
            # One fed fill item covering the whole 640x480 screen, opaque.
            # Feed f32 layout is TRS: [tx, ty, sx, sy, rot, opacity, tint..].
            u[0] = [sn.SRC_FILL, 0, 0, 0]
            f[0, 0] = 0.0     # tx
            f[0, 1] = 0.0     # ty
            f[0, 2] = 640.0   # sx (scale the unit fill quad to full width)
            f[0, 3] = 480.0   # sy (full height)
            f[0, 4] = 0.0     # rot
            f[0, 5] = 1.0     # opacity
            f[0, 6:9] = fed_color  # tint rgb
            if bytes_feed:
                # The real B3 shape: serialized buffers + a coverage dict.
                coverage = {'translated': 1, 'skipped_projective': 0,
                            'total': 1}
                return [notes], [1], u.tobytes(), f.tobytes(), coverage
            return [notes], [1], u, f

    return Bridge()


def _install_bridge(monkeypatch, bridge):
    monkeypatch.setattr(pl, '_load_bridge', lambda: bridge)


def _screen_image(rect):
    img = QImage(rect[0] + rect[2] + 20, rect[1] + rect[3] + 20,
                 QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    return img


def _build_pipeline(monkeypatch, bridge):
    _install_bridge(monkeypatch, bridge)
    player = _Player(compiled={'field_instances': [object()]})
    pipe = pl.build_pipeline(player)
    assert pipe is not None
    return player, pipe


def test_delegate_composes_and_blits_into_chart_rect(monkeypatch):
    player, pipe = _build_pipeline(monkeypatch, _fake_bridge())

    target = _screen_image(CHART_RECT)
    painter = QPainter(target)
    ctx = _Ctx(t_now=1.0, player=player)
    try:
        drew = pipe.delegate(frame=object(), ctx=ctx, painter=painter)
    finally:
        painter.end()

    assert drew is True
    x, y, w, h = CHART_RECT
    # The fed red fill covers the whole screen drawable, blitted into the
    # chart rect: a pixel well inside the rect is red, one outside is clear.
    inside = target.pixelColor(x + w // 2, y + h // 2)
    outside = target.pixelColor(x + w + 5, y + h + 5)
    assert inside.red() > 200 and inside.green() < 60 and inside.blue() < 60
    assert outside.alpha() == 0


def test_delegate_accepts_bridge_bytes_and_coverage(monkeypatch):
    # The real B3 feed_frame returns serialized buffers + a coverage dict;
    # the pipeline must unpack the 5-tuple and pass the bytes straight to
    # the evaluator, drawing the same red fill.
    player, pipe = _build_pipeline(
        monkeypatch, _fake_bridge(bytes_feed=True))
    target = _screen_image(CHART_RECT)
    painter = QPainter(target)
    ctx = _Ctx(t_now=0.5, player=player)
    try:
        drew = pipe.delegate(frame=object(), ctx=ctx, painter=painter)
    finally:
        painter.end()
    assert drew is True
    x, y, w, h = CHART_RECT
    inside = target.pixelColor(x + w // 2, y + h // 2)
    assert inside.red() > 200 and inside.green() < 60 and inside.blue() < 60


def test_delegate_leaves_chart_rect_bounds(monkeypatch):
    player, pipe = _build_pipeline(monkeypatch, _fake_bridge())
    target = _screen_image(CHART_RECT)
    painter = QPainter(target)
    ctx = _Ctx(t_now=0.0, player=player)
    try:
        pipe.delegate(frame=object(), ctx=ctx, painter=painter)
    finally:
        painter.end()
    x, y, w, h = CHART_RECT
    # Nothing painted above/left of the chart rect (the blit is clipped to
    # the rect region by the drawImage target rectangle).
    assert target.pixelColor(x - 1, y - 1).alpha() == 0
    assert target.pixelColor(x + w // 2, y - 1).alpha() == 0


def test_frame_exception_disables_permanently(monkeypatch):
    player, pipe = _build_pipeline(
        monkeypatch, _fake_bridge(raise_on_feed=True))
    target = _screen_image(CHART_RECT)
    painter = QPainter(target)
    ctx = _Ctx(t_now=0.0, player=player)
    try:
        drew = pipe.delegate(frame=object(), ctx=ctx, painter=painter)
    finally:
        painter.end()
    # The feed raised: the frame reports "not drawn" and the pipeline is
    # now permanently disabled for the session.
    assert drew is False
    assert pipe.healthy is False
    # A subsequent delegate also returns False without re-raising.
    target2 = _screen_image(CHART_RECT)
    painter2 = QPainter(target2)
    try:
        assert pipe.delegate(frame=object(), ctx=ctx, painter=painter2) is False
    finally:
        painter2.end()


def test_missing_bridge_reports_unavailable(monkeypatch):
    monkeypatch.setattr(pl, '_load_bridge', lambda: None)
    player = _Player(compiled={'field_instances': [object()]})
    assert pl.build_pipeline(player) is None


def test_no_compiled_document_reports_unavailable(monkeypatch):
    _install_bridge(monkeypatch, _fake_bridge())
    player = _Player(compiled={})
    assert pl.build_pipeline(player) is None


def test_pipeline_for_caches_and_marks_unavailable(monkeypatch):
    monkeypatch.setattr(pl, '_load_bridge', lambda: None)
    player = _Player(compiled={'field_instances': [object()]})
    assert pl.pipeline_for(player) is None
    # Cached as unavailable: a second probe does not rebuild (still None).
    assert pl.pipeline_for(player) is None
    assert getattr(player, pl._PLAYER_ATTR) is pl._UNAVAILABLE


def test_pipeline_for_returns_healthy_then_caches(monkeypatch):
    _install_bridge(monkeypatch, _fake_bridge())
    player = _Player(compiled={'field_instances': [object()]})
    first = pl.pipeline_for(player)
    assert first is not None and first.healthy
    # Same instance is returned on the next frame (cached on the player).
    assert pl.pipeline_for(player) is first


def test_end_to_end_through_the_real_bridge(monkeypatch):
    # Integration with B3's ACTUAL drawable_bridge (not the fake): a
    # synthetic single-proxy chart builds a real doc and feeds a real
    # entry stream; the pipeline composes and blits without error.
    bridge = pytest.importorskip('analysis.games.notitg.drawable_bridge')
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
    monkeypatch.setattr(pl, '_load_bridge', lambda: bridge)
    player = _Player(compiled=compiled)

    pipe = pl.build_pipeline(player)
    assert pipe is not None
    target = _screen_image(CHART_RECT)
    painter = QPainter(target)
    ctx = _Ctx(t_now=0.0, player=player)
    try:
        drew = pipe.delegate(frame=object(), ctx=ctx, painter=painter)
    finally:
        painter.end()
    # The real bridge translated the proxy entries into the feed and the
    # pipeline drew a frame without disabling itself.
    assert drew is True
    assert pipe.healthy is True
