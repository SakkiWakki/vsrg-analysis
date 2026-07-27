"""Per-column receptors (judgment layer) + per-note rotation/zoom draw.

The receptor marks must ride each lane's center (so lane_xs / animated
widths / field transforms carry them for free), pick up the per-column
the lane curve's own bend, and fall back to the legacy full-width line
under `receptor_style() == 'line'`. The note-draw path must apply
`ctx.candidate_rot_deg` / `candidate_zoom` about the note center only
when the arrays are present and non-identity.

All headless: a recording fake painter tracks the affine transform
stack (translate/rotate/scale/save/restore) so a fillRect's center can
be resolved back to device space without a real QPainter or display.
"""
from types import SimpleNamespace

import math

import numpy as np

from analysis.player.render import lane_path
from analysis.player.render.layers import field as _field
from tests.conftest import receptor_lane
from analysis.player.render.layers import notes as _notes


# ── recording fake painter with an affine transform stack ────────────

class _Mat:
    """2x3 affine (a, b, c, d, e, f): x' = a*x + c*y + e, etc."""

    def __init__(self, a=1.0, b=0.0, c=0.0, d=1.0, e=0.0, f=0.0):
        self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f

    def copy(self):
        return _Mat(self.a, self.b, self.c, self.d, self.e, self.f)

    def mul(self, o):
        return _Mat(
            self.a * o.a + self.c * o.b,
            self.b * o.a + self.d * o.b,
            self.a * o.c + self.c * o.d,
            self.b * o.c + self.d * o.d,
            self.a * o.e + self.c * o.f + self.e,
            self.b * o.e + self.d * o.f + self.f,
        )

    def apply(self, x, y):
        return (self.a * x + self.c * y + self.e,
                self.b * x + self.d * y + self.f)


class FakePainter:
    def __init__(self):
        self._m = _Mat()
        self._stack = []
        self._opacity = 1.0
        self.fill_rects = []   # (device_center_x, device_center_y, opacity)
        self.lines = []        # (x1, y1, x2, y2)

    # transform ops
    def save(self):
        self._stack.append((self._m.copy(), self._opacity))

    def restore(self):
        self._m, self._opacity = self._stack.pop()

    def translate(self, x, y):
        self._m = self._m.mul(_Mat(e=x, f=y))

    def rotate(self, deg):
        r = math.radians(deg)
        c, s = math.cos(r), math.sin(r)
        self._m = self._m.mul(_Mat(a=c, b=s, c=-s, d=c))

    def scale(self, sx, sy):
        self._m = self._m.mul(_Mat(a=sx, d=sy))

    def opacity(self):
        return self._opacity

    def setOpacity(self, v):
        self._opacity = v

    # no-op state
    def setPen(self, *a):
        pass

    def setBrush(self, *a):
        pass

    # draw ops -> record device-space centers
    def fillRect(self, rect, _brush):
        cx = rect.x() + rect.width() / 2.0
        cy = rect.y() + rect.height() / 2.0
        self.fill_rects.append((*self._m.apply(cx, cy), self._opacity))

    def drawRect(self, rect):
        pass

    def drawLine(self, p1, p2):
        self.lines.append((p1.x(), p1.y(), p2.x(), p2.y()))


# ── ctx / player fixtures ────────────────────────────────────────────

def _ctx(*, keycount=4, lane_w=80.0, x0=100.0, judge_y=400,
         style='bars', lane_xs=None, lane_ws=None, receptor_alpha=None,
         **lane_terms):
    adapter = SimpleNamespace(
        receptor_style=lambda: style,
        transparent_field=lambda: False,
    )
    player = SimpleNamespace(
        _adapter=adapter,
        keycount=keycount,
        H=600, W=800,
        sv_enabled=False,
        _sv_engine=SimpleNamespace(enabled=False),
        sv_render=None,
        windows=[],
        judge_colors={},
        replay={},
    )
    ctx = SimpleNamespace(
        player=player,
        keycount=keycount,
        t_now=0.0,
        scroll_speed=1000.0,
        judge_y=judge_y,
        x0=x0,
        lane_w=lane_w,
        lane_xs=lane_xs,
        lane_ws=lane_ws,
        time_to_y=lambda t: 0.0,
    )
    ctx.lane_x = lambda col: (lane_xs[col] if lane_xs is not None
                              else x0 + col * lane_w)
    ctx.lane_width = lambda col: (lane_ws[col] if lane_ws is not None
                                  else lane_w)
    ctx.lane_center = lambda col: ctx.lane_x(col) + ctx.lane_width(col) / 2.0
    ctx.receptor_alpha = receptor_alpha
    ctx.lane_path = (receptor_lane(ctx.lane_center, judge_y, **lane_terms)
                     if lane_terms
                     else lane_path.straight(ctx.lane_center,
                                             lambda offs: judge_y - offs))
    ctx.receptor_marks = ctx.lane_path.sample(np.arange(keycount),
                                              np.zeros(keycount))
    return ctx


# ── receptor marks follow the lane centers ──────────────────────────

def test_receptors_ride_lane_centers():
    ctx = _ctx(keycount=4, lane_w=80.0, x0=100.0, judge_y=400)
    p = FakePainter()
    _field.draw_judgment(ctx, p)
    assert len(p.fill_rects) == 4
    centers_x = sorted(cx for cx, _, _ in p.fill_rects)
    expected = [ctx.lane_center(c) for c in range(4)]
    assert centers_x == sorted(expected)
    for _, cy, _ in p.fill_rects:
        assert cy == 400


def test_receptors_follow_lane_xs():
    # A lane switch relocates + resizes columns; marks must track.
    lane_xs = (300.0, 380.0, 20.0, 500.0)
    lane_ws = (80.0, 80.0, 80.0, 80.0)
    ctx = _ctx(keycount=4, lane_xs=lane_xs, lane_ws=lane_ws, judge_y=250)
    p = FakePainter()
    _field.draw_judgment(ctx, p)
    got = sorted(cx for cx, _, _ in p.fill_rects)
    want = sorted(x + w / 2.0 for x, w in zip(lane_xs, lane_ws))
    assert got == want


def test_collapsed_lane_draws_no_mark():
    lane_xs = (100.0, 180.0, 180.0, 260.0)
    lane_ws = (80.0, 0.0, 80.0, 80.0)   # col 1 collapsed
    ctx = _ctx(keycount=4, lane_xs=lane_xs, lane_ws=lane_ws)
    p = FakePainter()
    _field.draw_judgment(ctx, p)
    assert len(p.fill_rects) == 3


# ── a bent lane displaces / rotates / fades its receptors ───────────

def test_a_bent_lane_moves_its_receptors():
    dx = np.array([10.0, 0.0, -5.0, 0.0])
    dy = np.array([0.0, 12.0, 0.0, -3.0])
    ctx = _ctx(keycount=4, dx=dx, dy=dy)
    p = FakePainter()
    _field.draw_judgment(ctx, p)
    base = [ctx.lane_center(c) for c in range(4)]
    # col 0 shifted +10 x, col 1 shifted +12 y, col 2 shifted -5 x.
    got = sorted((cx, cy) for cx, cy, _ in p.fill_rects)
    assert (base[0] + 10.0, 400) in got
    assert (base[1], 412) in got
    assert (base[2] - 5.0, 400) in got


def test_a_turned_receptor_goes_through_the_painter_transform():
    # A 90deg rotation about the mark center leaves the center fixed;
    # the transform path (save/restore) is exercised without moving it.
    rot = np.array([90.0, 0.0, 0.0, 0.0])
    zoom = np.array([2.0, 1.0, 1.0, 1.0])
    ctx = _ctx(keycount=4, rotation_deg=rot, zoom=zoom)
    p = FakePainter()
    _field.draw_judgment(ctx, p)
    # Center is invariant under rotation+uniform scale about itself.
    got = sorted((round(cx, 3), cy) for cx, cy, _ in p.fill_rects)
    want = sorted((round(ctx.lane_center(c), 3), 400) for c in range(4))
    assert got == want


def test_receptor_alpha_multiplies_opacity():
    # A receptor's visibility is its OWN, never the lane's alpha at that
    # point (an arrow there would take the stealth gradients).
    ctx = _ctx(keycount=4, receptor_alpha=np.array([0.25, 1.0, 1.0, 1.0]))
    p = FakePainter()
    _field.draw_judgment(ctx, p)
    ops = sorted(op for _, _, op in p.fill_rects)
    assert ops[0] == 0.25
    assert ops[-1] == 1.0


# ── style 'line' fallback ────────────────────────────────────────────

def test_style_line_draws_full_width_line_not_marks():
    ctx = _ctx(keycount=4, style='line', x0=100.0, lane_w=80.0)
    p = FakePainter()
    _field.draw_judgment(ctx, p)
    assert p.fill_rects == []
    # One judgment line spanning the field (death line absent).
    assert len(p.lines) == 1
    x1, y1, x2, y2 = p.lines[0]
    assert (x1, x2) == (100.0, 100.0 + 4 * 80.0)
    assert y1 == y2 == 400


# ── note rotation / zoom applied only when arrays present ───────────

def _tap_view(**over):
    base = dict(
        i=0, col=0, y=100, y_end=100, press_y=100, lx=200, off=0.0,
        press_t=0.0, release_t=None, rel_off=None, end_t=None,
        is_ln=False, is_roll=False, miss=False, state='tap',
        note_color=(255, 255, 255), jcolor=(255, 0, 0),
    )
    base.update(over)
    return _notes._NoteView(**base)


def _note_ctx():
    ctx = SimpleNamespace(lane_w=80.0)
    ctx.lane_width = lambda col: 80.0
    return ctx


def test_unmodded_note_uses_no_transform():
    ctx = _note_ctx()
    p = FakePainter()
    calls = []
    _notes._draw_view(ctx, p, _tap_view(),
                      lambda c, pt, n: calls.append('drew'))
    assert calls == ['drew']
    # No save was pushed for an identity note.
    assert p._stack == []


def test_note_rotation_zoom_transform_about_center():
    ctx = _note_ctx()
    p = FakePainter()
    # A 180deg rotation about the center reflects an offset point through
    # the center: probe by drawing a unit rect at the note center.
    n = _tap_view(rotation_deg=180.0, zoom=1.0)
    cx = n.lx + 80.0 / 2.0

    def drawer(c, pt, view):
        from PySide6.QtCore import QRectF
        pt.fillRect(QRectF(cx + 5.0 - 0.5, view.y - 0.5, 1.0, 1.0), None)

    _notes._draw_view(ctx, p, n, drawer)
    dcx, dcy, _ = p.fill_rects[0]
    # Point 5px right of center maps to 5px left of center under 180deg.
    assert round(dcx, 6) == round(cx - 5.0, 6)
    assert round(dcy, 6) == round(float(n.y), 6)


def test_note_zoom_scales_about_center():
    ctx = _note_ctx()
    p = FakePainter()
    n = _tap_view(zoom=3.0)
    cx = n.lx + 80.0 / 2.0
    from PySide6.QtCore import QRectF

    def drawer(c, pt, view):
        pt.fillRect(QRectF(cx + 2.0 - 0.5, view.y - 0.5, 1.0, 1.0), None)

    _notes._draw_view(ctx, p, n, drawer)
    dcx, _, _ = p.fill_rects[0]
    assert round(dcx, 6) == round(cx + 6.0, 6)  # 2px * 3


def _build_ctx_for_single_tap():
    """Minimal ctx/player that `_notes._build` can walk for one tap note
    at candidate position 0."""
    notes = SimpleNamespace(
        columns_list=[1], miss_head_suppressed=[False],
        ln_tail_times=np.array([float('nan')]),
        noterows_list=[48], roll_head_keys=set(),
    )
    player = SimpleNamespace(
        notes=notes, keycount=4,
        misses=np.array([False]),
        times=np.array([1.0]), offsets=np.array([0.0]),
        hold_release_offsets={},
        palette=[(1, 1, 1)] * 4,
        judge_colors={'marv': (0, 255, 0)},
        note_judges=['marv'],
        press_hide=False,
    )
    ctx = SimpleNamespace(
        player=player, t_now=0.0,
        candidates=[0],
        candidate_head_y=np.array([120.0]),
        candidate_tail_y=np.array([float('nan')]),
        candidate_press_y=np.array([120.0]),
        candidate_dx=None, candidate_alpha=None,
        candidate_rot_deg=np.array([45.0]),
        candidate_zoom=np.array([1.5]),
        lane_w=80.0,
    )
    ctx.lane_x = lambda col: 100.0 + col * 80.0
    return ctx


def test_build_populates_rotation_zoom_from_arrays():
    ctx = _build_ctx_for_single_tap()
    view = _notes._build(ctx, 0, 0)
    assert view is not None
    assert view.rotation_deg == 45.0
    assert view.zoom == 1.5


def test_build_defaults_identity_without_arrays():
    ctx = _build_ctx_for_single_tap()
    ctx.candidate_rot_deg = None
    ctx.candidate_zoom = None
    view = _notes._build(ctx, 0, 0)
    assert view.rotation_deg == 0.0
    assert view.zoom == 1.0


def test_absent_arrays_leave_identity():
    n = _tap_view()
    assert n.rotation_deg == 0.0
    assert n.zoom == 1.0
