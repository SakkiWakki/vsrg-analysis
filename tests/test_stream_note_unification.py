"""One note pipeline: chart streams (mines/lifts/fakes) are records in
the unified stream table and ride the SAME candidate machinery as taps.

Covers:
- the NotesModel stream table (kind column, time-sorted merge, fills);
- pixel-level parity of the unified draw path with the pre-refactor
  stream blits for a chart without mods (positions, sprites, order);
- the capabilities that unification makes free for streams:
  stealth/glow visibility (a stealthed mine renders through the same
  additive glow pass as a tap) and per-column reverse (a mine in a
  reversed column lands exactly where that column's taps do);
- hold-mine span endpoints coming from the unified kernel;
- a QImage smoke render through a real QPainter.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.games.notitg.note_mods import NotitgNoteMods
from analysis.player.init.notes_model import (
    KIND_FAKE, KIND_LIFT, KIND_MINE, NotesModel, copy_chart_streams,
    stream_groups_or_none)
from analysis.player.render import culling
from analysis.player.render.layers import chart_extras as _extras
from analysis.player.render.layers import notes as _notes
from analysis.player.render.mods import ModChannels, ModEvent
from analysis.player.render.qt_renderer import (_NoteView,
                                                 _precompute_candidate_ys)
from analysis.player.sv.engine import QuaverSVEngine
from analysis.player.sv.render import SvRenderController


# ── recording fakes ─────────────────────────────────────────────────


class _RecordCache:
    """Sprite cache double: records (name, kwargs) per get and hands out
    one shared pixmap so blit y-offsets are deterministic."""

    def __init__(self, size=20):
        from PySide6.QtGui import QPixmap
        self.calls = []
        self.pm = QPixmap(size, size)

    def get(self, name, ctx, **kw):
        self.calls.append((name, kw))
        return self.pm


class _RecordPainter:
    """Records pixmap blits with the painter state active at each blit
    (opacity, composition mode) so tests can assert the glow bracket."""

    def __init__(self):
        self.blits = []      # (x, y, opacity, composition)
        self._opacity = 1.0
        self._composition = 'normal'
        self._stack = []

    def drawPixmap(self, point, pm):
        self.blits.append((float(point.x()), float(point.y()),
                           self._opacity, self._composition))

    def opacity(self):
        return self._opacity

    def setOpacity(self, value):
        self._opacity = float(value)

    def setCompositionMode(self, mode):
        self._composition = mode

    def save(self):
        self._stack.append((self._opacity, self._composition))

    def restore(self):
        self._opacity, self._composition = self._stack.pop()

    def setTransform(self, *a, **k):
        pass

    def translate(self, *a):
        pass

    def rotate(self, *a):
        pass

    def scale(self, *a):
        pass


# ── game-agnostic harness (no mods) ─────────────────────────────────


NOTE_TS = {'mine': 3.0, 'lift': 3.2, 'fake': 3.4}
LANE_W = 64.0


def _default_engine():
    return QuaverSVEngine([], groups={
        '$Default': {'sections': [(0.0, 1.0)], 'initial_velocity': 1.0},
    })


def _stream_player(streams):
    engine = _default_engine()
    player = SimpleNamespace(H=800, hit_line_y_frac=0.5, scroll_speed=100.0,
                             judge_y_px=lambda: 400.0, keycount=4,
                             _sv_engine=engine, sv_enabled=True)
    ctrl = SvRenderController(player)
    player.batch_time_to_y = ctrl.batch_time_to_y

    notes = NotesModel()
    copy_chart_streams(notes, streams)
    player.notes = notes
    ctrl.build_ghost_sv_caches()
    return player, ctrl


def _stream_ctx(player, t, cache=None):
    engine = player._sv_engine
    frame = SimpleNamespace(use_sv=True, raw_t=t,
                            visual_cum_now=engine.cumulative_at(t),
                            render_multiplier=engine.render_multiplier_at(t))
    cum_now = engine.cumulative_at(t)
    ctx = SimpleNamespace(
        player=player, frame=frame, t_now=t, use_sv_space=True,
        target_lo=cum_now - 10.0, target_hi=cum_now + 10.0,
        candidates=[], screen_margin=80,
        lane_x=lambda col: 100.0 + col * LANE_W, lane_w=LANE_W,
        lane_width=lambda col: LANE_W,
        sprite_cache=cache or _RecordCache(),
    )
    culling.select_stream_candidates(ctx)
    _precompute_candidate_ys(ctx)
    _notes.prepare(ctx)
    return ctx


def _all_streams():
    return {
        'mine_times': [NOTE_TS['mine']], 'mine_cols': [0],
        'mine_until': [np.inf],
        'lift_times': [NOTE_TS['lift']], 'lift_cols': [1],
        'lift_until': [np.inf],
        'fake_times': [NOTE_TS['fake']], 'fake_cols': [2],
        'fake_until': [np.inf],
    }


def test_stream_table_merges_kinds_time_sorted():
    m = NotesModel()
    copy_chart_streams(m, {
        'mine_times': [5.0, 1.0], 'mine_cols': [0, 1],
        'lift_times': [3.0], 'lift_cols': [2],
        'fake_times': [1.0], 'fake_cols': [3],
    })
    assert list(m.stream_times) == [1.0, 1.0, 3.0, 5.0]
    # Stable sort: equal times keep family order (mines before fakes).
    assert list(m.stream_kinds) == [KIND_MINE, KIND_FAKE, KIND_LIFT,
                                    KIND_MINE]
    assert list(m.stream_cols) == [1, 3, 2, 0]
    # Fills for the columns these families never supplied.
    assert list(m.stream_rows) == [-1, -1, -1, -1]
    assert np.all(np.isinf(m.stream_until))
    assert np.all(np.isnan(m.stream_end_times))
    assert list(m.stream_groups) == [None, None, None, None]


def test_unmodded_streams_blit_at_prerefactor_positions():
    """Rest parity: without mods, each stream sprite blits at exactly
    the pre-refactor position (lane_x, projected y - pm.height/2), with
    the pre-refactor sprite key, in time order."""
    player, _ctrl = _stream_player(_all_streams())
    cache = _RecordCache()
    ctx = _stream_ctx(player, 1.0, cache)

    n = player.notes
    groups = stream_groups_or_none(n.stream_groups)
    expected_y = {}
    for name, kind in (('mine', KIND_MINE), ('lift', KIND_LIFT),
                       ('fake', KIND_FAKE)):
        k = int(np.flatnonzero(n.stream_kinds == kind)[0])
        expected_y[name] = float(_extras._chart_stream_ys(
            ctx, n.stream_times, n.stream_sv, groups,
            np.array([k], dtype=np.intp))[0])

    half = cache.pm.height() / 2
    for name, col in (('mine', 0), ('lift', 1), ('fake', 2)):
        painter = _RecordPainter()
        cache.calls.clear()
        draw = {'mine': _notes.draw_mines, 'lift': _notes.draw_lifts,
                'fake': _notes.draw_fakes}[name]
        draw(ctx, painter)
        assert len(painter.blits) == 1
        x, y_top, opacity, composition = painter.blits[0]
        assert x == pytest.approx(ctx.lane_x(col))
        assert y_top == pytest.approx(expected_y[name] - half)
        assert opacity == 1.0 and composition == 'normal'
        # Sprite selection is unchanged: mines are unkeyed glyphs,
        # lifts/fakes key on the column.
        expected_kw = {} if name == 'mine' else {'col': col}
        assert (name, expected_kw) in cache.calls


def test_expired_stream_records_do_not_draw():
    streams = _all_streams()
    streams['mine_until'] = [2.0]
    player, _ctrl = _stream_player(streams)
    ctx = _stream_ctx(player, 2.5)

    painter = _RecordPainter()
    _notes.draw_mines(ctx, painter)
    assert painter.blits == []
    # Lifts are still live (until inf).
    _notes.draw_lifts(ctx, painter)
    assert len(painter.blits) == 1


def test_hold_mine_span_endpoints_come_from_kernel(monkeypatch):
    streams = {
        'mine_times': [NOTE_TS['mine']], 'mine_cols': [0],
        'mine_until': [np.inf],
        'mine_end_times': [NOTE_TS['mine'] + 1.0],
    }
    player, _ctrl = _stream_player(streams)
    ctx = _stream_ctx(player, 1.0)

    lines = []
    monkeypatch.setattr(_extras, 'draw_lane_line',
                        lambda painter, color, lx, lane_w, y0, y1, width=1:
                        lines.append((lx, y0, y1)))
    painter = _RecordPainter()
    _notes.draw_mines(ctx, painter)

    v = ctx.stream_views[0]
    assert math.isfinite(v.y_end)
    # The span stroke runs between the kernel's head/tail ys, and both
    # the head sprite and the end sprite blit off those same values.
    assert lines == [(v.lx, v.y, v.y_end)]
    half = ctx.sprite_cache.pm.height() / 2
    tops = sorted(y for _x, y, _o, _c in painter.blits)
    assert tops == sorted([v.y - half, v.y_end - half])


def test_qimage_smoke_render():
    """The unified path drives a real QPainter without error and puts
    pixels where the kernel says the records are."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QImage, QPainter, QPixmap

    class _PixelCache:
        def __init__(self):
            self.pm = QPixmap(20, 20)
            self.pm.fill(Qt.red)

        def get(self, name, ctx, **kw):
            return self.pm

    player, _ctrl = _stream_player(_all_streams())
    ctx = _stream_ctx(player, 1.0, _PixelCache())

    img = QImage(600, 800, QImage.Format_ARGB32)
    img.fill(0)
    painter = QPainter(img)
    _notes.draw_mines(ctx, painter)
    _notes.draw_lifts(ctx, painter)
    _notes.draw_fakes(ctx, painter)
    painter.end()

    for v in ctx.stream_views:
        assert img.pixelColor(int(v.lx) + 5, int(v.y)).alpha() > 0


# ── NotITG mod harness (streams on the candidate axis) ──────────────


ROW = 144  # beat 3


class _ModNotes(NotesModel):
    pass


def _mod_player(tap_col=None, stream_cols=(0,), kinds=(KIND_MINE,)):
    """Fake player whose notes model carries stream records (and
    optionally one replay tap) for the NotITG kernel."""
    notes = NotesModel()
    streams = {
        'mine_times': [], 'mine_cols': [], 'mine_rows': [],
        'lift_times': [], 'lift_cols': [], 'lift_rows': [],
        'fake_times': [], 'fake_cols': [], 'fake_rows': [],
    }
    for col, kind in zip(stream_cols, kinds):
        family = {KIND_MINE: 'mine', KIND_LIFT: 'lift',
                  KIND_FAKE: 'fake'}[kind]
        streams[f'{family}_times'].append(3.0)
        streams[f'{family}_cols'].append(col)
        streams[f'{family}_rows'].append(ROW)
    copy_chart_streams(notes, streams)

    has_tap = tap_col is not None
    notes.noterows_list = [ROW] if has_tap else []
    notes.columns_list = [tap_col] if has_tap else []
    notes.ln_tail_times = np.full(1 if has_tap else 0, np.nan)
    return SimpleNamespace(
        keycount=4,
        columns=np.array([tap_col] if has_tap else [], dtype=np.int64),
        times=np.array([3.0] if has_tap else []),
        offsets=np.array([0.0] if has_tap else []),
        misses=np.array([False] if has_tap else [], dtype=bool),
        notes=notes,
    )


def _mod_ctx(player, head_y, *, n_taps):
    """Candidate axis with `n_taps` replay entries then the stream
    records, all at the same pre-mod head y."""
    total = n_taps + len(player.notes.stream_times)
    return SimpleNamespace(
        player=player, t_now=0.0, lane_w=64.0, judge_y=100,
        chart_rect=(0.0, 0.0, 400.0, 400.0),
        candidates=list(range(n_taps)),
        stream_candidates=np.arange(len(player.notes.stream_times),
                                    dtype=np.int64),
        stream_head_in_window=np.ones(len(player.notes.stream_times),
                                      dtype=bool),
        candidate_head_y=np.full(total, float(head_y)),
        candidate_tail_y=np.full(total, np.nan),
        candidate_press_y=np.full(total, float(head_y)),
        lane_x=lambda col: 100.0 + col * 64.0,
        lane_width=lambda col: 64.0,
    )


def _apply_mods(ctx, events):
    NotitgNoteMods(ModChannels.compile(events), [(0.0, 120.0)]).apply(ctx)


def test_mine_under_stealthglow_builds_glowing_view():
    """Regression for the invisible-intro bug family: a stream record
    under stealth 1.0 + stealthglow gets fill alpha ~0 and glow > 0
    through the SAME kernel stash as taps."""
    player = _mod_player(stream_cols=(0,), kinds=(KIND_FAKE,))
    ctx = _mod_ctx(player, head_y=40.0, n_taps=0)
    _apply_mods(ctx, [ModEvent(0.0, 1.0, -1, 'stealth'),
                      ModEvent(0.0, 1.0, -1, 'stealthglow')])

    _notes.prepare(ctx)
    (view,) = ctx.stream_views
    assert view.alpha == pytest.approx(0.0)
    assert view.glow > 0.0


def test_stealthglow_mine_draws_additive_glow_exactly_like_tap():
    from PySide6.QtGui import QPainter

    player = _mod_player(stream_cols=(0,), kinds=(KIND_MINE,))
    ctx = _mod_ctx(player, head_y=40.0, n_taps=0)
    _apply_mods(ctx, [ModEvent(0.0, 1.0, -1, 'stealth'),
                      ModEvent(0.0, 1.0, -1, 'stealthglow')])
    ctx.sprite_cache = _RecordCache()
    ctx.screen_margin = 80
    player.H = 400
    _notes.prepare(ctx)
    (view,) = ctx.stream_views

    painter = _RecordPainter()
    _notes.draw_mines(ctx, painter)
    # Exactly one blit: the additive glow pass (the hidden fill draws
    # nothing), at the glow strength.
    assert len(painter.blits) == 1
    _x, _y, opacity, composition = painter.blits[0]
    assert composition == QPainter.CompositionMode_Plus
    assert opacity == pytest.approx(view.glow)

    # A tap with the same per-note visibility renders the same bracket.
    tap = _NoteView(
        i=0, col=0, y=view.y, y_end=0, press_y=view.y, lx=int(view.lx),
        off=0.0, press_t=99.0, release_t=None, rel_off=None, end_t=None,
        is_ln=False, is_roll=False, miss=False, state='upcoming',
        note_color=(255, 255, 255), jcolor=(255, 0, 0),
        alpha=view.alpha, glow=view.glow)
    tap_painter = _RecordPainter()
    ctx.player.press_hide = False
    _notes._draw_view(ctx, tap_painter, tap, _notes._draw_replay_note)
    assert [(b[2], b[3]) for b in tap_painter.blits] \
        == [(b[2], b[3]) for b in painter.blits]


def test_stealth_only_mine_draws_nothing():
    player = _mod_player(stream_cols=(0,), kinds=(KIND_MINE,))
    ctx = _mod_ctx(player, head_y=40.0, n_taps=0)
    _apply_mods(ctx, [ModEvent(0.0, 1.0, -1, 'stealth')])
    ctx.sprite_cache = _RecordCache()
    ctx.screen_margin = 80
    player.H = 400
    _notes.prepare(ctx)

    painter = _RecordPainter()
    _notes.draw_mines(ctx, painter)
    assert painter.blits == []


def test_reversed_column_mine_lands_on_its_taps():
    """A mine shares its column's tap positions under the per-column
    reverse family: same y as a tap at the same (time, column), for the
    reversed and unreversed halves alike."""
    for events in ([], [ModEvent(0.0, 1.0, -1, 'split')],
                   [ModEvent(0.0, 1.0, -1, 'reverse0')]):
        for col in (0, 3):
            player = _mod_player(tap_col=col, stream_cols=(col,),
                                 kinds=(KIND_MINE,))
            ctx = _mod_ctx(player, head_y=40.0, n_taps=1)
            _apply_mods(ctx, events)
            tap_y, mine_y = ctx.candidate_head_y
            assert mine_y == pytest.approx(tap_y), (events, col)


def test_split_moves_mines_with_their_column_direction():
    """The shipped-bug shape: under split, a right-half mine must stay
    in native downscroll (like its taps) while a left-half mine mirrors
    to upscroll."""
    player = _mod_player(stream_cols=(0, 3), kinds=(KIND_MINE, KIND_MINE))
    ctx = _mod_ctx(player, head_y=40.0, n_taps=0)
    _apply_mods(ctx, [ModEvent(0.0, 1.0, -1, 'split')])
    left_y, right_y = ctx.candidate_head_y
    # judge_y=100, chart center 200 -> mirror line 300; a note 60 above
    # the judge line mirrors to 360.
    assert left_y == pytest.approx(360.0)
    assert right_y == pytest.approx(40.0)
