"""NotITG note-path splines (SetXSpline/SetZSpline) + arrowpath
gradient recording and consumption: the sim's per-(column, point)
channels, the note_path curve rebuild, and note_mods folding the
sampled displacement into heads, receptors, and depth (the W32_Stuxnet
helix reduced to synthetic control points)."""
import numpy as np
import pytest

from analysis.games.notitg import note_path
from analysis.games.notitg.note_mods import NotitgNoteMods
from analysis.games.notitg.mod_channels import ModChannels
from analysis.games.notitg.sim.actor import SimActor
from analysis.player.render.effects.timeline import Keyframe
from analysis.player.render.mods import arrow_effects as ae


# -- sample_note_path (the shared interpolation kernel) ----------------------

def test_single_point_is_constant():
    out = note_path.sample_note_path([100.0], [7.0], np.array([0.0, 500.0]))
    assert np.allclose(out, 7.0)


def test_two_points_are_the_linear_chord():
    out = note_path.sample_note_path(
        [0.0, 100.0], [10.0, 30.0], np.array([0.0, 50.0, 100.0]))
    assert np.allclose(out, [10.0, 20.0, 30.0])


def test_curve_passes_through_control_points_and_clamps():
    domains = np.array([0.0, 100.0, 200.0, 300.0])
    values = np.array([0.0, 50.0, -20.0, 10.0])
    at_controls = note_path.sample_note_path(domains, values, domains)
    assert np.allclose(at_controls, values)
    beyond = note_path.sample_note_path(
        domains, values, np.array([-500.0, 900.0]))
    assert np.allclose(beyond, [values[0], values[-1]])


def test_interpolation_is_smooth_not_piecewise_linear():
    # Sine control points: the Catmull-Rom midpoint overshoots the
    # linear chord toward the true sine (the reference helix is visibly
    # smooth; a linear kernel would facet it).
    domains = np.arange(5) * 100.0
    values = np.sin(domains / 400.0 * 2.0 * np.pi) * 300.0
    mid = 150.0
    smooth = note_path.sample_note_path(domains, values, np.array([mid]))[0]
    chord = (values[1] + values[2]) / 2.0
    true = np.sin(mid / 400.0 * 2.0 * np.pi) * 300.0
    assert abs(smooth - true) < abs(chord - true)


# -- sim recording ------------------------------------------------------------

def test_spline_pokes_record_per_point_channels():
    a = SimActor()
    a.poke('SetXSpline', [0, 2, 150.0, 0.0, -1])
    a.poke('SetXSpline', [1, 2, -75.0, 100.0, -1])
    a.poke('SetZSpline', [0, 2, 40.0, 0.0, -1])
    frames = a.keyframes()
    assert frames['spline_x:2:0'][-1].values == (150.0, 0.0)
    assert frames['spline_x:2:1'][-1].values == (-75.0, 100.0)
    assert frames['spline_z:2:0'][-1].values == (40.0, 0.0)


def test_gradient_pokes_record_stops_and_count():
    a = SimActor()
    a.poke('SetNumPathGradientPoints', [1, 2])
    a.poke('SetPathGradientColor', [0, 1, 0.0, 1.0, 0.0, 1.0])
    frames = a.keyframes()
    assert frames['pathgrad_n:1'][-1].values == (2.0,)
    assert frames['pathgrad:1:0'][-1].values == (0.0, 1.0, 0.0, 1.0)


def test_out_of_bounds_and_malformed_writes_drop_silently():
    a = SimActor()
    dropped = []
    a.dropped_notify = dropped.append
    a.poke('SetXSpline', [0, 99, 1.0, 0.0, -1])   # column out of range
    a.poke('SetXSpline', [999, 0, 1.0, 0.0, -1])  # point out of range
    a.poke('SetXSpline', [0, 0])                  # too few args
    a.poke('SetPathGradientColor', [0, 0])        # too few args
    assert a.keyframes() == {}
    assert dropped == []


# -- curve rebuild ------------------------------------------------------------

def _instant(t, values):
    return Keyframe(t, values, 0.0, 0)


def _helix_player(t0=0.0):
    """Two-point linear ramp on column 0 (x: 10 at domain 0 -> 30 at
    100) plus a constant z of 5, first poked at `t0`."""
    return note_path._build_one({
        'spline_x:0:0': [_instant(t0, (10.0, 0.0))],
        'spline_x:0:1': [_instant(t0, (30.0, 100.0))],
        'spline_z:0:0': [_instant(t0, (5.0, 0.0))],
        'spline_z:0:1': [_instant(t0, (5.0, 100.0))],
    })


def test_sampler_inert_before_first_poke_and_live_after():
    player = _helix_player(t0=2.0)
    assert player.sampler_at(1.0) is None
    sampler = player.sampler_at(3.0)
    assert sampler is not None
    out = sampler.offsets('x', np.array([0, 0]), np.array([0.0, 50.0]))
    assert np.allclose(out, [10.0, 20.0])
    # A column with no curve contributes zero.
    other = sampler.offsets('x', np.array([3]), np.array([50.0]))
    assert np.allclose(other, 0.0)


def test_all_zero_curves_stay_inert():
    player = note_path._build_one({
        'spline_x:0:0': [_instant(0.0, (0.0, 0.0))],
        'spline_x:0:1': [_instant(0.0, (0.0, 100.0))],
    })
    assert player.sampler_at(1.0) is None


def test_gradient_stops_truncate_to_count():
    player = note_path._build_one({
        'pathgrad_n:1': [_instant(0.0, (1.0,))],
        'pathgrad:1:0': [_instant(0.0, (0.0, 1.0, 0.0, 1.0))],
        'pathgrad:1:1': [_instant(0.0, (1.0, 0.0, 0.0, 1.0))],
    })
    assert player.gradient_at(1.0, 1) == [(0.0, 1.0, 0.0, 1.0)]
    assert player.gradient_at(1.0, 5) == [note_path._GRAD_REST]


# -- note_mods consumption ----------------------------------------------------

class _FakeNotes:
    def __init__(self, count):
        self.noterows_list = [0] * count


class _FakePlayer:
    def __init__(self, cols, keycount):
        self.columns = np.asarray(cols, dtype=np.int64)
        self.keycount = keycount
        self.notes = _FakeNotes(len(cols))


class _FakeCtx:
    def __init__(self, player, heads, judge_y, chart_h):
        self.player = player
        self.candidates = list(range(len(heads)))
        self.t_now = 0.0
        self.lane_w = ae.ARROW_SIZE  # scale = 1
        self.judge_y = judge_y
        self.chart_rect = (0.0, 0.0, 400.0, float(chart_h))
        self.candidate_head_y = np.asarray(heads, dtype=np.float64)
        self.candidate_tail_y = np.asarray(heads, dtype=np.float64)
        self.candidate_press_y = np.asarray(heads, dtype=np.float64)


def _note_mods(handle):
    return NotitgNoteMods(ModChannels.compile([]), [(0.0, 120.0)],
                          note_path=handle)


def _handle():
    return note_path.NotePathHandle({0: _helix_player()})


def test_spline_displaces_notes_at_their_y_offset():
    player = _FakePlayer([0, 0], 2)
    # Heads at the judge line (y_offset 0) and 64px above (y_offset 64).
    ctx = _FakeCtx(player, [100.0, 36.0], judge_y=100, chart_h=400)
    _note_mods(_handle()).apply(ctx)
    assert ctx.note_path_spline is not None
    assert ctx.candidate_dx == pytest.approx([10.0, 10.0 + 20.0 * 0.64])
    assert ctx.candidate_z == pytest.approx([5.0, 5.0])


def test_spline_displaces_receptors_at_domain_zero():
    player = _FakePlayer([0], 2)
    ctx = _FakeCtx(player, [100.0], judge_y=100, chart_h=400)
    _note_mods(_handle()).apply(ctx)
    assert ctx.receptor_offsets['dx'][0] == pytest.approx(10.0)
    assert ctx.receptor_offsets['dx'][1] == pytest.approx(0.0)


def test_no_note_path_keeps_the_inert_fast_path():
    player = _FakePlayer([0], 2)
    ctx = _FakeCtx(player, [100.0], judge_y=100, chart_h=400)
    _note_mods(None).apply(ctx)
    assert ctx.note_path_spline is None
    assert not hasattr(ctx, 'candidate_dx')


# -- arrowpath ribbons ---------------------------------------------------------

def _arrowpath_handle():
    player = note_path._build_one({
        'spline_x:0:0': [_instant(0.0, (10.0, 0.0))],
        'spline_x:0:1': [_instant(0.0, (30.0, 100.0))],
        'pathgrad_n:0': [_instant(0.0, (1.0,))],
        'pathgrad:0:0': [_instant(0.0, (0.0, 1.0, 0.0, 1.0))],
    })
    return note_path.NotePathHandle({0: player})


def test_arrowpath_mod_stashes_per_column_ribbons():
    from analysis.player.render.mods import ModEvent
    player = _FakePlayer([0], 2)
    ctx = _FakeCtx(player, [100.0], judge_y=100, chart_h=400)
    ctx.lane_x = lambda col: 100.0 + 64.0 * col
    mods = NotitgNoteMods(
        ModChannels.compile([ModEvent(0.0, 1.0, -1, 'arrowpath')]),
        [(0.0, 120.0)], note_path=_arrowpath_handle())
    mods.apply(ctx)
    ribbons = ctx.arrowpath_ribbons
    assert len(ribbons) == 2
    xs, ys, stops, width, alpha = ribbons[0]
    # First sample sits at the receptor (y_offset 0): lane center plus
    # the spline's domain-0 value; the trail carries the gradient color.
    assert xs[0] == pytest.approx(100.0 + 32.0 + 10.0)
    assert ys[0] == pytest.approx(300.0)  # default reverse mirror line
    assert stops == [(0.0, 1.0, 0.0, 1.0)]
    assert alpha == pytest.approx(1.0)
    assert width > 0
    # Column 1 has no spline: the trail is the straight lane center.
    xs1 = ribbons[1][0]
    assert np.allclose(xs1, 100.0 + 64.0 + 32.0)


def test_arrowpath_inactive_stashes_none():
    player = _FakePlayer([0], 2)
    ctx = _FakeCtx(player, [100.0], judge_y=100, chart_h=400)
    ctx.lane_x = lambda col: 100.0 + 64.0 * col
    _note_mods(_handle()).apply(ctx)
    assert ctx.arrowpath_ribbons is None


def test_gradient_segment_colors_interpolate_between_stops():
    from analysis.player.render.layers.field import _gradient_segment_colors
    colors = _gradient_segment_colors(
        [(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)], 2)
    assert colors[0] == pytest.approx((0.25, 0.25, 0.25, 1.0))
    assert colors[1] == pytest.approx((0.75, 0.75, 0.75, 1.0))
