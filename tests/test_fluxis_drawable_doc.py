"""fluXis storyboard -> DrawableDoc (the first non-NotITG Seam-A producer).

Synthetic element trees (built directly from `model.Element` / `build_timelines`)
compiled through `fluxis.drawable_doc.build_doc`, asserting: emission order across
the fluXis layer z bands (Background below the playfield, Foreground / Overlay
above), channel-backed motion (two sample times, the item transform differs per
the exported timeline), the image table contents, and per-kind skip counts. A real
fluXis map with a storyboard is not shipped in-tree, so a real-map smoke is
skipped-by-default; synthetic coverage is the bar.
"""
import numpy as np
import pytest

sn = pytest.importorskip('storyboard_native')

from analysis.games.fluxis import drawable_doc as dd
from analysis.games.fluxis.fsb_storyboard import _LAYER_Z
from analysis.player.render.effects.timeline import Keyframe
from analysis.player.render.storyboard.model import Element, build_timelines


_EASE_LINEAR = 0
_EASE_IN_QUAD = 2

# fluXis layer bands (fsb_storyboard._LAYER_Z): Background is below the playfield
# (negative), Foreground / Overlay above it.
_Z_BG = _LAYER_Z[0]      # -900
_Z_FG = _LAYER_Z[1]      # 400
_Z_OVER = _LAYER_Z[2]    # 700

_SCREEN_W = 1920.0
_SCREEN_H = 1080.0


def _instant(value, t=0.0):
    """One immediate keyframe holding `value` (a scalar) from `t`."""
    return [Keyframe(float(t), (float(value),), 0.0, _EASE_LINEAR)]


def _element(kind, z, *, asset=None, z_index=0, t_start=0.0, keyframes=None,
             children=(), **fields):
    """A synthetic fluXis storyboard Element at band `z`, resting at the SM
    defaults with `keyframes` (property -> Keyframe list) written on top."""
    return Element(
        kind=kind, z=z, z_index=z_index, t_start=t_start, t_end=float('inf'),
        anchor=(0.0, 0.0), origin=(0.5, 0.5),
        timelines=build_timelines(keyframes=keyframes or {}),
        asset=asset, children=tuple(children), **fields)


def _sprite(z, path, **kw):
    return _element('sprite', z, asset=path, **kw)


def _image_kinds(evaluator, t):
    return [k for (k, _sid, _m, _a) in dd.blit_stream(evaluator, t)]


# --------------------------------------------------------------------------
# export_channel - the duplicated primitive, exercised directly
# --------------------------------------------------------------------------

def test_export_channel_event_timeline_exact_for_instants_and_linear():
    from analysis.player.render.effects.timeline import EventTimeline
    tl = EventTimeline(
        [Keyframe(1.0, (10.0,), 0.0, _EASE_LINEAR),
         Keyframe(2.0, (30.0,), 1.0, _EASE_LINEAR)],
        rest=(0.0,))
    ts, vals, durs, rest = dd.export_channel(tl, 0.0, 10.0)
    assert rest == 0.0
    # instant hold at 1.0, then a linear ramp 10->30 across [2,3].
    assert ts[0] == 1.0 and vals[0] == 10.0 and durs[0] == 0.0
    assert (ts[1], vals[1], durs[1]) == (2.0, 10.0, 1.0)
    assert (ts[2], vals[2], durs[2]) == (3.0, 30.0, 0.0)


def test_export_channel_lazy_curve_dense_sampled():
    class _Curve:
        _rest = (0.0,)

        def sample(self, t):
            return (t * 2.0,)

    ts, vals, durs, rest = dd.export_channel(_Curve(), 0.0, 1.0)
    assert rest == 0.0
    assert len(ts) == len(vals) == len(durs)
    for bt, v in zip(ts, vals):
        assert abs(v - bt * 2.0) < 1e-6


# --------------------------------------------------------------------------
# build_doc - banded emission
# --------------------------------------------------------------------------

def test_layer_bands_emit_below_then_above_the_playfield():
    # A Background sprite (below the playfield) and a Foreground sprite (above):
    # the below image draws first, the above last, and a reserved playfield id
    # is minted between them.
    below = _sprite(_Z_BG, '/tmp/bg.png', keyframes={'x': _instant(100.0)})
    above = _sprite(_Z_FG, '/tmp/fg.png', keyframes={'x': _instant(500.0)})
    evaluator, id_maps, report = dd.build_doc([below, above],
                                              _SCREEN_W, _SCREEN_H)

    assert report['elements_below'] == 1 and report['elements_above'] == 1
    assert report['images'] == 2
    assert set(id_maps['images'].values()) == {'/tmp/bg.png', '/tmp/fg.png'}
    # A reserved, non-screen playfield drawable id is documented for later.
    assert id_maps['playfield'] != id_maps['screen']

    stream = dd.blit_stream(evaluator, 0.0)
    paths = [id_maps['images'][sid] for (_k, sid, _m, _a) in stream]
    assert paths == ['/tmp/bg.png', '/tmp/fg.png']


def test_cross_layer_and_within_layer_order_is_coarse_placement():
    # Overlay above Foreground above Background across layers; within a layer,
    # (z_index, t_start) breaks ties. Emitted order must be the (z, z_index,
    # t_start) sort - the drawable-ir coarse placement for a banded format.
    over = _sprite(_Z_OVER, '/tmp/over.png', keyframes={'x': _instant(1.0)})
    fg_late = _sprite(_Z_FG, '/tmp/fg_late.png', z_index=0, t_start=5.0,
                      keyframes={'x': _instant(2.0)})
    fg_early = _sprite(_Z_FG, '/tmp/fg_early.png', z_index=0, t_start=1.0,
                       keyframes={'x': _instant(3.0)})
    fg_zi = _sprite(_Z_FG, '/tmp/fg_zi.png', z_index=5, t_start=0.0,
                    keyframes={'x': _instant(4.0)})
    bg = _sprite(_Z_BG, '/tmp/bg.png', keyframes={'x': _instant(5.0)})

    # Pass out of order; the compiler must sort.
    evaluator, id_maps, report = dd.build_doc(
        [over, fg_late, bg, fg_zi, fg_early], _SCREEN_W, _SCREEN_H)

    stream = dd.blit_stream(evaluator, 0.0)
    paths = [id_maps['images'][sid] for (_k, sid, _m, _a) in stream]
    assert paths == ['/tmp/bg.png', '/tmp/fg_early.png', '/tmp/fg_late.png',
                     '/tmp/fg_zi.png', '/tmp/over.png']
    assert report['elements_below'] == 1
    assert report['elements_above'] == 4


def test_channel_backed_motion_differs_across_sample_times():
    # A sprite whose x ramps linearly 0 -> 400 across [0, 4]: the item's blit
    # mat3 translation must differ between t=0 and t=4 per the exported timeline.
    moving = _sprite(_Z_FG, '/tmp/move.png', keyframes={
        'x': [Keyframe(0.0, (0.0,), 0.0, _EASE_LINEAR),
              Keyframe(0.0, (400.0,), 4.0, _EASE_LINEAR)]})
    evaluator, id_maps, _report = dd.build_doc([moving], _SCREEN_W, _SCREEN_H)

    at0 = dd.blit_stream(evaluator, 0.0)
    at4 = dd.blit_stream(evaluator, 4.0)
    assert len(at0) == 1 and len(at4) == 1
    mat0 = at0[0][2]
    mat4 = at4[0][2]
    # The translation column (design-space x offset) must have advanced by ~400.
    dx = mat4[0, 2] - mat0[0, 2]
    assert abs(dx - 400.0) < 1.0, (mat0, mat4)


def test_hidden_bit_inverts_into_visible_gate():
    # An element hidden from t>=2 drops out of the blit stream after that time
    # (visible = 1 - hidden), while a never-hidden sibling stays.
    from analysis.player.render.effects.timeline import Keyframe as KF
    hidden_kf = [KF(2.0, (1.0,), 0.0, _EASE_LINEAR)]
    hiding = _sprite(_Z_FG, '/tmp/hiding.png',
                     keyframes={'x': _instant(10.0), 'hidden': hidden_kf})
    stays = _sprite(_Z_FG, '/tmp/stays.png', z_index=1,
                    keyframes={'x': _instant(20.0)})
    evaluator, id_maps, _report = dd.build_doc([hiding, stays],
                                               _SCREEN_W, _SCREEN_H)

    before = {id_maps['images'][sid] for (_k, sid, _m, _a)
              in dd.blit_stream(evaluator, 0.0)}
    after = {id_maps['images'][sid] for (_k, sid, _m, _a)
             in dd.blit_stream(evaluator, 3.0)}
    assert '/tmp/hiding.png' in before and '/tmp/stays.png' in before
    assert '/tmp/hiding.png' not in after and '/tmp/stays.png' in after


# --------------------------------------------------------------------------
# skip counts - the vocabulary-gap tally
# --------------------------------------------------------------------------

def test_unsupported_kinds_skipped_with_per_kind_counts():
    # fluXis's non-image element kinds (rect / ellipse / outline_* / text /
    # video) and an asset-less sprite are skipped, each tallied by kind.
    tree = [
        _sprite(_Z_BG, '/tmp/real.png', keyframes={'x': _instant(1.0)}),
        _element('rect', _Z_BG),
        _element('ellipse', _Z_FG),
        _element('outline_rect', _Z_FG),
        _element('outline_ellipse', _Z_OVER),
        _element('text', _Z_OVER, text='hi'),
        _element('video', _Z_BG),
        _sprite(_Z_BG, None),  # image kind, no asset -> 'no_asset'
    ]
    evaluator, id_maps, report = dd.build_doc(tree, _SCREEN_W, _SCREEN_H)

    assert report['elements_below'] == 1 and report['elements_above'] == 0
    assert report['element_skips'] == {
        'rect': 1, 'ellipse': 1, 'outline_rect': 1, 'outline_ellipse': 1,
        'text': 1, 'video': 1, 'no_asset': 1}
    assert report['images'] == 1


def test_group_is_flattened_to_its_image_children():
    # A group (Container/ActorFrame) draws nothing itself; its sprite children
    # band by their OWN z and the group is tallied as a skip.
    child_bg = _sprite(_Z_BG, '/tmp/ga.png', keyframes={'x': _instant(5.0)})
    child_fg = _sprite(_Z_FG, '/tmp/gb.png', keyframes={'x': _instant(6.0)})
    group = _element('group', _Z_FG, children=[child_bg, child_fg])
    evaluator, id_maps, report = dd.build_doc([group], _SCREEN_W, _SCREEN_H)

    assert report['elements_below'] == 1  # child_bg (Background)
    assert report['elements_above'] == 1  # child_fg (Foreground)
    assert report['element_skips'].get('group') == 1
    assert set(id_maps['images'].values()) == {'/tmp/ga.png', '/tmp/gb.png'}


def test_frames_kind_uses_first_frame_path():
    # A 'frames' element (sprite sequence) sources its first frame path into the
    # image table and emits one item.
    frames = _element('frames', _Z_FG, frames=('/tmp/f0.png', '/tmp/f1.png'),
                      keyframes={'x': _instant(7.0)})
    evaluator, id_maps, report = dd.build_doc([frames], _SCREEN_W, _SCREEN_H)

    assert report['elements_above'] == 1
    assert set(id_maps['images'].values()) == {'/tmp/f0.png'}


def test_shared_timeline_collapses_to_one_channel():
    # Two sprites sharing the SAME EventTimeline object for x must reuse one
    # channel (identity memoization) - both still emit and both track the curve.
    shared = build_timelines(keyframes={'x': _instant(123.0)})['x']
    a = Element(kind='sprite', z=_Z_FG, z_index=0, t_start=0.0,
                t_end=float('inf'), anchor=(0, 0), origin=(0.5, 0.5),
                timelines={**build_timelines(), 'x': shared}, asset='/tmp/a.png')
    b = Element(kind='sprite', z=_Z_FG, z_index=1, t_start=0.0,
                t_end=float('inf'), anchor=(0, 0), origin=(0.5, 0.5),
                timelines={**build_timelines(), 'x': shared}, asset='/tmp/b.png')
    evaluator, id_maps, report = dd.build_doc([a, b], _SCREEN_W, _SCREEN_H)

    assert report['elements_above'] == 2
    stream = dd.blit_stream(evaluator, 0.0)
    assert len(stream) == 2
    for (_k, _sid, mat, _a) in stream:
        assert abs(mat[0, 2] - 123.0) < 1.0


def test_accepts_storyboard_and_compiled_dict_shapes():
    # build_doc accepts a bare sequence, a Storyboard, and a compiled dict.
    from analysis.player.render.storyboard.model import Storyboard
    el = _sprite(_Z_FG, '/tmp/x.png', keyframes={'x': _instant(1.0)})

    sb = Storyboard(design_w=_SCREEN_W, design_h=_SCREEN_H, fit='min',
                    elements=(el,))
    _e1, _m1, r_sb = dd.build_doc(sb, _SCREEN_W, _SCREEN_H)
    _e2, _m2, r_dict = dd.build_doc({'tree': [el]}, _SCREEN_W, _SCREEN_H)
    _e3, _m3, r_seq = dd.build_doc([el], _SCREEN_W, _SCREEN_H)

    assert r_sb['elements_above'] == r_dict['elements_above'] == \
        r_seq['elements_above'] == 1


# --------------------------------------------------------------------------
# real-map smoke - skipped unless a fluXis storyboard is findable
# --------------------------------------------------------------------------

@pytest.mark.skip(reason="no in-tree fluXis map with a .fsb storyboard; "
                         "synthetic coverage is the bar")
def test_real_fluxis_storyboard_smoke():
    from analysis.games.fluxis.fsb_storyboard import parse_fsb
    sb = parse_fsb('<path to a real .fsb>')
    assert sb is not None
    evaluator, id_maps, report = dd.build_doc(sb, sb.design_w, sb.design_h)
    assert report['elements_below'] + report['elements_above'] >= 1
    dd.blit_stream(evaluator, 0.0)
