"""CompiledDocument model + design-space surface + storyboard wrapping.

Covers phase-1 skeleton invariants (compiled visibility required,
layer-slot mutable over time, capture-range validation, group membership
including note-subset arrays), the per-game `design_space()` surface, and
that the storyboard maps through the document header identically to the
pre-consolidation kwargs.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.core import game as game_mod
from analysis.player.render.document import (CaptureContent, CaptureRange,
                                             ClockTable, CompiledDocument,
                                             DEFAULT_STRATA, DesignSpace,
                                             FIT_HEIGHT, FIT_MIN, Node,
                                             NotefieldContent, RectContent,
                                             SpriteContent, StreamTable,
                                             TextContent, Timeline, FIT_STRETCH)
from analysis.player.render.document.builder import document_from_player
from analysis.player.render.effects.timeline import EventTimeline, Keyframe
from analysis.player.render.storyboard.model import Element, build_timelines


def _tl(*rest):
    return Timeline(EventTimeline((), rest=rest))


def _layer_tl(name='background'):
    return Timeline(EventTimeline((), rest=(name,)))


# ── design space ---------------------------------------------------------

def test_design_space_validates_fit_and_size():
    with pytest.raises(ValueError):
        DesignSpace(640, 480, fit='diagonal')
    with pytest.raises(ValueError):
        DesignSpace(0, 480)


def test_design_space_per_game():
    expected = {
        'etterna': (640.0, 480.0, FIT_HEIGHT, False),
        'quaver': (640.0, 480.0, FIT_HEIGHT, False),
        'osu': (640.0, 480.0, FIT_HEIGHT, False),
        'fluxis': (1366.0, 768.0, FIT_MIN, False),
        'notitg': (640.0, 480.0, FIT_STRETCH, True),
    }
    games = game_mod.all_games()
    for name, (w, h, fit, clip) in expected.items():
        ds = games[name].design_space()
        assert (ds.width, ds.height, ds.fit, ds.clip) == (w, h, fit, clip)


# ── node model invariants ------------------------------------------------

def test_visibility_is_required():
    # The primary constructor demands an explicit visibility timeline;
    # there is no last-value-hold default.
    with pytest.raises(TypeError):
        Node(node_id='n', parent=None, layer=_layer_tl(),
             t_start=0.0, t_end=1.0, content=RectContent())


def test_always_visible_is_the_only_default():
    n = Node.always_visible('bg', None, _layer_tl(), 0.0, 5.0,
                            content=SpriteContent('white'))
    assert n.visibility.sample(0.0) == (1.0,)
    assert n.visibility.sample(4.9) == (1.0,)


def test_leaf_and_children_are_mutually_exclusive():
    with pytest.raises(ValueError):
        Node.always_visible('bad', None, _layer_tl(), 0.0, 1.0,
                            content=SpriteContent('x'), children=('c',))


def test_content_must_be_a_leaf_variant():
    with pytest.raises(TypeError):
        Node.always_visible('bad', None, _layer_tl(), 0.0, 1.0,
                            content=object())


def test_group_node_has_no_content():
    g = Node.always_visible('grp', None, _layer_tl(), 0.0, 1.0,
                            children=('a', 'b'))
    assert g.is_group
    assert g.content is None


def test_layer_slot_is_mutable_over_time():
    # NotITG draworder: a node re-slots between strata as t advances. The
    # layer is a Timeline; a step curve returns different stratum names.
    class _StepCurve:
        def sample(self, t):
            return (('background',) if t < 2.0 else ('hud',))

    n = Node.always_visible('slot', None, Timeline(_StepCurve()), 0.0, 10.0,
                            content=RectContent())
    assert n.layer.sample(0.0) == ('background',)
    assert n.layer.sample(5.0) == ('hud',)


# ── capture ranges -------------------------------------------------------

def test_capture_range_validation():
    with pytest.raises(ValueError):
        CaptureRange(2, 1)               # low > high
    with pytest.raises(ValueError):
        CaptureRange(-1, 0)              # negative index
    ok = CaptureRange(0, 2)
    assert (ok.low, ok.high) == (0, 2)


def test_capture_content_declares_a_stratum_range():
    doc_strata = DEFAULT_STRATA
    rng = CaptureRange(doc_strata.index('field'), doc_strata.index('notes'))
    cap = CaptureContent(capture=rng)
    assert cap.capture.low == 1 and cap.capture.high == 2


# ── group membership incl. note subsets ----------------------------------

def test_notefield_membership_tags_a_note_subset():
    # The Quaver _note_sv_groups pattern generalized: a per-note array
    # names which group each note joins, so a NOTE SUBSET is a group
    # member without a distinct node kind.
    membership = np.array(['spin', 'spin', None, None], dtype=object)
    nf = NotefieldContent(membership=membership)
    tagged = [i for i, g in enumerate(nf.membership) if g == 'spin']
    assert tagged == [0, 1]


def test_notefield_default_membership_is_none():
    assert NotefieldContent().membership is None


# ── document header ------------------------------------------------------

def test_document_rejects_mismatched_node_key():
    n = Node.always_visible('real', None, _layer_tl(), 0.0, 1.0,
                            content=RectContent())
    with pytest.raises(ValueError):
        CompiledDocument(design=DesignSpace(640, 480), nodes={'wrong': n})


def test_document_root_must_be_in_table():
    with pytest.raises(ValueError):
        CompiledDocument(design=DesignSpace(640, 480), roots=('ghost',))


def test_document_defaults():
    doc = CompiledDocument(design=DesignSpace(640, 480))
    assert doc.strata == DEFAULT_STRATA
    assert isinstance(doc.clocks, ClockTable)
    assert isinstance(doc.streams, StreamTable)
    assert doc.clocks.keys == {'song': 'song'}
    assert doc.stratum_index('notes') == 2


# ── storyboard wrapping through the document -----------------------------

def _leaf_el(kind, z=0, **overrides):
    fields = dict(
        kind=kind, z=z, z_index=0, t_start=0.0, t_end=5.0,
        anchor=(0.0, 0.0), origin=(0.0, 0.0), timelines=build_timelines())
    fields.update(overrides)
    return Element(**fields)


def _fake_player(storyboard, game='notitg'):
    adapter = game_mod.get(game)
    return SimpleNamespace(
        _adapter=SimpleNamespace(
            design_space=adapter.design_space,
            storyboard=lambda replay: storyboard),
        replay={})


def _storyboard(elements):
    from analysis.player.render.storyboard import Storyboard
    return Storyboard(design_w=640, design_h=480, fit='min',
                      elements=tuple(elements), clip_design_box=True)


def test_builder_wraps_design_space():
    doc = document_from_player(_fake_player(None))
    assert (doc.design.width, doc.design.fit, doc.design.clip) == (
        640.0, FIT_STRETCH, True)
    assert doc.nodes == {} and doc.roots == ()


def test_builder_maps_leaf_kinds():
    sb = _storyboard([
        _leaf_el('sprite', asset='/tex.png'),
        _leaf_el('rect'),
        _leaf_el('ellipse'),
        _leaf_el('text', text='hi', font_px=24.0),
    ])
    doc = document_from_player(_fake_player(sb))
    kinds = [type(doc.nodes[r].content).__name__ for r in doc.roots]
    assert kinds == ['SpriteContent', 'RectContent', 'RectContent',
                     'TextContent']
    sprite = doc.nodes[doc.roots[0]].content
    assert isinstance(sprite, SpriteContent) and sprite.asset == '/tex.png'


def test_builder_visibility_inverts_hidden():
    # A storyboard element hidden after t=2 must read NOT-rendered there;
    # visibility is the compiled inverse of the SM hidden bit.
    hidden_kfs = {'hidden': [Keyframe(t=2.0, values=(1.0,), duration=0.0,
                                      easing=0)]}
    el = _leaf_el('rect', timelines=build_timelines(keyframes=hidden_kfs))
    doc = document_from_player(_fake_player(_storyboard([el])))
    vis = doc.nodes[doc.roots[0]].visibility
    assert vis.sample(0.0) == (1.0,)     # shown before the hidden keyframe
    assert vis.sample(3.0) == (0.0,)     # hidden after it


def test_builder_group_becomes_group_node_with_children():
    child = _leaf_el('rect')
    group = _leaf_el('group', children=(child,))
    doc = document_from_player(_fake_player(_storyboard([group])))
    root = doc.nodes[doc.roots[0]]
    assert root.is_group and len(root.children) == 1
    child_node = doc.nodes[root.children[0]]
    assert child_node.parent == root.node_id
    assert isinstance(child_node.content, RectContent)


def test_builder_layer_slot_from_z():
    sb = _storyboard([_leaf_el('rect', z=-5), _leaf_el('rect', z=3)])
    doc = document_from_player(_fake_player(sb))
    below = doc.nodes[doc.roots[0]].layer.sample(0.0)
    above = doc.nodes[doc.roots[1]].layer.sample(0.0)
    assert below == ('background',) and above == ('hud',)


def test_storyboard_design_mapping_reads_through_adapter():
    # The one real phase-1 consumer: the storyboard the adapter emits
    # takes its fit/clip from design_space(), so a document built from
    # the player and the storyboard the player renders agree on the
    # design mapping.
    for game in ('notitg', 'osu', 'fluxis', 'etterna'):
        ds = game_mod.get(game).design_space()
        assert ds.fit in (FIT_MIN, FIT_HEIGHT, FIT_STRETCH)
