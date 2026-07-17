"""Coverage + unit tests for the NotITG actor Lua API registry.

The coverage test is the point: every method name in the committed
called-surface list (analysis/games/notitg/gat_called_methods.txt) must
resolve to a registry entry - implemented, ignored-with-reason, or
deferred-with-reason - so a chart calling an unmapped method fails CI
instead of silently no-oping. The rest exercise the verbs this module
newly landed (crop family end-to-end, oscillator/vanish/capture
classification) and the generated Lua name sets the bridge builds from
the registry.
"""
from pathlib import Path

import pytest

from analysis.games.notitg import lua_api
from analysis.games.notitg.lua_api import (
    COMMAND_NAMES, DEFERRED, GETTER_NAMES, IGNORED, IMPLEMENTED,
    VERB_REGISTRY, resolve)
from analysis.games.notitg.mod_stubs import _lua_name_set
from analysis.games.notitg.recording_actor import RecordingActor
from analysis.player.render.effects.timeline import Keyframe
from analysis.player.render.storyboard.model import Element, build_timelines

_CALLED_SURFACE = Path(lua_api.__file__).with_name('gat_called_methods.txt')


def _called_names() -> list:
    lines = _CALLED_SURFACE.read_text().splitlines()
    return [line.strip() for line in lines
            if line.strip() and not line.strip().startswith('#')]


# -- coverage: the CI firewall ----------------------------------------------

def test_every_called_method_resolves():
    """Zero unmapped names: a chart-called method with no registry entry
    is the failure the test exists to catch."""
    unmapped = [name for name in _called_names() if resolve(name) is None]
    assert unmapped == [], f'unmapped called methods: {unmapped}'


def test_every_verb_has_a_valid_state():
    for name, verb in VERB_REGISTRY.items():
        assert verb.state in (IMPLEMENTED, IGNORED, DEFERRED), name


def test_ignored_and_deferred_carry_a_reason():
    """The coverage states that are not IMPLEMENTED must justify
    themselves - an ignored/deferred verb with no note is an unexplained
    gap."""
    for name, verb in VERB_REGISTRY.items():
        if verb.state in (IGNORED, DEFERRED):
            assert verb.note, f'{name} ({verb.state}) has no reason'


def test_no_duplicate_or_stray_registry_entries():
    """The registry and the called surface match exactly: every called
    name is registered and every registered name is a real called
    method (no registry-only cruft)."""
    called = set(_called_names())
    registered = set(VERB_REGISTRY)
    assert registered == called, {
        'registry_only': sorted(registered - called),
        'called_only': sorted(called - registered),
    }


def test_registry_stats_are_reported():
    impl = lua_api.names_by_state(IMPLEMENTED)
    ignored = lua_api.names_by_state(IGNORED)
    deferred = lua_api.names_by_state(DEFERRED)
    assert len(impl) + len(ignored) + len(deferred) == len(VERB_REGISTRY)
    assert impl and deferred  # both states are exercised by real verbs


# -- generated Lua name sets stay faithful to the recorder ------------------

def test_getter_names_match_recorder_answerable_set():
    """__GETTER routes a call to the recorder's value path; it must be
    exactly the getters the recorder answers (scalar getters + the AFT
    marker + getrotation), no more."""
    assert set(GETTER_NAMES) == {
        'GetX', 'GetY', 'GetZ', 'GetZoom', 'GetZoomX', 'GetZoomY',
        'GetRotationX', 'GetRotationY', 'GetRotationZ', 'GetTexture',
        'getrotation'}


def test_command_names_are_the_actor_command_verbs():
    assert set(COMMAND_NAMES) == {'playcommand', 'queuecommand'}


def test_lua_name_set_renders_a_set_literal():
    rendered = _lua_name_set(('GetX', 'GetY'))
    assert rendered == '{GetX=true, GetY=true}'


# -- crop family: end to end ------------------------------------------------

def test_crop_verbs_record_onto_crop_channels():
    actor = RecordingActor(clock=0.0)
    actor.poke('croptop', [0.25])
    actor.poke('cropbottom', [0.1])
    actor.poke('cropleft', [0.5])
    actor.poke('cropright', [0.05])
    kf = actor.keyframes()
    assert kf['crop_top'][0].values == (0.25,)
    assert kf['crop_bottom'][0].values == (0.1,)
    assert kf['crop_left'][0].values == (0.5,)
    assert kf['crop_right'][0].values == (0.05,)


def test_crop_verbs_are_crop_setters_in_the_registry():
    for name in ('croptop', 'cropbottom', 'cropleft', 'cropright'):
        assert resolve(name).category == lua_api.CROP_SETTER
        assert resolve(name).state == IMPLEMENTED


def test_crop_animates_under_a_tween():
    """A crop poke inside an open tween interval records with the tween's
    duration/easing, so a scrolling reveal keyframes like any scalar."""
    actor = RecordingActor(clock=0.0)
    actor.poke('linear', [2.0])
    actor.poke('cropright', [1.0])
    frame = actor.keyframes()['crop_right'][0]
    assert frame.duration == 2.0


def _crop_element(crop_keyframes) -> Element:
    rests = {p: 0.0 for p in ('crop_top', 'crop_bottom',
                              'crop_left', 'crop_right')}
    return Element(
        kind='rect', z=0, z_index=0, t_start=0.0, t_end=10.0,
        anchor=(0.0, 0.0), origin=(0.0, 0.0),
        timelines=build_timelines(rests=rests, keyframes=crop_keyframes))


def test_crop_reaches_the_rendered_inset():
    """The renderer insets the drawn rect by the crop fractions: a rect
    cropped 0.25 off the left and 0.5 off the bottom draws in the
    remaining 0.75 x 0.5 sub-region."""
    from PySide6.QtCore import QRectF
    from analysis.player.render.storyboard import render as sb_render

    element = _crop_element({
        'crop_left': [Keyframe(0.0, (0.25,), 0.0, 0)],
        'crop_bottom': [Keyframe(0.0, (0.5,), 0.0, 0)],
    })
    crop = sb_render._crop_fractions(element, 1.0)
    assert crop == (0.25, 0.0, 0.0, 0.5)
    inset = sb_render._inset_rect(QRectF(0.0, 0.0, 100.0, 80.0), crop)
    assert inset.left() == pytest.approx(25.0)
    assert inset.width() == pytest.approx(75.0)
    assert inset.height() == pytest.approx(40.0)


def test_uncropped_element_reads_zero_and_draws_whole():
    """An element with no crop timelines (fluXis/osu sprites) reads
    all-zero crop and its inset is the identity rect - the uncropped
    path is untouched."""
    from PySide6.QtCore import QRectF
    from analysis.player.render.storyboard import render as sb_render

    element = Element(kind='sprite', z=0, z_index=0, t_start=0.0, t_end=1.0,
                      anchor=(0.0, 0.0), origin=(0.0, 0.0),
                      timelines=build_timelines())
    assert sb_render._crop_fractions(element, 0.0) == (0.0, 0.0, 0.0, 0.0)
    rect = QRectF(0.0, 0.0, 10.0, 10.0)
    inset = sb_render._inset_rect(rect, (0.0, 0.0, 0.0, 0.0))
    assert (inset.width(), inset.height()) == (10.0, 10.0)


# -- new-verb classification ------------------------------------------------

def test_effect_oscillators_are_classified_with_engine_source():
    for name in ('vibrate', 'wag', 'bob', 'bounce', 'spin'):
        verb = resolve(name)
        assert verb.category == lua_api.EFFECT_OSCILLATOR
        assert verb.source == 'Actor.cpp'


def test_vanish_point_is_implemented_and_sourced():
    verb = resolve('SetVanishPoint')
    assert verb.category == lua_api.VANISH
    assert verb.state == IMPLEMENTED
    assert verb.native == ('vanish_x', 'vanish_y')


def test_depth_buffer_is_deferred_to_the_gl_executor():
    verb = resolve('EnableDepthBuffer')
    assert verb.state == DEFERRED
    assert 'GL executor' in verb.note


def test_texture_filtering_is_ignored_as_cosmetic():
    verb = resolve('SetTextureFiltering')
    assert verb.state == IGNORED
    assert 'cosmetic' in verb.note
