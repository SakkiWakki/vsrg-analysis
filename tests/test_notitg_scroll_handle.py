"""Lazy-path scroll multipliers: the xmod stream lives in the APPLIED
mods (the per-frame reader re-applies the chart's baseline and bursts
every frame), which the instant lazy compile does not have yet. The
compile therefore hands the player a sampleable HANDLE that rests at 1.0
and the background sweep swaps the resolved timeline in - the same
hot-swap shape as the screen-shake handle.
"""
import pytest

from analysis.games.notitg.sim.producers import ScrollMultiplierHandle


def test_handle_rests_at_unity_before_the_sweep():
    handle = ScrollMultiplierHandle()
    assert handle.sample(0.0) == (1.0,)
    assert handle.sample(123.4) == (1.0,)


def test_handle_samples_the_swapped_timeline():
    from analysis.player.render.effects.timeline import (
        EventTimeline, keyframes_from_events)

    handle = ScrollMultiplierHandle()
    events = [{'time': 0.0, 'duration': 1000.0, 'multiplier': 2.0,
               'ease': 0}]
    keyframes = keyframes_from_events(events, ('multiplier',), (1.0,))
    handle.timeline = EventTimeline(keyframes, rest=(1.0,))
    assert handle.sample(5.0) == (2.0,)


def test_init_state_prefers_the_adapter_live_timeline():
    from analysis.player.init.init_state import _build_scroll_mult_timeline

    handle = ScrollMultiplierHandle()

    class Adapter:
        def scroll_multiplier_timeline(self, replay):
            return handle

        def scroll_multipliers(self, replay):
            raise AssertionError('events path must not run with a '
                                 'live timeline present')

    class Player:
        _adapter = Adapter()

    assert _build_scroll_mult_timeline(Player(), {}) is handle


def test_init_state_falls_back_to_events():
    from analysis.player.init.init_state import _build_scroll_mult_timeline

    class Adapter:
        def scroll_multiplier_timeline(self, replay):
            return None

        def scroll_multipliers(self, replay):
            return [{'time': 0.0, 'duration': 0.0, 'multiplier': 1.5,
                     'ease': 0}]

    class Player:
        _adapter = Adapter()

    timeline = _build_scroll_mult_timeline(Player(), {})
    assert timeline.sample(1.0) == (1.5,)
