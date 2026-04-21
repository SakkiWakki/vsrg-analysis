from __future__ import annotations

import numpy as np

from analysis.overlay.api import (
    EVENT_KEY_PRESSED,
    EVENT_KEY_RELEASED,
    EVENT_SONG_ENDED,
    EVENT_SONG_STARTED,
    PHASE_IDLE,
    PHASE_PLAYING,
    OverlayGameState,
    OverlayStateTracker,
)
from plugins.unsafe.osu_live.client import LiveSnapshot
from plugins.unsafe.osu_live.state import snapshot_to_overlay_state


def test_tracker_emits_song_start_and_end():
    tracker = OverlayStateTracker()
    idle = OverlayGameState(game='test', phase=PHASE_IDLE,
                            song_id='song-a', keycount=4)
    playing = OverlayGameState(game='test', phase=PHASE_PLAYING,
                               song_id='song-a', keycount=4)

    assert [e.kind for e in tracker.update(idle)] == ['state']
    assert EVENT_SONG_STARTED in [e.kind for e in tracker.update(playing)]
    ended = tracker.update(idle)

    assert EVENT_SONG_ENDED in [e.kind for e in ended]
    end_event = next(e for e in ended if e.kind == EVENT_SONG_ENDED)
    assert end_event.state.phase == PHASE_PLAYING


def test_tracker_key_events_are_lane_count_agnostic():
    tracker = OverlayStateTracker()
    base = OverlayGameState(game='test', phase=PHASE_PLAYING,
                            song_id='song-a', keycount=7,
                            pressed_lanes=(0, 6))
    next_state = OverlayGameState(game='test', phase=PHASE_PLAYING,
                                  song_id='song-a', keycount=7,
                                  pressed_lanes=(1, 6, 9))

    tracker.update(base)
    events = tracker.update(next_state)

    pressed = [e.lane for e in events if e.kind == EVENT_KEY_PRESSED]
    released = [e.lane for e in events if e.kind == EVENT_KEY_RELEASED]
    assert pressed == [1]
    assert released == [0]


def test_tracker_does_not_restart_every_frame_without_song_id():
    tracker = OverlayStateTracker()
    first = OverlayGameState(game='test', phase=PHASE_PLAYING, keycount=4)
    second = OverlayGameState(game='test', phase=PHASE_PLAYING, keycount=4)

    tracker.update(first)
    events = tracker.update(second)

    assert EVENT_SONG_STARTED not in [e.kind for e in events]
    assert EVENT_SONG_ENDED not in [e.kind for e in events]


def test_osu_snapshot_adapts_to_overlay_state():
    snap = LiveSnapshot(
        connected=True,
        in_gameplay=True,
        map_title='Example Map',
        combo=123,
        max_combo=456,
        accuracy=98.76,
        unstable_rate=88.0,
        hits_300=10,
        hits_100=2,
        hits_50=1,
        hits_miss=0,
        offsets=np.asarray([0.01, -0.02], dtype=np.float64),
        columns=np.asarray([0, 3], dtype=np.int32),
        keycount=4,
    )

    state = snapshot_to_overlay_state(snap)

    assert state.game == 'osu'
    assert state.is_playing
    assert state.song_title == 'Example Map'
    assert state.keycount == 4
    assert state.combo == 123
    assert state.judgment('300') == 10
    assert state.hit_offsets_s == (0.01, -0.02)
    assert state.hit_lanes == (0, 3)
