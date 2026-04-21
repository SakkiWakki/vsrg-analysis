"""Sandbox-safe helpers for gamescope overlay plugins.

Overlay plugin modules may import this file from sandboxed bundles. It
intentionally contains only pure constants and small helpers; the shm
publisher that touches ``/dev/shm`` lives in ``analysis.overlay.publisher`` and is
owned by the trusted host process.
"""
from __future__ import annotations

from dataclasses import dataclass


ANCHOR_TL = 0
ANCHOR_TR = 1
ANCHOR_BL = 2
ANCHOR_BR = 3
ANCHOR_C = 4


def rgba(r: int, g: int, b: int, a: int = 255) -> int:
    """Pack 0..255 RGBA components into the uint32 layout read by the C
    overlay renderer: byte 0 is R and byte 3 is A."""
    return ((int(r) & 0xff)
            | ((int(g) & 0xff) << 8)
            | ((int(b) & 0xff) << 16)
            | ((int(a) & 0xff) << 24))


WHITE = rgba(250, 250, 250)
BLACK_DIM = rgba(10, 10, 15, 140)
BLUE_ACCENT = rgba(75, 164, 255, 230)
WARN_AMBER = rgba(255, 180, 50)
HIST_BAR = rgba(75, 164, 255, 230)


PHASE_DISCONNECTED = 'disconnected'
PHASE_IDLE = 'idle'
PHASE_SELECTING = 'selecting'
PHASE_PLAYING = 'playing'
PHASE_PAUSED = 'paused'
PHASE_RESULTS = 'results'

EVENT_STATE = 'state'
EVENT_SONG_STARTED = 'song_started'
EVENT_SONG_ENDED = 'song_ended'
EVENT_KEY_PRESSED = 'key_pressed'
EVENT_KEY_RELEASED = 'key_released'


@dataclass(frozen=True)
class OverlayGameState:
    """Game-agnostic live state for overlays and live viz.

    Game adapters should translate their native memory/API shape into this
    struct. Renderer code can then stay reusable across osu!, Etterna, or
    future games. Lanes are zero-based and bounded by ``keycount`` so 4K,
    7K, 10K, etc. all share one representation.
    """
    game: str
    phase: str = PHASE_DISCONNECTED
    song_id: str = ''
    song_title: str = ''
    artist: str = ''
    difficulty: str = ''
    rate: float = 1.0
    keycount: int = 0
    position_s: float = 0.0
    paused: bool = False
    combo: int = 0
    max_combo: int = 0
    accuracy: float = 0.0
    unstable_rate: float = 0.0
    judgments: tuple[tuple[str, int], ...] = ()
    hit_offsets_s: tuple[float, ...] = ()
    hit_lanes: tuple[int, ...] = ()
    pressed_lanes: tuple[int, ...] = ()

    @property
    def is_playing(self) -> bool:
        return self.phase == PHASE_PLAYING and not self.paused

    @property
    def song_label(self) -> str:
        if self.artist and self.song_title:
            return f'{self.artist} - {self.song_title}'
        return self.song_title or self.song_id

    def judgment(self, name: str, default: int = 0) -> int:
        for key, value in self.judgments:
            if key == name:
                return int(value)
        return int(default)


@dataclass(frozen=True)
class OverlayEvent:
    """Transition/event emitted from successive :class:`OverlayGameState`s."""
    kind: str
    state: OverlayGameState
    previous: OverlayGameState | None = None
    lane: int | None = None
    keycount: int = 0


class OverlayStateTracker:
    """Derive lifecycle/input events from game-agnostic state updates.

    This is deliberately small: callers feed it snapshots from whatever
    game adapter exists today. It emits song start/end and per-lane key
    transitions without knowing anything about osu!, Etterna, or the
    source of the data.
    """

    def __init__(self):
        self._state: OverlayGameState | None = None

    @property
    def state(self) -> OverlayGameState | None:
        return self._state

    def update(self, state: OverlayGameState) -> tuple[OverlayEvent, ...]:
        prev = self._state
        self._state = state
        events: list[OverlayEvent] = [
            OverlayEvent(EVENT_STATE, state, previous=prev,
                         keycount=int(state.keycount)),
        ]
        if prev is None:
            if state.is_playing:
                events.append(OverlayEvent(
                    EVENT_SONG_STARTED, state, previous=None,
                    keycount=int(state.keycount)))
            return tuple(events)

        prev_active = prev.is_playing
        cur_active = state.is_playing
        same_song = (prev.song_id == state.song_id) \
            if (prev.song_id or state.song_id) else True

        if prev_active and (not cur_active or not same_song):
            events.append(OverlayEvent(
                EVENT_SONG_ENDED, prev, previous=prev,
                keycount=int(prev.keycount)))
        if cur_active and (not prev_active or not same_song):
            events.append(OverlayEvent(
                EVENT_SONG_STARTED, state, previous=prev,
                keycount=int(state.keycount)))

        prev_pressed = set(_valid_lanes(prev.pressed_lanes, prev.keycount))
        cur_pressed = set(_valid_lanes(state.pressed_lanes, state.keycount))
        for lane in sorted(cur_pressed - prev_pressed):
            events.append(OverlayEvent(
                EVENT_KEY_PRESSED, state, previous=prev, lane=lane,
                keycount=int(state.keycount)))
        for lane in sorted(prev_pressed - cur_pressed):
            events.append(OverlayEvent(
                EVENT_KEY_RELEASED, state, previous=prev, lane=lane,
                keycount=int(state.keycount)))
        return tuple(events)


def widget_id(name: str) -> int:
    """Stable FNV-1a 32-bit id used by the renderer for drag layout."""
    h = 0x811c9dc5
    for b in str(name).encode('utf-8'):
        h ^= b
        h = (h * 0x01000193) & 0xffffffff
    return h or 0x811c9dc5


def _valid_lanes(lanes, keycount: int) -> tuple[int, ...]:
    n = max(0, int(keycount or 0))
    out = []
    for lane in lanes or ():
        lane = int(lane)
        if 0 <= lane < n:
            out.append(lane)
    return tuple(out)
