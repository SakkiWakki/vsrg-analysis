"""Passive read curves over the recording sim's segment timelines.

The lazy-replay read side, v2: `SegCurve` evaluates an actor property
at ANY published time by sampling the segment timelines the sim's
actors record as they run (`SimActor._seg`), with no `advance_to` and
no second simulation. Queries clamp to the recording sim's clock
(`sim.now`, the frontier): a playhead ahead of the sweep sees the
newest known state instead of blocking the caller, and a backward seek
is a cursor reset, never a rebuild.

Duck-typed drop-in for `LiveCurve`: `sample(t)` returns the same tuple
shapes, so the renderer cannot tell the difference.
"""
from __future__ import annotations

import os
from bisect import bisect_right

from analysis.player.render.segment_timeline import Cursor
from analysis.player.render.storyboard.model import (
    _COLOR_RESTS, _SCALAR_RESTS, LiveCurve, build_live_timelines)

_TRUE = {'1', 'true', 'yes', 'on'}


def segtl_enabled() -> bool:
    """Segment-timeline reads are the default lazy-replay read path;
    VSRG_NOTITG_SEGTL=0 reverts to LiveCurves driving the sim from the
    render thread (the pre-segment behavior, for differential testing)."""
    return os.environ.get('VSRG_NOTITG_SEGTL', '1').lower() in _TRUE


class SegCurve:
    __slots__ = ('_sim', '_rec_id', '_prop', '_rest', '_cursors')

    def __init__(self, sim, rec_id, prop, rest):
        self._sim = sim
        self._rec_id = rec_id
        self._prop = prop
        self._rest = rest if isinstance(rest, tuple) else (rest,)
        self._cursors: list = []

    def sample(self, t: float) -> tuple:
        sim = self._sim
        actor = sim.env._actors.get(self._rec_id)
        if actor is None:
            return self._rest
        if t > sim.frontier:
            # Beyond the sweep: the schedule-lowered preview knows the
            # declarative future; without one, hold the newest known
            # state (the frontier clamp).
            preview = getattr(actor, '_seg_preview', None)
            lanes = preview.get(self._prop) if preview else None
            if lanes:
                return tuple(lane.sample(t) for lane in lanes)
            t = sim.frontier

        tokens = actor._seg_tokens.get(self._prop)
        if tokens is not None:
            return self._token_at(tokens, t)

        lanes = actor._seg.get(self._prop)
        if not lanes:
            return self._rest
        cursors = self._cursors
        while len(cursors) < len(lanes):
            cursors.append(Cursor())
        return tuple(lane.sample(t, cur)
                     for lane, cur in zip(lanes, cursors))

    def is_static(self) -> bool:
        """True when this prop has NO recorded motion, so `sample` provably
        returns `rest` at every t and a consumer can skip walking the curve.

        The three sources `sample` reads are token steps, recorded lanes, and
        the beyond-frontier preview; with none of them the actor never touched
        this prop. Exporters dense-sample a curve at a fixed dt to discover its
        shape, which for the (overwhelmingly common) untouched prop is
        thousands of samples to learn it never moves - this answers that in
        three dict lookups."""
        actor = self._sim.env._actors.get(self._rec_id)
        if actor is None:
            return True
        preview = getattr(actor, '_seg_preview', None)
        return not (actor._seg_tokens.get(self._prop)
                    or actor._seg.get(self._prop)
                    or (preview and preview.get(self._prop)))

    def _token_at(self, tokens, t: float) -> tuple:
        ts, vals = tokens
        idx = bisect_right(ts, t) - 1
        return vals[idx] if idx >= 0 else self._rest


def build_seg_timelines(sim, rec_id, rests: dict | None = None) -> dict:
    """Like `build_live_timelines`, but each property reads the
    recorded segments passively. Same key set + rest defaults, so an
    element samples identically whichever curve family backs it."""
    rests = {**_SCALAR_RESTS, **_COLOR_RESTS, **(rests or {})}
    return {prop: SegCurve(sim, rec_id, prop, rest)
            for prop, rest in rests.items()}


def curve_for(sim, rec_id, prop, rest):
    """One live-value curve, in the session's read family."""
    family = SegCurve if segtl_enabled() else LiveCurve
    return family(sim, rec_id, prop, rest)


def timelines_for(sim, rec_id, rests: dict | None = None) -> dict:
    """A full property->curve dict, in the session's read family."""
    if segtl_enabled():
        return build_seg_timelines(sim, rec_id, rests=rests)
    return build_live_timelines(sim, rec_id, rests=rests)
