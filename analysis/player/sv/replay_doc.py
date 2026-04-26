"""Canonical SV input document.

Replaces the per-game ad-hoc replay keys (`_quaver_sv_sections`,
`_etterna_scrolls`, `_osu_bpms`, ...) with one shape that every parser
populates and the render controller reads. See DESIGN.tex Section 8 for
the model: a chart is described by an `engine_kind` + `engine_key` pair
plus whichever capability fields the kind needs.

Faithful port -- every field that any current adapter writes shows up
here. Reduction (e.g. dropping `flags['legacy_ln']` if it stops being
needed, or splitting time-space vs. beat-space into separate types) is
deferred. The fat-union shape keeps phase 1 of the port mechanical and
diff-reviewable; once the cutover lands we can prune.

The doc is placed on `replay['sv']`. Parsers also keep writing the
legacy `_<game>_*` keys during phase 1 (dual-write) so the existing
`SvRenderController._build_registry` keeps working ; phase 2 swaps the
controller to read the doc and the legacy keys go away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Engine-kind tags. Each chart's parser declares which integration model
# the chart's data lives in -- not which game it came from. The render
# controller dispatches on this to pick the native factory; cross-engine
# slots are derived from capability fields below (e.g. any chart with a
# `bpms` map can be projected into the beat-space engine).
KIND_TIME_SPACE = 'time_space'   # osu, Quaver
KIND_BEAT_SPACE = 'beat_space'   # Etterna XMOD
KIND_IDENTITY = 'identity'       # no SV; renderer uses scroll-speed only


@dataclass(frozen=True)
class SvReplayDoc:
    """Source-of-truth SV inputs for one parsed replay.

    The shape is a fat union: a `time_space` chart populates `sections`
    (and possibly `initial_velocity` / `groups`); a `beat_space` chart
    populates `scrolls` / `speeds` / `bpms` / `stops` / `delays` /
    `warps` / `sm_offset`. Cross-engine fields (`bpms`) are populated
    on every chart that has a BPM map regardless of native kind, since
    the renderer uses them to expose alternate-engine views.

    Construction is positional-arg-free; parsers always pass keyword
    args to keep the call site self-documenting against this many
    fields.
    """
    # --- identity -------------------------------------------------------
    engine_kind: str                          # KIND_TIME_SPACE | KIND_BEAT_SPACE | KIND_IDENTITY
    engine_key: str                           # 'quaver_time' | 'osu_time' | 'etterna_beat' | 'identity'

    # --- time-space inputs ---------------------------------------------
    # `sections` is the canonical (time_sec, multiplier) list every
    # time-space integrator consumes. `initial_velocity` is Quaver's
    # InitialScrollVelocity (default 1.0; osu charts always write 1.0).
    # `groups` is `{group_id -> {sections, initial_velocity}}` for
    # Quaver TimingGroups; None when the chart has only the default group.
    sections: list = field(default_factory=list)
    initial_velocity: float = 1.0
    groups: dict | None = None

    # --- beat-space inputs (Etterna SSC/SM) ----------------------------
    scrolls: list = field(default_factory=list)
    speeds: list = field(default_factory=list)
    stops: list = field(default_factory=list)
    delays: list = field(default_factory=list)
    warps: list = field(default_factory=list)

    # --- cross-engine BPM map ------------------------------------------
    # Populated whenever the chart has timing data, so any chart can be
    # played under the beat-space engine via `beat_space_engine(scrolls=[],
    # speeds=[], bpms=bpms, sm_offset=sm_offset)`. `bpms` shape is
    # `[(beat, bpm), ...]` ; `sm_offset` is Etterna's audio-vs-chart
    # offset in seconds (osu/Quaver always 0.0).
    bpms: list = field(default_factory=list)
    sm_offset: float = 0.0

    # --- per-note routing ----------------------------------------------
    # Parallel to `replay['noterows']` (post-sort). Each entry is the
    # group-id string a note belongs to; `None` when groups aren't used.
    # The renderer uses this for per-note y-projection in
    # `batch_time_to_y` so notes in different SV-streams render correctly.
    note_groups: np.ndarray | None = None

    # --- renderer flags ------------------------------------------------
    # Per-chart booleans the SV renderer needs but that don't fit any
    # other field. Today: {'legacy_ln': bool} (Quaver's
    # LegacyLNRendering chart flag). Kept as a dict so adding new flags
    # doesn't bloat the dataclass.
    flags: dict = field(default_factory=dict)


def identity_doc() -> SvReplayDoc:
    """Doc for charts with no SV ; the renderer uses scroll-speed only."""
    return SvReplayDoc(engine_kind=KIND_IDENTITY, engine_key='identity')


def replay_sv(replay: dict) -> SvReplayDoc:
    """Read the SV doc off a replay, falling back to identity when the
    parser hasn't dual-written it yet (during phase 1 only)."""
    doc = replay.get('sv')
    if isinstance(doc, SvReplayDoc):
        return doc
    return identity_doc()


__all__ = [
    'KIND_TIME_SPACE', 'KIND_BEAT_SPACE', 'KIND_IDENTITY',
    'SvReplayDoc', 'identity_doc', 'replay_sv',
]
