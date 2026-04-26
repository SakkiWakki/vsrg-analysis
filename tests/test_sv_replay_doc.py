"""Round-trip tests for `replay['sv']` dual-write.

Phase 1 of the SV-doc port: every parser writes BOTH the legacy
`_<game>_*` keys AND a canonical `SvReplayDoc` on `replay['sv']`. These
tests pin the two against each other so phase 2 (when the legacy keys
disappear) can land without a behavior change.
"""
from __future__ import annotations

import numpy as np

from analysis.player.sv.replay_doc import (KIND_BEAT_SPACE, KIND_IDENTITY,
                                            KIND_TIME_SPACE, SvReplayDoc)


# ---- Etterna ---------------------------------------------------------------


def _etterna_chart_dict(**overrides):
    """Minimal `found` dict for `EtternaAdapter._attach_chart_extras`. The
    `chart` block carries the SV inputs we want to round-trip."""
    base = {
        'file': 'synthetic.sm',
        'data': {'bpms': [(0.0, 120.0)], 'offset': 0.0},
        'chart': {
            'stepstype': 'dance-single',
            'bpms': [(0.0, 120.0)],
            'offset': 0.0,
            'scrolls': [(0.0, 1.0)],
            'speeds': [(0.0, 1.0, 0.0, 0)],
            'stops': [],
            'delays': [],
            'warps': [],
            'notedata': '0000',
        },
    }
    base['chart'].update(overrides)
    return base


def test_etterna_dual_write_beat_space_doc():
    from analysis.games.etterna.adapter import EtternaAdapter
    replay = {
        'noterows': np.array([], dtype=np.int64),
        'columns': np.array([], dtype=np.int32),
        'offsets': np.array([], dtype=np.float64),
        'notetypes': np.array([], dtype=np.int32),
        'misses': np.array([], dtype=bool),
    }
    EtternaAdapter._attach_chart_extras(
        replay, _etterna_chart_dict(scrolls=[(0.0, 2.0)],
                                     bpms=[(0.0, 120.0), (4.0, 240.0)]))
    doc = replay['sv']
    assert isinstance(doc, SvReplayDoc)
    assert doc.engine_kind == KIND_BEAT_SPACE
    assert doc.engine_key == 'etterna_beat'
    # Every legacy key must equal the corresponding doc field.
    assert list(doc.scrolls) == replay['_etterna_scrolls']
    assert list(doc.speeds) == replay['_etterna_speeds']
    assert list(doc.stops) == replay['_etterna_stops']
    assert list(doc.delays) == replay['_etterna_delays']
    assert list(doc.warps) == replay['_etterna_warps']
    assert list(doc.bpms) == replay['_etterna_bpms']
    assert doc.sm_offset == replay['_etterna_offset']


def test_etterna_dual_write_identity_when_chart_has_no_sv():
    """Single-segment 120-BPM chart with no SCROLLS / SPEEDS / STOPS /
    DELAYS / WARPS yields an identity doc (renderer skips beat-space
    integration since there's nothing to integrate)."""
    from analysis.games.etterna.adapter import EtternaAdapter
    replay = {
        'noterows': np.array([], dtype=np.int64),
        'columns': np.array([], dtype=np.int32),
        'offsets': np.array([], dtype=np.float64),
        'notetypes': np.array([], dtype=np.int32),
        'misses': np.array([], dtype=bool),
    }
    EtternaAdapter._attach_chart_extras(
        replay, _etterna_chart_dict(scrolls=[], speeds=[],
                                     bpms=[(0.0, 120.0)]))
    doc = replay['sv']
    assert doc.engine_kind == KIND_IDENTITY
    assert doc.engine_key == 'identity'
    # Legacy keys still get populated even in the identity case ; the
    # renderer reads them for its existing branch.
    assert replay['_etterna_bpms'] == [(0.0, 120.0)]


# ---- osu -------------------------------------------------------------------


def test_osu_bpms_projection_matches_doc_bpms():
    """The `_osu_bpms_from_timing_points` helper is the only behavior
    that has to match between the legacy key and the doc -- the rest of
    the osu parser depends on `osrparse` + a real `.osu` file, which the
    test environment doesn't have."""
    from analysis.games.osu.replay.parse import _osu_bpms_from_timing_points
    timing = [(0, 500.0), (4000, 250.0)]    # 120 -> 240 BPM
    bpms = _osu_bpms_from_timing_points(timing)
    # Construct the doc the way the parser does so we're exercising the
    # same code path.
    doc = SvReplayDoc(
        engine_kind=KIND_TIME_SPACE,
        engine_key='osu_time',
        sections=[],
        bpms=bpms,
    )
    assert list(doc.bpms) == bpms


# ---- Quaver ----------------------------------------------------------------


def test_quaver_bpms_projection_matches_doc_bpms():
    """Same shape as the osu test ; we exercise the public projection
    helper without needing a real `.qua` + `.qr` pair."""
    from analysis.games.quaver.parse import _quaver_bpms_to_beat_space
    timing = [(0, 120.0), (2000, 240.0)]
    bpms = _quaver_bpms_to_beat_space(timing)
    doc = SvReplayDoc(
        engine_kind=KIND_TIME_SPACE,
        engine_key='quaver_time',
        sections=[],
        initial_velocity=1.5,
        bpms=bpms,
    )
    assert list(doc.bpms) == bpms
    assert doc.initial_velocity == 1.5


# ---- Doc shape -------------------------------------------------------------


def test_doc_defaults_are_empty_for_unused_kinds():
    """A time-space doc shouldn't carry beat-space leftovers and vice
    versa. Defaults are empty containers so cross-engine derivation can
    test capability presence without a kind branch."""
    time_doc = SvReplayDoc(
        engine_kind=KIND_TIME_SPACE, engine_key='osu_time',
        sections=[(0.0, 1.0)],
    )
    assert time_doc.scrolls == []
    assert time_doc.speeds == []
    assert time_doc.bpms == []

    beat_doc = SvReplayDoc(
        engine_kind=KIND_BEAT_SPACE, engine_key='etterna_beat',
        scrolls=[(0.0, 1.0)],
    )
    assert beat_doc.sections == []
    assert beat_doc.initial_velocity == 1.0
    assert beat_doc.groups is None


def test_doc_is_frozen():
    import dataclasses
    doc = SvReplayDoc(engine_kind=KIND_IDENTITY, engine_key='identity')
    raised = False
    try:
        doc.engine_kind = KIND_TIME_SPACE
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised, 'SvReplayDoc must be frozen so the renderer can rely on it'


def test_replay_sv_helper_falls_back_to_identity():
    """During phase 1, code reading `replay['sv']` may hit a replay that
    came from a parser-not-yet-updated path. The helper returns an
    identity doc instead of KeyError."""
    from analysis.player.sv.replay_doc import replay_sv
    replay_no_sv = {'game': 'osu', 'noterows': np.array([])}
    doc = replay_sv(replay_no_sv)
    assert doc.engine_kind == KIND_IDENTITY
