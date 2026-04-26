"""Cross-engine slot matrix.

The render controller derives cross-engine slots from `replay['sv']`
capabilities (BPM map -> beat-space; time-space chart -> osu-time A/B
view; beat-space chart -> projected osu-time). These tests pin the
slot matrix so future SvReplayDoc fields don't accidentally drop a
view or expose one that shouldn't be there.
"""
from __future__ import annotations

from analysis.player.sv.registry import (KEY_ETTERNA_BEAT, KEY_IDENTITY,
                                          KEY_OSU_TIME, KEY_QUAVER_TIME)
from analysis.player.sv.replay_doc import (KIND_BEAT_SPACE, KIND_IDENTITY,
                                            KIND_TIME_SPACE, SvReplayDoc)


def _build(replay):
    """Run the render controller's _build_registry against a fake player.
    Returns the registered keys + the active key so tests can assert
    against the slot matrix directly."""
    from analysis.player.sv.render import SvRenderController
    from types import SimpleNamespace

    p = SimpleNamespace(
        game='etterna',
        scroll_mode='linear_ms',
        _state=lambda: {},
        _mode_state={'linear_ms': {}},
        _clock=SimpleNamespace(set_sv_engine=lambda _e: None),
    )
    ctrl = SvRenderController(p)
    ctrl._registry = ctrl._build_registry(replay)
    return ctrl._registry


# ---- Quaver native ---------------------------------------------------------


def test_quaver_native_with_bpms_offers_etterna_beat_and_osu_time():
    reg = _build({'sv': SvReplayDoc(
        engine_kind=KIND_TIME_SPACE,
        engine_key=KEY_QUAVER_TIME,
        sections=[(0.0, 1.0), (4.0, 2.0)],
        initial_velocity=1.0,
        bpms=[(0.0, 120.0), (4.0, 240.0)],
    )})
    assert reg.native_key() == KEY_QUAVER_TIME
    keys = set(reg.keys())
    assert keys == {KEY_QUAVER_TIME, KEY_OSU_TIME, KEY_ETTERNA_BEAT,
                    KEY_IDENTITY}


def test_quaver_native_without_bpms_skips_etterna_beat():
    reg = _build({'sv': SvReplayDoc(
        engine_kind=KIND_TIME_SPACE,
        engine_key=KEY_QUAVER_TIME,
        sections=[(0.0, 1.0)],
        bpms=[],
    )})
    assert KEY_ETTERNA_BEAT not in reg.keys(), \
        'no bpms = no beat-space cross slot'
    assert KEY_OSU_TIME in reg.keys()


# ---- osu native ------------------------------------------------------------


def test_osu_native_with_bpms_offers_etterna_beat_only():
    """osu doesn't get a quaver-time A/B slot ; quaver-time without
    InitialScrollVelocity is meaningless."""
    reg = _build({'sv': SvReplayDoc(
        engine_kind=KIND_TIME_SPACE,
        engine_key=KEY_OSU_TIME,
        sections=[(0.0, 1.0)],
        bpms=[(0.0, 120.0)],
    )})
    assert reg.native_key() == KEY_OSU_TIME
    keys = set(reg.keys())
    assert keys == {KEY_OSU_TIME, KEY_ETTERNA_BEAT, KEY_IDENTITY}
    assert KEY_QUAVER_TIME not in keys


def test_osu_native_without_bpms_only_offers_identity():
    reg = _build({'sv': SvReplayDoc(
        engine_kind=KIND_TIME_SPACE,
        engine_key=KEY_OSU_TIME,
        sections=[(0.0, 1.0)],
    )})
    assert set(reg.keys()) == {KEY_OSU_TIME, KEY_IDENTITY}


# ---- Etterna native --------------------------------------------------------


def test_etterna_native_offers_osu_time_via_projection():
    reg = _build({'sv': SvReplayDoc(
        engine_kind=KIND_BEAT_SPACE,
        engine_key=KEY_ETTERNA_BEAT,
        scrolls=[(0.0, 1.0), (4.0, 2.0)],
        bpms=[(0.0, 120.0)],
    )})
    assert reg.native_key() == KEY_ETTERNA_BEAT
    keys = set(reg.keys())
    assert keys == {KEY_ETTERNA_BEAT, KEY_OSU_TIME, KEY_IDENTITY}
    # Beat-space natives never get an etterna_beat *cross* slot
    # registered on top of themselves.
    assert KEY_QUAVER_TIME not in keys


# ---- Identity --------------------------------------------------------------


def test_identity_doc_only_registers_identity():
    reg = _build({'sv': SvReplayDoc(
        engine_kind=KIND_IDENTITY, engine_key=KEY_IDENTITY,
    )})
    assert set(reg.keys()) == {KEY_IDENTITY}
    assert reg.native_key() == KEY_IDENTITY


def test_replay_without_sv_key_falls_back_to_identity():
    """Phase 1 fallback: a replay with no `sv` key still loads (identity
    only). Lets the render controller stay safe if a future code path
    forgets to populate the doc."""
    reg = _build({})
    assert set(reg.keys()) == {KEY_IDENTITY}
