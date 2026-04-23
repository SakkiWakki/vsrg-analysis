from __future__ import annotations

import numpy as np

from analysis.core import gui_adapter as gui_adapter_mod
from analysis.viz.plugins import discover
from analysis.components import viz_backend


def _fake_replay():
    return {
        'game': 'etterna',
        'keycount': 4,
        'offsets': np.array([0.001, -0.002, 0.003, 0.0], dtype=np.float64),
        'columns': np.array([0, 1, 2, 3], dtype=np.int64),
        'noterows': np.array([0, 48, 96, 144], dtype=np.int64),
        'misses': np.array([False, False, False, False], dtype=bool),
        'notetypes': np.array([0, 0, 0, 0], dtype=np.int64),
        'bpms': [(0, 120.0)],
        'sm_offset': 0.0,
    }


def test_component_viz_plugins_are_discovered():
    names = {name for name, _builder, _category in discover()}
    assert 'Chord sizes' in names
    assert 'Full report (all plots)' in names
    assert 'Note visualizer (scrollable)' in names


def test_chart_component_viz_builder_returns_figure():
    match = next(
        (item for item in discover() if item[0] == 'Chord sizes'),
        None,
    )
    assert match is not None
    _name, builder, category = match
    assert category == 'chart'
    fig = builder(_fake_replay(), game='etterna')
    assert hasattr(fig, 'axes')
    assert len(fig.axes) == 1


def test_widget_runtime_prewarms_note_viz_tab():
    viz_backend._NOTE_VIZ_TAB = None
    viz_backend._prepare_widget_runtime()
    assert viz_backend._NOTE_VIZ_TAB is not None


def test_note_viz_config_works_for_all_gui_games():
    viz_backend._prepare_widget_runtime()
    replays = {
        'etterna': {},
        'osu': {'od': 8.0, 'mods': 0},
    }
    for name, adapter in gui_adapter_mod.all_games().items():
        replay = replays.get(name, {})
        cfg = adapter.note_viz_config(replay)
        assert 'windows' in cfg
        assert 'unit_label' in cfg
        assert 'rows_per_ms' in cfg
        assert 'win' in cfg
        assert isinstance(cfg['windows'], list)
