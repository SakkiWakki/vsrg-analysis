"""Timing offsets scattered over chart progression (x-axis in seconds)."""
import numpy as np

from analysis.viz.plots import plot_scatter_timeline
from ._common import clean_arrays, new_fig


def _noterows_to_seconds(rows, replay):
    """Convert a replay's noterows to seconds. osu rows are already ms; Etterna
    rows need the chart's BPM map if available, else a safe fallback."""
    if replay.get('chart_path'):  # osu
        return rows.astype(np.float64) / 1000.0
    bpms = replay.get('bpms')
    sm_offset = replay.get('sm_offset', 0.0)
    if bpms is not None:
        from analysis.games.etterna.sm_chart import row_to_time
        return np.array([row_to_time(int(r), bpms, sm_offset) for r in rows])
    # Fallback used elsewhere: 48 rows/beat @ 120 BPM = 96 rows/sec.
    return rows.astype(np.float64) / 96.0


def build(replay, game='etterna', **_):
    rows, offs, cols = clean_arrays(replay)
    t_sec = _noterows_to_seconds(rows, replay)
    fig, ax = new_fig(12, 5)
    plot_scatter_timeline(t_sec, offs, cols, ax=ax, xlabel='time (s)')
    return fig


def register(add):
    add('Timing scatter', build, category='chart')
