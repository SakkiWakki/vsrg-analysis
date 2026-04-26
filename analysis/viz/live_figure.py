"""Host-side live-viz wrapper: rebuild a static viz from live game memory.

Lives outside any plugin tree so sandboxed plugins can use it without
importing ``PySide6`` or ``matplotlib.backends`` themselves. The plugin
hands us a ``static_build_fn(replay, game)`` -- the same callable the
non-live viz registers -- and we host it inside a ``QWidget`` that
swaps in a fresh figure on each tick from the active osu memory poller.

A plugin's live-viz registration shrinks to one line::

    def build(replay=None, game='osu', **_):
        from analysis.viz.live_figure import build_live_figure
        from plugins.builtin.viz.drift import build as build_drift
        return build_live_figure(build_drift, game='osu')

Snapshot wiring:

The widget reads :func:`analysis.components.provider.current_game_memory`
each tick, translates it to a ``replay``-shaped dict via
:func:`game_memory_to_replay_dict`, and hands the result to
``static_build_fn``. The provider is installed by the host's osu live
client (:mod:`analysis.games.osu.live_client`); plugins don't have to
think about it.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from analysis.components.api import GameMemoryState
from analysis.components.provider import current_game_memory


def game_memory_to_replay_dict(snap: GameMemoryState | None,
                               *, game: str = 'osu') -> dict | None:
    """Translate a live memory snapshot into the ``replay``-shaped dict
    the static viz builders expect (``offsets``, ``columns``,
    ``noterows``, ``keycount``, ...).

    Returns None when the snapshot is unavailable or carries no hits
    yet -- callers should render a "waiting" state in that case rather
    than passing an empty dict to a viz builder that may not tolerate it.
    """
    if snap is None:
        return None
    hits = snap.hit_errors_ms
    if not hits:
        return None
    offsets_s = np.asarray(hits, dtype=np.float64) / 1000.0
    n = len(offsets_s)
    # Synthetic round-robin lanes -- live memory doesn't carry per-hit
    # column info on the standard osu reader. Static viz builders that
    # only use offsets work fine; ones that depend on real columns will
    # produce decorative-only hand splits, which is the same caveat the
    # previous in-plugin wrapper had.
    columns = np.arange(n, dtype=np.int32) % 4
    noterows = np.arange(n, dtype=np.int64)
    return {
        'game': game,
        'keycount': 4,
        'offsets': offsets_s,
        'columns': columns,
        'noterows': noterows,
        'misses': np.zeros(n, dtype=np.bool_),
        'notetypes': np.zeros(n, dtype=np.int32),
        'combo': snap.combo,
        'max_combo': snap.max_combo,
        'accuracy': snap.accuracy,
        'map_md5': snap.map_md5,
        'map_title': snap.map_title,
    }


def build_live_figure(static_build_fn: Callable, *, game: str = 'osu',
                      interval_ms: int = 100):
    """Return a live-rebuilding ``QWidget`` wrapping ``static_build_fn``.

    Plugin-side helper. We construct the widget here so the plugin
    doesn't import ``PySide6`` / ``matplotlib.backends`` directly --
    sandboxed plugins call this one function and get a working live
    visualization back.
    """
    return _LiveFigureWidget(static_build_fn, game=game,
                             interval_ms=interval_ms)


# ── Implementation ─────────────────────────────────────────────────


def _LiveFigureWidget(static_build_fn: Callable, *, game: str,
                      interval_ms: int):
    """Lazily-imported widget factory. The PySide6 / matplotlib imports
    happen here so this module is safe to import in headless tests --
    the widget itself only constructs when something asks for it."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    class _Widget(QWidget):
        def __init__(self):
            super().__init__()
            self._build_fn = static_build_fn
            self._game = game
            self._status = QLabel('Waiting for osu! ...')
            self._canvas: FigureCanvasQTAgg | None = None
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.addWidget(self._status)
            self._layout = layout
            self._timer = QTimer(self)
            self._timer.setInterval(int(interval_ms))
            self._timer.timeout.connect(self._refresh)
            self._timer.start()
            self._refresh()

        def closeEvent(self, ev):
            self._timer.stop()
            super().closeEvent(ev)

        def _refresh(self):
            snap = current_game_memory()
            replay = game_memory_to_replay_dict(snap, game=self._game)
            if replay is None:
                self._set_status(
                    'Waiting for osu! ; is the game running and a map active?')
                return

            try:
                fig = self._build_fn(replay, game=self._game)
            except Exception as exc:
                self._set_status(f'viz error: {exc}')
                return

            if self._canvas is not None:
                self._layout.removeWidget(self._canvas)
                self._canvas.setParent(None)
                self._canvas.deleteLater()
                self._canvas = None
            self._canvas = FigureCanvasQTAgg(fig)
            self._layout.addWidget(self._canvas, 1)

            n_hits = len(replay['offsets'])
            self._set_status(
                f'{replay["map_title"]}  |  '
                f'combo {replay["combo"]}/{replay["max_combo"]}  '
                f'acc {replay["accuracy"]:.2f}%  '
                f'({n_hits} hits)'
            )

        def _set_status(self, text: str):
            self._status.setText(text)

    return _Widget()
