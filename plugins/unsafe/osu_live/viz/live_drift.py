"""Wrap ``plugins/builtin/viz/drift.py`` for live tosu data.

The wrapping pattern here is the template for every other live viz:
import the static builder, hand it to :class:`LiveFigureWidget` so
the figure redraws from a ``TosuClient`` snapshot each tick. No
changes to the underlying viz.
"""
from __future__ import annotations


def build(replay=None, game='osu', **_):
    # Import inside build() so bundle discovery doesn't fail on machines
    # without PySide6 available at import time (the viz role should
    # tolerate partial availability).
    from plugins.unsafe.osu_live.live_viz import LiveFigureWidget
    from plugins.builtin.viz.drift import build as build_drift

    def _rebuild(rep):
        return build_drift(rep, game='osu')

    return LiveFigureWidget(_rebuild)


def register(add):
    add('Live: drift (hands × time)', build, category='chart')


def _open_live_stats_window():
    """Open the live-drift widget as a top-level window.

    Wired into the library toolbar via ``register_library_actions``.
    Kept in this module (rather than in a dedicated one) because the
    action and the widget are conceptually the same thing — a top-level
    "live viz" surface — and forcing a second module just to host the
    button would be noise."""
    from PySide6.QtWidgets import QApplication, QMessageBox
    try:
        widget = build()
    except Exception as exc:
        QMessageBox.warning(
            None, 'Live stats',
            f'Could not open the live stats panel:\n{exc}\n\n'
            'Make sure tosu is running on http://127.0.0.1:24050.')
        return
    widget.setParent(None)
    widget.setWindowTitle('osu! live stats')
    widget.resize(900, 500)
    widget.show()
    # Stash on the QApplication so GC doesn't kill the window as soon
    # as this function returns.
    app = QApplication.instance()
    if app is not None:
        windows = getattr(app, '_osu_live_windows', None)
        if windows is None:
            windows = []
            app._osu_live_windows = windows
        windows.append(widget)


def register_library_actions(add):
    add('Live stats', _open_live_stats_window)
