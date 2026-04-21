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
    """Open a tabbed top-level window with every live viz in this
    bundle. Each tab is the same ``LiveFigureWidget`` the viz picker
    would build — we just cohabit them so the user sees the whole live
    dashboard with one click instead of five separate windows."""
    from PySide6.QtWidgets import (QApplication, QMessageBox, QTabWidget,
                                   QVBoxLayout, QWidget)
    # Import each live viz module here (not at module scope) so a
    # broken sibling doesn't stop the button from opening the others.
    from plugins.unsafe.osu_live.viz import (live_drift,
                                             live_offset_distribution,
                                             live_rolling_stability,
                                             live_scatter_timeline,
                                             live_per_column)
    tabs = [
        ('Drift', live_drift.build),
        ('Offset distribution', live_offset_distribution.build),
        ('Rolling stability', live_rolling_stability.build),
        ('Scatter timeline', live_scatter_timeline.build),
        ('Per-column (synthetic)', live_per_column.build),
    ]

    window = QWidget()
    window.setWindowTitle('osu! live stats')
    window.resize(1000, 600)
    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    tab_widget = QTabWidget(window)
    layout.addWidget(tab_widget)

    for label, build_fn in tabs:
        try:
            w = build_fn()
        except Exception as exc:
            # Skip a broken tab rather than failing the whole dashboard.
            print(f'live viz {label!r} failed: {exc}')
            continue
        tab_widget.addTab(w, label)

    if tab_widget.count() == 0:
        QMessageBox.warning(
            None, 'Live stats',
            'No live visualizations could be built.\n\n'
            'Make sure either the native osu! reader is built or '
            'tosu is running on http://127.0.0.1:24050.')
        return

    window.show()
    # Stash on the QApplication so GC doesn't kill the window as soon
    # as this function returns.
    app = QApplication.instance()
    if app is not None:
        windows = getattr(app, '_osu_live_windows', None)
        if windows is None:
            windows = []
            app._osu_live_windows = windows
        windows.append(window)


def register_library_actions(add):
    from plugins.unsafe.osu_live.overlay import open_overlay
    add('Live stats', _open_live_stats_window)
    add('Live overlay', open_overlay)
    add('Start in-game overlay feed', _start_ingame_overlay_feed)


def _start_ingame_overlay_feed():
    """Start the /dev/shm publisher the gamescope overlay consumes.

    The overlay binary itself is launched from a gamescope session
    (see analysis/games/osu/gamescope_overlay/run-osu-gamescope-overlay.sh),
    but the *feed* into its shared-memory region is produced here by
    the Python live poller. Starting the publisher is a no-op after
    the first call thanks to the module singleton.
    """
    from PySide6.QtWidgets import QApplication, QMessageBox
    from plugins.unsafe.osu_live.shm_publisher import get_publisher
    pub = get_publisher()
    QMessageBox.information(
        QApplication.activeWindow(),
        'In-game overlay feed',
        'Publishing live stats to /dev/shm/osu_live_overlay.\n\n'
        'Launch osu! inside gamescope with:\n'
        '  gamescope -f -w 2560 -h 1440 -W 2560 -H 1440 -- \\\n'
        '      analysis/games/osu/gamescope_overlay/'
        'run-osu-gamescope-overlay.sh')
    # Keep a hard ref on the app so the publisher survives a
    # re-click (get_publisher handles the singleton either way,
    # this just makes the intent explicit).
    app = QApplication.instance()
    if app is not None:
        app._osu_live_shm_publisher = pub
