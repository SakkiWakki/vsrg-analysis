"""Wrap ``plugins/builtin/viz/drift.py`` for live osu data.

The pattern is the template for every other live viz: import the
static builder, hand it to :func:`analysis.viz.live_figure.build_live_figure`
so the figure redraws from each new memory snapshot. No changes to the
underlying viz.
"""
from __future__ import annotations


def build(replay=None, game='osu', **_):
    from analysis.viz.live_figure import build_live_figure
    from plugins.builtin.viz.drift import build as build_drift
    return build_live_figure(build_drift, game='osu')


def register(add):
    add('Live: drift (hands × time)', build, category='chart')


def _open_live_stats_window():
    """Open a tabbed top-level window with every live viz in this
    bundle. Each tab is the same live widget the viz picker
    would build ; we just cohabit them so the user sees the whole live
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
            'Make sure osu! is running and the native reader is built '
            '(run `make native` from the repo root).')
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
    add('Live stats', _open_live_stats_window)
    add('Start osu (with overlay)', _start_osu_with_overlay)


def _start_osu_with_overlay():
    """Ask the host to launch osu! with the overlay attached.

    All the OS branching (Linux GL preload / Vulkan layer / gamescope
    fallback / Windows DLL injection) lives in
    ``analysis/games/osu/launch.py`` and is reached via the host's
    game-adapter API. We just translate the structured
    :class:`LaunchResult` into a user-facing dialog when something
    goes wrong.
    """
    from PySide6.QtWidgets import QApplication, QMessageBox
    from analysis.plugins.host_api import game_proxy

    game = game_proxy('osu')
    if game is None:
        QMessageBox.warning(
            QApplication.activeWindow(), 'Start osu (with overlay)',
            'No osu! game adapter registered.')
        return

    result = game.launch(with_overlay=True)
    if not result.ok:
        QMessageBox.warning(
            QApplication.activeWindow(), 'Start osu (with overlay)',
            result.message or 'Launch failed.')
