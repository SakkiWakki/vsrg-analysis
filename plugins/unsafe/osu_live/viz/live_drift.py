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
    """Start the shm feed and launch osu! inside a gamescope session
    with our external overlay.

    One-click path: starts the /dev/shm publisher (so the overlay
    has data the moment it attaches), then spawns
    ``gamescope ... -- run-osu-gamescope-overlay.sh`` detached so
    the GUI stays responsive. The runner script handles the
    osu-first-then-overlay ordering that gamescope's surface
    promotion requires.
    """
    import os
    import shutil
    import subprocess
    from pathlib import Path

    from PySide6.QtWidgets import QApplication, QMessageBox
    from plugins.unsafe.osu_live.shm_publisher import get_publisher

    parent = QApplication.activeWindow()

    if shutil.which('gamescope') is None:
        QMessageBox.warning(
            parent, 'Start osu (with overlay)',
            'gamescope is not installed or not on PATH.\n\n'
            'Install it (pacman -S gamescope) and try again.')
        return
    if shutil.which('osu-wine') is None:
        QMessageBox.warning(
            parent, 'Start osu (with overlay)',
            'osu-wine is not installed or not on PATH.\n\n'
            'Install osu-winello and ensure osu-wine is on PATH.')
        return

    # plugins/unsafe/osu_live/viz/live_drift.py → repo root is 4 up.
    repo_root = Path(__file__).resolve().parents[4]
    runner = repo_root / 'analysis/games/osu/gamescope_overlay' \
                       / 'run-osu-gamescope-overlay.sh'
    overlay_bin = runner.parent / 'osu_overlay'
    if not runner.exists() or not overlay_bin.exists():
        QMessageBox.warning(
            parent, 'Start osu (with overlay)',
            f'Overlay not built yet.\n\nExpected:\n  {overlay_bin}\n\n'
            f'Run `make overlay` from the repo root first.')
        return

    pub = get_publisher()
    app = QApplication.instance()
    if app is not None:
        app._osu_live_shm_publisher = pub

    width  = os.environ.get('GAMESCOPE_WIDTH',  '2560')
    height = os.environ.get('GAMESCOPE_HEIGHT', '1440')
    cmd = [
        'gamescope', '-f',
        '-w', width, '-h', height,
        '-W', width, '-H', height,
        '--', str(runner),
    ]
    # start_new_session so closing the GUI doesn't SIGHUP gamescope.
    subprocess.Popen(cmd, start_new_session=True)
