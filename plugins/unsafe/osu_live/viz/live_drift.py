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


def _vulkan_layer_installed() -> bool:
    """True iff our implicit layer manifest is discoverable by the
    Vulkan loader. We check the per-user XDG path that
    ``make vulkan-layer-install`` writes to; we do not probe system
    paths because we never install there.
    """
    import os
    from pathlib import Path
    xdg = os.environ.get('XDG_DATA_HOME') or str(Path.home() / '.local/share')
    manifest = Path(xdg) / 'vulkan/implicit_layer.d/VkLayer_vsrg_overlay.json'
    return manifest.is_file()


def _start_osu_with_overlay():
    """Start the shm feed and launch osu! with an in-game HUD.

    Two paths, picked at runtime:

    * **Vulkan layer (preferred).** If our implicit layer manifest is
      installed, launch raw ``osu-wine`` with ``VSRG_OVERLAY_LAYER=1``.
      The layer attaches to DXVK in-process — no compositor, no
      input interception, no mouse-accel surprise.
    * **Gamescope external overlay (fallback).** If the layer is
      missing (or the user has explicitly disabled it via
      ``VSRG_OVERLAY_LAYER_DISABLE=1``), fall through to the original
      gamescope-wrapped path.

    The shm publisher is the same in both cases — the layer and the
    external overlay binary read the same ``/dev/shm/vsrg_overlay``
    segment.
    """
    import os
    import shutil
    import subprocess
    from pathlib import Path

    from PySide6.QtWidgets import QApplication, QMessageBox
    from analysis.overlay.publisher import discover_overlays

    parent = QApplication.activeWindow()

    if shutil.which('osu-wine') is None:
        QMessageBox.warning(
            parent, 'Start osu (with overlay)',
            'osu-wine is not installed or not on PATH.\n\n'
            'Install osu-winello and ensure osu-wine is on PATH.')
        return

    use_layer = (
        _vulkan_layer_installed()
        and os.environ.get('VSRG_OVERLAY_LAYER_DISABLE') != '1'
    )

    # Same in both paths: the publisher fills /dev/shm/vsrg_overlay.
    try:
        from analysis.config import get_config
        cfg = get_config()
    except Exception:
        cfg = None
    overlays = discover_overlays(config=cfg)
    # Layer path doesn't have an authoritative canvas size until the
    # swapchain shows up; pass the user's display size as a hint so
    # widget anchoring is sensible from the first frame.
    width  = os.environ.get('GAMESCOPE_WIDTH',  '2560')
    height = os.environ.get('GAMESCOPE_HEIGHT', '1440')
    pub = overlays.start(width=int(width), height=int(height),
                         config_store=cfg)

    from analysis import diag as _diag
    _p = _diag.path()
    if _p is not None:
        print(f'[osu_live] diagnostic log: {_p}', flush=True)
        _diag.log('osu_live',
                  f'=== overlay session start (path={"layer" if use_layer else "gamescope"}) ===')
    app = QApplication.instance()
    if app is not None:
        app._osu_live_overlay_registry = overlays
        app._osu_live_shm_publisher = pub

    if use_layer:
        # Hand the env var to the loader; DXVK will load our layer
        # below it inside osu!'s own process. start_new_session so
        # closing the GUI doesn't take osu down with it.
        env = dict(os.environ, VSRG_OVERLAY_LAYER='1')
        subprocess.Popen(['osu-wine'], env=env, start_new_session=True)
        return

    # ── gamescope fallback ────────────────────────────────────────
    if shutil.which('gamescope') is None:
        QMessageBox.warning(
            parent, 'Start osu (with overlay)',
            'Neither the Vulkan overlay layer nor gamescope is '
            'available.\n\n'
            'Install the layer (`make vulkan-layer-install`) or '
            'gamescope (`pacman -S gamescope`).')
        return

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

    cmd = [
        'gamescope', '-f',
        '-w', width, '-h', height,
        '-W', width, '-H', height,
        '--', str(runner),
    ]
    subprocess.Popen(cmd, start_new_session=True)
