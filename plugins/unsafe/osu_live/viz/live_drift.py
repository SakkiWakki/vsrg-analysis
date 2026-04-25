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


def _gl_layer_paths():
    """Built LD_PRELOAD hooks for osu!stable's OpenGL/EGL/GLX path."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[4]
    gl_dir = repo_root / 'analysis/games/osu/gl_layer/linux'
    return [
        gl_dir / 'lib/libvsrg_gl_overlay.so',
        gl_dir / 'lib64/libvsrg_gl_overlay.so',
        gl_dir / 'lib32/libvsrg_gl_overlay.so',
    ]


def _gl_layer_preload_path() -> str:
    """LD_PRELOAD path to the 64-bit renderer-enabled shim.

    We used to use glibc's bitness-dispatching ``$LIB`` token (which
    expands to ``lib``/``lib32``/``lib64`` based on the runtime
    process), but under osu-winello's yawl runtime (pressure-vessel
    sandbox) the token expands inside the container's re-mapped
    filesystem where our ``lib/`` subdirectory does not survive the
    bind-mount. Pressure-vessel logs the failure as::

        ERROR: ld.so: object '/tmp/pressure-vessel-libs-XXXXX/${LIB}/
            libvsrg_gl_overlay.so' from LD_PRELOAD cannot be preloaded

    osu!stable on modern Wine is 64-bit, so we pin to the explicit
    64-bit .so and skip the dispatch entirely.
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[4]
    return str(repo_root / 'analysis/games/osu/gl_layer/linux/lib64/libvsrg_gl_overlay.so')


def _gl_layer_built() -> bool:
    return any(p.is_file() for p in _gl_layer_paths())


def _start_osu_with_overlay():
    """Start the shm feed and launch osu! with an in-game HUD.

    Paths, picked at runtime:

    * **GL preload hook (stable default).** osu!stable under Wine uses
      OpenGL/EGL in current osu-winello logs, so launch raw
      ``osu-wine`` with our ``LD_PRELOAD`` hook when built.
    * **Vulkan layer (lazer/DXVK).** Kept available for lazer-style
      Vulkan paths, but only selected here with
      ``VSRG_FORCE_VULKAN_LAYER=1`` so stable does not silently take a
      layer that cannot see its presents.
    * **Gamescope external overlay (fallback).** If the layer is
      missing (or the user has explicitly disabled it via
      ``VSRG_GL_OVERLAY_DISABLE=1``), fall through to the original
      gamescope-wrapped path.

    The shm publisher is the same in both cases ; the layer and the
    external overlay binary read the same ``/dev/shm/vsrg_overlay``
    segment.
    """
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from PySide6.QtWidgets import QApplication, QMessageBox
    from analysis.overlay.publisher import discover_overlays

    parent = QApplication.activeWindow()

    if sys.platform == 'win32':
        _start_osu_with_overlay_windows(parent)
        return

    if shutil.which('osu-wine') is None:
        QMessageBox.warning(
            parent, 'Start osu (with overlay)',
            'osu-wine is not installed or not on PATH.\n\n'
            'Install osu-winello and ensure osu-wine is on PATH.')
        return

    use_gl_layer = (
        _gl_layer_built()
        and os.environ.get('VSRG_GL_OVERLAY_DISABLE') != '1'
        and os.environ.get('VSRG_FORCE_VULKAN_LAYER') != '1'
    )
    use_vulkan_layer = (
        _vulkan_layer_installed()
        and os.environ.get('VSRG_OVERLAY_LAYER_DISABLE') != '1'
        and os.environ.get('VSRG_FORCE_VULKAN_LAYER') == '1'
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
        path_label = (
            'gl-layer' if use_gl_layer else
            'vulkan-layer' if use_vulkan_layer else
            'gamescope'
        )
        _diag.log('osu_live',
                  f'=== overlay session start (path={path_label}) ===')
    app = QApplication.instance()
    if app is not None:
        app._osu_live_overlay_registry = overlays
        app._osu_live_shm_publisher = pub

    if use_gl_layer:
        # Inject into Wine's Unix process and catch EGL/GLX swap paths.
        # Keep any existing preload entries after ours so we get first
        # chance at the public swap/proc-address symbols.
        env = dict(os.environ, VSRG_GL_OVERLAY='1')
        # Point the renderer at a bundled font. The host's
        # /usr/share/fonts tree isn't bind-mounted into the
        # pressure-vessel sandbox yawl-winello uses, but $HOME is ;
        # so a path under the repo root resolves inside the game's
        # process too.
        repo_root = Path(__file__).resolve().parents[4]
        env['VSRG_OVERLAY_FONT'] = str(
            repo_root / 'analysis/overlay/assets/DejaVuSansMono.ttf')
        # Forward a debug flag from our own environment so we can
        # flip it on with ``VSRG_INPUT_DEBUG=1 <launch command>``
        # without touching the launcher. Harmless when unset.
        if os.environ.get('VSRG_INPUT_DEBUG'):
            env['VSRG_INPUT_DEBUG'] = os.environ['VSRG_INPUT_DEBUG']
        old_preload = env.get('LD_PRELOAD', '').strip()
        preload = _gl_layer_preload_path()
        env['LD_PRELOAD'] = (
            f'{preload} {old_preload}' if old_preload else preload
        )
        subprocess.Popen(['osu-wine'], env=env, start_new_session=True)
        return

    if use_vulkan_layer:
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
            'Neither the GL overlay hook nor gamescope is '
            'available.\n\n'
            'Build the hook (`make gl-layer`) or install '
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


def _start_osu_with_overlay_windows(parent):
    """Windows path: launch osu!.exe with ``VSRG_GL_OVERLAY=1`` set, then
    inject ``vsrg_gl_overlay.dll`` via our ``inject.exe`` helper.

    There's no ``osu-wine`` wrapper on Windows and no LD_PRELOAD, so the
    whole Linux launch machinery above is unusable here. Instead:
    MinHook patches ``wglSwapBuffers`` from inside osu!.exe once the DLL
    is loaded ; see ``analysis/games/osu/gl_layer/win/win_gl_layer.cpp``.
    """
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox
    from analysis.games.osu.replay import find_osu_dirs
    from analysis.overlay.publisher import discover_overlays

    repo_root = Path(__file__).resolve().parents[4]
    # One top-level CMake project builds both into build/win/. Path
    # matches what make.bat `overlay` target produces.
    overlay_out = (repo_root / 'build' / 'win' / 'analysis' / 'games'
                   / 'osu' / 'gl_layer' / 'win' / 'Release')
    dll_path = overlay_out / 'vsrg_gl_overlay.dll'
    injector = overlay_out / 'inject.exe'

    dirs = find_osu_dirs()
    root = dirs.get('root')
    osu_exe = Path(root) / 'osu!.exe' if root else None

    missing = []
    if not osu_exe or not osu_exe.is_file():
        missing.append(f'osu!.exe (looked at: {osu_exe})')
    if not dll_path.is_file():
        missing.append(f'overlay DLL: {dll_path}\n  Build with: make.bat overlay')
    if not injector.is_file():
        missing.append(f'injector: {injector}\n  Build with: make.bat overlay')
    if missing:
        QMessageBox.warning(
            parent, 'Start osu (with overlay)',
            'Cannot start overlay ; missing:\n\n  - '
            + '\n  - '.join(missing))
        return

    try:
        from analysis.config import get_config
        cfg = get_config()
    except Exception:
        cfg = None
    overlays = discover_overlays(config=cfg)
    width = int(os.environ.get('GAMESCOPE_WIDTH',  '2560'))
    height = int(os.environ.get('GAMESCOPE_HEIGHT', '1440'))
    pub = overlays.start(width=width, height=height, config_store=cfg)

    from analysis import diag as _diag
    if _diag.path() is not None:
        print(f'[osu_live] diagnostic log: {_diag.path()}', flush=True)
        _diag.log('osu_live',
                  '=== overlay session start (path=win-gl-layer) ===')
    app = QApplication.instance()
    if app is not None:
        app._osu_live_overlay_registry = overlays
        app._osu_live_shm_publisher = pub

    env = dict(os.environ, VSRG_GL_OVERLAY='1')
    env['VSRG_OVERLAY_FONT'] = str(
        repo_root / 'analysis/overlay/assets/DejaVuSansMono.ttf')
    if os.environ.get('VSRG_INPUT_DEBUG'):
        env['VSRG_INPUT_DEBUG'] = os.environ['VSRG_INPUT_DEBUG']

    # CREATE_NEW_PROCESS_GROUP so closing the GUI doesn't take osu! with
    # it (the Windows equivalent of start_new_session=True on Linux).
    proc = subprocess.Popen(
        [str(osu_exe)],
        env=env,
        cwd=str(osu_exe.parent),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    # Inject after a delay: DllMain reads VSRG_GL_OVERLAY at attach time
    # and hooks wglSwapBuffers, which means opengl32.dll must already be
    # loaded by the time we inject. Osu!.exe loads GL during startup, so
    # 2s is a comfortable margin. If the hook ever misses its first swap
    # on slow machines, bump this or retry on failure.
    def _do_inject():
        if proc.poll() is not None:
            print(f'[osu_live] osu! exited before injection (rc={proc.returncode})')
            return
        try:
            result = subprocess.run(
                [str(injector), str(proc.pid), str(dll_path)],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                QMessageBox.warning(
                    parent, 'Start osu (with overlay)',
                    f'Injection failed (rc={result.returncode}):\n\n'
                    f'{result.stderr or result.stdout}')
            else:
                print(f'[osu_live] {result.stdout.strip()}')
        except subprocess.TimeoutExpired:
            QMessageBox.warning(
                parent, 'Start osu (with overlay)',
                'Injector timed out. Is osu! hung, or did Windows block '
                'the remote thread?')

    QTimer.singleShot(2000, _do_inject)
