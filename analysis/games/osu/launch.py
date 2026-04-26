"""osu! launch helpers (Linux + Windows).

Builds the right argv + env to start osu! with our overlay attached.
The OS branches and overlay-injection mechanics live here so plugins
don't need ``subprocess`` / ``os`` access just to ask the host to
launch the game.

Three Linux paths, picked by environment + build presence:

* GL preload hook (default). osu!stable on Wine uses OpenGL/EGL; we
  ``LD_PRELOAD`` ``libvsrg_gl_overlay.so`` to hook swap calls.
* Vulkan layer (lazer/DXVK). Opt-in via ``VSRG_FORCE_VULKAN_LAYER=1``.
* Gamescope external overlay. Fallback when neither hook is available.

Windows uses a separate code path: spawn ``osu!.exe`` directly, then
inject ``vsrg_gl_overlay.dll`` via the ``inject.exe`` helper.

All entry points return a :class:`LaunchResult` describing what
happened. UI surfaces (warning dialogs, buttons) decide how to render
the result -- this module never imports Qt.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from analysis.core.game import LaunchResult


def launch_osu(*, with_overlay: bool = True) -> LaunchResult:
    """Start osu! with the overlay attached.

    On Linux, picks between GL preload, Vulkan layer, and gamescope
    fallback based on what's built and the environment. On Windows,
    launches ``osu!.exe`` and schedules DLL injection.

    ``with_overlay=False`` is reserved for a future "launch raw" mode
    that skips the overlay machinery entirely; not implemented yet.
    """
    if not with_overlay:
        return LaunchResult(ok=False, message='launch without overlay not supported yet')

    if sys.platform == 'win32':
        return _launch_windows()
    return _launch_linux()


# ── Linux ─────────────────────────────────────────────────────────


def _launch_linux() -> LaunchResult:
    if shutil.which('osu-wine') is None:
        return LaunchResult(
            ok=False,
            message=('osu-wine is not installed or not on PATH. '
                     'Install osu-winello and ensure osu-wine is on PATH.'),
        )

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

    overlay_size = _overlay_canvas_size_hint()
    pub_info = _start_overlay_publisher(*overlay_size)

    if use_gl_layer:
        env = dict(os.environ, VSRG_GL_OVERLAY='1')
        env['VSRG_OVERLAY_FONT'] = str(_repo_root() / 'analysis/overlay/assets/DejaVuSansMono.ttf')
        if os.environ.get('VSRG_INPUT_DEBUG'):
            env['VSRG_INPUT_DEBUG'] = os.environ['VSRG_INPUT_DEBUG']
        old_preload = env.get('LD_PRELOAD', '').strip()
        preload = _gl_layer_preload_path()
        env['LD_PRELOAD'] = f'{preload} {old_preload}' if old_preload else preload
        proc = subprocess.Popen(['osu-wine'], env=env, start_new_session=True)
        _diag('gl-layer')
        return LaunchResult(ok=True, pid=proc.pid, path_label='gl-layer', extra=pub_info)

    if use_vulkan_layer:
        env = dict(os.environ, VSRG_OVERLAY_LAYER='1')
        proc = subprocess.Popen(['osu-wine'], env=env, start_new_session=True)
        _diag('vulkan-layer')
        return LaunchResult(ok=True, pid=proc.pid, path_label='vulkan-layer', extra=pub_info)

    # ── gamescope fallback ─────────────────────────────────────────
    if shutil.which('gamescope') is None:
        return LaunchResult(
            ok=False,
            message=('Neither the GL overlay hook nor gamescope is available. '
                     'Build the hook (`make gl-layer`) or install gamescope '
                     '(`pacman -S gamescope`).'),
        )
    runner = _repo_root() / 'analysis/games/osu/gamescope_overlay/run-osu-gamescope-overlay.sh'
    overlay_bin = runner.parent / 'osu_overlay'
    if not runner.exists() or not overlay_bin.exists():
        return LaunchResult(
            ok=False,
            message=(f'Overlay not built yet.\n\nExpected:\n  {overlay_bin}\n\n'
                     f'Run `make overlay` from the repo root first.'),
        )
    width, height = overlay_size
    cmd = [
        'gamescope', '-f',
        '-w', str(width), '-h', str(height),
        '-W', str(width), '-H', str(height),
        '--', str(runner),
    ]
    proc = subprocess.Popen(cmd, start_new_session=True)
    _diag('gamescope')
    return LaunchResult(ok=True, pid=proc.pid, path_label='gamescope', extra=pub_info)


def _vulkan_layer_installed() -> bool:
    """True iff our implicit layer manifest is discoverable by the
    Vulkan loader. We check the per-user XDG path that
    ``make vulkan-layer-install`` writes to; we don't probe system
    paths because we never install there.
    """
    xdg = os.environ.get('XDG_DATA_HOME') or str(Path.home() / '.local/share')
    manifest = Path(xdg) / 'vulkan/implicit_layer.d/VkLayer_vsrg_overlay.json'
    return manifest.is_file()


def _gl_layer_paths():
    """Built LD_PRELOAD hooks for osu!stable's OpenGL/EGL/GLX path."""
    gl_dir = _repo_root() / 'analysis/games/osu/gl_layer/linux'
    return [
        gl_dir / 'lib/libvsrg_gl_overlay.so',
        gl_dir / 'lib64/libvsrg_gl_overlay.so',
        gl_dir / 'lib32/libvsrg_gl_overlay.so',
    ]


def _gl_layer_preload_path() -> str:
    """LD_PRELOAD path to the 64-bit renderer-enabled shim.

    osu!stable on modern Wine is 64-bit, so we pin to the explicit
    64-bit .so. The bitness-dispatching ``$LIB`` token doesn't survive
    osu-winello's pressure-vessel bind-mounts.
    """
    return str(_repo_root() / 'analysis/games/osu/gl_layer/linux/lib64/libvsrg_gl_overlay.so')


def _gl_layer_built() -> bool:
    return any(p.is_file() for p in _gl_layer_paths())


# ── Windows ────────────────────────────────────────────────────────


def _launch_windows() -> LaunchResult:
    from analysis.games.osu.replay import find_osu_dirs

    overlay_out = (_repo_root() / 'build' / 'win' / 'analysis' / 'games'
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
        return LaunchResult(
            ok=False,
            message='Cannot start overlay; missing:\n\n  - ' + '\n  - '.join(missing),
        )

    overlay_size = _overlay_canvas_size_hint()
    pub_info = _start_overlay_publisher(*overlay_size)

    env = dict(os.environ, VSRG_GL_OVERLAY='1')
    env['VSRG_OVERLAY_FONT'] = str(_repo_root() / 'analysis/overlay/assets/DejaVuSansMono.ttf')
    if os.environ.get('VSRG_INPUT_DEBUG'):
        env['VSRG_INPUT_DEBUG'] = os.environ['VSRG_INPUT_DEBUG']

    proc = subprocess.Popen(
        [str(osu_exe)],
        env=env,
        cwd=str(osu_exe.parent),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    _schedule_windows_injection(proc, injector, dll_path)
    _diag('win-gl-layer')
    return LaunchResult(
        ok=True,
        pid=proc.pid,
        path_label='win-gl-layer',
        extra={**pub_info, 'inject_pending': True},
    )


def _schedule_windows_injection(proc, injector: Path, dll_path: Path) -> None:
    """Inject the overlay DLL after a delay so opengl32.dll is loaded
    by the time DllMain runs. Uses a Qt timer so we don't block the
    calling thread; if Qt isn't available (headless tests), inject
    synchronously after a sleep."""
    try:
        from PySide6.QtCore import QTimer
    except ImportError:
        import time
        time.sleep(2.0)
        _do_inject(proc, injector, dll_path)
        return
    QTimer.singleShot(2000, lambda: _do_inject(proc, injector, dll_path))


def _do_inject(proc, injector: Path, dll_path: Path) -> None:
    if proc.poll() is not None:
        print(f'[osu_launch] osu! exited before injection (rc={proc.returncode})')
        return
    try:
        result = subprocess.run(
            [str(injector), str(proc.pid), str(dll_path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            print(f'[osu_launch] injection failed (rc={result.returncode}): '
                  f'{result.stderr or result.stdout}')
        else:
            print(f'[osu_launch] {result.stdout.strip()}')
    except subprocess.TimeoutExpired:
        print('[osu_launch] injector timed out')


# ── Shared host-side bring-up ─────────────────────────────────────


def _overlay_canvas_size_hint() -> tuple[int, int]:
    """Default canvas size for the overlay. The layer paths don't have
    an authoritative size until the swapchain shows up, so we pass the
    user's display size as a hint for first-frame anchoring."""
    width = int(os.environ.get('GAMESCOPE_WIDTH', '2560'))
    height = int(os.environ.get('GAMESCOPE_HEIGHT', '1440'))
    return width, height


def _start_overlay_publisher(width: int, height: int) -> dict:
    """Bring up the overlay publisher (the SHM feed both the layer and
    the gamescope external overlay read from). Stash the registry +
    publisher on the QApplication so they outlive this call without a
    module-level cache."""
    from analysis.overlay.publisher import discover_overlays
    try:
        from analysis.config import get_config
        cfg = get_config()
    except Exception:
        cfg = None
    overlays = discover_overlays(config=cfg)
    pub = overlays.start(width=width, height=height, config_store=cfg)
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app._osu_live_overlay_registry = overlays
            app._osu_live_shm_publisher = pub
    except ImportError:
        pass
    return {'overlay_registry': overlays, 'shm_publisher': pub}


def _diag(path_label: str) -> None:
    """Mirror the diagnostic-log entry the previous launcher wrote so
    log readers can still see overlay-session boundaries."""
    try:
        from analysis import diag as _diag_mod
    except ImportError:
        return
    p = _diag_mod.path()
    if p is not None:
        print(f'[osu_launch] diagnostic log: {p}', flush=True)
        _diag_mod.log('osu_launch',
                      f'=== overlay session start (path={path_label}) ===')


def _repo_root() -> Path:
    """``analysis/games/osu/launch.py`` -> repo root (3 parents up)."""
    return Path(__file__).resolve().parents[3]
