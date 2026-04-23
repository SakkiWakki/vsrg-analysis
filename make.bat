@echo off
REM vsrg-analysis — Windows build + launcher. Mirrors `make run` minus the
REM Linux-only overlay / gl-layer / vulkan-layer targets. Windows has its
REM own in-process overlay path: vsrg_gl_overlay.dll (hooks wglSwapBuffers)
REM + inject.exe (loader), both built from the top-level CMakeLists.txt.
REM
REM Usage:
REM   make.bat          - default: venv + native + gui (launches)
REM   make.bat all      - venv + native + overlay (dll + injector)
REM   make.bat venv     - create .venv and install requirements
REM   make.bat native   - build the Rust PyO3 extension
REM   make.bat overlay  - build vsrg_gl_overlay.dll + inject.exe (x86)
REM   make.bat gui      - launch the GUI (no build)
REM   make.bat clean    - remove build artifacts (keeps venv)

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

set "VENV=.venv"
set "VENV_PY=%VENV%\Scripts\python.exe"
set "VENV_PIP=%VENV%\Scripts\pip.exe"
set "VENV_MATURIN=%VENV%\Scripts\maturin.exe"
set "NATIVE_DIR=analysis\games\osu\native"
set "NATIVE_STAMP=%NATIVE_DIR%\.maturin-stamp"

set "OVERLAY_BUILD=build\win"
set "OVERLAY_DLL=%OVERLAY_BUILD%\analysis\games\osu\gl_layer\win\Release\vsrg_gl_overlay.dll"
set "OVERLAY_INJECT=%OVERLAY_BUILD%\analysis\games\osu\gl_layer\win\Release\inject.exe"

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=run"

if /i "%TARGET%"=="run"     goto :target_run
if /i "%TARGET%"=="all"     goto :target_all
if /i "%TARGET%"=="venv"    goto :target_venv
if /i "%TARGET%"=="native"  goto :target_native
if /i "%TARGET%"=="overlay" goto :target_overlay
if /i "%TARGET%"=="gui"     goto :target_gui
if /i "%TARGET%"=="clean"   goto :target_clean
echo unknown target: %TARGET%
echo run "make.bat" with no args, or see the header of this file.
exit /b 2

:target_run
call :do_venv   || goto :fail
call :do_native || goto :fail
goto :do_gui

:target_all
call :do_venv    || goto :fail
call :do_native  || goto :fail
call :do_overlay || goto :fail
exit /b 0

:target_venv
call :do_venv || goto :fail
exit /b 0

:target_native
call :do_venv   || goto :fail
call :do_native || goto :fail
exit /b 0

:target_overlay
call :do_overlay || goto :fail
exit /b 0

:target_gui
call :do_gui
exit /b %ERRORLEVEL%

:target_clean
call :do_clean
exit /b 0

:do_venv
if exist "%VENV_PY%" exit /b 0
echo [venv] creating %VENV%
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    py -3 -m venv "%VENV%" || exit /b 1
) else (
    python -m venv "%VENV%" || exit /b 1
)
"%VENV_PY%" -m pip install --upgrade pip wheel >nul || exit /b 1
echo [venv] pip install -r requirements.txt
"%VENV_PIP%" install -r requirements.txt || exit /b 1
"%VENV_PIP%" install maturin pytest || exit /b 1
exit /b 0

REM Rebuild the PyO3 extension if stamp is missing or older than any
REM Rust source / Cargo.toml. PowerShell compares mtimes because cmd
REM has no built-in.
:do_native
set "NEED_BUILD=0"
if not exist "%NATIVE_STAMP%" set "NEED_BUILD=1"
if "!NEED_BUILD!"=="0" (
    for /f %%R in ('powershell -NoProfile -Command ^
        "$s=(Get-Item '%NATIVE_STAMP%').LastWriteTime;" ^
        "$n=Get-ChildItem -Recurse -File '%NATIVE_DIR%\src','%NATIVE_DIR%\Cargo.toml' ^| Where-Object {$_.LastWriteTime -gt $s} ^| Select-Object -First 1;" ^
        "if ($n) { 'yes' } else { 'no' }"') do set "NEWER=%%R"
    if /i "!NEWER!"=="yes" set "NEED_BUILD=1"
)
if "!NEED_BUILD!"=="0" exit /b 0

echo [native] maturin develop --release
pushd "%NATIVE_DIR%" || exit /b 1
"..\..\..\..\%VENV_MATURIN%" develop --release
if errorlevel 1 ( popd & exit /b 1 )
popd
"%VENV_PY%" -c "import osu_memory_native" || (
    echo [native] post-build import failed — venv mismatch?
    exit /b 1
)
echo.>"%NATIVE_STAMP%"
exit /b 0

REM Build the overlay DLL + injector as one CMake project. Both targets
REM are 32-bit to match osu!.exe — a 64-bit DLL cannot be loaded into a
REM 32-bit process.
:do_overlay
call :ensure_platform "%OVERLAY_BUILD%" Win32 || exit /b 1
if not exist "%OVERLAY_BUILD%\CMakeCache.txt" (
    echo [overlay] cmake configure (Win32)
    cmake -S . -B "%OVERLAY_BUILD%" -A Win32 || exit /b 1
)
echo [overlay] cmake --build --config Release
cmake --build "%OVERLAY_BUILD%" --config Release || exit /b 1
if not exist "%OVERLAY_DLL%"    ( echo [overlay] missing: %OVERLAY_DLL%    & exit /b 1 )
if not exist "%OVERLAY_INJECT%" ( echo [overlay] missing: %OVERLAY_INJECT% & exit /b 1 )
exit /b 0

REM CMake refuses to reconfigure an existing build dir with a different
REM -A. Detect the stored platform and wipe the dir on mismatch.
:ensure_platform
set "EP_DIR=%~1"
set "EP_WANT=%~2"
set "EP_CACHE=%EP_DIR%\CMakeCache.txt"
if not exist "%EP_CACHE%" exit /b 0
set "EP_HAVE="
for /f "tokens=2 delims==" %%V in ('findstr /b /c:"CMAKE_GENERATOR_PLATFORM:" "%EP_CACHE%" 2^>nul') do set "EP_HAVE=%%V"
if /i "!EP_HAVE!"=="%EP_WANT%" exit /b 0
echo [cmake] platform mismatch in %EP_DIR% (have "!EP_HAVE!", want %EP_WANT%) — wiping
rmdir /s /q "%EP_DIR%" || exit /b 1
exit /b 0

:do_gui
echo [gui] launching
"%VENV_PY%" -m analysis.gui.app
exit /b %ERRORLEVEL%

:do_clean
echo [clean] removing build artifacts (venv preserved)
if exist "%NATIVE_STAMP%" del /f /q "%NATIVE_STAMP%"
if exist "%NATIVE_DIR%\target" rmdir /s /q "%NATIVE_DIR%\target"
if exist "%OVERLAY_BUILD%" rmdir /s /q "%OVERLAY_BUILD%"
exit /b 0

:fail
echo.
echo [make] failed with errorlevel %ERRORLEVEL%
exit /b 1
