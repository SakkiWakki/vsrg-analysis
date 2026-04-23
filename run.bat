@echo off
REM vsrg-analysis — end-user launcher for a prebuilt release zip.
REM Expects overlay\ and native\ next to this file. First run creates
REM a venv, installs dependencies + bundled native wheel, and stages
REM the overlay DLL/injector. Subsequent runs jump to the GUI.
REM
REM A prebuilt release only needs Python 3.10+. No Rust, CMake, or
REM MSVC required — all native pieces are pre-compiled.
REM
REM Source contributors should run make.bat instead.

setlocal enableextensions
cd /d "%~dp0"

set "VENV=.venv"
set "VENV_PY=%VENV%\Scripts\python.exe"
set "VENV_PIP=%VENV%\Scripts\pip.exe"

if exist "%VENV_PY%" goto :launch

echo [setup] first-run setup (this takes a minute)

REM Python check with install guidance.
set "HAS_PY=0"
where py >nul 2>&1     && set "HAS_PY=1"
where python >nul 2>&1 && set "HAS_PY=1"
if "%HAS_PY%"=="0" (
    echo.
    echo [setup] Python is not installed or not on PATH.
    echo.
    echo   Install option 1 ^(winget, recommended^):
    echo     winget install Python.Python.3.11
    echo.
    echo   Install option 2 ^(manual^):
    echo     https://www.python.org/downloads/
    echo     Tick "Add python.exe to PATH" during install.
    echo.
    echo After installing, close this window and run "run.bat" again.
    pause
    exit /b 1
)

where py >nul 2>&1
if %ERRORLEVEL%==0 (
    py -3 -m venv "%VENV%" || goto :fail_venv
) else (
    python -m venv "%VENV%" || goto :fail_venv
)

"%VENV_PY%" -m pip install --upgrade pip wheel >nul || goto :fail
echo [setup] installing requirements...
"%VENV_PIP%" install -r requirements.txt || goto :fail

REM Install the bundled native memory reader. If the wheel is missing
REM (not a release zip, or a stripped zip), fall back to source build
REM instructions rather than crashing silently.
if exist "native\*.whl" (
    echo [setup] installing bundled osu_memory_native wheel...
    for %%W in (native\*.whl) do "%VENV_PIP%" install "%%W" || goto :fail
) else (
    echo.
    echo [setup] native\*.whl not found.
    echo         This looks like a source checkout rather than a release
    echo         zip. For source builds, run "make.bat all" instead.
    echo         osu_live will fall back to the tosu HTTP bridge at runtime.
)

REM Stage the overlay DLL + injector under the path the plugin looks
REM for (matches what make.bat overlay produces, so runtime is identical).
REM Defining OVERLAY_DEST outside the `if (...)` block because cmd.exe
REM parses the whole block at once — a `set X=Y` inside only takes effect
REM after the block exits, so `%X%` references within would expand empty.
set "OVERLAY_DEST=build\win\analysis\games\osu\gl_layer\win\Release"
if exist "overlay\vsrg_gl_overlay.dll" (
    mkdir "%OVERLAY_DEST%" 2>nul
    copy /y "overlay\vsrg_gl_overlay.dll" "%OVERLAY_DEST%\" >nul || goto :fail
    copy /y "overlay\inject.exe"          "%OVERLAY_DEST%\" >nul || goto :fail
    echo [setup] overlay binaries staged.
)

echo [setup] done.

:launch
"%VENV_PY%" -m analysis.gui.app
exit /b %ERRORLEVEL%

:fail_venv
echo.
echo [setup] failed to create virtual environment.
echo.
echo Common causes:
echo   1. Your Python install is missing the "venv" module
echo      ^(standard; rare^).
echo   2. The current folder isn't writable. If you extracted this
echo      into C:\Program Files\ or a OneDrive-synced folder, move it
echo      somewhere like %%USERPROFILE%%\Downloads\vsrg-analysis and try again.
echo   3. Antivirus is blocking file creation. Whitelist this folder.
pause
exit /b 1

:fail
echo.
echo [setup] setup failed with errorlevel %ERRORLEVEL%.
pause
exit /b 1
