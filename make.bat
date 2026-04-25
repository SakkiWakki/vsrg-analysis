@echo off
REM vsrg-analysis ; Windows build + launcher. Mirrors `make run` minus the
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
REM   make.bat doctor   - check for required toolchains, print install hints
REM   make.bat release  - build everything and zip to dist\ for publishing
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

set "DIST_DIR=dist"

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=run"

if /i "%TARGET%"=="run"     goto :target_run
if /i "%TARGET%"=="all"     goto :target_all
if /i "%TARGET%"=="venv"    goto :target_venv
if /i "%TARGET%"=="native"  goto :target_native
if /i "%TARGET%"=="overlay" goto :target_overlay
if /i "%TARGET%"=="gui"     goto :target_gui
if /i "%TARGET%"=="doctor"  goto :target_doctor
if /i "%TARGET%"=="release" goto :target_release
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

:target_doctor
call :do_doctor
exit /b %ERRORLEVEL%

:target_release
call :do_venv    || goto :fail
call :do_native  || goto :fail
call :do_overlay || goto :fail
call :do_release || goto :fail
exit /b 0

:target_clean
call :do_clean
exit /b 0

:do_venv
if exist "%VENV_PY%" exit /b 0
call :check_python || exit /b 1
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
call :check_rust || exit /b 1
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
    echo [native] post-build import failed ; venv mismatch?
    exit /b 1
)
echo.>"%NATIVE_STAMP%"
exit /b 0

REM Build the overlay DLL + injector as one CMake project. Both targets
REM are 32-bit to match osu!.exe ; a 64-bit DLL cannot be loaded into a
REM 32-bit process.
:do_overlay
call :check_cmake || exit /b 1
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
echo [cmake] platform mismatch in %EP_DIR% (have "!EP_HAVE!", want %EP_WANT%) ; wiping
rmdir /s /q "%EP_DIR%" || exit /b 1
exit /b 0

REM ── toolchain checks ────────────────────────────────────────────────

:check_python
where py >nul 2>&1  && exit /b 0
where python >nul 2>&1 && exit /b 0
echo.
echo [doctor] Python not found on PATH.
echo   install:  winget install Python.Python.3.12
echo   or:       https://www.python.org/downloads/
echo   after install, open a NEW terminal so PATH picks it up.
exit /b 1

:check_rust
where cargo >nul 2>&1 && exit /b 0
REM cargo not on PATH ; but rustup may have installed it at ~/.cargo/bin
REM and the user hasn't opened a fresh shell yet. Add it temporarily so
REM builds work without reopening. Done via goto (not an `if (...)` block)
REM because %PATH% on Windows usually contains "Program Files (x86)", whose
REM close-paren terminates a cmd.exe compound statement early.
if not exist "%USERPROFILE%\.cargo\bin\cargo.exe" goto :check_rust_missing
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
exit /b 0

:check_rust_missing
echo.
echo [doctor] Rust toolchain not found (cargo not on PATH).
echo   install:  winget install Rustlang.Rustup
echo   then:     rustup-init  (accept defaults; it will offer to install MSVC Build Tools)
echo   after install, open a NEW terminal so PATH picks up %%USERPROFILE%%\.cargo\bin.
exit /b 1

:check_cmake
where cmake >nul 2>&1 && exit /b 0
echo.
echo [doctor] CMake not found.
echo   install:  winget install Kitware.CMake
echo   after install, open a NEW terminal so PATH picks it up.
echo.
echo   CMake also needs MSVC to drive; if you hit "no CMAKE_C_COMPILER could be found"
echo   later, install the MSVC Build Tools:
echo     winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
exit /b 1

:do_doctor
echo [doctor] checking toolchains...
set "DOCTOR_FAIL=0"
call :check_python || set "DOCTOR_FAIL=1"
call :check_rust   || set "DOCTOR_FAIL=1"
call :check_cmake  || set "DOCTOR_FAIL=1"
if "!DOCTOR_FAIL!"=="0" (
    echo [doctor] all required toolchains present.
    exit /b 0
)
echo.
echo [doctor] one or more toolchains missing ; see hints above.
exit /b 1

REM ── release: zip built artifacts into dist\ for GitHub release ──────
REM Users download this, unzip into the repo root, and skip `make.bat all`.
:do_release
call :check_python || exit /b 1
REM Stamp with a short git rev so release zips are traceable back to a commit.
set "REV=unknown"
for /f %%H in ('git rev-parse --short HEAD 2^>nul') do set "REV=%%H"
for /f %%V in ('"%VENV_PY%" -c "import sys;print(f'{sys.version_info.major}{sys.version_info.minor}')"') do set "PYVER=%%V"

set "RELEASE_NAME=vsrg-analysis-windows-%REV%-py%PYVER%"
set "RELEASE_DIR=%DIST_DIR%\%RELEASE_NAME%"

if exist "%RELEASE_DIR%" rmdir /s /q "%RELEASE_DIR%"
mkdir "%RELEASE_DIR%" || exit /b 1
mkdir "%RELEASE_DIR%\overlay" || exit /b 1
mkdir "%RELEASE_DIR%\native"  || exit /b 1

echo [release] staging Python runtime (analysis\, plugins\)
REM robocopy exit codes: 0-7 = success (files copied / nothing to do),
REM 8+ = real failure. We need to swallow the "success" codes because
REM cmd treats anything != 0 as error.
robocopy analysis "%RELEASE_DIR%\analysis" /E /NFL /NDL /NJH /NJS /NP ^
    /XD __pycache__ .maturin-stamp target >nul
if errorlevel 8 exit /b 1
robocopy plugins "%RELEASE_DIR%\plugins" /E /NFL /NDL /NJH /NJS /NP ^
    /XD __pycache__ >nul
if errorlevel 8 exit /b 1

echo [release] staging launcher scripts + requirements
copy /y run.bat           "%RELEASE_DIR%\" >nul || exit /b 1
copy /y requirements.txt  "%RELEASE_DIR%\" >nul || exit /b 1
if exist analyze copy /y analyze "%RELEASE_DIR%\" >nul

echo [release] staging overlay DLL + injector
copy /y "%OVERLAY_DLL%"    "%RELEASE_DIR%\overlay\" >nul || exit /b 1
copy /y "%OVERLAY_INJECT%" "%RELEASE_DIR%\overlay\" >nul || exit /b 1

echo [release] maturin build --release (distributable wheel)
pushd "%NATIVE_DIR%" || exit /b 1
"..\..\..\..\%VENV_MATURIN%" build --release
if errorlevel 1 ( popd & exit /b 1 )
popd
copy /y "%NATIVE_DIR%\target\wheels\*.whl" "%RELEASE_DIR%\native\" >nul || exit /b 1

REM Quick-start guide inside the zip.
> "%RELEASE_DIR%\README.txt" echo vsrg-analysis %REV% (Windows, Python 3.%PYVER:~1%)
>>"%RELEASE_DIR%\README.txt" echo.
>>"%RELEASE_DIR%\README.txt" echo HOW TO RUN:
>>"%RELEASE_DIR%\README.txt" echo   Double-click run.bat, or from a terminal: run.bat
>>"%RELEASE_DIR%\README.txt" echo.
>>"%RELEASE_DIR%\README.txt" echo First run installs Python dependencies into a local .venv
>>"%RELEASE_DIR%\README.txt" echo folder next to this file. Subsequent runs jump straight to
>>"%RELEASE_DIR%\README.txt" echo the GUI. Only Python 3.10+ is required; Rust/CMake/MSVC are
>>"%RELEASE_DIR%\README.txt" echo NOT needed (the native pieces are prebuilt).
>>"%RELEASE_DIR%\README.txt" echo.
>>"%RELEASE_DIR%\README.txt" echo If Python is not installed, run.bat will tell you how to
>>"%RELEASE_DIR%\README.txt" echo install it.

echo [release] zipping to %DIST_DIR%\%RELEASE_NAME%.zip
powershell -NoProfile -Command ^
    "Compress-Archive -Force -Path '%RELEASE_DIR%\*' -DestinationPath '%DIST_DIR%\%RELEASE_NAME%.zip'" || exit /b 1

echo [release] done: %DIST_DIR%\%RELEASE_NAME%.zip
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
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
exit /b 0

:fail
echo.
echo [make] failed with errorlevel %ERRORLEVEL%
exit /b 1
