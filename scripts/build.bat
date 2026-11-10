@echo off
rem ============================================================================
rem Build ct-fbp-cuda (Windows + MSVC + CUDA + Ninja)
rem Usage: scripts\build.bat [extra CMake args]
rem   e.g. scripts\build.bat -DCMAKE_CUDA_ARCHITECTURES=86
rem
rem NOTE: nvcc from CUDA <=13.1 only supports MSVC 2019~2022 (19.2x~19.4x).
rem       This machine has VS2026 (19.51) as default toolset, so we pin the
rem       installed 14.44 (VS2022-era) toolset via -vcvars_ver. CUDA 13.5 也接受
rem       14.44，因此同一套参数对 13.1 / 13.5 都可用。
rem ============================================================================
setlocal
if not defined VSCMD_VER (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul || goto :err
)
where ninja >nul 2>nul || set "PATH=%PATH%;C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

set "ROOT=%~dp0.."
rem build\ 可能已被 VS 生成器（CMake Tools / cmake --build build）占用；
rem 那种情况下改用 build-ninja\，避免 "generator does not match"。
set "BUILD_DIR=%ROOT%\build"
if exist "%BUILD_DIR%\CMakeCache.txt" (
    findstr /c:"CMAKE_GENERATOR:INTERNAL=Ninja" "%BUILD_DIR%\CMakeCache.txt" >nul 2>nul || set "BUILD_DIR=%ROOT%\build-ninja"
)
cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release %* || goto :err
cmake --build "%BUILD_DIR%" || goto :err
echo.
echo Build dir: %BUILD_DIR%
echo Artifacts: %BUILD_DIR%\ct_fbp.exe  %BUILD_DIR%\ct_tour.exe  %BUILD_DIR%\test_pipeline.exe
exit /b 0

:err
echo Build FAILED
exit /b 1
