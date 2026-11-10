@echo off
rem ============================================================================
rem Build ct-fbp-cuda (Windows + MSVC + CUDA + Ninja)
rem Usage: scripts\build.bat [extra CMake args]
rem   e.g. scripts\build.bat -DCMAKE_CUDA_ARCHITECTURES=86
rem
rem NOTE: nvcc from CUDA 13.1 only supports MSVC 2019~2022 (19.2x~19.4x).
rem       This machine has VS2026 (19.51) as default toolset, so we pin the
rem       installed 14.44 (VS2022-era) toolset via -vcvars_ver.
rem ============================================================================
setlocal
if not defined VSCMD_VER (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul || goto :err
)
where ninja >nul 2>nul || set "PATH=%PATH%;C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

set "ROOT=%~dp0.."
cmake -S "%ROOT%" -B "%ROOT%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release %* || goto :err
cmake --build "%ROOT%\build" || goto :err
echo.
echo Artifacts: build\ct_fbp.exe  build\test_pipeline.exe
exit /b 0

:err
echo Build FAILED
exit /b 1
