@echo off
rem Build sqlite-vec as a native ARM64 Windows extension: third_party\sqlite-vec\vec0.dll
rem Sets compiler paths directly because vcvars did not set up the environment on this machine.
rem Set NEON=1 to enable NEON-accelerated distance functions.
setlocal
set MSVC=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207
set KITS=%ProgramFiles(x86)%\Windows Kits\10
set SDK=10.0.26100.0
set "PATH=%MSVC%\bin\Hostarm64\arm64;%KITS%\bin\%SDK%\arm64;%PATH%"
set "INCLUDE=%MSVC%\include;%KITS%\Include\%SDK%\ucrt;%KITS%\Include\%SDK%\um;%KITS%\Include\%SDK%\shared"
set "LIB=%MSVC%\lib\arm64;%KITS%\Lib\%SDK%\ucrt\arm64;%KITS%\Lib\%SDK%\um\arm64"
cd /d "%~dp0..\third_party\sqlite-vec" || exit /b 1
set FLAGS=
if "%NEON%"=="1" set FLAGS=/DSQLITE_VEC_ENABLE_NEON
cl /nologo /O2 /LD %FLAGS% /Isqlite\sqlite-amalgamation-3490100 /Isrc src\sqlite-vec.c /Fevec0.dll /Fo:sqlite-vec.obj
exit /b %ERRORLEVEL%
