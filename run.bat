@echo off
rem Launch DouZero_For_HLDDZ_FullAuto with Administrator privileges.
rem The QQ game (HLDDZ3D.exe) usually runs elevated. A non-elevated process is blocked
rem by UIPI from touching its window (SetActiveWindow / MoveWindow / PrintWindow denied).
rem
rem Output: the console window shows the log, and it is also appended to logs\console.log
rem Usage: double-click this file and accept UAC, or run `run.bat --check` to verify paths.
setlocal
cd /d "%~dp0"
title DouZero HLDDZ

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

if /i "%~1"=="--check" (
    echo python  : %PYTHON%
    echo workdir : %CD%
    echo logfile : %~dp0logs\console.log
    exit /b 0
)

net session >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [Administrator rights required]
    echo   Requesting elevation, please click YES in the UAC dialog.
    echo   A NEW console window will open; this one closes after 3 seconds.
    echo.
    timeout /t 3 >nul
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -WorkingDirectory '%~dp0' -Verb RunAs"
    exit /b
)

echo ============================================================
echo   DouZero_For_HLDDZ_FullAuto
echo   console log: %~dp0logs\console.log
echo ============================================================
echo.

"%PYTHON%" -u -m hlddz

echo.
echo [process exited, code %errorlevel%]
echo Log file: %~dp0logs\console.log
pause
endlocal
