@echo off
rem Launch DouZero_For_HLDDZ_FullAuto with Administrator privileges.
rem The QQ game (HLDDZ3D.exe) usually runs elevated. A non-elevated process is blocked
rem by UIPI from touching its window (SetActiveWindow / MoveWindow / PrintWindow all denied).
rem Usage: double-click this file and accept UAC, or run `run.bat --check` to verify paths.
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

if /i "%~1"=="--check" (
    echo python  : %PYTHON%
    echo workdir : %CD%
    exit /b 0
)

net session >nul 2>&1
if errorlevel 1 (
    echo Requesting administrator privileges...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -WorkingDirectory '%~dp0' -Verb RunAs"
    exit /b
)

echo Starting DouZero_For_HLDDZ_FullAuto as Administrator...
"%PYTHON%" mian.py
if errorlevel 1 pause
endlocal
