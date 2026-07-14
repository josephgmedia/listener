@echo off
REM Starts the calendar watcher silently using whatever Python is on PATH.
REM Prefers the Listener venv if it exists.
set "VENV_PYW=%~dp0..\.venv\Scripts\pythonw.exe"
if exist "%VENV_PYW%" (
    start "" "%VENV_PYW%" "%~dp0calendar_watcher.py"
) else (
    start "" pythonw "%~dp0calendar_watcher.py"
)
