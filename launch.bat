@echo off
cd /d "%~dp0app"
echo.
echo  Starting Meeting Recorder...

start "Listener Server" python server.py

timeout /t 2 /nobreak > nul

start "" "http://localhost:8765"

echo  Done - minimise this window, close "Listener Server" to stop.
echo.
pause
