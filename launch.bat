@echo off
setlocal
cd /d "%~dp0app"
echo.
echo  Starting Meeting Recorder...

REM Prefer the venv created by Listener.exe, fall back to system Python.
set "PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
start "Listener Server" cmd /k ""%PY%" listener_server.py"

REM Wait for the server to actually come up before opening the browser.
REM Whisper/Torch imports can take 5-15s, so a fixed timeout caused 404s.
set /a tries=0
:wait_loop
curl -s -o nul -m 1 http://localhost:8765/status >nul 2>nul
if not errorlevel 1 goto server_up
set /a tries+=1
if %tries% geq 60 goto server_timeout
<nul set /p="."
timeout /t 1 /nobreak >nul
goto wait_loop

:server_up
echo.
echo  Server ready - opening browser.
start "" "http://localhost:8765/desktop"
goto done

:server_timeout
echo.
echo  [WARN] Server did not respond after 60s. Check the "Listener Server"
echo         window for errors, then open http://localhost:8765/desktop manually.

:done
echo.
echo  Minimise this window. Close "Listener Server" to stop the recorder.
echo.
pause
endlocal
