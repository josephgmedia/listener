@echo off
cd /d "%~dp0app"
echo.
echo  Starting Meeting Recorder...

start "Listener Server" cmd /k ""C:\Users\artist\AppData\Local\Programs\Python\Python313\python.exe" listener_server.py"

timeout /t 3 /nobreak > nul

start "" "http://localhost:8765"

echo  Done - minimise this window, close "Listener Server" to stop.
echo.
pause
