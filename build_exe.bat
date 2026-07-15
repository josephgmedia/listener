@echo off
REM Builds Listener.exe from launcher.py.
REM --windowed so there is NO console window on normal runs; the launcher shows a
REM console only during first-time setup and otherwise sits in the system tray.
cd /d "%~dp0"

echo Installing build dependencies (pyinstaller, pystray, pillow)...
python -m pip install --upgrade pyinstaller pystray pillow
if errorlevel 1 (
    echo Could not install build dependencies.
    pause
    exit /b 1
)

echo Building Listener.exe...
python -m PyInstaller --onefile --windowed --name Listener --icon app\icon.ico ^
    --distpath . launcher.py
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)
rmdir /s /q build 2>nul
del Listener.spec 2>nul
echo.
echo Built Listener.exe  ^(double-click to run — no cmd windows^)
pause
