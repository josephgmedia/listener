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
REM --onedir, NOT --onefile: onefile unpacks the whole ~23MB bundle into a temp
REM folder on EVERY launch, which added seconds to startup. onedir just runs the
REM exe in place. The launcher walks up to find the project root, so it works
REM from the Listener\ subfolder (see _find_root in launcher.py).
python -m PyInstaller --onedir --windowed --name Listener --icon app\icon.ico ^
    --distpath . --noconfirm launcher.py
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)
rmdir /s /q build 2>nul
del Listener.spec 2>nul
echo.
echo Built Listener\Listener.exe
echo Use the "Listener" shortcut in this folder to launch it.
pause
