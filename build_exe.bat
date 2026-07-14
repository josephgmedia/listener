@echo off
REM Builds Listener.exe from launcher.py. Requires: pip install pyinstaller
cd /d "%~dp0"
python -m PyInstaller --onefile --console --name Listener --icon app\icon.ico --distpath . launcher.py
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)
rmdir /s /q build 2>nul
del Listener.spec 2>nul
echo.
echo Built Listener.exe
pause
