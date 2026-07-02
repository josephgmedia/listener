@echo off
setlocal

echo.
echo ============================================================
echo   LISTENER - one-time setup
echo ============================================================
echo.

REM Find a working Python on PATH
where python >nul 2>nul
if errorlevel 1 (
    echo  [ERROR] No "python" found on PATH.
    echo         Install Python 3.10-3.12 from python.org and tick "Add to PATH",
    echo         or activate your conda env, then re-run this script.
    pause
    exit /b 1
)

echo  Using:
python --version
where python
echo.

echo  Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo  [WARN] pip upgrade failed - continuing anyway.
)

echo.
echo  Installing dependencies from requirements.txt...
python -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo.
    echo  [ERROR] Dependency install failed.
    echo         If torch failed, you may need a specific CUDA wheel:
    echo           python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
    echo         If openai-whisper failed, ffmpeg may be required system-wide.
    pause
    exit /b 1
)

echo.
echo  Verifying imports...
python -c "import anthropic, whisper, sounddevice, soundfile, numpy, torch; print('  core deps OK'); import sys;\
    import importlib; spec = importlib.util.find_spec('pyaudiowpatch');\
    print('  pyaudiowpatch:', 'OK' if spec else 'MISSING (USB-headset loopback unavailable)')"
if errorlevel 1 (
    echo  [ERROR] One or more packages failed to import.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Setup complete. Next steps:
echo     1. Set your Anthropic API key:   setx ANTHROPIC_API_KEY "sk-ant-..."
echo        (open a NEW terminal afterwards so the variable is picked up)
echo     2. Run launch.bat to start the recorder.
echo     3. If the transcript comes back as just "Thank you.", run:
echo          python diagnose_audio.py
echo ============================================================
echo.
pause
endlocal
