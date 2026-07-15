"""
Listener — Smart Launcher
Built into Listener.exe with PyInstaller (see build_exe.bat).

On first run it finds a system Python, creates a virtual environment next to
this exe, and installs everything from requirements.txt. On every run it then
starts the recording server and opens the browser UI once the server is up.
Delete the .venv folder to force a fresh reinstall.
"""

import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

PORT = 8765

# When frozen by PyInstaller, the app root is the folder containing the exe.
if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).parent
else:
    ROOT = Path(__file__).parent

VENV_DIR = ROOT / ".venv"
VENV_PY = VENV_DIR / "Scripts" / "python.exe"
SERVER = ROOT / "app" / "listener_server.py"
REQUIREMENTS = ROOT / "requirements.txt"
SETUP_MARKER = VENV_DIR / ".setup-complete"


def say(msg):
    print(f"  {msg}", flush=True)


def fail(msg):
    print(f"\n  [ERROR] {msg}\n", flush=True)
    input("  Press Enter to close...")
    sys.exit(1)


def find_system_python():
    """Locate a real Python 3.10+ install (not this frozen exe)."""
    candidates = []
    for name in ("python", "python3"):
        p = shutil.which(name)
        if p:
            candidates.append(p)
    # Common per-user install locations, newest first
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
    if local.exists():
        for d in sorted(local.glob("Python3*"), reverse=True):
            exe = d / "python.exe"
            if exe.exists():
                candidates.append(str(exe))

    for cand in candidates:
        try:
            out = subprocess.run(
                [cand, "-c", "import sys; print(sys.version_info[:2])"],
                capture_output=True, text=True, timeout=15,
            )
            major, minor = eval(out.stdout.strip())
            if (major, minor) >= (3, 10):
                return cand
        except Exception:
            continue
    return None


def ensure_venv():
    if VENV_PY.exists() and SETUP_MARKER.exists():
        return

    say("First-time setup — this takes a few minutes and only happens once.")
    sys_py = find_system_python()
    if not sys_py:
        fail(
            "No Python 3.10+ found on this machine.\n"
            "  Install it from https://python.org (tick 'Add to PATH'),\n"
            "  then run Listener again."
        )
    say(f"Using Python: {sys_py}")

    if not VENV_PY.exists():
        say("Creating virtual environment...")
        r = subprocess.run([sys_py, "-m", "venv", str(VENV_DIR)])
        if r.returncode != 0:
            fail("Could not create the virtual environment.")

    say("Installing dependencies (grab a coffee — torch is big)...")
    r = subprocess.run([
        str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "-q",
    ])

    # On an NVIDIA machine, install a CUDA torch build FIRST. Plain
    # `pip install torch` from PyPI gives a CPU-only wheel by default, which makes
    # Whisper transcription painfully slow even on a 4090. Installing the CUDA
    # wheel first means the later `-r requirements.txt` (torch>=2.2) is already
    # satisfied and won't replace it.
    if shutil.which("nvidia-smi"):
        say("NVIDIA GPU detected — installing CUDA build of torch...")
        subprocess.run([
            str(VENV_PY), "-m", "pip", "install", "torch",
            "--index-url", "https://download.pytorch.org/whl/cu124",
        ])

    r = subprocess.run([
        str(VENV_PY), "-m", "pip", "install", "-r", str(REQUIREMENTS),
    ])
    if r.returncode != 0:
        fail(
            "Dependency install failed. Check the output above.\n"
            "  Delete the .venv folder and run Listener again to retry."
        )
    SETUP_MARKER.write_text("ok")
    say("Setup complete.")


def server_running():
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}/status", timeout=1)
        return True
    except Exception:
        return False


def main():
    print()
    print("  ============================================")
    print("   LISTENER")
    print("  ============================================")
    print()

    if not SERVER.exists():
        fail(f"Server script not found: {SERVER}\n  Keep Listener.exe in the project folder.")

    if server_running():
        say("Server already running — opening browser.")
        webbrowser.open(f"http://localhost:{PORT}/desktop")
        return

    ensure_venv()

    say("Starting server...")
    proc = subprocess.Popen(
        [str(VENV_PY), str(SERVER)],
        cwd=str(SERVER.parent),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )

    say("Waiting for server (model load can take 5-30s)...")
    for _ in range(120):
        if server_running():
            say("Server ready — opening browser.")
            webbrowser.open(f"http://localhost:{PORT}/desktop")
            return
        if proc.poll() is not None:
            fail("Server exited during startup. Check the 'Listener Server' window for errors.")
        time.sleep(1)
        print(".", end="", flush=True)

    fail(f"Server did not respond after 120s. Try opening http://localhost:{PORT}/desktop manually.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
