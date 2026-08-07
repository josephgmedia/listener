"""
Listener — Smart Launcher (windowless + system tray)
Built into Listener.exe with PyInstaller (see build_exe.bat, --windowed).

Behaviour:
  • First run: pops a console window, finds a system Python, creates a .venv next
    to this exe and installs requirements.txt (a CUDA torch build if an NVIDIA GPU
    is present). This is the only time a console appears.
  • Every run after that: starts the recording server HIDDEN (no console window),
    opens the browser UI, and sits in the system tray. Right-click the tray icon to
    open the UI, open the log, or quit.

Delete the .venv folder to force a fresh reinstall.
"""

import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

PORT = 8765
CREATE_NO_WINDOW   = 0x08000000   # server runs with no console window
CREATE_NEW_CONSOLE = 0x00000010

# When frozen by PyInstaller the app root is the folder containing the exe.
FROZEN = getattr(sys, "frozen", False)
ROOT = Path(sys.executable).parent if FROZEN else Path(__file__).parent

VENV_DIR     = ROOT / ".venv"
VENV_PY      = VENV_DIR / "Scripts" / "python.exe"
SERVER       = ROOT / "app" / "listener_server.py"
REQUIREMENTS = ROOT / "requirements.txt"
SETUP_MARKER = VENV_DIR / ".setup-complete"
ICON         = ROOT / "app" / "icon.ico"
LOG          = ROOT / "listener.log"


# ── Output that survives a windowed (no-console) exe ──────────────────────────
def say(msg):
    """Print if we have a console; silently no-op in windowed mode."""
    try:
        if sys.stdout:
            print(f"  {msg}", flush=True)
    except Exception:
        pass


def _msgbox(text, title="Listener"):
    """Native dialog — works with no console (windowed exe)."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, title, 0x40)  # MB_ICONINFORMATION
    except Exception:
        say(text)


# ── First-run console (only allocated during setup) ───────────────────────────
_console_allocated = False

def _alloc_console():
    """Attach a real console to the windowed process so the one-time install is
    visible. No-op in dev (we already have a console)."""
    global _console_allocated
    if _console_allocated or not FROZEN:
        return
    try:
        import ctypes
        if ctypes.windll.kernel32.AllocConsole():
            sys.stdout = open("CONOUT$", "w", buffering=1)
            sys.stderr = open("CONOUT$", "w", buffering=1)
            sys.stdin  = open("CONIN$",  "r")
            _console_allocated = True
    except Exception:
        pass


def _free_console():
    global _console_allocated
    if not _console_allocated:
        return
    try:
        import ctypes
        try:
            if sys.stdout:
                sys.stdout.flush()
        except Exception:
            pass
        ctypes.windll.kernel32.FreeConsole()
    except Exception:
        pass
    # Back to windowed defaults so later say() calls are harmless.
    sys.stdout = sys.stderr = sys.stdin = None
    _console_allocated = False


def _hold(msg):
    """Show an error and keep the setup console open so it can be read."""
    say("[ERROR] " + msg)
    _msgbox(msg, "Listener — setup failed")
    try:
        if sys.stdin:
            input("  Press Enter to close...")
    except Exception:
        pass


def find_system_python():
    """Locate a real Python 3.10+ install (not this frozen exe)."""
    candidates = []
    for name in ("python", "python3"):
        p = shutil.which(name)
        if p:
            candidates.append(p)
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
    """Create the venv + install deps if needed. Returns True when ready."""
    if VENV_PY.exists() and SETUP_MARKER.exists():
        return True

    _alloc_console()  # make the one-time install visible
    print("\n  ============================================")
    print("   LISTENER — First-time setup")
    print("  ============================================\n")
    say("Installs Python dependencies. This only happens once.")

    sys_py = find_system_python()
    if not sys_py:
        _hold("No Python 3.10+ found on this machine.\n"
              "Install it from https://python.org (tick 'Add to PATH'),\n"
              "then run Listener again.")
        return False
    say(f"Using Python: {sys_py}")

    if not VENV_PY.exists():
        say("Creating virtual environment...")
        if subprocess.run([sys_py, "-m", "venv", str(VENV_DIR)]).returncode != 0:
            _hold("Could not create the virtual environment.")
            return False

    say("Upgrading pip...")
    subprocess.run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "-q"])

    # On an NVIDIA machine install a CUDA torch build FIRST. Plain `pip install
    # torch` from PyPI gives a CPU-only wheel, which makes Whisper transcription
    # painfully slow even on a good GPU. Installing it here means the later
    # `-r requirements.txt` (torch>=2.2) is already satisfied and won't replace it.
    if shutil.which("nvidia-smi"):
        say("NVIDIA GPU detected — installing CUDA build of torch (big; a few min)...")
        subprocess.run([str(VENV_PY), "-m", "pip", "install", "torch",
                        "--index-url", "https://download.pytorch.org/whl/cu124"])

    say("Installing dependencies (grab a coffee — torch is big)...")
    if subprocess.run([str(VENV_PY), "-m", "pip", "install", "-r", str(REQUIREMENTS)]).returncode != 0:
        _hold("Dependency install failed. Check the output above.\n"
              "Delete the .venv folder and run Listener again to retry.")
        return False

    SETUP_MARKER.write_text("ok")
    say("Setup complete — starting Listener.")
    time.sleep(1.2)
    _free_console()
    return True


def server_running():
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}/status", timeout=1)
        return True
    except Exception:
        return False


def start_server_hidden():
    """Launch the server with no console window; its output goes to listener.log."""
    logf = open(LOG, "w", buffering=1)
    env = dict(os.environ)
    env["LISTENER_NO_BROWSER"] = "1"   # we open the browser; avoid a duplicate tab
    # -u / PYTHONUNBUFFERED: with stdout redirected to a file Python switches to
    # block buffering, so the last few KB of output — which is exactly where a
    # crash traceback lives — never reaches listener.log. Unbuffered means the
    # log always tells you why something failed.
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [str(VENV_PY), "-u", str(SERVER)],
        cwd=str(SERVER.parent),
        stdout=logf, stderr=subprocess.STDOUT,
        creationflags=CREATE_NO_WINDOW,
        env=env,
    )


def open_ui():
    webbrowser.open(f"http://localhost:{PORT}/desktop")


def open_log():
    try:
        os.startfile(str(LOG))
    except Exception:
        pass


def run_tray(proc):
    """Sit in the system tray until the user quits. Falls back to a blocking wait
    if pystray isn't available, so the app still works either way."""
    try:
        import pystray
        from PIL import Image
        try:
            image = Image.open(str(ICON))
        except Exception:
            image = Image.new("RGB", (64, 64), (57, 49, 223))

        def _open(icon, item): open_ui()
        def _log(icon, item):  open_log()
        def _quit(icon, item):
            try:
                proc.terminate()
            except Exception:
                pass
            icon.stop()

        menu = pystray.Menu(
            pystray.MenuItem("Open Listener", _open, default=True),
            pystray.MenuItem("Open log", _log),
            pystray.MenuItem("Quit", _quit),
        )
        icon = pystray.Icon("Listener", image, "Listener — recording server", menu)

        # If the server dies on its own, surface the log and drop the tray.
        def _watch():
            proc.wait()
            open_log()
            try:
                icon.stop()
            except Exception:
                pass
        threading.Thread(target=_watch, daemon=True).start()

        icon.run()
    except Exception as e:
        say(f"Tray unavailable ({e}); server is running. Closing this stops it.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()


def main():
    if not SERVER.exists():
        _msgbox(f"Server script not found:\n{SERVER}\n\n"
                "Keep Listener.exe in the project folder.")
        return

    if server_running():
        open_ui()   # already running (maybe a second launch) — just open the UI
        return

    if not ensure_venv():
        return

    proc = start_server_hidden()

    # Wait for the server to accept connections, then open the browser.
    for _ in range(180):
        if server_running():
            open_ui()
            break
        if proc.poll() is not None:
            _msgbox("The Listener server exited during startup.\n"
                    "The log will open so you can see what happened.")
            open_log()
            return
        time.sleep(1)

    run_tray(proc)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
