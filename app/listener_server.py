"""
Meeting Recorder — Local server
Records system audio + mic simultaneously, transcribes with Whisper, formats with Claude.
Run this first, then open recorder.html in your browser.
"""

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("whisperx").setLevel(logging.ERROR)
logging.getLogger("whisperx.asr").setLevel(logging.ERROR)
logging.getLogger("whisperx.vads").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)

import anthropic
# NOTE: whisperx is imported lazily inside the transcription functions, not here.
# It drags in torch/faster-whisper/pyannote/lightning/transformers — a ~10-30s
# cold import that used to block server startup (and the browser) for no reason,
# since it's only needed the moment you stop a recording. See transcribe_*().
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import json
import io
import torch
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# ── FFmpeg: use bundled binary from imageio-ffmpeg so system PATH doesn't matter ─
import os
try:
    import imageio_ffmpeg
    _ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass  # ffmpeg not available via imageio-ffmpeg, fall back to system PATH

# pyaudiowpatch provides true WASAPI loopback for USB audio devices (Jabra, Bose, etc.)
# that sounddevice's WASAPI implementation cannot capture.
try:
    import pyaudiowpatch as pyaudio
    HAVE_PYAUDIO = True
except ImportError:
    HAVE_PYAUDIO = False
    print("  [warn] pyaudiowpatch not installed — USB audio loopback unavailable.")

# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_MODEL   = "large-v2"
SAMPLE_RATE     = 48000
OUTPUT_DIR      = Path("recordings")
OUTPUT_DIR.mkdir(exist_ok=True)
PORT            = 8765
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE    = "float16" if torch.cuda.is_available() else "int8"
HF_TOKEN        = os.environ.get("HF_TOKEN", "")  # set HF_TOKEN env var for speaker diarization

# ── State ─────────────────────────────────────────────────────────────────────
recording             = False
loopback_chunks       = []
mic_chunks            = []
stop_event            = threading.Event()
rec_thread_loopback   = None
rec_thread_mic        = None
whisper_model         = None   # whisperx model, loaded once and reused
align_model_cache     = None   # (model_a, metadata) for "en", loaded once and reused
current_status        = "idle"
last_summary          = ""
last_transcript       = ""
last_saved_path       = ""
transcription_ready   = threading.Event()
loopback_rate_saved   = 48000
mic_rate_saved        = 48000
_device_cache         = None   # cache (loopback_id, loopback_name, mic_id, mic_name)
manual_loopback_id    = None   # None = auto-detect; int = force this sounddevice index
manual_mic_id         = None   # None = auto-detect; int = force this sounddevice index
recording_session_ts  = None   # timestamp string for the current recording's live-flush files
_flush_stop           = threading.Event()
_flush_thread         = None

# ── Device detection ──────────────────────────────────────────────────────────
def _can_open_as_input(device_id, channels, samplerate):
    """Return True if the device can actually be opened as an input stream."""
    try:
        sd.check_input_settings(device=device_id, channels=channels, samplerate=samplerate)
        return True
    except Exception:
        return False


def _detect_loopback_device():
    """Find best system audio loopback device across different machine configs."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    wasapi_index = None
    for i, api in enumerate(hostapis):
        if 'wasapi' in api['name'].lower():
            wasapi_index = i
            break

    # Priority 1: Stereo Mix — Windows virtual device that captures all system audio
    # output regardless of which physical device (headset, speakers) is active.
    # More reliable than direct WASAPI loopback, especially for USB audio devices.
    for i, d in enumerate(devices):
        if 'stereo mix' in d['name'].lower() and d['max_input_channels'] > 0:
            ch = max(1, d['max_input_channels'])
            if _can_open_as_input(i, ch, SAMPLE_RATE):
                return i, d['name']

    # Priority 2: VB-Cable Output (user has installed VB-Cable)
    if wasapi_index is not None:
        for i, d in enumerate(devices):
            if (d.get('hostapi') == wasapi_index
                    and 'cable output' in d['name'].lower()
                    and d['max_input_channels'] > 0):
                return i, d['name']

    # Priority 3: Focusrite WASAPI loopback (enable in Focusrite Control app)
    if wasapi_index is not None:
        for i, d in enumerate(devices):
            if (d.get('hostapi') == wasapi_index
                    and 'analogue' in d['name'].lower()
                    and 'focusrite' in d['name'].lower()
                    and d['max_input_channels'] > 0):
                return i, d['name']

    # Priority 4: WASAPI loopback directly on the default output device.
    # Works on integrated Realtek/Intel audio but NOT on most USB audio devices
    # (Jabra, Bose, Sony etc.) whose drivers don't expose PortAudio loopback.
    if wasapi_index is not None:
        try:
            default_out = sd.query_devices(kind='output')
            default_name = default_out['name'].strip()
            for i, d in enumerate(devices):
                if d.get('hostapi') == wasapi_index and (
                    d['name'] == default_name
                    or d['name'].startswith(default_name)
                    or default_name in d['name']
                ):
                    # Only use if it can actually be opened — USB audio devices fail here
                    ch = max(1, d['max_output_channels'])
                    if _can_open_as_input(i, ch, SAMPLE_RATE) or _can_open_as_input(i, 1, SAMPLE_RATE):
                        return i, f"{d['name']} [loopback]"
        except Exception:
            pass

    return None, None


def _detect_mic_device():
    """Find the default mic input device."""
    try:
        default = sd.query_devices(kind='input')
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['name'] == default['name'] and d['max_input_channels'] > 0:
                return i, d['name']
    except Exception:
        pass

    # Fallback: first available input
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            return i, d['name']

    return None, None


def detect_pyaudio_loopback():
    """
    Use pyaudiowpatch to find the WASAPI loopback device for the default output.
    pyaudiowpatch adds hidden [Loopback] variants for every output device, including
    USB audio (Jabra, Bose, Sony) which sounddevice/PortAudio cannot capture.
    Returns (pyaudio_device_index, display_name, sample_rate) or (None, None, 48000).
    """
    if not HAVE_PYAUDIO:
        return None, None, 48000
    try:
        p = pyaudio.PyAudio()
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)

        # Get the current default output device name
        default_out = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        default_name = default_out["name"]

        # Find the loopback variant of the default output device
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if (d.get("hostApi") == wasapi_info["index"]
                    and d.get("isLoopbackDevice")
                    and default_name in d["name"]):
                rate = int(d.get("defaultSampleRate", 48000))
                name = d["name"]
                p.terminate()
                return i, name, rate

        # Fallback: any loopback device
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d.get("isLoopbackDevice") and d.get("maxInputChannels", 0) > 0:
                rate = int(d.get("defaultSampleRate", 48000))
                name = d["name"]
                p.terminate()
                return i, name, rate

        p.terminate()
    except Exception as e:
        print(f"  pyaudiowpatch loopback detection error: {e}")

    return None, None, 48000


def get_devices(refresh=False):
    """Return (loopback_id, loopback_name, mic_id, mic_name), cached after first call."""
    global _device_cache
    if _device_cache is None or refresh:
        l_id, l_name = _detect_loopback_device()
        m_id, m_name = _detect_mic_device()
        _device_cache = (l_id, l_name, m_id, m_name)
    return _device_cache


def get_loopback_device(refresh=False):
    l_id, l_name, _, _ = get_devices(refresh)
    return l_id, l_name


def get_mic_device(refresh=False):
    _, _, m_id, m_name = get_devices(refresh)
    return m_id, m_name


def get_device_sample_rate(device_id, preferred=48000):
    """Find a working sample rate for a given device."""
    for rate in [preferred, 44100, 16000, 22050, 96000]:
        try:
            sd.check_input_settings(device=device_id, samplerate=rate, channels=1)
            return rate
        except Exception:
            continue
    return preferred


def get_all_devices():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    result = []
    for i, d in enumerate(devices):
        api_name = hostapis[d['hostapi']]['name'] if 'hostapi' in d else '?'
        result.append({
            "id": i,
            "name": d['name'],
            "api": api_name,
            "in": d['max_input_channels'],
            "out": d['max_output_channels']
        })
    return result


# ── Device resolution (honours manual overrides from the UI) ──────────────────
# resolve_loopback()/resolve_mic() are the single source of truth for WHICH
# device gets captured. Both /start and the signal-test endpoints call them, so a
# "Test" always exercises exactly what a real recording would use.

def resolve_loopback():
    """How the system-audio loopback will be captured.
    Returns {method: 'pyaudio'|'sounddevice'|None, id, name, rate}."""
    if manual_loopback_id is not None:
        devices = sd.query_devices()
        name = (devices[manual_loopback_id]['name']
                if 0 <= manual_loopback_id < len(devices) else f"#{manual_loopback_id}")
        return {"method": "sounddevice", "id": manual_loopback_id, "name": name,
                "rate": get_device_sample_rate(manual_loopback_id)}
    # Auto: prefer pyaudiowpatch WASAPI loopback (captures USB headsets)
    if HAVE_PYAUDIO:
        pa_id, pa_name, pa_rate = detect_pyaudio_loopback()
        if pa_id is not None:
            return {"method": "pyaudio", "id": pa_id, "name": pa_name, "rate": pa_rate}
    l_id, l_name = get_loopback_device(refresh=True)
    if l_id is not None:
        return {"method": "sounddevice", "id": l_id, "name": l_name,
                "rate": get_device_sample_rate(l_id)}
    return {"method": None, "id": None, "name": None, "rate": SAMPLE_RATE}


def resolve_mic():
    """How the mic will be captured. Returns {id, name, rate}."""
    if manual_mic_id is not None:
        devices = sd.query_devices()
        name = (devices[manual_mic_id]['name']
                if 0 <= manual_mic_id < len(devices) else f"#{manual_mic_id}")
        return {"id": manual_mic_id, "name": name, "rate": get_device_sample_rate(manual_mic_id)}
    m_id, m_name = get_mic_device()
    return {"id": m_id, "name": m_name,
            "rate": get_device_sample_rate(m_id) if m_id is not None else SAMPLE_RATE}


# ── Signal test ───────────────────────────────────────────────────────────────
def _level(captured):
    """Peak/RMS of captured chunks. 'signal' True if clearly above the noise floor."""
    if not captured:
        return {"ok": True, "peak": 0.0, "rms": 0.0, "signal": False}
    data = np.concatenate(captured, axis=0)
    if data.ndim == 2:
        data = data.mean(axis=1)
    peak = float(np.max(np.abs(data)))
    rms  = float(np.sqrt(np.mean(data ** 2)))
    return {"ok": True, "peak": round(peak, 4), "rms": round(rms, 5), "signal": peak > 0.005}


def _measure_sd(device_id, seconds=1.5):
    """Capture briefly from a sounddevice input (mic, Stereo Mix, VB-Cable) and
    report level."""
    captured = []
    devices = sd.query_devices()
    if not (0 <= device_id < len(devices)):
        return {"ok": False, "error": f"Device #{device_id} out of range"}
    d = devices[device_id]
    channels = max(1, int(d['max_input_channels'] or d['max_output_channels'] or 1))
    rate = get_device_sample_rate(device_id)
    try:
        with sd.InputStream(samplerate=rate, channels=channels, dtype='float32',
                            device=device_id,
                            callback=lambda indata, frames, t, s: captured.append(indata.copy())):
            sd.sleep(int(seconds * 1000))
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return _level(captured)


def _measure_pyaudio(device_id, rate, seconds=1.5):
    """Capture briefly from a pyaudiowpatch WASAPI loopback device and report level."""
    captured = []
    try:
        p = pyaudio.PyAudio()
        d = p.get_device_info_by_index(device_id)
        channels = max(1, int(d["maxInputChannels"]))
        stream = p.open(format=pyaudio.paFloat32, channels=channels, rate=int(rate),
                        input=True, input_device_index=device_id, frames_per_buffer=1024)
        need = int(rate * seconds)
        got = 0
        while got < need:
            buf = stream.read(1024, exception_on_overflow=False)
            captured.append(np.frombuffer(buf, dtype=np.float32).reshape(-1, channels))
            got += 1024
        stream.stop_stream(); stream.close(); p.terminate()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return _level(captured)


def test_signal(kind):
    """Run a ~1.5s capture on the loopback or mic exactly as /start would, so the
    user can confirm signal before a real meeting. Returns level + device name."""
    if current_status == "recording":
        return {"ok": False, "error": "Stop the recording before testing devices"}
    if kind == "loopback":
        plan = resolve_loopback()
        if plan["id"] is None:
            return {"ok": False, "error": "No loopback device found", "device": None}
        res = (_measure_pyaudio(plan["id"], plan["rate"]) if plan["method"] == "pyaudio"
               else _measure_sd(plan["id"]))
        res["device"] = plan["name"]
        return res
    else:
        plan = resolve_mic()
        if plan["id"] is None:
            return {"ok": False, "error": "No mic device found", "device": None}
        res = _measure_sd(plan["id"])
        res["device"] = plan["name"]
        return res


# ── Crash-safe live flush ─────────────────────────────────────────────────────
# During recording audio lives only in RAM (loopback_chunks/mic_chunks) and is
# written on Stop. If the app or PC dies mid-meeting that audio is lost. This
# background thread appends new audio to on-disk WAVs every few seconds, so a
# crash costs at most the last ~5s. On a clean Stop the completed recording is
# saved from RAM as usual and these live files are deleted (see transcribe_recording).

def _flush_worker(loopback_rate, mic_rate, session_ts):
    loop_path = OUTPUT_DIR / f"audio_{session_ts}_live_loopback.wav"
    mic_path  = OUTPUT_DIR / f"audio_{session_ts}_live_mic.wav"
    loop_sf = mic_sf = None
    loop_written = mic_written = 0

    def _drain(chunks, written, handle, path, rate, channels_hint=1):
        end = len(chunks)               # snapshot length first — list only ever grows
        if end <= written:
            return written, handle
        block = chunks[written:end]
        data = np.concatenate(block, axis=0)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if handle is None:
            handle = sf.SoundFile(str(path), mode='w', samplerate=int(rate), channels=1)
        handle.write(data)
        handle.flush()
        return end, handle

    try:
        while not _flush_stop.is_set():
            _flush_stop.wait(5)
            try:
                loop_written, loop_sf = _drain(loopback_chunks, loop_written, loop_sf, loop_path, loopback_rate)
                mic_written,  mic_sf  = _drain(mic_chunks,      mic_written,  mic_sf,  mic_path,  mic_rate)
            except Exception as e:
                print(f"  [flush] write error: {e}")
        # Final drain after stop so the live file is complete right up to Stop
        try:
            _drain(loopback_chunks, loop_written, loop_sf, loop_path, loopback_rate)
            _drain(mic_chunks,      mic_written,  mic_sf,  mic_path,  mic_rate)
        except Exception:
            pass
    finally:
        if loop_sf: loop_sf.close()
        if mic_sf:  mic_sf.close()


def _cleanup_live_files(session_ts):
    """Remove the crash-recovery WAVs once the real recording has been saved."""
    if not session_ts:
        return
    for suffix in ("live_loopback", "live_mic"):
        p = OUTPUT_DIR / f"audio_{session_ts}_{suffix}.wav"
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


# ── Calendar (Granola-style meeting awareness) ────────────────────────────────
# Looks up the user's Google Calendar for a meeting happening right now or
# starting within the next ~10 minutes. Used by the recorder UI to auto-fill
# the Context field on page load. Token + credentials live at the repo root
# (same files calendar_watcher.py uses), not the /app directory.

def _list_calendar_tokens():
    """All calendar_token*.json files in the repo root. Sorted for determinism."""
    from pathlib import Path as _P
    repo_root = _P(__file__).parent.parent
    return sorted(repo_root.glob("calendar_token*.json"))


def _meeting_from_account(token_file):
    """Check ONE Google account for an in-progress or imminent meeting.

    Returns either:
      - {"meeting": payload, "account": email}   on success with a match
      - {"meeting": None,    "account": email}   on success, nothing scheduled
      - {"error": "...",     "account": email}   on auth/network failure
    """
    from datetime import datetime, timezone, timedelta
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
    try:
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                token_file.write_text(creds.to_json())
            else:
                return {"error": "auth_expired", "account": token_file.stem}

        service = build("calendar", "v3", credentials=creds)
        # Identify which account this token represents (for the UI to show)
        try:
            cal_info = service.calendarList().get(calendarId="primary").execute()
            account_label = cal_info.get("id") or token_file.stem
        except Exception:
            account_label = token_file.stem

        now      = datetime.now(timezone.utc)
        time_min = (now - timedelta(minutes=60)).isoformat()
        time_max = (now + timedelta(minutes=15)).isoformat()

        result = service.events().list(
            calendarId   = "primary",
            timeMin      = time_min,
            timeMax      = time_max,
            singleEvents = True,
            orderBy      = "startTime",
            maxResults   = 10,
        ).execute()

        best = None  # (priority, event_dict) — lower priority wins
        for item in result.get("items", []):
            start = item["start"].get("dateTime")
            end   = item["end"].get("dateTime") if "end" in item else None
            if not start: continue
            sdt = datetime.fromisoformat(start)
            edt = datetime.fromisoformat(end) if end else (sdt + timedelta(hours=1))
            if sdt.tzinfo is None: sdt = sdt.replace(tzinfo=timezone.utc)
            if edt.tzinfo is None: edt = edt.replace(tzinfo=timezone.utc)

            secs_until_start = (sdt - now).total_seconds()
            secs_until_end   = (edt - now).total_seconds()

            if secs_until_start <= 0 and secs_until_end > 0:
                priority = 0   # in-progress now
            elif 0 < secs_until_start <= 600:
                priority = 1   # starting within 10 minutes
            else:
                continue

            attendees = [a.get("email", "") for a in item.get("attendees", []) if not a.get("self")]
            payload = {
                "title":            item.get("summary", "Meeting"),
                "start":            sdt.isoformat(),
                "end":              edt.isoformat(),
                "secs_until_start": int(secs_until_start),
                "in_progress":      priority == 0,
                "attendees":        attendees[:10],
                "uid":              item.get("id", ""),
                "account":          account_label,
            }
            if best is None or priority < best[0]:
                best = (priority, payload)

        return {"meeting": best[1] if best else None, "account": account_label}

    except Exception as e:
        return {"error": str(e), "account": token_file.stem}


def get_current_meeting():
    """Return the in-progress or imminent meeting across ALL connected Google
    accounts, or {} if nothing. Adds multi-account support: drop additional
    calendar_token_*.json files into the repo root and they get checked too."""
    from pathlib import Path as _P

    repo_root  = _P(__file__).parent.parent
    creds_file = repo_root / "credentials.json"

    if not creds_file.exists():
        return {"connected": False, "reason": "no_credentials"}

    tokens = _list_calendar_tokens()
    if not tokens:
        return {"connected": False, "reason": "no_accounts_configured",
                "hint": "Run add_calendar_account.py to connect a Google account."}

    try:
        # Touch imports here so we fail fast if deps are missing
        from google.oauth2.credentials import Credentials   # noqa: F401
    except ImportError:
        return {"connected": False, "reason": "missing_deps",
                "hint": "pip install google-auth google-auth-oauthlib google-api-python-client"}

    # Check every account, pick best meeting (in-progress beats starting-soon;
    # within same priority, earliest start wins)
    best = None
    accounts_checked = []
    errors = []
    for token in tokens:
        res = _meeting_from_account(token)
        accounts_checked.append(res.get("account"))
        if res.get("error"):
            errors.append({"account": res.get("account"), "error": res["error"]})
            continue
        m = res.get("meeting")
        if not m:
            continue
        # Priority: in_progress wins, then earliest secs_until_start
        score = (0 if m["in_progress"] else 1, m["secs_until_start"])
        if best is None or score < best[0]:
            best = (score, m)

    if best:
        payload = dict(best[1])
        payload["connected"]        = True
        payload["accounts_checked"] = accounts_checked
        if errors:
            payload["account_errors"] = errors
        return payload

    return {"connected": True, "title": None,
            "accounts_checked": accounts_checked,
            "account_errors": errors or None}


# ── Ad-hoc call detection (Granola-style "is the user on a call?") ────────────
# Three independent signals that combine into a confidence score. Each detector
# fails gracefully if its dependency isn't installed — never blocks the server.
#
#   Process:      Is a known comms app running? (cheap, eager — fires on idle Teams)
#   Window title: Is any visible window titled like an active call? (very accurate)
#   Audio session: Is any process actively using the mic? (most reliable, catches anything)

KNOWN_COMMS_APPS = {
    "zoom.exe":             "Zoom",
    "teams.exe":            "Teams",
    "ms-teams.exe":         "Teams",
    "slack.exe":            "Slack",
    "discord.exe":          "Discord",
    "webex.exe":            "Webex",
    "webexmta.exe":         "Webex",
    "whatsapp.exe":         "WhatsApp",
    "skype.exe":            "Skype",
    "gotomeeting.exe":      "GoToMeeting",
    "bluejeans.exe":        "BlueJeans",
    "chime.exe":            "Chime",
}

# (regex pattern, friendly app name) — patterns are case-insensitive
WINDOW_TITLE_PATTERNS = [
    (r"Zoom Meeting",                        "Zoom"),
    (r"Meeting \| Microsoft Teams",          "Teams"),
    (r"Microsoft Teams.*Meeting",            "Teams"),
    # Google Meet — multiple title formats Chrome uses across states:
    #   in-call:   "Meet — abc-defg-hij - Google Chrome"
    #   lobby:     "Meet - Google Chrome"
    #   joined:    "abc-defg-hij | Meet - Google Chrome"
    #   classic:   "meet.google.com/abc-defg-hij - Google Chrome"
    (r"Meet\s*[—\-:·]\s*[a-z]{3,4}-[a-z]{4}-[a-z]{3,4}", "Google Meet"),  # in-call w/ code
    (r"[a-z]{3,4}-[a-z]{4}-[a-z]{3,4}\s*\|\s*Meet",      "Google Meet"),  # joined view
    (r"meet\.google\.com/[a-z\-]{8,}",                    "Google Meet"),  # URL with code
    (r"Webex Meeting",                       "Webex"),
    (r"GoTo Meeting",                        "GoToMeeting"),
    (r"Slack \| huddle",                     "Slack Huddle"),
    (r"Discord.*Voice Connected",            "Discord"),
]


def _detect_processes():
    """Return list of comms-app process names currently running."""
    apps_seen = set()
    try:
        import psutil
        for p in psutil.process_iter(["name"]):
            try:
                name = (p.info["name"] or "").lower()
                if name in KNOWN_COMMS_APPS:
                    apps_seen.add(KNOWN_COMMS_APPS[name])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return {"available": True, "apps": sorted(apps_seen)}
    except ImportError:
        # Fallback: parse `tasklist` output (slower, Windows-only, no extra deps)
        try:
            import subprocess
            out = subprocess.check_output(
                ["tasklist", "/FO", "CSV", "/NH"],
                creationflags=0x08000000, text=True, timeout=3,
            ).lower()
            for proc, friendly in KNOWN_COMMS_APPS.items():
                if proc in out:
                    apps_seen.add(friendly)
            return {"available": True, "apps": sorted(apps_seen), "via": "tasklist"}
        except Exception as e:
            return {"available": False, "reason": "no_psutil_and_tasklist_failed",
                    "detail": str(e)}


def _detect_window_titles():
    """Return list of (app, title) for visible windows matching call patterns."""
    import re
    matches = []
    try:
        import win32gui
    except ImportError:
        return {"available": False, "reason": "missing_pywin32",
                "hint": "pip install pywin32"}

    def _cb(hwnd, acc):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd) or ""
        if not title.strip():
            return
        for pattern, app in WINDOW_TITLE_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                acc.append({"app": app, "title": title})
                break

    try:
        win32gui.EnumWindows(_cb, matches)
        return {"available": True, "matches": matches}
    except Exception as e:
        return {"available": False, "reason": "enum_failed", "detail": str(e)}


def _detect_audio_sessions():
    """Return processes currently in active audio sessions (rendering or capturing)."""
    try:
        from pycaw.pycaw import AudioUtilities, AudioSessionState
    except ImportError:
        return {"available": False, "reason": "missing_pycaw",
                "hint": "pip install pycaw comtypes"}

    active_apps = []
    try:
        sessions = AudioUtilities.GetAllSessions()
        for s in sessions:
            if s.State != AudioSessionState.Active:
                continue
            proc = s.Process
            if proc is None:
                continue
            try:
                name = (proc.name() or "").lower()
            except Exception:
                continue
            friendly = KNOWN_COMMS_APPS.get(name)
            active_apps.append({
                "process":  name,
                "app":      friendly or name.replace(".exe", "").title(),
                "is_comms": bool(friendly),
            })
        return {"available": True, "active_sessions": active_apps}
    except Exception as e:
        return {"available": False, "reason": "enum_failed", "detail": str(e)}


def detect_active_call():
    """
    Combine all three signals into a single confidence score.

    Confidence ladder (highest wins):
      high    — comms app AND (window title OR active audio session) agree
      medium  — only one strong signal (window title OR active audio session
                from a comms app)
      low     — comms app is running but no active call signal
      none    — nothing detected
    """
    proc_result   = _detect_processes()
    window_result = _detect_window_titles()
    audio_result  = _detect_audio_sessions()

    proc_apps    = set(proc_result.get("apps", []))
    window_apps  = {m["app"] for m in window_result.get("matches", [])}
    audio_comms  = {s["app"] for s in audio_result.get("active_sessions", []) if s.get("is_comms")}
    audio_any    = bool(audio_result.get("active_sessions"))

    evidence = []
    if proc_apps:    evidence.append("process")
    if window_apps:  evidence.append("window_title")
    if audio_comms:  evidence.append("audio_session")

    # Confidence logic
    # Window-title patterns are very specific (e.g. "Meet — abc-defg-hij" with a
    # meeting code regex) — a match almost certainly means a real call. We treat
    # title + any audio activity as ironclad so browser-based calls (Meet in
    # Chrome) score high even though chrome.exe isn't in our comms-app list.
    if window_apps and (proc_apps or audio_comms or audio_any):
        confidence = "high"
    elif window_apps:
        # Title matched but nothing else — could be a lobby/preview state.
        confidence = "medium"
    elif audio_comms:
        # Comms app actively using audio without a visible call window.
        confidence = "medium"
    elif audio_any and not proc_apps:
        # Something's using audio but we don't recognise it — could be a custom
        # WebRTC app, browser-based call, or just Spotify. Flag it as a hint.
        confidence = "medium"
        evidence.append("unknown_audio")
    elif proc_apps:
        confidence = "low"
    else:
        confidence = "none"

    detected_apps = sorted(window_apps | audio_comms | proc_apps)

    return {
        "detected":    confidence not in ("none", "low"),
        "confidence":  confidence,
        "apps":        detected_apps,
        "evidence":    evidence,
        "raw": {
            "process":       proc_result,
            "window_title":  window_result,
            "audio_session": audio_result,
        },
    }


# ── Recording ─────────────────────────────────────────────────────────────────
def loopback_callback(indata, frames, time, status):
    if recording:
        loopback_chunks.append(indata.copy())


def mic_callback(indata, frames, time, status):
    if recording:
        mic_chunks.append(indata.copy())


def do_loopback_recording(device_id, sample_rate):
    global current_status
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        use_wasapi = False
        if 'hostapi' in devices[device_id]:
            api_name = hostapis[devices[device_id]['hostapi']]['name'].lower()
            use_wasapi = 'wasapi' in api_name

        # Output devices (headsets, speakers) need WASAPI loopback mode to be captured.
        # Input-only devices (Stereo Mix) are opened normally.
        is_output_device = (devices[device_id]['max_output_channels'] > 0
                            and devices[device_id]['max_input_channels'] == 0)

        # Use the device's actual input channel count (Stereo Mix reports this correctly).
        # For WASAPI direct-output loopback devices, fall back to output channel count.
        if devices[device_id]['max_input_channels'] > 0:
            channels = max(1, int(devices[device_id]['max_input_channels']))
        else:
            channels = max(1, int(devices[device_id]['max_output_channels']))

        kwargs = dict(
            samplerate=sample_rate,
            channels=channels,
            dtype='float32',
            device=device_id,
            callback=loopback_callback
        )
        # Only add WASAPI exclusive settings for Focusrite-style loopback devices
        if use_wasapi and ('analogue' in devices[device_id]['name'].lower()
                           or 'focusrite' in devices[device_id]['name'].lower()):
            kwargs['extra_settings'] = sd.WasapiSettings(True)

        with sd.InputStream(**kwargs):
            while not stop_event.is_set():
                sd.sleep(100)
    except Exception as e:
        print(f"Loopback recording error: {e}")


def do_loopback_recording_pyaudio(device_id, sample_rate):
    """
    Record WASAPI loopback using pyaudiowpatch.
    Works for USB audio devices (Jabra, Bose, etc.) that sounddevice cannot capture.
    """
    try:
        p = pyaudio.PyAudio()
        d = p.get_device_info_by_index(device_id)
        channels = max(1, int(d["maxInputChannels"]))

        def _callback(in_data, frame_count, time_info, status):
            if recording:
                chunk = np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels)
                loopback_chunks.append(chunk.copy())
            return (None, pyaudio.paContinue)

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_id,
            frames_per_buffer=1024,
            stream_callback=_callback
        )
        stream.start_stream()
        print(f"  Loopback stream open: {channels}ch @ {sample_rate}Hz (pyaudiowpatch WASAPI loopback)")

        while not stop_event.is_set():
            sd.sleep(100)

        stream.stop_stream()
        stream.close()
        p.terminate()
    except Exception as e:
        print(f"Loopback recording error: {e}")


def do_mic_recording(device_id, sample_rate):
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            device=device_id,
            callback=mic_callback
        ):
            while not stop_event.is_set():
                sd.sleep(100)
    except Exception as e:
        print(f"Mic recording error: {e}")


def mix_audio(chunks_a, rate_a, chunks_b, rate_b, target_rate):
    """Mix two mono audio streams, resampling if needed, returning mono mix."""
    def to_array(chunks, rate):
        if not chunks:
            return np.array([], dtype=np.float32)
        data = np.concatenate(chunks, axis=0)
        if data.ndim == 2:
            data = data.mean(axis=1)
        # Resample to target_rate if needed
        if rate != target_rate:
            ratio = target_rate / rate
            new_len = int(len(data) * ratio)
            data = np.interp(
                np.linspace(0, len(data), new_len),
                np.arange(len(data)),
                data
            ).astype(np.float32)
        return data

    a = to_array(chunks_a, rate_a)
    b = to_array(chunks_b, rate_b)

    # Pad shorter stream
    max_len = max(len(a), len(b))
    if len(a) < max_len:
        a = np.pad(a, (0, max_len - len(a)))
    if len(b) < max_len:
        b = np.pad(b, (0, max_len - len(b)))

    # Mix and normalise
    mixed = a + b
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak

    return mixed


# ── Step 1: Stop recording → mix + transcribe (runs immediately on /stop) ──────
def wav_to_mp3(wav_path, bitrate="128k"):
    """Convert WAV to MP3 using bundled ffmpeg, delete the WAV, return MP3 path."""
    import subprocess
    mp3_path = Path(str(wav_path).replace(".wav", ".mp3"))
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"
    subprocess.run(
        [ffmpeg_exe, "-y", "-i", str(wav_path), "-b:a", bitrate, str(mp3_path)],
        capture_output=True
    )
    if mp3_path.exists():
        wav_path.unlink()
        return mp3_path
    return wav_path   # fallback: keep WAV if conversion failed


def load_audio_file(path, target_sr=16000):
    """Load WAV or MP3 to 16kHz mono float32. WAV uses soundfile; MP3 uses ffmpeg."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    else:
        # Use ffmpeg to decode MP3 → raw pcm → numpy
        import subprocess
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"
        proc = subprocess.run(
            [ffmpeg_exe, "-y", "-i", str(path), "-f", "f32le", "-ac", "1",
             "-ar", str(target_sr), "-"],
            capture_output=True
        )
        audio = np.frombuffer(proc.stdout, dtype=np.float32)
        return audio
    if sr != target_sr:
        target_len = int(len(audio) * target_sr / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)), audio
        ).astype(np.float32)
    return audio


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def get_align_model():
    """Load the English wav2vec2 alignment model once and reuse it.

    Previously this was reloaded on every transcription, adding needless latency
    to each recording. Cached here since the language is always 'en'."""
    global align_model_cache
    if align_model_cache is None:
        import whisperx
        align_model_cache = whisperx.load_align_model(language_code="en", device=DEVICE)
    return align_model_cache


def transcribe_recording(loopback_rate, mic_rate):
    global current_status, last_transcript, whisper_model

    current_status = "transcribing"

    if not loopback_chunks and not mic_chunks:
        print("No audio captured.")
        current_status = "error"
        transcription_ready.set()
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    print(f"Mixing audio (loopback: {len(loopback_chunks)} chunks, mic: {len(mic_chunks)} chunks)...")

    # Per-stream stats — useful when transcripts come back empty / "Thank you."
    def _stream_stats(chunks):
        if not chunks:
            return 0.0, 0.0, 0
        d = np.concatenate(chunks, axis=0)
        if d.ndim == 2:
            d = d.mean(axis=1)
        return float(np.max(np.abs(d))), float(np.sqrt(np.mean(d**2))), len(d)

    loop_peak, loop_rms, loop_samples = _stream_stats(loopback_chunks)
    mic_peak,  mic_rms,  mic_samples  = _stream_stats(mic_chunks)
    print(f"  Loopback signal: peak={loop_peak:.4f}  rms={loop_rms:.5f}  samples={loop_samples}")
    print(f"  Mic signal     : peak={mic_peak:.4f}  rms={mic_rms:.5f}  samples={mic_samples}")

    # Save loopback and mic separately so we can diagnose which side was silent
    if loop_samples:
        loop_path = OUTPUT_DIR / f"audio_{timestamp}_loopback.wav"
        loop_data = np.concatenate(loopback_chunks, axis=0)
        if loop_data.ndim == 2:
            loop_data = loop_data.mean(axis=1)
        sf.write(str(loop_path), loop_data, loopback_rate)
    if mic_samples:
        mic_path = OUTPUT_DIR / f"audio_{timestamp}_mic.wav"
        mic_data = np.concatenate(mic_chunks, axis=0)
        if mic_data.ndim == 2:
            mic_data = mic_data.mean(axis=1)
        sf.write(str(mic_path), mic_data, mic_rate)

    mixed = mix_audio(loopback_chunks, loopback_rate, mic_chunks, mic_rate, SAMPLE_RATE)
    duration = len(mixed) / SAMPLE_RATE

    audio_path = OUTPUT_DIR / f"audio_{timestamp}.wav"
    sf.write(str(audio_path), mixed, SAMPLE_RATE)
    print(f"Audio saved as WAV, converting to MP3...")
    audio_path = wav_to_mp3(audio_path)
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"Audio saved: {audio_path} ({duration:.1f}s, {size_mb:.1f}MB)")

    # Real recording is now safely on disk — drop the crash-recovery live files.
    _cleanup_live_files(recording_session_ts)

    # Bail out early if BOTH streams were essentially silent — Whisper would
    # otherwise hallucinate "Thank you." or similar from pure silence.
    SILENCE_PEAK = 0.005
    if loop_peak < SILENCE_PEAK and mic_peak < SILENCE_PEAK:
        print("Both streams are silent — skipping transcription to avoid Whisper hallucination.")
        last_transcript = ""
        current_status = "error_silent_audio"
        transcription_ready.set()
        return

    # Resample to 16000Hz for Whisper (passes array directly — no ffmpeg needed)
    WHISPER_SR = 16000
    new_len = int(len(mixed) * WHISPER_SR / SAMPLE_RATE)
    whisper_audio = np.interp(
        np.linspace(0, len(mixed) - 1, new_len),
        np.arange(len(mixed)),
        mixed
    ).astype(np.float32)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Transcribing with whisperx {WHISPER_MODEL} on {gpu_name}...")

    try:
        import whisperx  # lazy — see import note at top of file
        if whisper_model is None:
            whisper_model = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)

        audio = load_audio_file(audio_path)
        result = whisper_model.transcribe(audio, batch_size=16 if DEVICE == "cuda" else 4, language="en")

        # Align for accurate word timestamps (align model cached across recordings)
        model_a, metadata = get_align_model()
        result = whisperx.align(result["segments"], model_a, metadata, audio, device=DEVICE,
                                return_char_alignments=False)

        # Speaker diarization if HF token is set
        if HF_TOKEN:
            print("Diarizing speakers...")
            from whisperx.diarize import DiarizationPipeline
            diarize_model    = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
            diarize_segments = diarize_model(str(audio_path))
            result           = whisperx.assign_word_speakers(diarize_segments, result)

        # Build transcript string
        lines = []
        current_speaker = None
        for seg in result["segments"]:
            speaker = seg.get("speaker", None)
            text    = seg["text"].strip()
            time    = fmt_time(seg["start"])
            if speaker and speaker != current_speaker:
                current_speaker = speaker
                lines.append(f"\n[{speaker}] {time}")
            lines.append(f"  {text}" if speaker else f"{time}  {text}")

        last_transcript = "\n".join(lines).strip()
        print(f"Transcript ready: {len(last_transcript.split())} words")

        if not last_transcript:
            print("Empty transcript — no speech detected.")
            current_status = "error"

    except Exception as e:
        print(f"Transcription error: {e}")
        current_status = "error"

    # Signal that transcription is done — /format can now proceed
    transcription_ready.set()


# ── Step 2: Format with Claude (runs after /format is called) ─────────────────
def format_with_claude(role, fmt):
    global current_status, last_summary, last_saved_path

    # Wait for Whisper to finish (may already be done if user was slow picking format)
    print("Waiting for transcription to complete...")
    transcription_ready.wait()

    if current_status == "error_silent_audio" or not last_transcript:
        # Write a clearly-labelled markdown so the failure is visible in the
        # recordings folder instead of silently producing nothing.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_path = OUTPUT_DIR / f"meeting_{timestamp}_NO_AUDIO.md"
        msg = ("Both the system-audio loopback and the mic captured silence "
               "(peak amplitude < 0.005). Whisper was skipped to avoid the "
               "well-known 'Thank you.' silence hallucination.\n\n"
               "Run `python diagnose_audio.py` from the listener folder to "
               "find out which side is broken.")
        out_path.write_text(
            f"# Meeting — {datetime.now().strftime('%d %B %Y, %H:%M')} (NO AUDIO)\n\n{msg}\n",
            encoding="utf-8"
        )
        last_summary = msg
        last_saved_path = str(out_path)
        current_status = "error"
        return
    if current_status == "error":
        return

    current_status = "processing"

    format_instructions = {
        "brief":    "Create a concise debrief: key topics, key decisions, and action items.",
        "detailed": "Create a detailed breakdown: Meeting Overview, Key Discussion Points, Their Asks, My Commitments, Open Questions, Next Steps.",
        "actions":  "Extract ONLY action items and commitments — mine and theirs. Clear list format.",
        "email":    "Draft a professional follow-up email I can send to the other party. Summarise what was discussed and confirm next steps."
    }

    try:
        client = anthropic.Anthropic()
        print("Formatting with Claude...")
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system="""You are a meeting assistant for Joe, a freelance motion designer, videographer,
photographer, and musician based in Sydney, Australia. He works across projection design,
animation, content creation, and video production.

The transcript may contain speaker labels like [SPEAKER_00], [SPEAKER_01], etc.
Read through the transcript and identify speakers by name where they are addressed or
introduce themselves (e.g. "Thanks Joe", "Peter, do you want to...", "Hi, I'm Lucy").
Replace [SPEAKER_XX] labels with the person's actual name wherever you can identify them.
For any speakers you cannot identify, use a short description (e.g. "Unknown" or keep the label).
Joe is usually the person recording — he is likely the one presenting or being referred to as the creative lead.

Format meeting transcripts clearly and practically.
Use ### for section headings, bullet points with - for lists.
Bold key names and decisions using **text**.""",
            messages=[{
                "role": "user",
                "content": f"""My role in this meeting: {role}

Format requested: {format_instructions.get(fmt, fmt)}

Raw transcript:
{last_transcript}"""
            }]
        )
        last_summary = message.content[0].text
    except Exception as e:
        print(f"Claude error: {e}")
        last_summary = "### Claude Unavailable\n\nTranscript captured but formatting failed. Check your API credits.\n\nError: " + str(e)

    # Save markdown
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = OUTPUT_DIR / f"meeting_{timestamp}.md"
    md_content = f"""# Meeting — {datetime.now().strftime("%d %B %Y, %H:%M")}
**Role:** {role}

---

## Summary

{last_summary}

---

## Raw Transcript

{last_transcript}
"""
    out_path.write_text(md_content, encoding="utf-8")
    last_saved_path = str(out_path)
    print(f"Saved: {out_path}")
    current_status = "done"


# ── HTTP Server ───────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        def serve_html(filename):
            html_path = Path(__file__).parent / filename
            if html_path.exists():
                body = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_json({"error": f"{filename} not found"}, 404)

        if path in ("/", "/desktop"):
            serve_html("recorder.html")
            return

        if path == "/mobile":
            serve_html("index.html")
            return

        if path == "/status":
            # Kept cheap (polled every 5s): use the cached auto-detected names, or
            # the manually-selected device name if the user overrode it.
            loopback_id, loopback_name = get_loopback_device()
            mic_id, mic_name = get_mic_device()
            all_devs = None
            if manual_loopback_id is not None or manual_mic_id is not None:
                all_devs = sd.query_devices()
            if manual_loopback_id is not None and 0 <= manual_loopback_id < len(all_devs):
                loopback_name = all_devs[manual_loopback_id]['name']
                loopback_id   = manual_loopback_id
            if manual_mic_id is not None and 0 <= manual_mic_id < len(all_devs):
                mic_name = all_devs[manual_mic_id]['name']
                mic_id   = manual_mic_id
            self.send_json({
                "status": current_status,
                "loopback_device": loopback_name or "Not found",
                "loopback_device_id": loopback_id,
                "mic_device": mic_name or "Not found",
                "mic_device_id": mic_id,
                "manual_loopback_id": manual_loopback_id,
                "manual_mic_id": manual_mic_id,
                "loopback_chunks": len(loopback_chunks),
                "mic_chunks": len(mic_chunks),
                "summary": last_summary,
                "transcript": last_transcript,
                "saved": last_saved_path
            })

        elif path == "/devices":
            self.send_json(get_all_devices())

        elif path == "/calendar-now":
            self.send_json(get_current_meeting())

        elif path == "/call-detect":
            self.send_json(detect_active_call())

        else:
            # Try to serve static files (fonts, etc.) from the app directory
            static_path = Path(__file__).parent / path.lstrip('/')
            if static_path.exists() and static_path.is_file():
                ext = static_path.suffix.lower()
                content_types = {
                    '.otf': 'font/otf', '.ttf': 'font/ttf',
                    '.woff': 'font/woff', '.woff2': 'font/woff2',
                    '.js': 'application/javascript', '.css': 'text/css',
                    '.png': 'image/png', '.jpg': 'image/jpeg', '.ico': 'image/x-icon',
                }
                body = static_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_types.get(ext, 'application/octet-stream'))
                self.send_header("Content-Length", len(body))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        global recording, stop_event, rec_thread_loopback, rec_thread_mic
        global current_status, loopback_chunks, mic_chunks
        global last_summary, last_transcript, last_saved_path
        global loopback_rate_saved, mic_rate_saved, transcription_ready

        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        content_type = self.headers.get("Content-Type", "")

        # ── Multipart upload (WAV file) ────────────────────────────────────────
        if path == "/upload":
            global current_status, last_summary, last_transcript, last_saved_path, transcription_ready
            if current_status == "recording":
                self.send_json({"error": "Cannot upload while recording"}); return

            if "multipart/form-data" not in content_type:
                self.send_json({"error": "Expected multipart/form-data"}); return

            # Parse boundary from Content-Type header
            boundary = None
            for part in content_type.split(";"):
                part = part.strip()
                if part.startswith("boundary="):
                    boundary = part[len("boundary="):].strip().encode()
                    break
            if not boundary:
                self.send_json({"error": "No boundary in Content-Type"}); return

            raw = self.rfile.read(length)
            # Extract the file bytes between the boundary markers
            b_start = raw.find(b"\r\n\r\n") + 4
            b_end   = raw.rfind(b"\r\n--" + boundary)
            wav_bytes = raw[b_start:b_end]

            # Save to recordings/ with a timestamp name
            ts       = datetime.now().strftime("%Y-%m-%d_%H-%M")
            out_path = OUTPUT_DIR / f"audio_{ts}_uploaded.wav"
            out_path.write_bytes(wav_bytes)
            print(f"Uploaded WAV received ({len(wav_bytes)//1024} KB), converting to MP3...")
            out_path = wav_to_mp3(out_path)
            print(f"Saved: {out_path} ({out_path.stat().st_size//1024} KB)")

            # Reset state and kick off transcription (same as /stop)
            last_summary    = ""
            last_transcript = ""
            last_saved_path = ""
            transcription_ready.clear()
            current_status = "transcribing"

            def transcribe_uploaded():
                global current_status, last_transcript, whisper_model
                try:
                    import whisperx  # lazy — see import note at top of file
                    if whisper_model is None:
                        print("  Loading whisperx model...")
                        whisper_model = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)
                    print("  Loading audio...")
                    audio    = load_audio_file(out_path)
                    print("  Transcribing... (this takes a few minutes on GPU)")
                    result   = whisper_model.transcribe(audio, batch_size=16 if DEVICE == "cuda" else 4, language="en")
                    print("  Aligning timestamps...")
                    model_a, metadata = get_align_model()
                    result   = whisperx.align(result["segments"], model_a, metadata, audio, device=DEVICE,
                                              return_char_alignments=False)
                    if HF_TOKEN:
                        print("  Diarizing speakers...")
                        from whisperx.diarize import DiarizationPipeline
                        dm     = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
                        dsegs  = dm(str(out_path))
                        result = whisperx.assign_word_speakers(dsegs, result)

                    lines, cur_spk = [], None
                    for seg in result["segments"]:
                        speaker = seg.get("speaker")
                        text    = seg["text"].strip()
                        t       = fmt_time(seg["start"])
                        if speaker and speaker != cur_spk:
                            cur_spk = speaker
                            lines.append(f"\n[{speaker}] {t}")
                        lines.append(f"  {text}" if speaker else f"{t}  {text}")
                    last_transcript = "\n".join(lines).strip()
                    print(f"Transcript ready: {len(last_transcript.split())} words")
                    current_status = "awaiting_format"
                except Exception as e:
                    print(f"Transcription error: {e}")
                    current_status = "error"
                finally:
                    transcription_ready.set()

            threading.Thread(target=transcribe_uploaded, daemon=True).start()
            self.send_json({"ok": True, "status": "transcribing"})
            return

        body = json.loads(self.rfile.read(length)) if length else {}

        if path == "/set-devices":
            # Override auto-detection. Pass an int device id, or null/"auto" to
            # revert that device to automatic detection.
            global manual_loopback_id, manual_mic_id
            if current_status == "recording":
                self.send_json({"error": "Stop recording before changing devices"}); return

            def _norm(v):
                if v is None or v == "auto" or v == "":
                    return None
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return None

            if "loopback_id" in body:
                manual_loopback_id = _norm(body.get("loopback_id"))
            if "mic_id" in body:
                manual_mic_id = _norm(body.get("mic_id"))
            get_devices(refresh=True)  # refresh auto-detect cache for /status
            self.send_json({"ok": True,
                            "manual_loopback_id": manual_loopback_id,
                            "manual_mic_id": manual_mic_id})
            return

        if path == "/test-signal":
            # Capture ~1.5s from the loopback or mic (exactly as /start would) and
            # report the level so the user can confirm signal before a meeting.
            kind = body.get("kind", "mic")
            self.send_json(test_signal("loopback" if kind == "loopback" else "mic"))
            return

        if path == "/start":
            global recording_session_ts, _flush_thread
            if current_status == "recording":
                self.send_json({"error": "Already recording"})
                return

            # Resolve devices (honours manual overrides picked in the UI, else
            # auto-detects: pyaudiowpatch WASAPI loopback first, then sounddevice).
            loop_plan = resolve_loopback()
            mic_plan  = resolve_mic()

            loopback_id   = loop_plan["id"]
            loopback_name = loop_plan["name"]
            loopback_rate = loop_plan["rate"]
            use_pyaudio_loopback = (loop_plan["method"] == "pyaudio")

            mic_id   = mic_plan["id"]
            mic_name = mic_plan["name"]
            mic_rate = mic_plan["rate"]

            get_loopback_device(refresh=True)  # refresh cache for /status display

            if loopback_id is None and mic_id is None:
                self.send_json({"error": "No audio devices found"})
                return

            # Save rates for use by /stop later
            loopback_rate_saved = loopback_rate
            mic_rate_saved      = mic_rate

            # Reset state
            loopback_chunks.clear()
            mic_chunks.clear()
            last_summary = ""
            last_transcript = ""
            last_saved_path = ""
            transcription_ready.clear()
            stop_event.clear()
            recording = True
            current_status = "recording"
            recording_session_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            print(f"Starting recording:")
            if loopback_id is not None:
                method = "pyaudiowpatch" if use_pyaudio_loopback else "sounddevice"
                print(f"  Loopback: {loopback_name} (#{loopback_id}) @ {loopback_rate}Hz [{method}]")
                rec_fn = do_loopback_recording_pyaudio if use_pyaudio_loopback else do_loopback_recording
                rec_thread_loopback = threading.Thread(
                    target=rec_fn,
                    args=(loopback_id, loopback_rate),
                    daemon=True
                )
                rec_thread_loopback.start()
            else:
                print("  No loopback device — capturing mic only")

            if mic_id is not None:
                print(f"  Mic: {mic_name} (#{mic_id}) @ {mic_rate}Hz")
                rec_thread_mic = threading.Thread(
                    target=do_mic_recording,
                    args=(mic_id, mic_rate),
                    daemon=True
                )
                rec_thread_mic.start()
            else:
                print("  No mic device found")

            # Crash-safe: flush captured audio to disk every few seconds
            _flush_stop.clear()
            _flush_thread = threading.Thread(
                target=_flush_worker,
                args=(loopback_rate, mic_rate, recording_session_ts),
                daemon=True
            )
            _flush_thread.start()

            self.send_json({
                "ok": True,
                "status": "recording",
                "loopback": loopback_name,
                "mic": mic_name
            })

        elif path == "/pause":
            recording = False
            current_status = "paused"
            self.send_json({"ok": True, "status": "paused"})

        elif path == "/resume":
            recording = True
            current_status = "recording"
            self.send_json({"ok": True, "status": "recording"})

        elif path == "/stop":
            # Stop recording immediately and kick off Whisper in background.
            # Format choice comes later via /format.
            recording = False
            stop_event.set()
            _flush_stop.set()          # stop crash-safe flusher (final drain happens inside)
            transcription_ready.clear()
            current_status = "transcribing"

            t = threading.Thread(
                target=transcribe_recording,
                args=(loopback_rate_saved, mic_rate_saved),
                daemon=True
            )
            t.start()
            self.send_json({"ok": True, "status": "transcribing"})

        elif path == "/format":
            # Receive format choice. If Whisper is still running this will wait internally.
            role = body.get("role", "Meeting participant")
            fmt  = body.get("format", "brief")

            t = threading.Thread(
                target=format_with_claude,
                args=(role, fmt),
                daemon=True
            )
            t.start()
            self.send_json({"ok": True, "status": "processing"})

        elif path == "/notion-save":
            import urllib.request as _ur, re as _re

            token     = body.get("token", "")
            db_id     = body.get("db_id", "")
            title     = body.get("title", "Meeting")
            date      = body.get("date", "")
            context   = body.get("context", "")
            project   = body.get("project", "")
            output    = body.get("output", "")
            transcript = body.get("transcript", "")

            if not token or not db_id:
                self.send_json({"error": "Missing token or db_id"}); return

            def rt(text):
                """Convert **bold** markdown to Notion rich_text array."""
                parts = _re.split(r'\*\*(.+?)\*\*', text)
                result = []
                for i, part in enumerate(parts):
                    if not part: continue
                    result.append({"type": "text", "text": {"content": part},
                                   "annotations": {"bold": i % 2 == 1}})
                return result or [{"type": "text", "text": {"content": text}}]

            def md_to_blocks(text):
                """Convert markdown text to Notion block children."""
                blocks = []
                for line in text.split('\n'):
                    h = _re.match(r'^#{1,3}\s+(.+)', line)
                    li = _re.match(r'^[-•*]\s+(.+)', line)
                    nl = _re.match(r'^(\d+)\.\s+(.+)', line)
                    hr = _re.match(r'^---+$', line)
                    if h:
                        blocks.append({"object": "block", "type": "heading_2",
                                       "heading_2": {"rich_text": rt(h[1])}})
                    elif li:
                        blocks.append({"object": "block", "type": "bulleted_list_item",
                                       "bulleted_list_item": {"rich_text": rt(li[1])}})
                    elif nl:
                        blocks.append({"object": "block", "type": "numbered_list_item",
                                       "numbered_list_item": {"rich_text": rt(nl[2])}})
                    elif hr:
                        blocks.append({"object": "block", "type": "divider", "divider": {}})
                    elif line.strip():
                        blocks.append({"object": "block", "type": "paragraph",
                                       "paragraph": {"rich_text": rt(line)}})
                return blocks

            def speaker_blocks(text):
                """Format transcript speaker turns as Notion blocks."""
                blocks = []
                blocks.append({"object": "block", "type": "heading_2",
                                "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Transcript"}}]}})
                for line in text.split('\n'):
                    spk = _re.match(r'^\[([^\]]+)\]\s*(\d{2}:\d{2})', line)
                    txt = line.strip().lstrip()
                    if spk:
                        blocks.append({"object": "block", "type": "paragraph",
                                       "paragraph": {"rich_text": [
                                           {"type": "text", "text": {"content": f"{spk[1]}  {spk[2]}"},
                                            "annotations": {"bold": True, "color": "blue"}}
                                       ]}})
                    elif txt:
                        blocks.append({"object": "block", "type": "paragraph",
                                       "paragraph": {"rich_text": [{"type": "text", "text": {"content": txt},
                                                                     "annotations": {"color": "gray"}}]}})
                return blocks

            # Build children: summary blocks + divider + transcript blocks
            children = md_to_blocks(output)
            if transcript:
                children.append({"object": "block", "type": "divider", "divider": {}})
                children += speaker_blocks(transcript)

            # Properties — metadata + searchable copies of output/transcript
            # (full content also lives in page body; properties capped at Notion's 2000-char limit)
            def truncate(text, limit=2000):
                if not text: return ""
                return text if len(text) <= limit else text[:limit - 15] + "...[truncated]"

            props = {
                "Name":   {"title": [{"text": {"content": title}}]},
                "Status": {"select": {"name": "Done"}},
            }
            if date:       props["Date"]       = {"date": {"start": date}}
            if context:    props["Context"]    = {"rich_text": [{"text": {"content": truncate(context)}}]}
            if project:    props["Project"]    = {"select": {"name": project}}
            if output:     props["Output"]     = {"rich_text": [{"text": {"content": truncate(output)}}]}
            if transcript: props["Transcript"] = {"rich_text": [{"text": {"content": truncate(transcript)}}]}

            # Notion API limits 100 children per request — chunk if needed
            import urllib.request as _ur2
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
                "Notion-Version": "2022-06-28",
            }
            try:
                payload = json.dumps({
                    "parent":   {"database_id": db_id},
                    "properties": props,
                    "children": children[:100]
                }).encode()
                req = _ur2.Request("https://api.notion.com/v1/pages", data=payload,
                                   headers=headers, method="POST")
                with _ur2.urlopen(req) as resp:
                    data = json.loads(resp.read())
                if data.get("object") == "error":
                    self.send_json({"error": data.get("message")}); return
                page_id = data["id"]

                # Append remaining blocks if over 100
                remaining = children[100:]
                while remaining:
                    chunk = remaining[:100]; remaining = remaining[100:]
                    patch = json.dumps({"children": chunk}).encode()
                    req2 = _ur2.Request(f"https://api.notion.com/v1/blocks/{page_id}/children",
                                        data=patch, headers=headers, method="PATCH")
                    _ur2.urlopen(req2)

                self.send_json({"ok": True, "id": page_id})
            except Exception as e:
                self.send_json({"error": str(e)})

        else:
            self.send_json({"error": "Not found"}, 404)


def _local_ip():
    """Best-effort LAN IP for the phone URL. Never blocks startup: a bad network
    or slow reverse-DNS used to hang gethostbyname() for seconds right before the
    server started serving."""
    import socket as _s
    try:
        # Doesn't actually send packets; just asks the OS which local interface
        # would route to an external address. Fast and doesn't hit DNS.
        s = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
        try:
            s.settimeout(0.2)
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return "localhost"


def main():
    # Bind and start serving FIRST, in a background thread, so /status answers
    # almost immediately and the browser opens fast. All the banner/device/GPU
    # probing below is diagnostic only and must not sit on the critical path.
    import threading as _t
    # Threaded so a 1.5s device signal-test can't block the status poll.
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    _t.Thread(target=server.serve_forever, daemon=True).start()

    # Open the browser ourselves when run directly (launch.bat / python).
    # The .exe launcher opens it instead and sets LISTENER_NO_BROWSER to avoid
    # a duplicate tab.
    if not os.environ.get("LISTENER_NO_BROWSER"):
        import webbrowser
        def _open_browser():
            import time; time.sleep(0.3)
            webbrowser.open(f"http://localhost:{PORT}/desktop")
        _t.Thread(target=_open_browser, daemon=True).start()

    print("\n" + "=" * 55)
    print("  MEETING RECORDER — Server")
    print("=" * 55)
    print(f"  Desktop  : http://localhost:{PORT}/desktop")
    print(f"  Phone    : http://{_local_ip()}:{PORT}/mobile")
    print(f"  Ctrl+C to stop\n")

    # Device + GPU detection (runs after the server is already up)
    if HAVE_PYAUDIO:
        pa_id, pa_name, pa_rate = detect_pyaudio_loopback()
        if pa_id is not None:
            print(f"  Loopback : {pa_name} @ {pa_rate}Hz [pyaudiowpatch WASAPI]")
        else:
            _, loopback_name, _, _ = get_devices(refresh=True)
            print(f"  Loopback : {loopback_name or 'Not found'} [sounddevice fallback]")
    else:
        _, loopback_name, _, _ = get_devices(refresh=True)
        print(f"  Loopback : {loopback_name or 'Not found'}")

    _, _, mic_id, mic_name = get_devices()
    print(f"  Mic      : {mic_name or 'Not found'}" + (f" (#{mic_id})" if mic_id is not None else ""))
    print(f"  GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None — using CPU'}")
    print()

    try:
        while True:
            import time; time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
