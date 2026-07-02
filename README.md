# Listener

A discreet meeting recorder for Windows and iPhone. Records system audio or room mic, transcribes locally with Whisper, formats notes with Claude.

No bots. Nothing joins your call.

---

## What's in the repo

| File | Purpose |
|------|---------|
| `server.py` | Python backend — WASAPI loopback recording, Whisper transcription, Claude formatting |
| `recorder.html` | Desktop UI — served from localhost:8765 |
| `launch.bat` | Double-click to start server and open UI |
| `listener-mobile.html` | Standalone iPhone tool — uses device mic and Claude API directly, no server needed |

---

## Desktop setup (Windows)

### 1. Install dependencies
```
pip install openai-whisper anthropic sounddevice soundfile numpy keyboard torch pydub
```

**Optional — ad-hoc call detection** (auto-detects when you're on a Zoom/Teams/Meet/etc. call without a calendar match):
```
pip install psutil pywin32 pycaw comtypes
```
All three are optional and independent — install some, all, or none. Listener will use whatever's available and degrade gracefully when something's missing. Without any of them, Listener still works exactly as before, just without ad-hoc call detection.

**Optional — calendar awareness** (auto-fills meeting context from Google Calendar):
```
pip install google-auth google-auth-oauthlib google-api-python-client
```

Add multiple Google accounts (personal + work Gmail):
```
python add_calendar_account.py personal
python add_calendar_account.py work
```
Each command opens Google sign-in for ONE account and saves its token as `calendar_token_<label>.json`. Listener checks all connected accounts and surfaces whichever meeting is starting first (or in progress). The original `calendar_token.json` from `calendar_watcher.py` still works as the primary account — nothing breaks.

### 2. Install FFmpeg
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH, or drop `ffmpeg.exe` in the same folder as `server.py`.

### 3. Pre-download Whisper model
Do this before running the server — avoids a 3GB download mid-recording:
```
python -c "import whisper; whisper.load_model('large')"
```

### 4. Set your Claude API key
```
$env:ANTHROPIC_API_KEY = "your-key-here"
```
Or add permanently via Windows Environment Variables.

### 5. Focusrite users — enable loopback
Open Focusrite Control → enable **Send Direct Monitor to Loopback**.

### 6. Run it
Double-click `launch.bat`. Server starts, browser opens automatically at `http://localhost:8765`.

---

## Desktop usage

- **Start Recording** — captures system audio via WASAPI loopback
- **Pause** — temporarily stops capture without ending session
- **Stop & Summarise** — stops recording, transcribes with Whisper, formats with Claude
- **Clear** — resets for a new session

Controls during recording: `SPACE` to pause, `S` to stop, `Q` to quit.

---

## Audio device priority

The script auto-detects the best loopback device in this order:

1. Focusrite WASAPI loopback (enable in Focusrite Control)
2. VB-Cable Output (install from [vb-audio.com](https://vb-audio.com/Cable/) if needed)
3. Stereo Mix (enable in Windows Sound settings → Recording → Show Disabled Devices)

---

## Other Windows machines (no Focusrite)

Enable Stereo Mix in Sound settings (Recording tab → Show Disabled Devices → Enable). If Stereo Mix isn't available, install VB-Cable. Everything else is the same.

---

## iPhone setup

`listener-mobile.html` is a standalone tool — no server, no Python.

1. Download the file and AirDrop it to your iPhone
2. Open in Safari
3. Share → Add to Home Screen
4. Open it, go to Setup tab, paste your Claude API key

Uses your device mic to capture the room. Best for in-person meetings.

---

## Output

Each desktop recording saves to the `recordings/` folder:

```
recordings/
  meeting_2026-04-09_09-30.md   ← Claude summary + raw transcript
  audio_2026-04-09_09-30.mp3    ← raw audio at 128kbps (delete when done)
```

---

## Cost

- **Whisper** — free, runs locally on your GPU
- **Claude API** — ~$0.03–0.05 per meeting summary

---

## Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU recommended (4090 = near-instant transcription)
- Anthropic API key — get one at [console.anthropic.com](https://console.anthropic.com)
