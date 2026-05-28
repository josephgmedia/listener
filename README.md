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
Easiest — double-click `setup.bat`. It uses whichever `python` is on your PATH and installs everything from `requirements.txt`.

Or run manually:
```
pip install -r requirements.txt
```

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

## Co-pilot mode (real-time Q&A)

Open `http://localhost:8765/copilot` (or click "Co-pilot mode" from the recorder).

While you're on a call, Listener transcribes the loopback + your mic every 8 seconds. Claude Haiku watches the rolling transcript and only chimes in when it detects the **other party** has asked you a question — then it surfaces a 1–3 sentence suggested answer in big readable text on the right pane.

**Latency:** ~10–13 seconds from Peter's question to suggestion appearing (8s chunk + Whisper "small" + Haiku roundtrip). Whisper's first chunk also has a one-off 5–10s warm-up. This isn't instant but it's fast enough to read while Peter is still talking through their point.

**Limitations:**
- Audio is mixed (loopback + mic). Claude tries to ignore questions you ask yourself, but Whisper can't perfectly attribute who said what.
- Suggestions are best-effort — treat them as prompts, not scripts.
- Every 8s chunk that contains audio calls Claude Haiku. Cheap, but not free over long calls.
- Co-pilot and batch recording can't run simultaneously (same audio devices).

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
