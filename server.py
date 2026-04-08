"""
Meeting Recorder — Local server
Handles recording, Whisper transcription, Claude formatting.
Run this first, then open recorder.html in your browser.
"""

import anthropic
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import sys
import json
import torch
from pydub import AudioSegment
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_MODEL   = "large"
SAMPLE_RATE     = 48000
CHANNELS        = 1        # Mono — Focusrite loopback doesn't support stereo
OUTPUT_DIR      = Path("recordings")
OUTPUT_DIR.mkdir(exist_ok=True)
PORT            = 8765
USE_FP16        = torch.cuda.is_available()  # Auto-detect GPU

# ── State ─────────────────────────────────────────────────────────────────────
recording       = False
audio_chunks    = []
stop_event      = threading.Event()
rec_thread      = None
whisper_model   = None
current_status  = "idle"
current_device  = None
last_summary    = ""
last_transcript = ""
last_saved_path = ""
state_lock      = threading.Lock()

# ── Audio device ──────────────────────────────────────────────────────────────
def get_loopback_device():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    wasapi_index = None
    for i, api in enumerate(hostapis):
        if 'wasapi' in api['name'].lower():
            wasapi_index = i
            break

    if wasapi_index is None:
        return None, None

    # Priority 1: Focusrite loopback (Analogue input via WASAPI — loopback enabled in Focusrite Control)
    for i, d in enumerate(devices):
        if (d.get('hostapi') == wasapi_index
                and 'analogue' in d['name'].lower()
                and 'focusrite' in d['name'].lower()
                and d['max_input_channels'] > 0):
            return i, d['name']

    # Priority 2: VB-Cable Output if present
    for i, d in enumerate(devices):
        if (d.get('hostapi') == wasapi_index
                and 'cable output' in d['name'].lower()
                and d['max_input_channels'] > 0):
            return i, d['name']

    # Priority 3: Stereo Mix — built-in Windows loopback on non-interface machines
    for i, d in enumerate(devices):
        if ('stereo mix' in d['name'].lower()
                and d['max_input_channels'] > 0):
            return i, d['name']

    # Priority 4: Any WASAPI input that isnt a mic or analogue input
    for i, d in enumerate(devices):
        if (d.get('hostapi') == wasapi_index
                and d['max_input_channels'] > 0
                and 'microphone' not in d['name'].lower()
                and 'analogue' not in d['name'].lower()
                and 'cable output' not in d['name'].lower()):
            return i, d['name']

    return None, None


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


# ── Recording ─────────────────────────────────────────────────────────────────
def audio_callback(indata, frames, time, status):
    if recording:
        audio_chunks.append(indata.copy())


def do_recording(device_id):
    global current_status
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            device=device_id,
            callback=audio_callback
        ):
            while not stop_event.is_set():
                sd.sleep(100)
    except Exception as e:
        current_status = "error"
        print(f"Recording error: {e}")


# ── Process after stop ────────────────────────────────────────────────────────
def process_recording(role, fmt):
    global current_status, last_summary, last_transcript, last_saved_path, whisper_model

    current_status = "processing"

    if not audio_chunks:
        print("No audio chunks captured.")
        current_status = "error"
        return

    # Save audio — flatten mono (n,1) → (n,)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    audio_data = np.concatenate(audio_chunks, axis=0)
    if audio_data.ndim == 2 and audio_data.shape[1] == 1:
        audio_data = audio_data.flatten()
    duration = len(audio_data) / SAMPLE_RATE

    # Save as MP3 at 128kbps — WAV is enormous for long meetings
    audio_path = OUTPUT_DIR / f"audio_{timestamp}.mp3"
    audio_int16 = (audio_data * 32767).astype(np.int16)
    segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1
    )
    segment.export(str(audio_path), format="mp3", bitrate="128k")
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"Audio saved: {audio_path} ({duration:.1f}s, {size_mb:.1f}MB)")

    # Transcribe
    print(f"Loading Whisper {WHISPER_MODEL} (fp16={USE_FP16}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else chr(67)+chr(80)+chr(85)})...")
    if whisper_model is None:
        whisper_model = whisper.load_model(WHISPER_MODEL)
    print("Transcribing...")
    result = whisper_model.transcribe(
        str(audio_path),
        language="en",
        verbose=False,
        fp16=USE_FP16
    )
    last_transcript = result["text"].strip()
    print(f"Transcript: {len(last_transcript.split())} words")

    if not last_transcript:
        print("Empty transcript — no speech detected.")
        current_status = "error"
        return

    # Claude
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
animation, content creation, and video production. Format meeting transcripts clearly and
practically. Use ### for section headings, bullet points with - for lists.""",
            messages=[{
                "role": "user",
                "content": f"""My role in this meeting: {role}

Format requested: {format_instructions.get(fmt, format_instructions['brief'])}

Raw transcript:
{last_transcript}"""
            }]
        )
        last_summary = message.content[0].text
    except Exception as e:
        print(f"Claude error: {e}")
        last_summary = "### Claude Unavailable\n\nTranscript captured but Claude formatting failed. Check your API credits at console.anthropic.com\n\nError: " + str(e)

    # Save markdown
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

        if path == "/" or path == "/index.html":
            html_path = Path(__file__).parent / "recorder.html"
            if html_path.exists():
                body = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_json({"error": "recorder.html not found"}, 404)
            return

        if path == "/status":
            device_id, device_name = get_loopback_device()
            self.send_json({
                "status": current_status,
                "device": device_name or "Not found",
                "device_id": device_id,
                "chunks": len(audio_chunks),
                "summary": last_summary,
                "transcript": last_transcript,
                "saved": last_saved_path
            })

        elif path == "/devices":
            self.send_json(get_all_devices())

        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        global recording, stop_event, rec_thread, current_status
        global audio_chunks, current_device, last_summary, last_transcript, last_saved_path

        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if path == "/start":
            if current_status == "recording":
                self.send_json({"error": "Already recording"})
                return

            device_id = body.get("device_id")
            if device_id is None:
                device_id, _ = get_loopback_device()
            if device_id is None:
                self.send_json({"error": "No loopback device found"})
                return

            # Reset state for new recording
            audio_chunks.clear()
            last_summary = ""
            last_transcript = ""
            last_saved_path = ""
            stop_event.clear()
            recording = True
            current_status = "recording"
            current_device = device_id
            rec_thread = threading.Thread(target=do_recording, args=(device_id,), daemon=True)
            rec_thread.start()
            self.send_json({"ok": True, "status": "recording"})

        elif path == "/pause":
            recording = False
            current_status = "paused"
            self.send_json({"ok": True, "status": "paused"})

        elif path == "/resume":
            recording = True
            current_status = "recording"
            self.send_json({"ok": True, "status": "recording"})

        elif path == "/stop":
            role = body.get("role", "Meeting participant")
            fmt = body.get("format", "brief")
            recording = False
            stop_event.set()
            current_status = "processing"
            t = threading.Thread(target=process_recording, args=(role, fmt), daemon=True)
            t.start()
            self.send_json({"ok": True, "status": "processing"})

        else:
            self.send_json({"error": "Not found"}, 404)


def main():
    print("\n" + "=" * 50)
    print("  MEETING RECORDER — Server")
    print("=" * 50)

    device_id, device_name = get_loopback_device()
    if device_id is not None:
        print(f"  Loopback device: {device_name} (#{device_id})")
    else:
        print("  Warning: No loopback device found")

    print(f"  GPU acceleration: {USE_FP16}")
    print(f"  Server running at http://localhost:{PORT}")
    print(f"  Press Ctrl+C to stop\n")

    server = HTTPServer(("localhost", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
