"""
Meeting Recorder — Local server
Records system audio + mic simultaneously, transcribes with Whisper, formats with Claude.
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
OUTPUT_DIR      = Path("recordings")
OUTPUT_DIR.mkdir(exist_ok=True)
PORT            = 8765
USE_FP16        = torch.cuda.is_available()

# ── State ─────────────────────────────────────────────────────────────────────
recording           = False
loopback_chunks     = []
mic_chunks          = []
stop_event          = threading.Event()
rec_thread_loopback = None
rec_thread_mic      = None
whisper_model       = None
current_status      = "idle"
last_summary        = ""
last_transcript     = ""
last_saved_path     = ""

# ── Device detection ──────────────────────────────────────────────────────────
def get_loopback_device():
    """Find best system audio loopback device across different machine configs."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    wasapi_index = None
    for i, api in enumerate(hostapis):
        if 'wasapi' in api['name'].lower():
            wasapi_index = i
            break

    if wasapi_index is None:
        return None, None

    # Priority 1: Focusrite WASAPI loopback (enable in Focusrite Control)
    for i, d in enumerate(devices):
        if (d.get('hostapi') == wasapi_index
                and 'analogue' in d['name'].lower()
                and 'focusrite' in d['name'].lower()
                and d['max_input_channels'] > 0):
            return i, d['name']

    # Priority 2: VB-Cable Output
    for i, d in enumerate(devices):
        if (d.get('hostapi') == wasapi_index
                and 'cable output' in d['name'].lower()
                and d['max_input_channels'] > 0):
            return i, d['name']

    # Priority 3: Stereo Mix (standard Windows loopback)
    for i, d in enumerate(devices):
        if ('stereo mix' in d['name'].lower()
                and d['max_input_channels'] > 0):
            return i, d['name']

    # Priority 4: Any WASAPI input that isn't a mic
    for i, d in enumerate(devices):
        if (d.get('hostapi') == wasapi_index
                and d['max_input_channels'] > 0
                and 'microphone' not in d['name'].lower()
                and 'analogue' not in d['name'].lower()
                and 'cable output' not in d['name'].lower()):
            return i, d['name']

    return None, None


def get_mic_device():
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
        use_wasapi = False
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        if 'hostapi' in devices[device_id]:
            api_name = hostapis[devices[device_id]['hostapi']]['name'].lower()
            use_wasapi = 'wasapi' in api_name

        kwargs = dict(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            device=device_id,
            callback=loopback_callback
        )
        if use_wasapi and ('analogue' in devices[device_id]['name'].lower() 
                           or 'focusrite' in devices[device_id]['name'].lower()):
            kwargs['extra_settings'] = sd.WasapiSettings(True)

        with sd.InputStream(**kwargs):
            while not stop_event.is_set():
                sd.sleep(100)
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


# ── Process after stop ────────────────────────────────────────────────────────
def process_recording(role, fmt, loopback_rate, mic_rate):
    global current_status, last_summary, last_transcript, last_saved_path, whisper_model

    current_status = "processing"

    if not loopback_chunks and not mic_chunks:
        print("No audio captured.")
        current_status = "error"
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    print(f"Mixing audio (loopback: {len(loopback_chunks)} chunks, mic: {len(mic_chunks)} chunks)...")
    mixed = mix_audio(loopback_chunks, loopback_rate, mic_chunks, mic_rate, SAMPLE_RATE)
    duration = len(mixed) / SAMPLE_RATE

    # Save as MP3 128kbps
    audio_path = OUTPUT_DIR / f"audio_{timestamp}.mp3"
    audio_int16 = (mixed * 32767).astype(np.int16)
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
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Loading Whisper {WHISPER_MODEL} (fp16={USE_FP16}, device={gpu_name})...")
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
        last_summary = "### Claude Unavailable\n\nTranscript captured but formatting failed. Check your API credits.\n\nError: " + str(e)

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
            loopback_id, loopback_name = get_loopback_device()
            mic_id, mic_name = get_mic_device()
            self.send_json({
                "status": current_status,
                "loopback_device": loopback_name or "Not found",
                "loopback_device_id": loopback_id,
                "mic_device": mic_name or "Not found",
                "mic_device_id": mic_id,
                "loopback_chunks": len(loopback_chunks),
                "mic_chunks": len(mic_chunks),
                "summary": last_summary,
                "transcript": last_transcript,
                "saved": last_saved_path
            })

        elif path == "/devices":
            self.send_json(get_all_devices())

        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        global recording, stop_event, rec_thread_loopback, rec_thread_mic
        global current_status, loopback_chunks, mic_chunks
        global last_summary, last_transcript, last_saved_path

        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if path == "/start":
            if current_status == "recording":
                self.send_json({"error": "Already recording"})
                return

            loopback_id, loopback_name = get_loopback_device()
            mic_id, mic_name = get_mic_device()

            if loopback_id is None and mic_id is None:
                self.send_json({"error": "No audio devices found"})
                return

            # Find working sample rates for each device
            loopback_rate = get_device_sample_rate(loopback_id) if loopback_id is not None else SAMPLE_RATE
            mic_rate = get_device_sample_rate(mic_id) if mic_id is not None else SAMPLE_RATE

            # Reset state
            loopback_chunks.clear()
            mic_chunks.clear()
            last_summary = ""
            last_transcript = ""
            last_saved_path = ""
            stop_event.clear()
            recording = True
            current_status = "recording"

            print(f"Starting recording:")
            if loopback_id is not None:
                print(f"  Loopback: {loopback_name} (#{loopback_id}) @ {loopback_rate}Hz")
                rec_thread_loopback = threading.Thread(
                    target=do_loopback_recording,
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
            role = body.get("role", "Meeting participant")
            fmt = body.get("format", "brief")
            recording = False
            stop_event.set()
            current_status = "processing"

            # Get the rates used
            loopback_id, _ = get_loopback_device()
            mic_id, _ = get_mic_device()
            loopback_rate = get_device_sample_rate(loopback_id) if loopback_id is not None else SAMPLE_RATE
            mic_rate = get_device_sample_rate(mic_id) if mic_id is not None else SAMPLE_RATE

            t = threading.Thread(
                target=process_recording,
                args=(role, fmt, loopback_rate, mic_rate),
                daemon=True
            )
            t.start()
            self.send_json({"ok": True, "status": "processing"})

        else:
            self.send_json({"error": "Not found"}, 404)


def main():
    print("\n" + "=" * 55)
    print("  MEETING RECORDER — Server")
    print("=" * 55)

    loopback_id, loopback_name = get_loopback_device()
    mic_id, mic_name = get_mic_device()

    print(f"  Loopback : {loopback_name or 'Not found'}" + (f" (#{loopback_id})" if loopback_id is not None else ""))
    print(f"  Mic      : {mic_name or 'Not found'}" + (f" (#{mic_id})" if mic_id is not None else ""))
    print(f"  GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None — using CPU'}")
    print(f"  Server   : http://localhost:{PORT}")
    print(f"  Ctrl+C to stop\n")

    server = HTTPServer(("localhost", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
