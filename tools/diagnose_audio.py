"""
Listener — audio capture diagnostic.

Runs the same device-selection logic the server uses, then records 5 seconds
from the loopback and the mic simultaneously and reports peak/RMS for each.

Usage (from the listener folder):
    python diagnose_audio.py

While it's recording, play some audio (a YouTube clip, anything) AND speak
into your mic. If a stream's peak is below ~0.01 it captured silence.
"""

import sys
import time
import threading
import numpy as np
import sounddevice as sd

try:
    import pyaudiowpatch as pyaudio
    HAVE_PYAUDIO = True
except ImportError:
    HAVE_PYAUDIO = False

SAMPLE_RATE = 48000
RECORD_SECONDS = 5


def list_devices():
    print("\n--- All audio devices ---")
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for i, d in enumerate(devices):
        api = hostapis[d['hostapi']]['name'] if 'hostapi' in d else '?'
        print(f"  [{i:2}] {d['name']:<60} api={api:<14} in={d['max_input_channels']} out={d['max_output_channels']}")
    print()


def detect_sd_loopback():
    """Mirror of listener_server._detect_loopback_device."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    wasapi_index = next((i for i, a in enumerate(hostapis) if 'wasapi' in a['name'].lower()), None)

    # Stereo Mix
    for i, d in enumerate(devices):
        if 'stereo mix' in d['name'].lower() and d['max_input_channels'] > 0:
            return i, d['name'], 'sounddevice/Stereo Mix'

    # VB-Cable
    if wasapi_index is not None:
        for i, d in enumerate(devices):
            if (d.get('hostapi') == wasapi_index and 'cable output' in d['name'].lower()
                    and d['max_input_channels'] > 0):
                return i, d['name'], 'sounddevice/VB-Cable'

    # Focusrite
    if wasapi_index is not None:
        for i, d in enumerate(devices):
            if (d.get('hostapi') == wasapi_index and 'focusrite' in d['name'].lower()
                    and 'analogue' in d['name'].lower() and d['max_input_channels'] > 0):
                return i, d['name'], 'sounddevice/Focusrite'

    # Direct WASAPI loopback on default output
    if wasapi_index is not None:
        try:
            default_out = sd.query_devices(kind='output')['name'].strip()
            for i, d in enumerate(devices):
                if d.get('hostapi') == wasapi_index and (
                        d['name'] == default_out or default_out in d['name']):
                    return i, f"{d['name']} [loopback]", 'sounddevice/WASAPI default-output'
        except Exception:
            pass

    return None, None, None


def detect_pa_loopback():
    if not HAVE_PYAUDIO:
        return None, None, None
    try:
        p = pyaudio.PyAudio()
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = p.get_device_info_by_index(wasapi['defaultOutputDevice'])
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if (d.get('hostApi') == wasapi['index'] and d.get('isLoopbackDevice')
                    and default_out['name'] in d['name']):
                p.terminate()
                return i, d['name'], int(d.get('defaultSampleRate', 48000))
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d.get('isLoopbackDevice') and d.get('maxInputChannels', 0) > 0:
                p.terminate()
                return i, d['name'], int(d.get('defaultSampleRate', 48000))
        p.terminate()
    except Exception as e:
        print(f"  pyaudiowpatch error: {e}")
    return None, None, None


def report(name, chunks):
    if not chunks:
        print(f"  {name:<10}  NO CHUNKS CAPTURED — stream never delivered any data")
        return
    data = np.concatenate(chunks, axis=0)
    if data.ndim == 2:
        data = data.mean(axis=1)
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data**2)))
    verdict = (
        "SILENT (probably wrong device or muted)" if peak < 0.005
        else "very quiet — check input level" if peak < 0.05
        else "OK — signal present"
    )
    print(f"  {name:<10}  peak={peak:.4f}  rms={rms:.5f}  -> {verdict}")


def record_sd(device_id, channels, rate, chunks, stop_evt):
    def cb(indata, frames, t, status):
        if not stop_evt.is_set():
            chunks.append(indata.copy())
    try:
        with sd.InputStream(samplerate=rate, channels=channels,
                            dtype='float32', device=device_id, callback=cb):
            stop_evt.wait()
    except Exception as e:
        print(f"  [sd capture error on dev {device_id}: {e}]")


def record_pa(device_id, channels, rate, chunks, stop_evt):
    try:
        p = pyaudio.PyAudio()
        def cb(in_data, frame_count, time_info, status):
            if not stop_evt.is_set():
                chunk = np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels)
                chunks.append(chunk.copy())
            return (None, pyaudio.paContinue)
        stream = p.open(format=pyaudio.paFloat32, channels=channels, rate=rate,
                        input=True, input_device_index=device_id,
                        frames_per_buffer=1024, stream_callback=cb)
        stream.start_stream()
        stop_evt.wait()
        stream.stop_stream()
        stream.close()
        p.terminate()
    except Exception as e:
        print(f"  [pyaudio capture error on dev {device_id}: {e}]")


def main():
    print("=" * 60)
    print("  LISTENER AUDIO DIAGNOSTIC")
    print("=" * 60)
    print(f"  pyaudiowpatch installed : {HAVE_PYAUDIO}")
    print(f"  sounddevice version     : {sd.__version__}")

    list_devices()

    # Loopback selection
    pa_id, pa_name, pa_rate = detect_pa_loopback()
    sd_id, sd_name, sd_method = detect_sd_loopback()

    print("--- Loopback selection ---")
    if pa_id is not None:
        print(f"  WILL USE: pyaudiowpatch  [{pa_id}] {pa_name} @ {pa_rate}Hz")
        loop_method = 'pa'
        loop_id = pa_id
        loop_rate = pa_rate
    elif sd_id is not None:
        print(f"  WILL USE: {sd_method}  [{sd_id}] {sd_name}")
        loop_method = 'sd'
        loop_id = sd_id
        loop_rate = SAMPLE_RATE
    else:
        print("  NO LOOPBACK DEVICE FOUND — system audio cannot be captured.")
        loop_method = None
        loop_id = None
        loop_rate = SAMPLE_RATE

    # Mic selection
    print("\n--- Mic selection ---")
    try:
        mic_default = sd.query_devices(kind='input')
        mic_id = None
        for i, d in enumerate(sd.query_devices()):
            if d['name'] == mic_default['name'] and d['max_input_channels'] > 0:
                mic_id = i
                break
        print(f"  WILL USE: [{mic_id}] {mic_default['name']}")
    except Exception as e:
        mic_id = None
        print(f"  NO MIC FOUND: {e}")

    # Capture
    print(f"\n--- Recording {RECORD_SECONDS}s — play audio AND speak now ---")
    loop_chunks = []
    mic_chunks = []
    stop_evt = threading.Event()

    threads = []
    if loop_method == 'pa':
        try:
            p = pyaudio.PyAudio()
            d = p.get_device_info_by_index(loop_id)
            ch = max(1, int(d['maxInputChannels']))
            p.terminate()
        except Exception:
            ch = 2
        threads.append(threading.Thread(target=record_pa, args=(loop_id, ch, loop_rate, loop_chunks, stop_evt)))
    elif loop_method == 'sd':
        d = sd.query_devices(loop_id)
        ch = max(1, int(d['max_input_channels'] or d['max_output_channels']))
        threads.append(threading.Thread(target=record_sd, args=(loop_id, ch, loop_rate, loop_chunks, stop_evt)))

    if mic_id is not None:
        threads.append(threading.Thread(target=record_sd, args=(mic_id, 1, SAMPLE_RATE, mic_chunks, stop_evt)))

    for t in threads:
        t.start()
    for s in range(RECORD_SECONDS, 0, -1):
        sys.stdout.write(f"\r  ...{s}s ")
        sys.stdout.flush()
        time.sleep(1)
    print("\r  done.       ")
    stop_evt.set()
    for t in threads:
        t.join(timeout=2)

    print("\n--- Per-stream signal report ---")
    report("loopback", loop_chunks)
    report("mic", mic_chunks)
    print()
    print("Hints:")
    print("  - If LOOPBACK is silent: nothing was playing through the default output,")
    print("    OR Stereo Mix is muted in Windows Sound > Recording, OR a USB headset's")
    print("    driver doesn't expose loopback (install VB-Cable as a fix).")
    print("  - If MIC is silent: mic muted in Windows / hardware mute, wrong default")
    print("    input device, or input level too low.")
    print("  - If both are silent: probably the wrong default devices selected at OS level.")


if __name__ == '__main__':
    main()
