
# multichannel_fasterwhisper_vadfilter.py

import sounddevice as sd
from sounddevice import AsioSettings
import numpy as np
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
import threading, queue
from pythonosc.udp_client import SimpleUDPClient

# ── INSTALLATION ────────────────────────────────────────────────────────────
# pip install faster-whisper sounddevice scipy python-osc
# ──────────────────────────────────────────────────────────────────────────

# ── CONFIGURATION ───────────────────────────────────────────────────────────
DEVICE_NAME       = "AudioFuse ASIO Driver"
NUM_CHANNELS      = 8
CHANNEL_OFFSET    = 17      # 1-based first ASIO channel
INTERFACE_SR      = 48000   # ASIO sample rate
TARGET_SR         = 16000   # Whisper requires 16 kHz
CHUNK_SEC         = 5       # seconds per transcription flush
MODEL_NAME        = "base.en"
COMPUTE_TYPE      = "int8" # int8 quant for speed
BEAM_SIZE         = 2       # small beam search for accuracy
VAD_FILTER_PARAMS = {
    # Silero VAD thresholds and durations
    "threshold": 0.9,
    "neg_threshold": 0.75,
    "min_speech_duration_ms": 10,
    "max_speech_duration_s": float("inf"),
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 100
}
SUPPRESS_BLANK    = True
OSC_IP            = "127.0.0.1"
OSC_PORT          = 8000
# ──────────────────────────────────────────────────────────────────────────

# 1) Setup ASIO host API & device
hostapis   = sd.query_hostapis()
asio_api   = next(i for i,a in enumerate(hostapis) if "asio" in a["name"].lower())
devices    = sd.query_devices()
device_idx = next(i for i,d in enumerate(devices) if d["hostapi"] == asio_api and d["name"] == DEVICE_NAME)

# 2) Expose only the selected ASIO channels
selectors = list(range(CHANNEL_OFFSET - 1, CHANNEL_OFFSET - 1 + NUM_CHANNELS))
sd.default.device         = device_idx
sd.default.samplerate     = INTERFACE_SR
sd.default.channels       = NUM_CHANNELS
sd.default.extra_settings = AsioSettings(channel_selectors=selectors)

# 3) Per-channel buffers and queues
buffers = {ch: [] for ch in range(NUM_CHANNELS)}
queues  = {ch: queue.Queue(maxsize=1) for ch in range(NUM_CHANNELS)}

# 4) OSC client setup
osc = SimpleUDPClient(OSC_IP, OSC_PORT)

# 5) Load the faster-whisper model onto the GPU
model = WhisperModel(
    MODEL_NAME,
    device       = "cuda",
    compute_type = COMPUTE_TYPE
)

# 6) Audio callback: collect and flush fixed-duration chunks
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Stream status: {status}")
    for ch in range(NUM_CHANNELS):
        buffers[ch].append(indata[:, ch].copy())
    # flush when the first channel reaches CHUNK_SEC
    if sum(buf.shape[0] for buf in buffers[0]) >= CHUNK_SEC * INTERFACE_SR:
        for ch in range(NUM_CHANNELS):
            chunk = np.concatenate(buffers[ch], axis=0)
            buffers[ch].clear()
            q = queues[ch]
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put(chunk)

# 7) Worker thread: resample, transcribe with built-in Silero VAD, send OSC
def transcribe_worker(ch):
    while True:
        chunk48 = queues[ch].get()
        if chunk48 is None:
            break
        # down-sample to 16 kHz
        chunk16 = resample_poly(chunk48, TARGET_SR, INTERFACE_SR).astype(np.float32)
        # use faster-whisper with Silero VAD filter
        segments, _ = model.transcribe(
            chunk16,
            beam_size           = BEAM_SIZE,
            word_timestamps     = False,
            vad_filter          = True,
            vad_parameters      = VAD_FILTER_PARAMS,
            suppress_blank      = SUPPRESS_BLANK
        )
        for seg in segments:
            text = seg.text.strip()
            if text:
                print(f"[Ch{ch+1}] {text}")
                osc.send_message(f"/channel/{ch+1}", text)

# 8) Start one worker thread per channel
threads = []
for ch in range(NUM_CHANNELS):
    t = threading.Thread(target=transcribe_worker, args=(ch,), daemon=True)
    t.start()
    threads.append(t)

# 9) Run the ASIO input stream
with sd.InputStream(
    samplerate = INTERFACE_SR,
    channels   = NUM_CHANNELS,
    blocksize  = int(CHUNK_SEC * INTERFACE_SR),
    callback   = audio_callback):
    print(f"Recording channels {selectors} @ {INTERFACE_SR} Hz, chunks={CHUNK_SEC}s…")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Stopping…")

# 10) Signal shutdown and join threads
for ch in queues:
    queues[ch].put(None)
for t in threads:
    t.join()
print("Done.")
