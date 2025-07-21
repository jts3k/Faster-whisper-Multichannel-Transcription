# Faster-whisper Multichannel Transcription

This repo provides two scripts:

- **list_devices.py**  
  Enumerates all PortAudio/ASIO devices and channel counts so you can pick the right driver & channel offset.

- **multichannel.py**  
  Captures N ASIO channels, runs `faster-whisper` locally on your GPU with Silero VAD filtering, and emits transcripts via OSC.

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/YOUR_USERNAME/whisper-live-multichannel.git
   cd whisper-live-multichannel


2. **Create & activate a virtual environment**

   * **Windows (PowerShell)**

     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   * **macOS/Linux**

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Make cuDNN available**

   `faster-whisper` relies on CTranslate2 + cuDNN for GPU acceleration. You have two options:

   * **(a) pip approach**

     ```bash
     pip install nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12
     ```

     This will install the CUDA 12.x / cuDNN DLLs under your venv. If you still see “Could not locate cudnn\_ops64\_9.dll,” add the cuDNN package folder to your PATH:

     ```powershell
     # (PowerShell)
     $cu = python -c "import nvidia.cudnn as cd, os; print(os.path.dirname(cd.__file__))"
     setx PATH "$env:PATH;$cu"
     ```

   * **(b) manual copy**

     1. Find your cuDNN wheel’s DLL folder:

        ```bash
        python -c "import nvidia.cudnn as cd, os; print(os.path.dirname(cd.__file__))"
        ```
     2. Copy all `*.dll` from that folder (and its `bin/` subfolder, if present) into your `venv/Scripts` (Windows) or `venv/bin` (macOS/Linux) directory.
     3. Restart your shell.

5. **(Optional) ASIO environment variable**
   If you’re on Windows and using sounddevice’s bundled PortAudio with ASIO support, ensure:

   ```powershell
   setx SD_ASIO 1
   ```

---

## Usage

1. **List your devices**

   ```bash
   python list_devices.py
   ```

   Note the **index** and **name** of your ASIO host API and the channel counts.

2. **Edit `multichannel.py`**

   * Set `DEVICE_NAME`, `NUM_CHANNELS`, `CHANNEL_OFFSET`, and `OSC_IP/OSC_PORT` at the top of the file.
   * Adjust model parameters (e.g. `MODEL_NAME`, `COMPUTE_TYPE`, `BEAM_SIZE`, `VAD_FILTER_PARAMS`) as desired.

3. **Run the transcription**

   ```bash
   python multichannel.py
   ```

   You should see console prints and OSC messages on `/channel/1`, `/channel/2`, etc.

---

## Tips & Troubleshooting

* **No cuDNN DLL found?**
  Double‑check you either installed the NVIDIA wheel or manually copied the DLLs to your venv’s `Scripts/` (Windows) or `bin/` (macOS/Linux) folder.
* **High latency?**
  Try smaller `CHUNK_SEC` (e.g. `1.0`), lower `BEAM_SIZE`, or switch to `compute_type="float16"`.
* **Hallucinations in silence?**
  Tweak `VAD_FILTER_PARAMS` (see [faster-whisper/vad.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py) for defaults).

---

This software has been tested on a Windows 11 machine with NVIDIA 5090 GPU.

