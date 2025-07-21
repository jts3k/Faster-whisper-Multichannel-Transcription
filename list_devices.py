import sounddevice as sd

# List all devices (with their indices)
for idx, dev in enumerate(sd.query_devices()):
    print(f"{idx}: {dev['name']}  ({sd.query_hostapis()[dev['hostapi']]['name']})")

# Example output:
#  0: Speakers (Realtek(R) Audio)  (Windows WASAPI)
# …
# 10: Focusrite USB ASIO Driver  (ASIO)
