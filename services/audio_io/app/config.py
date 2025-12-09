# services/audio_io/app/config.py

import pyaudio
from dataclasses import dataclass

@dataclass
class MicConfig:
    chunk: int = 12000
    rate: int = 48000
    channels: int = 1
    record_seconds: int = 5
    fmt: int = pyaudio.paInt16
    device_index: int | None = None
    buffer_size: int = 24000
