# services/audio_io/app/mic.py

import io
import wave
import pyaudio

from .config import MicConfig


class MicController:
    def __init__(self, config: MicConfig | None = None):
        self.config = config or MicConfig()
        self.frames: list[bytes] = []
        self.audio: pyaudio.PyAudio | None = None
        self.stream: pyaudio.Stream | None = None
        self.sample_width: int | None = None

    def open_stream(self):
        """새로운 PyAudio 인스턴스를 생성하고 입력 스트림을 엽니다."""
        if self.audio is not None and self.stream is not None:
            return

        self.audio = pyaudio.PyAudio()
        self.sample_width = self.audio.get_sample_size(self.config.fmt)

        stream_kwargs = dict(
            format=self.config.fmt,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            frames_per_buffer=self.config.chunk,
        )
        if self.config.device_index is not None:
            stream_kwargs["input_device_index"] = self.config.device_index

        self.stream = self.audio.open(**stream_kwargs)

    def close_stream(self):
        """스트림과 PyAudio 인스턴스를 종료합니다."""
        print("[Mic] stop recording / close stream")
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.audio:
            self.audio.terminate()
            self.audio = None

    def record_audio(self) -> bytes:
        """
        config.record_seconds 동안 마이크에서 녹음하고
        메모리 내 WAV 바이트를 반환합니다.
        """
        if self.audio is None or self.stream is None:
            self.open_stream()

        print("[Mic] start recording...")
        frames: list[bytes] = []

        num_chunks = int(
            self.config.rate / self.config.chunk * self.config.record_seconds
        )
        for _ in range(num_chunks):
            data = self.stream.read(self.config.chunk, exception_on_overflow=False)
            frames.append(data)

        print("[Mic] recording done")

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(self.config.channels)
            if self.sample_width is None and self.audio is not None:
                self.sample_width = self.audio.get_sample_size(self.config.fmt)
            wf.setsampwidth(self.sample_width or 2)
            wf.setframerate(self.config.rate)
            wf.writeframes(b"".join(frames))

        return wav_io.getvalue()
