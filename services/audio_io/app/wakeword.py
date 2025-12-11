# services/audio_io/app/wakeword.py

import os
import time
from typing import Callable, Optional
from pathlib import Path

import numpy as np
from scipy.signal import resample
import pvporcupine

from .mic import MicController

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
JARVIS_PPN = MODEL_DIR / "jarvis.ppn"
DUMMY_PPN = MODEL_DIR / "dummy.ppn"


class WakeupWord:
    def __init__(self, mic: MicController):
        self.mic = mic

        self.access_key = os.getenv("PICOVOICE_ACCESS_KEY")
        if not self.access_key:
            raise RuntimeError(
                "PICOVOICE_ACCESS_KEY 환경변수가 설정되어 있지 않습니다. "
                "Picovoice Console에서 AccessKey를 발급받아 설정해주세요."
            )

        self.porcupine: pvporcupine.Porcupine | None = None
        self.keyword_paths: list[str] = [
            str(JARVIS_PPN),
            str(DUMMY_PPN),
        ]
        self.keyword_names: list[str] = ["jarvis", "dummy"]

        self.running = False
        self.last_detected_keyword: Optional[str] = None

    def init_model(self):
        if self.porcupine is not None:
            return

        for p in self.keyword_paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"[Wakeword] Keyword file not found: {p}")

        print(f"[Wakeword] Initializing Porcupine...")
        print(f"[Wakeword] Keywords: {self.keyword_names}")
        self.porcupine = pvporcupine.create(
            access_key=self.access_key,
            keyword_paths=self.keyword_paths,
            sensitivities=[0.6] * len(self.keyword_paths),
        )
        print(
            f"[Wakeword] Porcupine ready. "
            f"sample_rate={self.porcupine.sample_rate}, "
            f"frame_length={self.porcupine.frame_length}"
        )

    def is_wakeup_once(self) -> Optional[str]:
        """
        버퍼 한 번 읽고 wakeword 여부를 판정.
        감지되면 감지된 키워드 이름("jarvis"/"dummy"), 아니면 None 리턴.
        blocking 함수. 백그라운드 스레드에서 호출할 것.
        """
        if self.mic.stream is None:
            self.mic.open_stream()
        if self.porcupine is None:
            self.init_model()

        assert self.porcupine is not None

        chunk_size = self.mic.config.chunk
        input_rate = self.mic.config.rate

        raw = self.mic.stream.read(chunk_size, exception_on_overflow=False)
        audio_chunk = np.frombuffer(raw, dtype=np.int16)

        if input_rate != self.porcupine.sample_rate:
            target_len = int(
                len(audio_chunk)
                * self.porcupine.sample_rate
                / input_rate
            )
            if target_len > 0:
                audio_chunk = resample(audio_chunk, target_len).astype(np.int16)

        frame_length = self.porcupine.frame_length

        detected_keyword: Optional[str] = None

        for offset in range(0, len(audio_chunk) - frame_length + 1, frame_length):
            frame = audio_chunk[offset : offset + frame_length]
            keyword_index = self.porcupine.process(frame.tolist())

            if keyword_index >= 0:
                detected_keyword = self.keyword_names[keyword_index]
                print(f"[Wakeword] Detected: {detected_keyword}")
                self.last_detected_keyword = detected_keyword
                break

        return detected_keyword


def start_wakeword_loop(
    wake: WakeupWord,
    on_detect: Callable[[str], None] | None = None,
    poll_interval: float = 0.0,
):
    """
    WakeupWord를 계속 폴링하는 blocking 루프.
    별도 스레드에서 돌리면 됨.
    on_detect 콜백이 있으면, 감지 시 호출.
    """
    print("[Wakeword] loop started")
    wake.running = True
    try:
        while wake.running:
            detected_keyword = wake.is_wakeup_once()
            if detected_keyword is not None and on_detect is not None:
                on_detect(detected_keyword)
                # 같은 발화로 여러 번 연속 감지되는 것 방지
                time.sleep(1.0)
            if poll_interval > 0:
                time.sleep(poll_interval)
    finally:
        print("[Wakeword] loop stopped")
