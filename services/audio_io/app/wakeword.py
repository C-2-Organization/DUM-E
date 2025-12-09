# services/audio_io/app/wakeword.py

import time
from typing import Callable
from pathlib import Path

import numpy as np
from scipy.signal import resample
import openwakeword
from openwakeword.model import Model

from .mic import MicController

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
# TODO: Replace to custom model
MODEL_PATH = MODEL_DIR / "hello_rokey_8332_32.tflite"


class WakeupWord:
    def __init__(self, mic: MicController):
        openwakeword.utils.download_models()
        self.mic = mic
        self.model: Model | None = None
        self.model_name = MODEL_PATH.stem
        self.running = False

    def init_model(self):
        if self.model is None:
            print(f"[Wakeword] Loading model from: {MODEL_PATH}")
            self.model = Model(wakeword_models=[str(MODEL_PATH)])

    def is_wakeup_once(self) -> bool:
        """
        버퍼 한 번 읽고 wakeword 여부를 판정.
        blocking 함수. 백그라운드 스레드에서 호출할 것.
        """
        if self.mic.stream is None:
            self.mic.open_stream()
        if self.model is None:
            self.init_model()

        buffer_size = self.mic.config.buffer_size
        audio_chunk = np.frombuffer(
            self.mic.stream.read(buffer_size, exception_on_overflow=False),
            dtype=np.int16,
        )
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))

        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs.get(self.model_name, 0.0)
        print(f"[Wakeword] confidence: {confidence:.3f}")

        if confidence > 0.3:
            print("[Wakeword] Wakeword detected!")
            return True
        return False


def start_wakeword_loop(
    wake: WakeupWord,
    on_detect: Callable[[], None] | None = None,
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
            detected = wake.is_wakeup_once()
            if detected and on_detect is not None:
                on_detect()
                time.sleep(1.0)
            if poll_interval > 0:
                time.sleep(poll_interval)
    finally:
        print("[Wakeword] loop stopped")
