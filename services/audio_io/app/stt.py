# services/audio_io/app/stt.py

import os
import time
import tempfile
from typing import Optional

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from openai import OpenAI

from services.common.env_loader import load_env, get_env

load_env()

class StreamingSTT:
    """
    - wakeword가 감지된 이후에 호출되는 STT 모듈
    - 사용자가 말하는 동안 계속 녹음
    - '5초 이상' 음성이 없으면 녹음을 종료하고 Whisper로 전송
    """

    def __init__(
        self,
        samplerate: int = 16000,
        chunk_duration: float = 0.5,   # 한 번에 0.5초씩 읽기
        silence_sec: float = 3.0,      # 5초 이상 조용하면 종료
        max_total_sec: float = 60.0,   # 안전장치: 최대 60초까지만 듣기
        energy_threshold: float = 500, # 이 값 이상이면 '사람이 말하는 중'이라고 간주
    ):
        api_key = get_env("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.samplerate = samplerate
        self.chunk_duration = chunk_duration
        self.silence_sec = silence_sec
        self.max_total_sec = max_total_sec
        self.energy_threshold = energy_threshold

    def _record_until_silence(self) -> np.ndarray:
        """
        sounddevice로 마이크를 조금씩 읽으면서
        - 최초로 음성이 감지될 때까지 기다렸다가
        - 그 이후로 5초 이상 조용하면 종료
        """
        print("[STT] 🎙 녹음을 시작합니다. 말이 끊기면 자동으로 종료됩니다.")
        num_samples_per_chunk = int(self.samplerate * self.chunk_duration)

        chunks: list[np.ndarray] = []
        start_time = time.time()
        last_voice_time = time.time()
        heard_voice = False

        while True:
            audio_block = sd.rec(
                num_samples_per_chunk,
                samplerate=self.samplerate,
                channels=1,
                dtype="int16",
            )
            sd.wait()

            block_energy = float(np.abs(audio_block).mean())

            chunks.append(audio_block.copy())

            now = time.time()

            if block_energy > self.energy_threshold:
                heard_voice = True
                last_voice_time = now

            if heard_voice and (now - last_voice_time) >= self.silence_sec:
                print("[STT] 🤫 3초 이상 조용해서 녹음을 종료합니다.")
                break

            if (now - start_time) >= self.max_total_sec:
                print("[STT] ⏱ 최대 녹음 시간 초과로 종료합니다.")
                break

        audio_all = np.concatenate(chunks, axis=0)
        return audio_all

    def listen_and_transcribe(self) -> str:
        """
        - 마이크에서 streaming으로 음성을 받다가
        - 5초 이상 무음 구간이 나오면 종료
        - Whisper로 전송 후 텍스트 반환
        """
        audio_all = self._record_until_silence()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio_all)
            temp_path = temp_wav.name

        print(f"[STT] 🎧 Whisper로 전송 중... ({temp_path})")

        with open(temp_path, "rb") as f:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )

        text = transcript.text
        print(f"[STT] ✅ 인식 결과: {text}")
        return text


if __name__ == "__main__":
    stt = StreamingSTT()
    msg = stt.listen_and_transcribe()
    print("최종 텍스트:", msg)
