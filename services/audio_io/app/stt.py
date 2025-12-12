# services/audio_io/app/stt.py

import time
import tempfile
from typing import Optional
import webrtcvad

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
    - 'silence_sec 이상' 음성이 없으면 녹음을 종료하고 Whisper로 전송
    - WebRTC VAD + Noise Gate + Adaptive Threshold 적용
    """

    def __init__(
        self,
        samplerate: int = 16000,
        chunk_duration: float = 0.5,   # 한 번에 0.5초씩 읽기
        silence_sec: float = 2.0,      # 2초 이상 조용하면 종료
        max_total_sec: float = 60.0,   # 안전장치: 최대 60초까지만 듣기
        energy_threshold: float = 200, # 이 값 이상이면 '사람이 말하는 중'이라고 간주
    ):
        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in .env (StreamingSTT)")

        self.client = OpenAI(api_key=api_key)
        self.samplerate = samplerate
        self.chunk_duration = chunk_duration
        self.silence_sec = silence_sec
        self.max_total_sec = max_total_sec
        self.energy_threshold = energy_threshold

        # 🔥 추가: WebRTC VAD + ambient 에너지 추정용
        self.vad = webrtcvad.Vad(2)  # 0~3, 클수록 더 aggressive
        self.ambient_energy: float | None = None

        print(
            f"[STT] 🔧 Initialized: samplerate={self.samplerate}, "
            f"chunk_duration={self.chunk_duration}, silence_sec={self.silence_sec}, "
            f"max_total_sec={self.max_total_sec}, energy_threshold={self.energy_threshold}"
        )

    def _record_until_silence(self) -> np.ndarray:
        """
        sounddevice로 마이크를 조금씩 읽으면서
        - 초기에는 ambient noise를 측정해서 adaptive threshold 설정
        - WebRTC VAD + 에너지를 동시에 만족하는 chunk만 '말하는 중'으로 간주
        - 그 이후로 silence_sec 이상 조용하면 종료
        """
        print("[STT] 🎙 녹음을 시작합니다. 말이 끊기면 자동으로 종료됩니다.")

        num_samples_per_chunk = int(self.samplerate * self.chunk_duration)
        vad_frame_ms = 20  # WebRTC VAD 허용: 10 / 20 / 30 ms
        vad_frame_len = int(self.samplerate * vad_frame_ms / 1000)  # 20ms → 320 샘플

        chunks: list[np.ndarray] = []
        start_time = time.time()
        last_voice_time = time.time()
        heard_voice = False

        # ambient noise 추정용
        ambient_samples: list[float] = []
        ambient_collect_sec = 1.0  # 처음 1초 정도는 주변 소음 기준 잡기
        ambient_end_time = start_time + ambient_collect_sec

        while True:
            # 1) 마이크에서 chunk_duration 만큼 읽기
            audio_block = sd.rec(
                num_samples_per_chunk,
                samplerate=self.samplerate,
                channels=1,
                dtype="int16",
            )
            sd.wait()

            # shape (N, 1) → (N,)
            audio_block = audio_block.reshape(-1)

            # 2) 이 chunk의 에너지 계산
            block_energy = float(np.abs(audio_block).mean())
            now = time.time()

            # --- ambient noise 업데이트 (처음 일정 시간 동안) ---
            if self.ambient_energy is None:
                ambient_samples.append(block_energy)
                if now >= ambient_end_time and ambient_samples:
                    self.ambient_energy = float(np.mean(ambient_samples))
                    print(f"[STT] 🌡 ambient_energy 추정: {self.ambient_energy:.2f}")
            ambient = self.ambient_energy or block_energy

            # Adaptive threshold: 주변 소음에 비례해서 가중
            adaptive_threshold = max(self.energy_threshold, ambient * 2.0)

            print(
                f"[STT] 🔊 block_energy={block_energy:.2f}, "
                f"ambient={ambient:.2f}, adaptive_th={adaptive_threshold:.2f}"
            )

            # 3) 전체 녹음 버퍼에는 계속 추가 (앞뒤 약간의 무음 포함용)
            chunks.append(audio_block.copy())

            # 4) 이 chunk 안에서 VAD 프레임 단위로 '말하는 구간' 비율 계산
            num_frames = len(audio_block) // vad_frame_len
            if num_frames <= 0:
                speech_ratio = 0.0
            else:
                speech_frames = 0
                for i in range(num_frames):
                    frame = audio_block[i * vad_frame_len : (i + 1) * vad_frame_len]
                    # WebRTC VAD는 16bit PCM mono bytes 입력
                    if self.vad.is_speech(frame.tobytes(), self.samplerate):
                        speech_frames += 1
                speech_ratio = speech_frames / float(num_frames)

            print(f"[STT] 🗣 VAD speech_ratio={speech_ratio:.2f}")

            # 5) noise gate + VAD 동시 조건
            is_speech_block = (
                block_energy > adaptive_threshold and speech_ratio > 0.3
            )

            if is_speech_block:
                heard_voice = True
                last_voice_time = now

            # 6) 사람이 한 번이라도 말한 이후 + silence_sec 이상 조용하면 종료
            if heard_voice and (now - last_voice_time) >= self.silence_sec:
                print(f"[STT] 🤫 {self.silence_sec}초 이상 조용해서 녹음을 종료합니다.")
                break

            # 7) 안전 장치: 전체 최대 길이
            if (now - start_time) >= self.max_total_sec:
                print("[STT] ⏱ 최대 녹음 시간 초과로 종료합니다.")
                break

        if not chunks:
            print("[STT] ⚠ 녹음된 chunk가 없습니다.")
            return np.zeros((0,), dtype=np.int16)

        audio_all = np.concatenate(chunks, axis=0)

        # --- 전체 구간에 대해 '진짜 말이 거의 없으면' 그냥 빈 배열 반환 (잡음만 있는 경우) ---
        total_frames = len(audio_all) // vad_frame_len
        if total_frames > 0:
            total_speech_frames = 0
            for i in range(total_frames):
                frame = audio_all[i * vad_frame_len : (i + 1) * vad_frame_len]
                if self.vad.is_speech(frame.tobytes(), self.samplerate):
                    total_speech_frames += 1
            total_speech_ratio = total_speech_frames / float(total_frames)
        else:
            total_speech_ratio = 0.0

        print(f"[STT] 📊 전체 total_speech_ratio={total_speech_ratio:.2f}")

        if total_speech_ratio < 0.1:
            print("[STT] ⚠ 음성 비율이 너무 낮아서 '말이 없는 잡음'으로 간주합니다.")
            return np.zeros((0,), dtype=np.int16)

        return audio_all

    def listen_and_transcribe(self) -> str:
        """
        - 마이크에서 streaming으로 음성을 받다가
        - silence_sec 이상 무음 구간이 나오면 종료
        - Whisper로 전송 후 텍스트 반환
        - 잡음만 있을 경우 빈 문자열 반환
        """
        audio_all = self._record_until_silence()

        # 🔥 유효한 음성이 없으면 바로 빈 문자열 리턴
        if audio_all.size == 0:
            print("[STT] ⚠ 유효한 음성이 없어서 빈 결과를 반환합니다.")
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio_all)
            temp_path = temp_wav.name

        print(f"[STT] 🎧 Whisper로 전송 중... ({temp_path})")

        # 🔥 Whisper에 prompt 추가: 로봇 명령어 환경이라고 힌트 주기
        with open(temp_path, "rb") as f:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                prompt=(
                    "This audio comes from a robot command environment. "
                    "Ignore background noise, random conversations, and TV or music. "
                    "Only transcribe clear commands or questions addressed to the robot, "
                    "in Korean or English. If there is no clear speech, return an empty result."
                ),
            )

        text = transcript.text.strip()
        print(f"[STT] ✅ 인식 결과: {text!r}")
        return text

    # 🔥 main.py 호환용: 기존 구조를 유지하기 위해 추가
    def transcribe_once(self) -> str:
        """
        main.py에서 사용하는 API.
        내부적으로 listen_and_transcribe()를 그대로 호출한다.
        """
        return self.listen_and_transcribe()


if __name__ == "__main__":
    stt = StreamingSTT()
    msg = stt.listen_and_transcribe()
    print("최종 텍스트:", msg)
