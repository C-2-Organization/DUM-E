# services/audio_io/app/tts.py

import tempfile
from typing import Optional

import numpy as np

from services.audio_io.app.tts_base import BaseTTS

import sounddevice as sd
import scipy.io.wavfile as wav
from openai import OpenAI

from services.common.env_loader import load_env, get_env


class TTS(BaseTTS):
    """
    단순 TTS 래퍼.
    - OpenAI audio.speech API로 텍스트를 음성으로 변환
    - 생성된 WAV 파일을 sounddevice로 재생
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini-tts",
        voice: str = "verse",
        effect: str = "jarvis",  # none / jarvis
    ):
        load_env()
        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in .env")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.voice = voice
        self.effect = effect  # 🔹 효과 모드 (기본: jarvis)

        print(
            f"[TTS] 🔧 기본 설정: model={self.model}, voice={self.voice}, effect={self.effect}"
        )

    def _apply_jarvis_voice(self, data, sr):
        """
        자비스/로봇 분위기 필터 (강화 버전):
        - 저역을 더 강하게 올려서 묵직한 톤
        - 고역을 더 많이 깎아서 메탈릭·통신 느낌
        - 가벼운 bit-crush로 디지털 느낌 추가
        - chorus/delay를 더 키워서 'AI 보이스' 스타일 강화
        """
        # float32로 변환
        x = data.astype(np.float32)

        # ---------------------------
        # 1) FFT 기반 EQ
        # ---------------------------
        X = np.fft.rfft(x, axis=0)
        n = X.shape[0]

        low = int(n * 0.05)   # 저역
        mid = int(n * 0.25)   # 중역
        high = int(n * 0.55)  # 고역 시작 구간 (이전보다 더 낮은 지점부터 깎기)

        # 저역 40% 증가 → 더 무거운 느낌
        X[:low] *= 1.2

        # 중역 약간 보정 → 명료도 유지
        X[low:mid] *= 1.10

        # 고역은 절반 수준까지 감쇄 → 시스텀/전화기 같은 느낌
        X[high:] *= 0.65

        y = np.fft.irfft(X, n=data.shape[0], axis=0)

        # ---------------------------
        # 2) soft clipping + 가벼운 bit-crush
        # ---------------------------
        y = y / 32768.0

        # 살짝 세게 태닝해서 선명도 확보
        y = np.tanh(y * 1.4)

        # bit depth를 약간 줄여서 디지털스러운 질감 추가
        # (너무 심하면 지지직거리니 256~512 정도로만 조정)
        levels = 512.0
        y = np.round(y * levels) / levels

        y = y * 32768.0

        # ---------------------------
        # 3) metallic chorus / short delays
        # ---------------------------
        num_samples = y.shape[0]
        y_mix = y.copy()

        # 더 자비스스럽게: 짧은 딜레이를 여러 개 섞음
        delays_ms = [4, 9, 13]      # 밀리초 단위 딜레이
        gains = [0.18, 0.12, 0.08]  # 각 딜레이 볼륨 비율

        for d_ms, g in zip(delays_ms, gains):
            d_samples = int(sr * d_ms / 1000.0)
            if 0 < d_samples < num_samples:
                delayed = np.zeros_like(y_mix)
                delayed[d_samples:] = y[:-d_samples] * g
                y_mix += delayed

        # ---------------------------
        # 4) 최종 볼륨 조정 + 클리핑
        # ---------------------------
        y_mix *= 0.85

        return np.clip(y_mix, -32768, 32767).astype(np.int16)

    def set_voice(self, voice: str):
        """
        런타임에 보이스를 바꾸고 싶을 때 사용.
        예: tts.set_voice("alloy"), tts.set_voice("onyx")
        """
        print(f"[TTS] 🔄 voice 변경: {self.voice} -> {voice}")
        self.voice = voice

    def speak(self, text: str) -> Optional[str]:
        """
        주어진 텍스트를 TTS로 재생.
        - text가 비어 있으면 아무 것도 하지 않음
        - tmp wav 파일 경로를 리턴 (디버깅/로그용)
        """
        text = text.strip()
        if not text:
            print("[TTS] 빈 텍스트라서 재생을 건너뜁니다.")
            return None

        print(
            f"[TTS] ▶ TTS 시작 (len={len(text)} chars, voice={self.voice}, "
            f"model={self.model}, effect={self.effect})"
        )

        # 임시 wav 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        # OpenAI TTS 호출
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="wav",
        ) as response:
            response.stream_to_file(tmp_path)

        # WAV 로드
        sr, data = wav.read(tmp_path)

        # 자비스 효과 적용
        if self.effect == "jarvis":
            data = self._apply_jarvis_voice(data, sr)

        # 🔹 재생 속도 조절 (자비스 모드일 때만 살짝 느리게)
        if self.effect == "jarvis":
            speed_factor = 0.95  # 0.85 = 15% 느리게 (0.7~0.9 사이에서 취향대로 조정 가능)
            playback_sr = int(sr * speed_factor)
        else:
            playback_sr = sr

        print(
            f"[TTS] 재생 sample_rate={playback_sr}, shape={data.shape}, dtype={data.dtype}"
        )
        sd.play(data, playback_sr)
        sd.wait()

        print("[TTS] ✅ 재생 완료")
        return tmp_path
