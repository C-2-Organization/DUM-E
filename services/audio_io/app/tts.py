# services/audio_io/app/tts.py

import tempfile
from typing import Optional

import sounddevice as sd
import scipy.io.wavfile as wav
from openai import OpenAI

from services.common.env_loader import load_env, get_env


class TTS:
    """
    단순 TTS 래퍼.
    - OpenAI audio.speech API로 텍스트를 음성으로 변환
    - 생성된 WAV 파일을 sounddevice로 재생
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini-tts",   # 또는 "tts-1"
        voice: str = "alloy",
    ):
        load_env()
        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in .env")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.voice = voice

    def speak(self, text: str) -> Optional[str]:
        """
        주어진 텍스트를 TTS로 재생.
        - text가 비어 있으면 아무 것도 하지 않음
        - tmp wav 파일 경로를 리턴 (디버깅용)
        """
        text = text.strip()
        if not text:
            print("[TTS] 빈 텍스트라서 재생을 건너뜁니다.")
            return None

        print(f"[TTS] ▶ TTS 시작 (len={len(text)} chars)")

        # 임시 wav 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        # OpenAI TTS 호출
        # 참고: audio/speech endpoint, response_format="wav"
        # 공식 가이드: https://platform.openai.com/docs/guides/text-to-speech :contentReference[oaicite:0]{index=0}
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="wav",   # wav로 받아서 바로 재생
        ) as response:
            response.stream_to_file(tmp_path)

        # WAV 로드 & 재생
        sr, data = wav.read(tmp_path)
        print(f"[TTS] 재생 sample_rate={sr}, shape={data.shape}, dtype={data.dtype}")
        sd.play(data, sr)
        sd.wait()  # 재생 끝날 때까지 블록

        print("[TTS] ✅ 재생 완료")
        return tmp_path
