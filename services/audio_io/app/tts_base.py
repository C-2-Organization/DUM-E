# services/audio_io/app/tts_base.py

from typing import Optional

class BaseTTS:
    """
    모든 TTS 백엔드(OpenAI, Mureka)가 따라야 하는 공통 인터페이스.
    Dum-E / Jarvis는 이 클래스를 기준으로 작동함.
    """
    def speak(self, text: str) -> Optional[str]:
        raise NotImplementedError("speak() must be implemented by subclass")
