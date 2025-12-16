# services/audio_io/app/jarvis_assistant.py

from typing import Optional
from openai import OpenAI

from services.audio_io.app.tts import TTS
from services.common.env_loader import load_env, get_env


class JarvisAssistant:
    """
    Generates Jarvis-style responses and plays them via TTS.
    """

    def __init__(
        self,
        llm_model: str = "gpt-4.1-mini",
        tts: Optional[TTS] = None,
    ):
        load_env()
        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in .env (JarvisAssistant)")

        self.client = OpenAI(api_key=api_key)
        self.llm_model = llm_model
        self.tts = tts or TTS()

    def _build_system_prompt(self) -> str:
        """
        Jarvis-style tone guideline.
        Language follows environment variable JARVIS_LANG (ko/en).
        Not imitating any specific actor or copyrighted character.
        """
        lang = (get_env("JARVIS_LANG") or "ko").lower()
        base = (
            "You are a highly advanced AI assistant reminiscent of a futuristic "
            "scientific support system. Do NOT imitate any copyrighted characters "
            "or real actors. Your tone should be calm, polite, concise, and analytical. "
            "Limit responses to 1–3 sentences unless otherwise required. "
            "Provide clear status, system context, or guidance for the user."
        )
        if lang.startswith("ko"):
            return base + " Always respond in Korean."
        else:
            return base + " Always respond in English."

    def generate_reply(self, user_text: str) -> str:
        """
        Converts user_text into a refined, Jarvis-style response in the configured language.
        """
        user_text = user_text.strip()
        if not user_text:
            return ""

        print(f"[Jarvis] 🧠 LLM Request: {user_text!r}")

        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": self._build_system_prompt(),
                },
                {
                    "role": "user",
                    "content": user_text,
                },
            ],
        )
        answer = resp.choices[0].message.content.strip()
        print(f"[Jarvis] 🧠 LLM Response: {answer!r}")
        return answer

    def reply_and_speak(self, user_text: str) -> str:
        """
        Creates an English Jarvis-style response and plays it via TTS.
        """
        answer = self.generate_reply(user_text)
        if answer:
            try:
                self.tts.speak(answer)
            except Exception as e:
                print(f"[Jarvis] ❌ TTS Error: {e}")
        return answer
