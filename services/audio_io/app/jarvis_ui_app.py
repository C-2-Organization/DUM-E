#!/usr/bin/env python3
# services/audio_io/app/jarvis_ui_app.py
"""
Jarvis HUD + 전체 음성 파이프라인 통합
- Wakeword → STT → LLM → TTS
- PySide6 HUD에 실시간 상태/파형 표시
"""

import sys
import os
import threading
import time
import signal
from pathlib import Path

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from services.audio_io.app.jarvis_hud import JarvisHUD
from services.audio_io.app.tts_streaming import StreamingTTS
from services.audio_io.app.wakeword import WakeupWord, start_wakeword_loop
from services.audio_io.app.stt import StreamingSTT
from services.audio_io.app.mic import MicController
from services.audio_io.app.config import MicConfig
from services.audio_io.app.jarvis_assistant import JarvisAssistant
from services.common.env_loader import load_env


class JarvisApp:
    """
    Jarvis 통합 애플리케이션
    - HUD UI
    - Wakeword
    - STT
    - LLM (JarvisAssistant)
    - TTS (StreamingTTS)
    """

    def __init__(self):
        load_env()

        # Qt Application
        self.qt_app = QApplication(sys.argv)

        # HUD
        self.hud = JarvisHUD()

        # 마이크
        self.mic = MicController(MicConfig())

        # TTS (스트리밍)
        self.tts = StreamingTTS(
            model="gpt-4o-mini-tts",
            voice="onyx",
            effect="jarvis",
            chunk_size=2048,
        )

        # TTS → HUD 콜백
        self.tts.set_audio_callback(self.on_audio_chunk)
        self.tts.set_speaking_callbacks(
            on_start=lambda: self.hud.set_state("speaking"),
            on_end=lambda: self.hud.set_state("idle"),
        )

        # Jarvis Assistant (LLM)
        self.jarvis = JarvisAssistant(tts=None)  # TTS는 별도 관리

        # STT
        self.stt = StreamingSTT()

        # Wakeword
        self.wake = WakeupWord(self.mic)

        # 상태
        self.busy = False
        self.wake_thread = None
        self.conversation_mode = False  # 연속 대화 모드
        self.conversation_timeout = 10  # 초 (무응답 시 자동 종료)

        print("[JarvisApp] 초기화 완료")

    def on_audio_chunk(self, samples, sr):
        """TTS PCM 샘플 → HUD"""
        self.hud.on_audio_chunk(samples, sr)

    def start(self):
        """애플리케이션 시작"""
        # Ctrl+C 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Qt가 Python 시그널을 처리할 수 있도록 타이머 설정
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        timer.start(500)  # 0.5초마다 Python 인터프리터 실행
        
        self.hud.show()
        # 시작 인사 제거 (wakeword 이후에만 말하도록)
        # 필요하면 JARVIS_GREETING=1 설정 시에만 활성화
        if os.getenv("JARVIS_GREETING", "0") == "1":
            lang = os.getenv("JARVIS_LANG", "ko")
            msg = (
                "모든 시스템 정상입니다. 어떻게 도와드릴까요?"
                if lang == "ko"
                else "All systems functional. How may I assist, sir?"
            )
            self._speak(msg)

        # 웨이크워드 루프 시작
        self.start_wakeword_loop()

        # Qt 이벤트 루프
        sys.exit(self.qt_app.exec())

    def _signal_handler(self, signum, frame):
        """Ctrl+C 시그널 핸들러"""
        print("\n[JarvisApp] 종료 시그널 수신, 정리 중...")
        self.shutdown()

    def shutdown(self):
        """전체 애플리케이션 종료"""
        try:
            print("[JarvisApp] 종료 시작...")
            self.conversation_mode = False
            self.busy = False
            
            # 웨이크워드 루프 정지
            try:
                self.wake.running = False
            except Exception as e:
                print(f"[JarvisApp] Wake stop error: {e}")
            
            # 웨이크워드 스레드 대기
            if self.wake_thread and self.wake_thread.is_alive():
                try:
                    self.wake_thread.join(timeout=1.0)
                except Exception as e:
                    print(f"[JarvisApp] Wake thread join error: {e}")
            
            # 마이크 정지
            try:
                self.mic.close_stream()
            except Exception as e:
                print(f"[JarvisApp] Mic close error: {e}")
            
            # TTS 정지
            try:
                self.tts.stop()
            except Exception as e:
                print(f"[JarvisApp] TTS stop error: {e}")
            
            # HUD 종료
            try:
                self.hud.set_state("idle")
                self.hud.close()
            except Exception as e:
                print(f"[JarvisApp] HUD close error: {e}")
            
            # Qt 종료 요청
            try:
                QTimer.singleShot(0, self.qt_app.quit)
            except Exception as e:
                print(f"[JarvisApp] Qt quit error: {e}")
            
            print("[JarvisApp] 종료 완료")
        except Exception as e:
            print(f"[JarvisApp] Shutdown error: {e}")
        finally:
            # 강제 종료 (1초 후)
            QTimer.singleShot(1000, lambda: sys.exit(0))

    def _speak(self, text: str):
        """TTS 재생 (별도 스레드)"""
        def _thread():
            # 새 재생 전에 이전 재생 중단
            self.tts.stop()
            self.tts.speak(text)

        threading.Thread(target=_thread, daemon=True).start()

    def start_wakeword_loop(self):
        """웨이크워드 감지 루프"""
        if self.wake_thread and self.wake_thread.is_alive():
            print("[JarvisApp] 웨이크워드 루프 이미 실행 중")
            return

        def loop():
            try:
                self.wake.init_model()
                print("[JarvisApp] 웨이크워드 루프 시작")
                start_wakeword_loop(self.wake, on_detect=self.on_wakeword_detected, poll_interval=0.0)
            except Exception as e:
                print(f"[JarvisApp] 웨이크워드 루프 오류: {e}")

        self.wake_thread = threading.Thread(target=loop, daemon=True)
        self.wake_thread.start()

    def on_wakeword_detected(self, keyword: str):
        """웨이크워드 감지 → 연속 대화 모드 진입"""
        if self.busy:
            print("[JarvisApp] 이미 처리 중, 웨이크워드 무시")
            return

        self.busy = True
        self.conversation_mode = True
        print(f"[JarvisApp] 🎤 연속 대화 모드 시작 (keyword: {keyword})")

        try:
            # 초기 응답 (한국어/영어)
            lang = os.getenv("JARVIS_LANG", "ko")
            ack_msg = "네, 말씀하세요." if lang == "ko" else "Yes, sir?"
            
            self.hud.set_state("listening")
            self._speak(ack_msg)
            time.sleep(1.5)

            # 연속 대화 루프
            while self.conversation_mode:
                # STT
                self.hud.set_state("listening")
                print("[JarvisApp] STT 대기 중...")
                
                user_text = self.stt.listen_and_transcribe()

                if not user_text or user_text.strip() == "":
                    print("[JarvisApp] 무응답 - 대화 종료")
                    lang = os.getenv("JARVIS_LANG", "ko")
                    bye_msg = "대기 모드로 돌아갑니다." if lang == "ko" else "Returning to standby, sir."
                    self._speak(bye_msg)
                    break

                print(f"[JarvisApp] 👤 User: {user_text}")

                # 종료 명령 체크
                if self._is_exit_command(user_text):
                    print("[JarvisApp] 종료 명령 감지 (종료 기능 비활성화: 대기 모드로 전환)")
                    lang = os.getenv("JARVIS_LANG", "ko")
                    msg = "대기 모드로 전환합니다." if lang == "ko" else "Switching to standby."
                    self._speak(msg)
                    # 대화 루프만 종료하여 정상 대기 상태로 복귀
                    break

                # LLM
                self.hud.set_state("thinking")
                print("[JarvisApp] LLM 처리 중...")
                response = self.jarvis.generate_reply(user_text)

                print(f"[JarvisApp] 🤖 Jarvis: {response}")

                # TTS
                self._speak(response)
                time.sleep(1.0)  # 잠깐 대기 후 다시 듣기

        except Exception as e:
            print(f"[JarvisApp] 오류: {e}")
            self.hud.set_state("idle")
        finally:
            self.conversation_mode = False
            self.busy = False
            self.hud.set_state("idle")
            print("[JarvisApp] 연속 대화 모드 종료")

    def _is_exit_command(self, text: str) -> bool:
        """대화 종료 명령어 체크 (한국어/영어)"""
        exit_keywords = [
            # 영어
            "that's all", "that is all",
            # 한국어
            "그만", "끝", "됐어", "됐습니다",
        ]
        text_lower = text.lower().strip()
        return any(kw in text_lower for kw in exit_keywords)


def main():
    """메인 엔트리포인트"""
    try:
        app = JarvisApp()
        app.start()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt - 종료")
        sys.exit(0)
    except Exception as e:
        print(f"[Main] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
