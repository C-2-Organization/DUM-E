# services/audio_io/app/main.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # /home/rokey/DUM-E
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from fastapi import FastAPI, Response
import threading

from .config import MicConfig
from .mic import MicController
from .wakeword import WakeupWord, start_wakeword_loop
from .stt import StreamingSTT
from .tts import TTS

app = FastAPI(title="Dummy Audio IO Service")

mic = MicController(MicConfig())
wake = WakeupWord(mic)
stt = StreamingSTT()
tts = TTS()

wake_thread: threading.Thread | None = None
_last_wakeup_flag = False


def _on_wake_detected():
    """
    wakeword 루프 스레드에서 호출되는 콜백.
    여기서 STT를 동기적으로 실행하면,
    STT 동안 wakeword는 자연스럽게 '일시정지'된 효과가 난다.
    """
    global _last_wakeup_flag
    print("[AudioIO] >>> WAKE WORD DETECTED! STT 시작")
    _last_wakeup_flag = True

    # 1) wakeword가 계속 마이크를 읽고 있으니 잠시 멈추고 싶다면:
    wake.running = False  # wakeword loop 종료

    # 2) STT 실행 (blocking)
    text = stt.listen_and_transcribe()

    # 3) 여기서 LLM 에이전트 호출, 로그 저장 등 추가 작업 가능
    print(f"[AudioIO] 💬 사용자의 발화: {text}")

    # 3) TTS로 그대로 말해주기
    try:
        tts.speak(text)
    except Exception as e:
        print(f"[AudioIO] ❌ TTS 에러: {e}")

    # 4) STT/TTS 끝나면 다시 wakeword 루프 재시작
    wake_thread = threading.Thread(
        target=start_wakeword_loop,
        args=(wake, _on_wake_detected, 0.0),
        daemon=True,
    )
    wake_thread.start()


@app.on_event("startup")
def on_startup():
    global wake_thread
    print("[AudioIO] FastAPI startup")
    mic.open_stream()
    wake.init_model()

    wake_thread = threading.Thread(
        target=start_wakeword_loop,
        args=(wake, _on_wake_detected, 0.0),
        daemon=True,
    )
    wake_thread.start()


@app.on_event("shutdown")
def on_shutdown():
    print("[AudioIO] FastAPI shutdown")
    wake.running = False
    mic.close_stream()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/last_wakeup")
def last_wakeup():
    """
    마지막으로 wakeword가 감지되었는지 확인.
    (아주 단순한 플래그; 나중에는 timestamp나 카운터로 확장 가능)
    """
    global _last_wakeup_flag
    flag = _last_wakeup_flag
    _last_wakeup_flag = False
    return {"detected": flag}


@app.post("/record_wav")
def record_wav():
    """
    config.record_seconds 동안 마이크 녹음해서 WAV 바이너리를 그대로 반환.
    (나중에 STT 서비스에 바로 넘기거나, 디버깅용으로 사용 가능)
    """
    wav_bytes = mic.record_audio()
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="record.wav"'},
    )
