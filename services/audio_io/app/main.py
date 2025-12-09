# services/audio_io/app/main.py

from fastapi import FastAPI, Response
import threading

from .config import MicConfig
from .mic import MicController
from .wakeword import WakeupWord, start_wakeword_loop

app = FastAPI(title="Dummy Audio IO Service")

mic = MicController(MicConfig())
wake = WakeupWord(mic)

wake_thread: threading.Thread | None = None
_last_wakeup_flag = False


def _on_wake_detected():
    global _last_wakeup_flag
    print("[AudioIO] >>> WAKE WORD DETECTED!")
    _last_wakeup_flag = True


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
