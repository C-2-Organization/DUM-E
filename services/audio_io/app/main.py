# services/audio_io/app/main.py
import sys
from pathlib import Path
import subprocess
import random
import time
import json
import os
import threading

ROOT = Path(__file__).resolve().parents[3]  # /home/rokey/DUM-E
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from fastapi import FastAPI, Response

from .config import MicConfig
from .mic import MicController
from .wakeword import WakeupWord, start_wakeword_loop
from .stt import StreamingSTT
from .jarvis_assistant import JarvisAssistant
from .webcam import capture_webcam_jpeg, jpeg_bytes_to_data_url
from .context_memory import ContextMemory

from services.llm_agent.app.skill_planner import plan_skill_flow, analyze_scene_only
from services.llm_agent.ros_bridge import call_run_skill
from dum_e_interfaces.msg import SkillCommand

try:
    from pynput import keyboard
except ImportError:
    keyboard = None
    print("[AudioIO] ⚠ pynput 미설치 상태입니다. push_to_talk 모드는 동작하지 않을 수 있습니다.")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DUM_E_UI = os.getenv("DUM_E_UI", "0") == "1"
AUDIO_MODE = os.getenv("DUM_E_AUDIO_MODE", "wakeword").lower()
print(f"[AudioIO] 🔧 AUDIO_MODE = {AUDIO_MODE}")
print(f"[AudioIO] 🔧 DUM_E_UI   = {int(DUM_E_UI)}")

# ---------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------
app = FastAPI(title="Dummy Audio IO Service")

# ---------------------------------------------------------------------
# Optional UI (lazy import; DO NOT create QApplication at import time)
# ---------------------------------------------------------------------
ui = None

def _ui_enabled() -> bool:
    return ui is not None

def _ui_set_state(state: str):
    if ui is None:
        return
    try:
        ui.set_state(state)
    except Exception:
        pass

def _ui_on_audio_chunk(samples, sr):
    if ui is None:
        return
    try:
        ui.on_audio_chunk(samples, sr)
    except Exception:
        pass


class _NullUI:
    """Headless 모드에서 ui 호출 안전하게 무시하기 위한 더미"""
    def show(self): pass
    def set_state(self, state: str): pass
    def on_audio_chunk(self, samples, sr): pass
    def run(self): return 0
    def quit(self): pass


# ---------------------------------------------------------------------
# Core components (safe at import time)
# ---------------------------------------------------------------------
mic = MicController(MicConfig())
wake = WakeupWord(mic)
stt = StreamingSTT()

# TTS는 UI 여부에 따라 런타임(=__main__)에서 결정/생성한다.
tts = None  # type: ignore
jarvis = None  # type: ignore

if not DUM_E_UI:
    from .tts import TTS as HeadlessTTS
    tts = HeadlessTTS(model="gpt-4o-mini-tts", voice="onyx", effect="jarvis")
    try:
        tts.set_voice("onyx")
    except Exception:
        pass
    jarvis = JarvisAssistant(tts=tts)

    # headless에서는 ui 더미로
    ui = _NullUI()

# ---------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------
wake_thread: threading.Thread | None = None
_last_wakeup_flag = False
_busy = False
_robot_proc: subprocess.Popen | None = None
_push_to_talk_active = False
_pending_clarify: dict | None = None
ctx_mem = ContextMemory(maxlen=5)

# ---------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------
GREETING_RESPONSES = [
    "Systems online, sir. Standing by for your command.",
    "Initialization complete. Ready when you are, sir.",
    "All systems functional. How may I assist, sir?",
    "Wakeword monitoring activated. I'm here, sir.",
    "Operational and awaiting your direction, sir.",
    "Diagnostics clear. At your service, sir.",
    "Startup sequence complete. Listening now, sir.",
    "Good day, sir. Ready for deployment.",
    "Everything’s set. Please proceed when ready, sir.",
    "Full system readiness achieved. How can I help, sir?",
]

WAKE_RESPONSES = [
    "Yes, sir?",
    "At your service, sir.",
    "How can I assist, sir?",
    "I'm listening, sir.",
    "Ready when you are.",
    "Standing by, sir.",
    "Awaiting your command.",
    "What can I do for you, sir?",
    "Online and attentive, sir.",
    "Yes, I'm here.",
    "Go ahead, sir.",
    "Online and awaiting orders.",
    "Here, sir.",
    "What do you need, sir?",
]

COMMAND_ACK_RESPONSES = [
    "I'm on it, sir.",
    "For you, Sir, Always.",
    "Understood, sir. Executing now.",
    "Right away, sir.",
    "As you command, sir.",
    "Consider it done.",
    "On your order, sir.",
    "Initializing protocol, sir.",
    "Affirmative. Processing.",
    "Certainly, sir. Handling it now.",
    "Your wish is my command.",
    "Acknowledged. Beginning operation.",
    "At your service, sir.",
    "Execution confirmed.",
    "Working on it immediately.",
    "Standing by, action engaged.",
    "Task received. Proceeding.",
    "Always, sir.",
    "Directive accepted. Moving forward.",
    "Command priority elevated. Executing.",
    "Very well, sir. Activating sequence.",
    "All systems aligned. Carrying out your request.",
]

COMPLETE_RESPONSES = [
    "Task completed, sir.",
    "Operation successful. Anything else you require?",
    "The process has finished, sir.",
    "Execution complete. Awaiting further instructions.",
    "Mission accomplished, sir.",
    "Your request has been fulfilled.",
    "All done, sir. Ready for the next task.",
    "The action has been carried out successfully.",
    "Procedure finalized, sir.",
    "Complete. Standing by for your next command.",
]

STOP_KEYWORDS_KO = ["멈춰", "잠깐", "스톱", "그만", "정지", "잠만", "기다려"]
STOP_KEYWORDS_EN = ["stop", "pause", "hold on", "wait"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _is_stop_command(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in STOP_KEYWORDS_EN) or any(k in text for k in STOP_KEYWORDS_KO)

def _is_robot_already_running() -> bool:
    global _robot_proc
    return _robot_proc is not None and _robot_proc.poll() is None

def _launch_robot_bringup() -> bool:
    global _robot_proc
    if _is_robot_already_running():
        print("[AudioIO] 🤖 로봇 bringup 이 이미 실행 중인 것 같아요.")
        return False

    cmd = ["ros2", "launch", "dum_e_bringup", "dum_e_bringup.launch.py"]
    print(f"[AudioIO] 🚀 로봇 bringup 실행: {' '.join(cmd)}")

    try:
        _robot_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print(f"[AudioIO] ❌ ros2 launch 실행 실패: {e}")
        _robot_proc = None
        return False

def _request_skill_stop():
    """현재 실행 중인 스킬/모션 중단 요청"""
    global _pending_clarify
    _pending_clarify = None

    print("[AudioIO] 🛑 STOP 요청을 ROS로 전송합니다.")
    try:
        cmd = [
            "ros2", "topic", "pub",
            "/dum_e_control",
            "std_msgs/String",
            "data: 'stop'",
            "-r", "10",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3.0)
        proc.terminate()
    except Exception as e:
        print(f"[AudioIO] ❌ STOP 호출 중 에러: {e}")
        try:
            tts.speak("I could not send a proper stop command to the robot, sir.")  # type: ignore
        except Exception:
            pass


# ---------------------------------------------------------------------
# Planner/Execution
# ---------------------------------------------------------------------
def _execute_plan(plan: dict) -> bool:
    steps = plan.get("steps") or []
    if not isinstance(steps, list):
        print("[AudioIO] ⚠ plan.steps 가 리스트가 아님:", steps)
        return False

    executed_any = False

    for step in steps:
        skill = (step.get("skill") or "").upper()

        if skill == "ROBOT_WAKEUP":
            print("[AudioIO] 🤖 ROBOT_WAKEUP 스텝 실행 시도")
            started = _launch_robot_bringup()
            try:
                if started:
                    tts.speak("Initializing system.")  # type: ignore
                else:
                    if _is_robot_already_running():
                        tts.speak("Dummy is already running, sir.")  # type: ignore
                    else:
                        tts.speak("There was a problem waking up dummy. Please try again later, sir.")  # type: ignore
            except Exception as e:
                print(f"[AudioIO] ❌ TTS 에러 (ROBOT_WAKEUP): {e}")
            executed_any = True
            continue

        elif skill in ("PICK", "FIND", "PLACE"):
            obj = step.get("object") or {}
            obj_name = obj.get("canonical_en") or obj.get("raw") or ""
            if not obj_name:
                print(f"[AudioIO] ⚠ {skill} 스텝에 object_name 이 없음:", step)
                continue

            params = step.get("params") or {}
            params_json = json.dumps(params, ensure_ascii=False)

            print(f"[AudioIO] 🦾 실행: {skill} '{obj_name}', params={params}")

            try:
                resp = call_run_skill(
                    skill_type=getattr(SkillCommand, skill),
                    object_name=obj_name,
                    target_pose=None,
                    params_json=params_json,
                    timeout_sec=60.0,
                )
            except Exception as e:
                print(f"[AudioIO] ❌ /run_skill 호출 중 에러: {e}")
                return False

            print(
                f"[AudioIO] ✅ /run_skill 응답: success={resp.success}, "
                f"confidence={resp.confidence:.2f}, message='{resp.message}'"
            )
            executed_any = True
            continue

        elif skill in ("HOME", "DROP"):
            try:
                resp = call_run_skill(
                    skill_type=getattr(SkillCommand, skill),
                    object_name="",
                    target_pose=None,
                    params_json={},
                    timeout_sec=30.0,
                )
                print(
                    f"[AudioIO] ✅ /run_skill 응답: success={resp.success}, "
                    f"confidence={resp.confidence:.2f}, message='{resp.message}'"
                )
            except Exception as e:
                print(f"[AudioIO] ❌ /run_skill 호출 중 에러: {e}")
                return False
            executed_any = True
            continue

        else:
            print(f"[AudioIO] ℹ 아직 지원하지 않는 스킬: {skill}")

    return executed_any


def _handle_text_high_priority(user_text: str) -> bool:
    """busy 상태에서도 STOP/간단 chat 같은 것 처리"""
    global _pending_clarify

    if not user_text or not user_text.strip():
        return True

    if _is_stop_command(user_text):
        print("[AudioIO] 🛑 (HP) STOP 계열 명령 감지")
        _request_skill_stop()
        try:
            tts.speak("Stopping, sir.")  # type: ignore
        except Exception:
            pass
        return True

    # busy 중에도 chat은 가능하게(기존 로직 유지)
    try:
        mem = ctx_mem.snapshot()
        plan = plan_skill_flow(user_text, scene_image_url=None, memory_context=mem)

        context_update = (plan.get("context_update") or "").strip()
        if context_update:
            ctx_mem.push(context_update)

        intent = (plan.get("intent") or "").lower().strip()

        if intent == "chat":
            _pending_clarify = None
            reply = (plan.get("chat_reply") or "").strip() or "Understood, sir."
            tts.speak(reply)  # type: ignore
            return True

        mode = (plan.get("command_mode") or "").lower().strip()
        if intent == "command" and mode == "plan":
            _pending_clarify = None
            return False

        if intent == "command" and mode == "clarify":
            q = ((plan.get("clarification") or {}).get("question") or "").strip() or "Could you clarify, sir?"
            tts.speak(q)  # type: ignore
            return True

    except Exception as e:
        print(f"[AudioIO] ⚠ busy chat 처리 실패: {e}")

    return False


def _listen_one_utterance_even_if_busy(preface: str | None = None):
    try:
        if preface:
            try:
                tts.speak(preface)  # type: ignore
                time.sleep(0.2)
            except Exception:
                pass

        user_text = stt.transcribe_once()
        print(f"[AudioIO] (BUSY LISTEN) 🎙 '{user_text}'")

        if _handle_text_high_priority(user_text):
            return

        try:
            tts.speak("I'm currently executing a task, sir. Say 'stop' to interrupt or try again shortly.")  # type: ignore
        except Exception:
            pass

    except Exception as e:
        print(f"[AudioIO] ❌ busy listen 중 에러: {e}")


def _run_single_command_flow(preface_msg: str | None = None, transcribe_fn=None):
    global _busy, _pending_clarify

    if transcribe_fn is None:
        transcribe_fn = stt.transcribe_once

    if _busy:
        print("[AudioIO] ⚠ 이미 명령 처리 중입니다. (일반 명령은 거부, STOP은 허용)")
        try:
            _ui_set_state("listening")
            user_text = transcribe_fn()
            if _handle_text_high_priority(user_text):
                return
        except Exception as e:
            print(f"[AudioIO] ❌ busy 상태 처리 중 에러: {e}")
        return

    _busy = True
    try:
        if preface_msg:
            try:
                print(f"[AudioIO] 💬 Preface: {preface_msg}")
                tts.speak(preface_msg)  # type: ignore
                time.sleep(1.0)
            except Exception as e:
                print(f"[AudioIO] ❌ TTS 에러 (preface): {e}")

        _ui_set_state("listening")

        user_text = transcribe_fn()
        raw_user_text = user_text
        print(f"[AudioIO] 🎙 사용자가 말한 내용: '{user_text}'")

        if not user_text.strip():
            print("[AudioIO] ⚠ STT 결과가 비어있음. 다시 대기.")
            return

        # webcam (optional)
        image_data_url = None
        try:
            jpg_bytes, saved_path = capture_webcam_jpeg()
            image_data_url = jpeg_bytes_to_data_url(jpg_bytes)
            if saved_path:
                print(f"[AudioIO] 📷 Webcam captured: {saved_path}")
        except Exception as e:
            print(f"[AudioIO] ⚠ Webcam capture failed (continue without image): {e}")

        # pending clarify follow-up packaging
        if _pending_clarify is not None:
            age = time.time() - float(_pending_clarify.get("timestamp", 0))
            if age > 60.0:
                _pending_clarify = None
            else:
                user_text = (
                    "FOLLOW-UP ANSWER TO YOUR LAST CLARIFICATION.\n"
                    f"Previous command: {_pending_clarify.get('original_user_text','')}\n"
                    f"You asked: {_pending_clarify.get('question','')}\n"
                    f"User answer: {user_text}\n"
                    "Now infer the full intended robot command and produce the final plan.\n"
                )

        _ui_set_state("thinking")

        # planner
        try:
            mem = ctx_mem.snapshot()
            plan = plan_skill_flow(
                user_text,
                scene_image_url=image_data_url,
                memory_context=mem,
            )

            context_update = (plan.get("context_update") or "").strip()
            if context_update:
                ctx_mem.push(context_update)
                print(f"[AudioIO] 🧠 Context saved ({len(ctx_mem.snapshot())}/5): {context_update}")
            else:
                print("[AudioIO] ⚠ No context_update returned by planner")

            intent = (plan.get("intent") or "").lower().strip()

            if intent == "chat":
                _pending_clarify = None
                reply = (plan.get("chat_reply") or "").strip() or "Understood, sir."
                tts.speak(reply)  # type: ignore
                _ui_set_state("idle")
                return

            mode = (plan.get("command_mode") or "").lower().strip()
            if intent == "command" and mode == "clarify":
                clar = plan.get("clarification") or {}
                _pending_clarify = {
                    "question": (clar.get("question") or "").strip(),
                    "expected_answer_type": clar.get("expected_answer_type"),
                    "choices": clar.get("choices"),
                    "original_user_text": raw_user_text,
                    "scene_summary": (plan.get("scene") or {}).get("summary", ""),
                    "timestamp": time.time(),
                }
                q = _pending_clarify["question"] or "Could you clarify, sir?"
                tts.speak(q)  # type: ignore
                _ui_set_state("idle")
                return

        except Exception as e:
            print(f"[AudioIO] ❌ Planner 에러: {e}")
            try:
                jarvis.reply_and_speak(  # type: ignore
                    "A system issue occurred while organizing the internal task sequence. "
                    "Please apologize to the user in a concise and respectful manner, and inform them to try again shortly."
                )
            except Exception as tts_err:
                print(f"[AudioIO] ❌ TTS 에러: {tts_err}")
            _ui_set_state("idle")
            return

        print("[AudioIO] 🧠 Planner 결과:")
        print(plan)

        can_execute = bool(plan.get("can_execute_now"))
        user_message = plan.get("user_message") or ""

        if not can_execute:
            msg = user_message or "Process execution failed."
            print(f"[AudioIO] ❌ Process execution failed: {msg}")
            try:
                tts.speak(msg)  # type: ignore
            except Exception as e:
                print(f"[AudioIO] ❌ TTS 에러: {e}")
            _ui_set_state("idle")
            return

        ack_msg = random.choice(COMMAND_ACK_RESPONSES)
        print(f"[AudioIO] 💬 Command ack: {ack_msg}")
        tts.speak(ack_msg)  # type: ignore
        time.sleep(1.0)

        executed = _execute_plan(plan)
        if executed:
            complete_msg = random.choice(COMPLETE_RESPONSES)
            print(f"[AudioIO] ✅ Plan execution complete: {complete_msg}")
            tts.speak(complete_msg)  # type: ignore
            time.sleep(0.5)
        else:
            fallback_msg = user_message or "Process execution failed."
            print(f"[AudioIO] ⚠ 계획은 가능하다고 했지만 실제 실행 실패: {fallback_msg}")

        _ui_set_state("idle")

    finally:
        _busy = False


def _on_wake_detected(keyword: str):
    global _last_wakeup_flag

    _ui_set_state("listening")

    print(f"[AudioIO] >>> WAKE WORD DETECTED! ({keyword}) STT 시작")
    _last_wakeup_flag = True

    wake_msg = random.choice(WAKE_RESPONSES)

    if _busy:
        print("[AudioIO] (WAKE) busy 상태에서도 1회 명령 청취")
        _listen_one_utterance_even_if_busy(preface=wake_msg)
        return

    _run_single_command_flow(preface_msg=wake_msg)


def _on_space_pressed():
    print("[AudioIO] ⌨ Space pressed → push-to-talk command flow 시작")

    def is_active():
        return _push_to_talk_active

    _ui_set_state("listening")

    user_text = stt.transcribe_while(is_active)
    print(f"[AudioIO] (PTT) 🎙 사용자가 말한 내용: '{user_text}'")

    if not user_text.strip():
        print("[AudioIO] (PTT) ⚠ STT 결과가 비어있음. 무시.")
        _ui_set_state("idle")
        return

    if _is_stop_command(user_text):
        print("[AudioIO] (PTT) 🛑 STOP 계열 명령 감지")
        _request_skill_stop()
        _ui_set_state("idle")
        return

    def _return_existing_text():
        return user_text

    _run_single_command_flow(preface_msg=None, transcribe_fn=_return_existing_text)


def _start_push_to_talk_loop():
    global _push_to_talk_active

    if keyboard is None:
        print("[AudioIO] ❌ pynput 모듈이 없어 push_to_talk 모드를 사용할 수 없습니다.")
        return

    def on_press(key):
        global _push_to_talk_active
        try:
            if key == keyboard.Key.space:
                if not _push_to_talk_active:
                    _push_to_talk_active = True
                    threading.Thread(target=_on_space_pressed, daemon=True).start()
        except Exception as e:
            print(f"[AudioIO] ⚠ on_press 에러: {e}")

    def on_release(key):
        global _push_to_talk_active
        try:
            if key == keyboard.Key.space:
                _push_to_talk_active = False
        except Exception as e:
            print(f"[AudioIO] ⚠ on_release 에러: {e}")

    print("[AudioIO] ⌨ push_to_talk 키 리스너 시작 (space 키)")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


# ---------------------------------------------------------------------
# FastAPI lifecycle
# ---------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    global wake_thread
    print("[AudioIO] FastAPI startup")

    mic.open_stream()

    mode = AUDIO_MODE
    enable_wake = mode in ("wakeword", "hybrid", "both")
    enable_ptt = mode in ("push_to_talk", "hybrid", "both")

    if enable_wake:
        wake.init_model()
        wake_thread = threading.Thread(
            target=start_wakeword_loop,
            args=(wake, _on_wake_detected, 0.0),
            daemon=True,
        )
        wake_thread.start()
        print("[AudioIO] ✅ Wakeword loop started")

    if enable_ptt:
        pt_thread = threading.Thread(target=_start_push_to_talk_loop, daemon=True)
        pt_thread.start()
        print("[AudioIO] ✅ Push-to-talk loop started (space key)")

    # greeting
    if enable_wake and enable_ptt:
        greeting_msg = "Systems online, sir."
    elif enable_wake:
        greeting_msg = random.choice(GREETING_RESPONSES)
    elif enable_ptt:
        greeting_msg = "Systems online, sir. Push and hold the space bar to issue a command."
    else:
        greeting_msg = random.choice(GREETING_RESPONSES)

    print(f"[AudioIO] 💬 Greeting: {greeting_msg}")
    try:
        tts.speak(greeting_msg)  # type: ignore
    except Exception:
        pass


@app.on_event("shutdown")
def on_shutdown():
    print("[AudioIO] FastAPI shutdown")
    wake.running = False
    mic.close_stream()


# ---------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/last_wakeup")
def last_wakeup():
    global _last_wakeup_flag
    flag = _last_wakeup_flag
    _last_wakeup_flag = False
    return {"detected": flag}


@app.post("/record_wav")
def record_wav():
    wav_bytes = mic.record_audio()
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="record.wav"'},
    )


@app.get("/webcam_snapshot")
def webcam_snapshot():
    try:
        jpg_bytes, _ = capture_webcam_jpeg()
        return Response(content=jpg_bytes, media_type="image/jpeg")
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/scene_probe")
def scene_probe():
    jpg_bytes, saved_path = capture_webcam_jpeg()
    img_url = jpeg_bytes_to_data_url(jpg_bytes)

    result = analyze_scene_only(img_url)
    result["_debug"] = {"saved_path": saved_path}
    return result


# ---------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------
def run_server():
    import uvicorn
    # ✅ IMPORTANT: pass the app object directly (avoid re-importing this module)
    uvicorn.run(app, host="0.0.0.0", port=9000, reload=False, log_level="info")


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # TTS / UI setup must happen ONLY here (not at import time)
    if DUM_E_UI:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer
        from services.audio_io.app.jarvis_hud import JarvisHUD
        from services.audio_io.app.tts_streaming import StreamingTTS

        class UIRunner:
            def __init__(self):
                self.qt_app = QApplication(sys.argv)
                self.hud = JarvisHUD()

                self._sig_timer = QTimer()
                self._sig_timer.timeout.connect(lambda: None)
                self._sig_timer.start(500)

            def show(self):
                self.hud.show()

            def set_state(self, state: str):
                try:
                    self.hud.set_state(state)
                except Exception:
                    pass

            def on_audio_chunk(self, samples, sr):
                try:
                    self.hud.on_audio_chunk(samples, sr)
                except Exception:
                    pass

            def run(self):
                return self.qt_app.exec()

            def quit(self):
                try:
                    QTimer.singleShot(0, self.qt_app.quit)
                except Exception:
                    pass

        ui = UIRunner()
        ui.show()

        # StreamingTTS + HUD
        tts = StreamingTTS(
            model="gpt-4o-mini-tts",
            voice="onyx",
            effect="jarvis",
            chunk_size=2048,
        )
        tts.set_audio_callback(lambda samples, sr: _ui_on_audio_chunk(samples, sr))
        tts.set_speaking_callbacks(
            on_start=lambda: _ui_set_state("speaking"),
            on_end=lambda: _ui_set_state("idle"),
        )

        jarvis = JarvisAssistant(tts=tts)

    else:
        # Headless TTS
        from .tts import TTS as HeadlessTTS
        tts = HeadlessTTS(model="gpt-4o-mini-tts", voice="onyx", effect="jarvis")
        try:
            tts.set_voice("onyx")
        except Exception:
            pass
        ui = _NullUI()

    # 서버는 백그라운드, UI는 메인스레드(있을 때)
    th = threading.Thread(target=run_server, daemon=True)
    th.start()

    if DUM_E_UI:
        sys.exit(ui.run())
    else:
        # headless는 메인스레드에서 대기만 (서버 thread가 돌아감)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
