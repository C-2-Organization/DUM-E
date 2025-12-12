# services/audio_io/app/main.py
import sys
from pathlib import Path
import subprocess
import random
import time
import json
import os

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
from .jarvis_assistant import JarvisAssistant

from services.llm_agent.app.skill_planner import plan_skill_flow
from services.llm_agent.ros_bridge import call_run_skill
from dum_e_interfaces.msg import SkillCommand

try:
    from pynput import keyboard
except ImportError:
    keyboard = None
    print("[AudioIO] ⚠ pynput 미설치 상태입니다. push_to_talk 모드는 동작하지 않을 수 있습니다.")

AUDIO_MODE = os.getenv("DUM_E_AUDIO_MODE", "wakeword").lower()
print(f"[AudioIO] 🔧 AUDIO_MODE = {AUDIO_MODE}")

app = FastAPI(title="Dummy Audio IO Service")

mic = MicController(MicConfig())
wake = WakeupWord(mic)
stt = StreamingSTT()
tts = TTS(
    model="gpt-4o-mini-tts",  # 기본값이라 사실 안 써도 되지만 명시해둘게
    voice="onyx",             # 제일 저음 보이스
    effect="jarvis",          # 기계음 + 자비스 느낌 DSP 필터 ON
)
tts.set_voice("onyx")   # 시작할 때 한 번만 호출해도 됨
jarvis = JarvisAssistant(tts=tts)

wake_thread: threading.Thread | None = None
_last_wakeup_flag = False
_busy = False

_robot_proc: subprocess.Popen | None = None

_push_to_talk_active = False

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

def _is_robot_already_running() -> bool:
    """
    이미 ros2 launch가 떠 있는지 간단히 체크.
    """
    global _robot_proc
    return _robot_proc is not None and _robot_proc.poll() is None

def _launch_robot_bringup() -> bool:
    """
    ros2 launch dum_e_bringup dum_e_bringup.launch.py 를 백그라운드로 실행.
    성공적으로 프로세스를 띄우면 True, 실패하면 False.
    """
    global _robot_proc

    if _is_robot_already_running():
        print("[AudioIO] 🤖 로봇 bringup 이 이미 실행 중인 것 같아요.")
        return False

    cmd = ["ros2", "launch", "dum_e_bringup", "dum_e_bringup.launch.py"]
    print(f"[AudioIO] 🚀 로봇 bringup 실행: {' '.join(cmd)}")

    try:
        # stdout/stderr는 필요하면 로그 파일로 돌려도 됨
        _robot_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except Exception as e:
        print(f"[AudioIO] ❌ ros2 launch 실행 실패: {e}")
        _robot_proc = None
        return False

def _execute_plan(plan: dict) -> bool:
    """
    planner가 만들어준 JSON(plan)을 보고 실제 ROS 스킬을 실행한다.

    - 성공적으로 지원 가능한 스킬을 하나라도 실행하면 True
    - 아무 것도 실행하지 못하면 False
    """
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
                    tts.speak("Waking up dummy, sir.")
                else:
                    if _is_robot_already_running():
                        tts.speak("Dummy is already running, sir.")
                    else:
                        tts.speak("There was a problem waking up dummy. Please try again later, sir.")
            except Exception as e:
                print(f"[AudioIO] ❌ TTS 에러 (ROBOT_WAKEUP): {e}")
            executed_any = True
            continue

        elif skill == "PICK":
            obj = step.get("object") or {}
            obj_name = obj.get("canonical_en") or obj.get("raw") or ""
            if not obj_name:
                print("[AudioIO] ⚠ PICK 스텝에 object_name 이 없음:", step)
                continue

            params = step.get("params") or {}
            params_json = json.dumps(params, ensure_ascii=False)

            print(f"[AudioIO] 🦾 실행: PICK '{obj_name}', params={params}")

            try:
                resp = call_run_skill(
                    skill_type=SkillCommand.PICK,
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

        elif skill == "FIND":
            obj = step.get("object") or {}
            obj_name = obj.get("canonical_en") or obj.get("raw") or ""
            if not obj_name:
                print("[AudioIO] ⚠ FIND 스텝에 object_name 이 없음:", step)
                continue

            params = step.get("params") or {}
            params_json = json.dumps(params, ensure_ascii=False)

            print(f"[AudioIO] 🦾 실행: FIND '{obj_name}', params={params}")

            try:
                resp = call_run_skill(
                    skill_type=SkillCommand.FIND,
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

        else:
            print(f"[AudioIO] ℹ 아직 지원하지 않는 스킬: {skill}")

    return executed_any

def _run_single_command_flow(
    preface_msg: str | None = None,
    transcribe_fn=None,
):
    """
    - (선택) preface_msg 를 먼저 TTS로 말해주고
    - STT 1회 → planner → ROS 실행까지 한 번에 처리.
    - transcribe_fn 이 None이면 기본적으로 stt.transcribe_once() 사용.
    """
    global _busy

    if transcribe_fn is None:
        transcribe_fn = stt.transcribe_once

    if _busy:
        print("[AudioIO] ⚠ 이미 명령 처리 중입니다. 이번 호출은 무시합니다.")
        return

    _busy = True
    try:
        if preface_msg:
            try:
                print(f"[AudioIO] 💬 Preface: {preface_msg}")
                tts.speak(preface_msg)
                time.sleep(1.0)
            except Exception as e:
                print(f"[AudioIO] ❌ TTS 에러 (preface): {e}")

        # 1) STT 실행 (blocking)
        user_text = transcribe_fn()
        print(f"[AudioIO] 🎙 사용자가 말한 내용: '{user_text}'")

        if not user_text.strip():
            print("[AudioIO] ⚠ STT 결과가 비어있음. 다시 대기.")
            return

        ack_msg = random.choice(COMMAND_ACK_RESPONSES)
        print(f"[AudioIO] 💬 Command ack: {ack_msg}")
        tts.speak(ack_msg)
        time.sleep(1.0)

        # 2) Planner 호출: 자연어 → 스킬 플로우(JSON)
        try:
            plan = plan_skill_flow(user_text)
        except Exception as e:
            print(f"[AudioIO] ❌ Planner 에러: {e}")
            try:
                # 자비스 스타일로 사과 + 재시도 안내
                jarvis.reply_and_speak(
                    "A system issue occurred while organizing the internal task sequence. "
                    "Please apologize to the user in a concise and respectful manner, and inform them to try again shortly."
                )
            except Exception as tts_err:
                print(f"[AudioIO] ❌ TTS 에러: {tts_err}")
            return

        print("[AudioIO] 🧠 Planner 결과:")
        print(plan)

        can_execute = bool(plan.get("can_execute_now"))
        user_message = plan.get("user_message") or ""

        if not can_execute:
            # 3-A) 현재 스킬셋으로는 수행 불가능한 명령
            msg = user_message or "Process execution failed."
            print(f"[AudioIO] ❌ Process execution failed: {msg}")

            try:
                tts.speak(msg)
            except Exception as e:
                print(f"[AudioIO] ❌ TTS 에러: {e}")

        else:
            # 3-B) 수행 가능한 경우 → 실제 ROS 스킬 실행
            executed = _execute_plan(plan)

            if not executed:
                # 계획 상으로는 can_execute_now=True 인데,
                # 우리가 실제로 지원하는 스킬이 없거나 실행 실패한 경우
                fallback_msg = (
                    user_message
                    or "Process execution failed."
                )
                print(f"[AudioIO] ⚠ 계획은 가능하다고 했지만 실제 실행 실패: {fallback_msg}")
                try:
                    tts.speak(fallback_msg)
                except Exception as e:
                    print(f"[AudioIO] ❌ TTS 에러: {e}")
            else:
                complete_msg = random.choice(COMPLETE_RESPONSES)
                print(f"[AudioIO] ✅ Plan execution complete: {complete_msg}")
                tts.speak(complete_msg)
                time.sleep(0.5)

    finally:
        _busy = False

def _on_wake_detected(keyword: str):
    """
    wakeword 루프 스레드에서 호출되는 콜백.
    여기서 STT를 동기적으로 실행하고,
    플래너 → ROS 실행까지 처리한다.
    """
    global _last_wakeup_flag

    print(f"[AudioIO] >>> WAKE WORD DETECTED! ({keyword}) STT 시작")
    _last_wakeup_flag = True

    wake_msg = random.choice(WAKE_RESPONSES)
    _run_single_command_flow(preface_msg=wake_msg)

def _on_space_pressed():
    """
    스페이스 키를 눌렀을 때 한 번의 명령을 처리.
    - _push_to_talk_active 가 True인 동안만 STT 녹음
    - 키를 떼면 녹음 종료 후 Whisper 전송
    """
    print("[AudioIO] ⌨ Space pressed → push-to-talk command flow 시작")

    # 현재 스레드에서 보는 플래그를 캡쳐하기 위한 클로저
    def is_active():
        return _push_to_talk_active

    # push-to-talk에서는 굳이 "I'm listening" 같은 프리페이스는 안 해도 됨
    _run_single_command_flow(
        preface_msg=None,
        transcribe_fn=lambda: stt.transcribe_while(is_active),
    )


def _start_push_to_talk_loop():
    """
    pynput 키보드 리스너를 이용해 space 키를 감지.
    space 누를 때마다 _on_space_pressed() 호출.
    """
    global _push_to_talk_active

    if keyboard is None:
        print("[AudioIO] ❌ pynput 모듈이 없어 push_to_talk 모드를 사용할 수 없습니다.")
        return

    def on_press(key):
        global _push_to_talk_active
        try:
            if key == keyboard.Key.space:
                # 이미 처리 중이면 중복 호출 방지
                if not _push_to_talk_active:
                    _push_to_talk_active = True
                    # 명령 처리는 별도 스레드에서
                    threading.Thread(
                        target=_on_space_pressed,
                        daemon=True,
                    ).start()
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

@app.on_event("startup")
def on_startup():
    global wake_thread
    print("[AudioIO] FastAPI startup")
    mic.open_stream()

    if AUDIO_MODE == "wakeword":
        # 기존 wakeword 모드
        wake.init_model()

        wake_thread = threading.Thread(
            target=start_wakeword_loop,
            args=(wake, _on_wake_detected, 0.0),
            daemon=True,
        )
        wake_thread.start()

        greeting_msg = random.choice(GREETING_RESPONSES)
        print(f"[AudioIO] 💬 Greeting (wakeword): {greeting_msg}")
        tts.speak(greeting_msg)
        time.sleep(0.5)
        print("[AudioIO] ✅ Wakeword loop started")

    elif AUDIO_MODE == "push_to_talk":
        # push-to-talk 모드
        greeting_msg = (
            "Systems online, sir. Push and hold the space bar to issue a command."
        )
        print(f"[AudioIO] 💬 Greeting (push_to_talk): {greeting_msg}")
        tts.speak(greeting_msg)
        time.sleep(0.5)

        pt_thread = threading.Thread(
            target=_start_push_to_talk_loop,
            daemon=True,
        )
        pt_thread.start()
        print("[AudioIO] ✅ Push-to-talk loop started (space key)")

    else:
        # 알 수 없는 모드인 경우 안전하게 wakeword 모드로 폴백
        print(f"[AudioIO] ⚠ Unknown AUDIO_MODE='{AUDIO_MODE}', falling back to wakeword mode.")
        wake.init_model()

        wake_thread = threading.Thread(
            target=start_wakeword_loop,
            args=(wake, _on_wake_detected, 0.0),
            daemon=True,
        )
        wake_thread.start()

        greeting_msg = random.choice(GREETING_RESPONSES)
        print(f"[AudioIO] 💬 Greeting (fallback wakeword): {greeting_msg}")
        tts.speak(greeting_msg)
        time.sleep(0.5)
        print("[AudioIO] ✅ Wakeword loop started (fallback)")


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
