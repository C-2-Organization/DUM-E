# services/audio_io/app/main.py
import sys
from pathlib import Path
import subprocess
import random
import time
import json

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

GREETING_RESPONSES = [
    "good morning, sir. Standing by for your command.",
    "For you, Sir, Always."
    # "Initialization complete. Ready when you are, sir.",
    # "All systems functional. How may I assist, sir?",
    # "Wakeword monitoring activated. I'm here, sir.",
    # "Operational and awaiting your direction, sir.",
    # "Diagnostics clear. At your service, sir.",
    # "Startup sequence complete. Listening now, sir.",
    # "Good day, sir. Ready for deployment.",
    # "Everything’s set. Please proceed when ready, sir.",
    # "Full system readiness achieved. How can I help, sir?",
]

WAKE_RESPONSES = [
    "Yes, sir?",
    "At your service, sir.",
    "How can I assist, sir?",
    "I'm listening, sir.",
    "Ready and waiting, sir.",
    "Standing by, sir.",
    "Awaiting your command.",
    "What can I do for you, sir?",
    "Online and attentive, sir.",
    "Yes, I'm here.",
]

EXECUTE_RESPONSES = [
    "Understood. Executing your command, sir.",
    "Acknowledged. Initiating the requested sequence.",
    "Your instructions are clear. Proceeding now.",
    "Command received. Beginning operations.",
    "I'm on it, sir.",
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
    "Operation successful. Anything else?",
    "The requested process has been completed.",
    "Execution finished. Awaiting your next command.",
    "Mission accomplished, sir.",
    "All done. How else can I help?",
    "Sequence complete. Standing by.",
    "Your instructions have been fully carried out.",
    "Everything is done as requested.",
    "Process completed without issues, sir.",
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
        print("[AudioIO] ℹ 이미 ros2 bringup이 떠 있습니다.")
        return False

    try:
        print("[AudioIO] 🚀 ros2 bringup 실행 시도...")
        _robot_proc = subprocess.Popen(
            [
                "ros2",
                "launch",
                "dum_e_bringup",
                "dum_e_bringup.launch.py",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"[AudioIO] ✅ bringup 프로세스 시작 (pid={_robot_proc.pid})")
        return True
    except Exception as e:
        print(f"[AudioIO] ❌ bringup 실행 실패: {e}")
        _robot_proc = None
        return False


def _is_robot_wakeup_command(text: str) -> bool:
    """
    사용자의 명령이 '로봇 깨우기' 관련인지 간단 판별.
    """
    text = text.lower()
    wake_keywords = [
        "wake up",
        "wakeup",
        "wake dummy",
        "turn on robot",
        "turn on dummy",
        "로봇 켜",
        "더미 켜",
        "더미 깨워",
    ]
    return any(k in text for k in wake_keywords)


def _execute_plan(plan: dict) -> bool:
    """
    Planner가 생성한 plan(JSON)을 실제 ROS 스킬 실행으로 연결.
    """
    skills = plan.get("skills", [])
    if not skills:
        print("[AudioIO] ⚠ plan에 skills가 비어 있습니다.")
        return False

    executed_any = False

    for skill in skills:
        skill_name = skill.get("name")
        params = skill.get("params", {})

        if not skill_name:
            print(f"[AudioIO] ⚠ 잘못된 skill 항목: {skill}")
            continue

        print(f"[AudioIO] ▶ 스킬 실행 요청: {skill_name} (params={params})")

        msg = SkillCommand()
        msg.skill_name = skill_name
        msg.json_param = str(params)

        try:
            result = call_run_skill(msg)
            print(f"[AudioIO] ✅ 스킬 결과: {result}")
            executed_any = True
        except Exception as e:
            print(f"[AudioIO] ❌ 스킬 실행 에러: {e}")
            continue

    return executed_any

def _on_wake_detected(keyword: str):
    """
    wakeword 루프 스레드에서 호출되는 콜백.
    여기서 STT를 동기적으로 실행하고,
    플래너 → ROS 실행까지 처리한다.
    """
    global _last_wakeup_flag, _busy

    if _busy:
        print(f"[AudioIO] ⚠ 이미 명령 처리 중이므로 이번 wakeword('{keyword}')는 무시합니다.")
        return

    print(f"[AudioIO] >>> WAKE WORD DETECTED! ({keyword}) STT 시작")
    _last_wakeup_flag = True

    try:
        # 0) Busy 플래그 설정
        _busy = True

        # 1) STT로 사용자 발화 인식
        user_text = stt.transcribe_once()
        print(f"[AudioIO] 🗣 STT 결과: {user_text!r}")

        if not user_text:
            print("[AudioIO] ⚠ STT 결과가 비어 있습니다. 명령 처리 중단.")
            return

        # 1-A) 로봇 깨우기 전용 명령인지 먼저 체크
        if _is_robot_wakeup_command(user_text):
            print("[AudioIO] 🤖 로봇 깨우기 명령으로 인식됨")

            started = _launch_robot_bringup()
            try:
                if started:
                    # 로봇이 꺼져 있었다 → 새로 켜는 중
                    tts.speak("Waking up dummy")
                else:
                    # 이미 켜져 있거나 실행 실패
                    if _is_robot_already_running():
                        tts.speak("Dummy is already running.")
                    else:
                        tts.speak("There was a problem waking up dummy. Please try again later.")
            except Exception as e:
                print(f"[AudioIO] ❌ TTS 에러: {e}")
            return

        # 2) Planner 호출: 자연어 → 스킬 플로우(JSON)
        try:
            plan = plan_skill_flow(user_text)
        except Exception as e:
            print(f"[AudioIO] ❌ Planner 에러: {e}")
            try:
                # 자비스 스타일로 사과 + 재시도 안내
                jarvis.reply_and_speak(
                    "A system issue occurred while organizing the internal task sequence. "
                    "Please apologize to the user in a concise and respectful manner, and inform them to try again shortly."                )
            except Exception as tts_err:
                print(f"[AudioIO] ❌ Jarvis/TTS 에러: {tts_err}")
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
                # planner가 준 메시지를 바탕으로, 현재 명령을 수행할 수 없는 이유를 정중하게 설명
                jarvis.reply_and_speak(
                    f"Based on the following information, explain in a concise and polite manner why the requested command cannot be executed: {msg}"                )
            except Exception as e:
                print(f"[AudioIO] ❌ Jarvis/TTS 에러: {e}")

        else:
            # 3-B) 수행 가능한 경우 → 실제 ROS 스킬 실행
            executed = _execute_plan(plan)

            if not executed:
                # 계획 상으로는 can_execute_now = True였으나,
                # 실제 스킬 실행은 1개도 성공하지 못한 경우
                try:
                    tts.speak("I tried to execute the process, but there was an issue. Please check the system, sir.")
                except Exception as e:
                    print(f"[AudioIO] ❌ TTS 에러: {e}")
                return

            # 4) 수행 완료 후 짧은 피드백
            complete_msg = random.choice(COMPLETE_RESPONSES)
            print(f"[AudioIO] 💬 COMPLETE: {complete_msg}")
            try:
                tts.speak(complete_msg)
            except Exception as e:
                print(f"[AudioIO] ❌ TTS 에러: {e}")

    finally:
        _busy = False


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

    greeting_msg = random.choice(GREETING_RESPONSES)
    print(f"[AudioIO] 💬 Greeting: {greeting_msg}")
    tts.speak(greeting_msg)
    time.sleep(0.5)
    print("[AudioIO] ✅ Wakeword loop started")


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
    return {"last_wakeup": _last_wakeup_flag}


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
