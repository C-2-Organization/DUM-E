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

from services.llm_agent.app.skill_planner import plan_skill_flow
from services.llm_agent.ros_bridge import call_run_skill
from dum_e_interfaces.msg import SkillCommand

app = FastAPI(title="Dummy Audio IO Service")

mic = MicController(MicConfig())
wake = WakeupWord(mic)
stt = StreamingSTT()
tts = TTS()

wake_thread: threading.Thread | None = None
_last_wakeup_flag = False


def _execute_plan(plan: dict) -> bool:
    """
    planner가 만들어준 JSON(plan)을 보고 실제 ROS 스킬을 실행한다.

    - 현재는 PICK 스킬만 지원
    - 성공적으로 지원 가능한 스킬을 하나라도 실행하면 True
    - 아무 것도 실행하지 못하면 False
    """
    steps = plan.get("steps") or []
    if not isinstance(steps, list):
        print("[AudioIO] ⚠ plan.steps 가 리스트가 아님:", steps)
        return False

    executed_any = False

    for step in steps:
        skill = step.get("skill")
        if skill == "PICK":
            obj = step.get("object") or {}
            # canonical_en 있으면 그걸 우선 사용, 없으면 raw
            obj_name = obj.get("canonical_en") or obj.get("raw") or ""
            if not obj_name:
                print("[AudioIO] ⚠ PICK 스텝에 object_name 이 없음:", step)
                continue

            print(f"[AudioIO] 🦾 실행: PICK '{obj_name}'")

            try:
                resp = call_run_skill(
                    skill_type=SkillCommand.PICK,
                    object_name=obj_name,
                    target_pose=None,      # pose는 내부 스킬 로직에 맡김
                    params_json="",        # 옵션 필요시 나중에 추가
                    timeout_sec=60.0,      # 실제 동작 고려해서 넉넉히
                )
            except Exception as e:
                print(f"[AudioIO] ❌ /run_skill 호출 중 에러: {e}")
                # 여기서 바로 실패 반환할지, 다음 step 시도할지는 정책 문제
                return False

            print(
                f"[AudioIO] ✅ /run_skill 응답: success={resp.success}, "
                f"confidence={resp.confidence:.2f}, message='{resp.message}'"
            )

            executed_any = True
            # 현재는 PICK 하나만 지원하니까 첫 PICK 실행 후 바로 종료
            break

        else:
            # 지금은 PICK 외에는 직접 실행하지 않음
            print(f"[AudioIO] ℹ 아직 지원하지 않는 스킬: {skill}")

    return executed_any


def _on_wake_detected(keyword: str):
    """
    wakeword 루프 스레드에서 호출되는 콜백.
    여기서 STT를 동기적으로 실행하고,
    플래너 → ROS 실행까지 처리한다.
    """
    global _last_wakeup_flag, wake_thread
    print(f"[AudioIO] >>> WAKE WORD DETECTED! ({keyword}) STT 시작")
    _last_wakeup_flag = True

    # wakeword loop 종료 (STT/로봇 동작 동안은 잠시 쉬게)
    wake.running = False

    # 1) STT 실행 (blocking)
    user_text = stt.listen_and_transcribe()
    print(f"[AudioIO] 🎙 사용자가 말한 내용: '{user_text}'")

    if not user_text.strip():
        print("[AudioIO] ⚠ STT 결과가 비어있음. 다시 대기.")
        # 바로 다시 wakeword 루프 재시작
        wake_thread = threading.Thread(
            target=start_wakeword_loop,
            args=(wake, _on_wake_detected, 0.0),
            daemon=True,
        )
        wake_thread.start()
        return

    # 2) Planner 호출: 자연어 → 스킬 플로우(JSON)
    try:
        plan = plan_skill_flow(user_text)
    except Exception as e:
        print(f"[AudioIO] ❌ Planner 에러: {e}")
        try:
            tts.speak("생각을 정리하는 중에 문제가 생겼어요. 잠시 후에 다시 시도해 주세요.")
        except Exception as tts_err:
            print(f"[AudioIO] ❌ TTS 에러: {tts_err}")

        # 다시 wakeword 루프 재시작
        wake_thread = threading.Thread(
            target=start_wakeword_loop,
            args=(wake, _on_wake_detected, 0.0),
            daemon=True,
        )
        wake_thread.start()
        return

    print("[AudioIO] 🧠 Planner 결과:")
    print(plan)

    can_execute = bool(plan.get("can_execute_now"))
    user_message = plan.get("user_message") or ""

    if not can_execute:
        # 3-A) 현재 스킬셋으로는 수행 불가능한 명령
        msg = user_message or "현재 이 명령은 수행할 수 없습니다."
        print(f"[AudioIO] ❌ 실행 불가: {msg}")

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
                or "아직 이 명령을 완전히 실행할 수 있는 스킬이 구현되어 있지 않습니다."
            )
            print(f"[AudioIO] ⚠ 계획은 가능하다고 했지만 실제 실행 실패: {fallback_msg}")
            try:
                tts.speak(fallback_msg)
            except Exception as e:
                print(f"[AudioIO] ❌ TTS 에러: {e}")
        else:
            # 정책상: 성공 시에는 조용히 동작만 할 수도 있고,
            # 간단한 안내를 음성으로 줄 수도 있다.
            # 지금 요구사항은 "실행할 수 없는 경우에만 TTS"라서 여기서는 말하지 않음.
            print("[AudioIO] ✅ 플랜 실행 완료 (TTS는 생략)")

    # 4) 끝나면 다시 wakeword 루프 재시작
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
