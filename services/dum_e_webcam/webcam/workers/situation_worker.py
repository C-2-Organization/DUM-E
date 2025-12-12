# webcam/workers/situation_worker.py
import threading
import time
import queue

import cv2

from webcam.services import CameraCapture, MotionDetector, analyze_situation, dispatch

# 카메라에서 '이상 프레임'만 넣는 큐
frame_queue: "queue.Queue" = queue.Queue(maxsize=5)


def worker_loop():
    """큐에서 프레임 꺼내서 GPT 상황 분석 + 액션 디스패치"""
    while True:
        frame = frame_queue.get()  # 블로킹 대기
        try:
            result = analyze_situation(frame)
            dispatch(result)
        except Exception as e:
            print("[GPT Worker Error]", e)
        finally:
            frame_queue.task_done()
        time.sleep(0.2)  # 과도한 호출 방지


def camera_loop():
    """실시간 카메라 루프: 모션 감지 후 이상 프레임만 큐로 전송"""
    cam = CameraCapture()
    md = MotionDetector()
    last_sent = 0.0
    min_interval = 2.0  # 초 단위 GPT 호출 최소 간격

    print("[CAM] camera_loop 시작")

    while True:
        frame = cam.read()
        if frame is None:
            continue

        suspicious = md.is_suspicious(frame)
        now = time.time()

        if suspicious and (now - last_sent) > min_interval:
            if not frame_queue.full():
                frame_queue.put(frame.copy())
                last_sent = now
                print("[CAM] suspicious frame queued")

        # 디버깅용 화면
        cv2.imshow("Dum-E Situation Cam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


def start_worker_and_camera():
    """FastAPI startup 이벤트에서 호출할 함수"""
    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=camera_loop, daemon=True).start()
    print("[SYSTEM] GPT worker + camera loop started")
