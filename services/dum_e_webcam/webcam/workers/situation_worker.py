# webcam/workers/situation_worker.py
from __future__ import annotations

import os
import threading
import time
import queue
import cv2
import math

from webcam.services import CameraCapture, MotionDetector, analyze_situation, dispatch
from webcam.services.table.hole_detector import detect_table_holes
from webcam.services.vision.yolo_locator import YoloLocator
from webcam.monitor_state import update_state, push_event

import rclpy
from webcam.services.ros_bridge.target_pub import PerceptionPublisher

# baseline(홀 id ↔ img(x,y) ↔ robot(x,y)) 로드 + KNN 추정
from webcam.services.table.hole_robot_map import (
    load_holes_baseline,
    estimate_robot_xy_from_center,
)

# =========================
# 설정
# =========================
BASELINE_JSON = os.getenv(
    "TABLE_HOLES_BASELINE_JSON",
    "/home/ilhoon/DUM-E/services/dum_e_webcam/webcam/table_holes_baseline.json",
)
BASELINE_HOLES = load_holes_baseline(BASELINE_JSON)

frame_queue: "queue.Queue" = queue.Queue(maxsize=5)

YOLO_MODEL = os.getenv("YOLO_MODEL_PATH", "/home/ilhoon/Tutorial/OD_Tutorial/YOLO_P/yolov8s-worldv2.pt")
yolo = YoloLocator(model_path=YOLO_MODEL, conf=0.15, imgsz=1280, device="cpu")

# =========================
# ROS2: 프로세스당 1회 init + 퍼블리셔 전역 1개 유지
# =========================
_ROS_INIT_LOCK = threading.Lock()
_ROS_INIT_DONE = False

_PUB_LOCK = threading.Lock()
_PUB_NODE: PerceptionPublisher | None = None
_PUB_TOPIC = os.getenv("WEBCAM_PUB_TOPIC", "/dum_e/webcam/webcam")


def ensure_ros_init() -> None:
    """uvicorn/스레드 환경에서 rclpy.init 반복 호출 방지."""
    global _ROS_INIT_DONE
    if _ROS_INIT_DONE:
        return
    with _ROS_INIT_LOCK:
        if _ROS_INIT_DONE:
            return
        if not rclpy.ok():
            rclpy.init(args=None)
        _ROS_INIT_DONE = True


def get_pub_node() -> PerceptionPublisher:
    """퍼블리셔 노드는 1개만 만들어 계속 재사용(중간 destroy/shutdown 금지)."""
    global _PUB_NODE
    with _PUB_LOCK:
        if _PUB_NODE is None:
            ensure_ros_init()
            _PUB_NODE = PerceptionPublisher(topic=_PUB_TOPIC)
        return _PUB_NODE


# =========================
# GPT worker (이상 프레임만)
# =========================
def worker_loop():
    while True:
        frame = frame_queue.get()
        update_state({"queue_size": frame_queue.qsize()})

        try:
            update_state({"gpt_inference": True, "gpt_inference_since": time.time()})
            push_event("GPT", "analyze_situation start")

            result = analyze_situation(frame)
            dispatch(result)

            update_state({
                "gpt_last_done_ts": time.time(),
                "gpt": {
                    "scene_summary": result.get("scene_summary"),
                    "risk_level": result.get("risk_level"),
                },
                "action": {
                    "recommended_action": result.get("recommended_action"),
                }
            })
            push_event("GPT", f"done: {result.get('risk_level', '-')}, {result.get('recommended_action', '-')}")
        except Exception as e:
            push_event("ERR", f"GPT Worker Error: {e}")
            print("[GPT Worker Error]", e)
        finally:
            update_state({"gpt_inference": False, "gpt_inference_since": None})
            frame_queue.task_done()

        time.sleep(0.2)


# =========================
# Camera loop (실시간 상태 + ROS 전송)
# =========================
def camera_loop():
    def describe_between_holes(cx, cy, holes, radius, hole_safe=1.15):
        if not holes or len(holes) < 2:
            return None

        dist2 = []
        for i, (hx, hy) in enumerate(holes, start=1):
            dx = float(cx) - float(hx)
            dy = float(cy) - float(hy)
            dist2.append((i, dx * dx + dy * dy))
        dist2.sort(key=lambda x: x[1])

        nearest_id, nearest_d2 = dist2[0]
        second_id, second_d2 = dist2[1]

        nearest_r = None
        if isinstance(radius, (int, float)):
            nearest_r = float(radius)
        elif isinstance(radius, (list, tuple)) and len(radius) >= nearest_id:
            try:
                nearest_r = float(radius[nearest_id - 1])
            except Exception:
                nearest_r = None

        if nearest_r is not None:
            if math.sqrt(nearest_d2) <= nearest_r * hole_safe:
                return f"{nearest_id}번 위에 있습니다"

        a, b = sorted([nearest_id, second_id])
        return f"{a}~{b}번 사이에 있습니다"

    cam = CameraCapture()
    md = MotionDetector()

    last_sent = 0.0
    min_interval = 2.0
    dropped = 0

    ROI_MARGIN = 30
    HOLE_SAFE = 1.15

    # robot_xy 추정 파라미터 (KNN)
    KNN_K = int(os.getenv("ROBOT_XY_KNN_K", "3"))
    KNN_MODE = os.getenv("ROBOT_XY_KNN_MODE", "idw")
    IDW_POWER = float(os.getenv("ROBOT_XY_IDW_POWER", "1"))

    # fallback: ROI -> robot 범위 (대충 값, env로 튜닝)
    RX_MIN = float(os.getenv("ROI_TO_ROBOT_X_MIN", "-200"))
    RX_MAX = float(os.getenv("ROI_TO_ROBOT_X_MAX", "200"))
    RY_MIN = float(os.getenv("ROI_TO_ROBOT_Y_MIN", "-200"))
    RY_MAX = float(os.getenv("ROI_TO_ROBOT_Y_MAX", "200"))

    push_event("CAM", "camera_loop start")
    cv2.namedWindow("Dum-E Situation Cam", cv2.WINDOW_NORMAL)

    # 디버그 출력 주기
    last_ros_log = 0.0

    try:
        while True:
            frame = cam.read()
            now = time.time()

            if frame is None:
                update_state({"camera_ok": False})
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            update_state({"camera_ok": True, "last_frame_ts": now})

            # =========================
            # 0) 홀 검출
            # =========================
            holes = []
            radius = None
            debug_img = frame

            try:
                holes, radius, debug_img = detect_table_holes(frame, debug=True)
                if holes is None:
                    holes = []
            except Exception as e:
                push_event("ERR", f"hole_detector error: {e}")
                holes = []
                radius = None
                debug_img = frame

            # 테이블 ROI 경계(홀 좌표 기반) 계산
            x_min = x_max = y_min = y_max = None
            if len(holes) >= 3:
                xs = [p[0] for p in holes]
                ys = [p[1] for p in holes]
                x_min = min(xs) - ROI_MARGIN
                x_max = max(xs) + ROI_MARGIN
                y_min = min(ys) - ROI_MARGIN
                y_max = max(ys) + ROI_MARGIN

            # =========================
            # 1) YOLO 탐지(여러 개) + enrich(ROI/between/robot_xy)
            # =========================
            yolo_state = {
                "cls": None,
                "conf": None,
                "center": None,
                "bbox": None,
                "in_table_roi": None,
                "between_holes": None,
                "confirmed": [],
            }

            try:
                confirmed = yolo.detect_confirmed(frame)  # 안정화된 것만
                enriched = []

                for d in (confirmed or []):
                    cx, cy = d.get("center", (None, None))
                    in_roi = None
                    between = None

                    # ROI/between_holes
                    if cx is not None and cy is not None and x_min is not None:
                        fx, fy = float(cx), float(cy)
                        in_roi = (x_min <= fx <= x_max) and (y_min <= fy <= y_max)
                        if in_roi:
                            between = describe_between_holes(fx, fy, holes, radius, hole_safe=HOLE_SAFE)

                    robot_xy = None
                    robot_dbg = None

                    if cx is not None and cy is not None:
                        est = estimate_robot_xy_from_center(
                            cx,
                            cy,
                            BASELINE_HOLES,
                            k=2,   # 사이값은 무조건 2
                        )
                        if est:
                            robot_xy = est["robot_xy"]
                            robot_dbg = {
                                "k": est["k"],
                                "neighbor_ids": est["neighbor_ids"],
                            }


                    d2 = dict(d)
                    d2["in_table_roi"] = bool(in_roi) if in_roi is not None else None
                    d2["between_holes"] = between
                    d2["robot_xy"] = robot_xy
                    d2["robot_dbg"] = robot_dbg
                    enriched.append(d2)

                yolo_state["confirmed"] = enriched

                # 대표 1개 선택
                rep = None
                for dd in enriched:
                    if dd.get("in_table_roi") is True:
                        rep = dd
                        break
                if rep is None and enriched:
                    rep = enriched[0]

                if rep:
                    conf = rep.get("conf")
                    yolo_state.update({
                        "cls": rep.get("cls_name"),
                        "conf": round(float(conf), 2) if conf is not None else None,
                        "center": rep.get("center"),
                        "bbox": rep.get("bbox"),
                        "track_id": rep.get("track_id"),
                        "hit": rep.get("hit"),
                        "in_table_roi": rep.get("in_table_roi"),
                        "between_holes": rep.get("between_holes"),
                    })
                else:
                    yolo_state.update({
                        "cls": None, "conf": None, "center": None, "bbox": None,
                        "in_table_roi": None, "between_holes": None
                    })

            except Exception as e:
                push_event("ERR", f"YOLO error: {e}")
                # yolo_state는 직전 값 유지

            update_state({"yolo": yolo_state})

            # =========================
            # 1.5) 대표 로봇 XY 상태
            # =========================
            robot_target_xy = None
            robot_target_dbg = None

            rep_xy = None
            rep_dbg = None
            if yolo_state.get("confirmed"):
                for dd in yolo_state["confirmed"]:
                    if dd.get("in_table_roi") is True and dd.get("robot_xy") is not None:
                        rep_xy = dd.get("robot_xy")
                        rep_dbg = dd.get("robot_dbg")
                        break
                if rep_xy is None:
                    for dd in yolo_state["confirmed"]:
                        if dd.get("robot_xy") is not None:
                            rep_xy = dd.get("robot_xy")
                            rep_dbg = dd.get("robot_dbg")
                            break

            if rep_xy is not None:
                robot_target_xy = rep_xy
                robot_target_dbg = rep_dbg

            update_state({
                "robot_target_xy": robot_target_xy,
                "robot_target_dbg": robot_target_dbg,
            })

            # =========================
            # 2) ROS 전송: candidates + recommended_action
            # =========================
            candidates = []
            for dd in (yolo_state.get("confirmed") or [])[:5]:
                candidates.append({
                    "track_id": dd.get("track_id"),
                    "cls_name": dd.get("cls_name"),
                    "conf": dd.get("conf"),
                    "hit": dd.get("hit"),
                    "miss": dd.get("miss"),
                    "center": dd.get("center"),
                    "bbox": dd.get("bbox"),
                    "in_table_roi": dd.get("in_table_roi"),
                    "between_holes": dd.get("between_holes"),
                    "robot_xy": dd.get("robot_xy"),
                    "robot_dbg": dd.get("robot_dbg"),
                })

            has_table_obj = any(c.get("in_table_roi") is True for c in candidates)
            recommended_action = "look_at" if has_table_obj else ("idle" if not candidates else "look_at")

            # best 1개 선정(ROI 우선, 없으면 첫번째)
            best = None
            for c in candidates:
                if c.get("in_table_roi") is True:
                    best = c
                    break
            if best is None and candidates:
                best = candidates[0]

            msg = {
                "stamp": time.time(),
                "source": "webcam",
                "recommended_action": recommended_action,
                "risk_level": "low",
                "human_present": False,
                "hand_near_target": False,
                "best": best,
                "candidates": candidates,
            }
            
            # =========================
            # 2.5) ROS publish rate limit (예: 5Hz)
            # =========================
            if not hasattr(camera_loop, "_last_pub"):
                camera_loop._last_pub = 0.0

            if now - camera_loop._last_pub < 0.2:   # 0.2초=5Hz
                # publish만 스킵하고, 아래 overlay/키입력은 계속 돌려도 됨
                # (진짜로 전체 루프를 느리게 하고 싶으면 continue를 아래로 옮기면 됨)
                pass
            else:
                camera_loop._last_pub = now

                # 디버그: 2초마다 구독자 수 확인
                if now - last_ros_log > 2.0:
                    last_ros_log = now
                    try:
                        sub_cnt = get_pub_node().pub.get_subscription_count()
                    except Exception:
                        sub_cnt = -1
                    print(f"[ROS PUB] topic={_PUB_TOPIC} subs={sub_cnt} candidates={len(candidates)} action={recommended_action}")

                try:
                    get_pub_node().publish_dict(msg)
                except Exception as e:
                    push_event("ERR", f"ROS publish error: {e}")
                    print("[ROS publish error]", e)

            # =========================
            # 3) 모션 감지 -> 큐 (GPT용)
            # =========================
            suspicious = md.is_suspicious(frame)
            if suspicious and (now - last_sent) > min_interval:
                if not frame_queue.full():
                    frame_queue.put(frame.copy())
                    last_sent = now
                    push_event("CAM", "suspicious frame queued")
                else:
                    dropped += 1
                    push_event("CAM", "queue full -> dropped")

            update_state({"queue_size": frame_queue.qsize(), "queue_dropped": dropped})

            # =========================
            # 4) 오버레이
            # =========================
            for dd in yolo_state.get("confirmed", []):
                bbox = dd.get("bbox")
                center = dd.get("center")
                if bbox:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                if center:
                    ccx, ccy = [int(v) for v in center]
                    cv2.circle(debug_img, (ccx, ccy), 5, (255, 0, 0), -1)

            cv2.imshow("Dum-E Situation Cam", debug_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()
        # 여기서 destroy/shutdown 하지 않는다 (uvicorn에서 꼬임 방지)


def start_worker_and_camera():
    # GPT worker 1개
    threading.Thread(target=worker_loop, daemon=True).start()

    # camera loop 1개
    def _cam_safe():
        try:
            camera_loop()
        except Exception as e:
            push_event("ERR", f"camera_loop crashed: {e}")
            print("[camera_loop crashed]", e)

    threading.Thread(target=_cam_safe, daemon=True).start()
    push_event("SYSTEM", "GPT worker + camera loop started")