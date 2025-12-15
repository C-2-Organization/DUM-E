# webcam/workers/situation_worker.py
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

from webcam.services.ros_bridge import PerceptionPublisherThread
from webcam.services.ros_bridge.schema import build_perception_msg

# ✅ baseline(홀 id ↔ img(x,y) ↔ robot(x,y)) 로드
from webcam.services.table.hole_robot_map import (
    load_holes_baseline,
    estimate_robot_xy_knn_mean,
    estimate_robot_xy_knn_idw,
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

# ROS2 publish (webcam -> robot)
_pub = None

def _ensure_pub():
    global _pub
    if _pub is None:
        _pub = PerceptionPublisherThread(topic="/dum_e/webcam/webcam")
        _pub.start()
    return _pub

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
        """
        holes: [(x,y), ...]  # hole_detector가 그리는 번호 순서가 holes 인덱스+1이라고 가정
        radius: 숫자 or [r1,r2,...] (홀마다 반지름) or None

        return: "8번 위에 있습니다" / "8~9번 사이에 있습니다" / None
        """
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

    # ✅ robot_xy 추정 파라미터 (KNN)
    KNN_K = int(os.getenv("ROBOT_XY_KNN_K", "3"))          # 2 또는 3 추천
    KNN_MODE = os.getenv("ROBOT_XY_KNN_MODE", "idw")       # "mean" | "idw"
    IDW_POWER = float(os.getenv("ROBOT_XY_IDW_POWER", "1"))# 1.0 추천, 2.0은 더 쏠림

    push_event("CAM", "camera_loop start")
    cv2.namedWindow("Dum-E Situation Cam", cv2.WINDOW_NORMAL)

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
                # 대표 1개(모니터 ROI 문장 등 기존 호환)
                "cls": None,
                "conf": None,
                "center": None,
                "bbox": None,
                "in_table_roi": None,
                "between_holes": None,

                # 다중 물체
                "confirmed": [],
            }

            try:
                confirmed = yolo.detect_confirmed(frame)  # ✅ 안정화된 것만
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

                    # ✅ robot_xy (KNN mean / KNN idw)
                    robot_xy = None
                    robot_dbg = None
                    if cx is not None and cy is not None:
                        if KNN_MODE.lower() == "mean":
                            est = estimate_robot_xy_knn_mean(cx, cy, BASELINE_HOLES, k=KNN_K)
                        else:
                            est = estimate_robot_xy_knn_idw(cx, cy, BASELINE_HOLES, k=KNN_K, power=IDW_POWER)

                        if est:
                            robot_xy = est.get("robot_xy")
                            robot_dbg = {
                                "neighbor_ids": est.get("neighbor_ids"),
                                "pix_dists": est.get("pix_dists"),
                                "weights": est.get("weights"),
                                "k": est.get("k"),
                                "mode": KNN_MODE,
                            }

                    d2 = dict(d)
                    d2["in_table_roi"] = bool(in_roi) if in_roi is not None else None
                    d2["between_holes"] = between
                    d2["robot_xy"] = robot_xy
                    d2["robot_dbg"] = robot_dbg
                    enriched.append(d2)

                yolo_state["confirmed"] = enriched

                # ✅ 대표 1개 선택: 테이블 위(in_roi=True) 우선, 없으면 첫 번째
                rep = None
                for d in enriched:
                    if d.get("in_table_roi") is True:
                        rep = d
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

            # ✅ 모니터 상태 업데이트
            update_state({"yolo": yolo_state})

            # =========================
            # 1.5) 대표 로봇 XY 상태(모니터 RobotXY 카드용)
            # =========================
            robot_target_xy = None
            robot_target_dbg = None

            # 대표는 테이블 위 우선으로 뽑혔으니 그걸 사용
            rep_xy = None
            rep_dbg = None
            if yolo_state.get("confirmed"):
                # yolo_state 대표 1개가 rep를 반영하므로, rep center/between 기반으로 찾기보단
                # enriched에서 테이블 위 우선 후보를 그대로 다시 고르는 게 안전
                for d in yolo_state["confirmed"]:
                    if d.get("in_table_roi") is True and d.get("robot_xy") is not None:
                        rep_xy = d.get("robot_xy")
                        rep_dbg = d.get("robot_dbg")
                        break
                if rep_xy is None:
                    for d in yolo_state["confirmed"]:
                        if d.get("robot_xy") is not None:
                            rep_xy = d.get("robot_xy")
                            rep_dbg = d.get("robot_dbg")
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
            # ✅ 웹캠 단계 기본 룰:
            # - 테이블 위 후보가 있으면 look_at
            # - 없으면 idle
            candidates = []
            for d in (yolo_state.get("confirmed") or [])[:5]:
                candidates.append({
                    "track_id": d.get("track_id"),
                    "cls_name": d.get("cls_name"),
                    "conf": d.get("conf"),
                    "hit": d.get("hit"),
                    "miss": d.get("miss"),
                    "center": d.get("center"),
                    "bbox": d.get("bbox"),
                    "in_table_roi": d.get("in_table_roi"),
                    "between_holes": d.get("between_holes"),
                    "robot_xy": d.get("robot_xy"),
                })

            has_table_obj = any(c.get("in_table_roi") is True for c in candidates)
            recommended_action = "look_at" if has_table_obj else ("idle" if not candidates else "look_at")

            try:
                msg = build_perception_msg(
                    candidates=candidates,
                    recommended_action=recommended_action,
                    risk_level="low",
                    human_present=False,
                    hand_near_target=False,
                )
                _ensure_pub().publish(msg)
            except Exception as e:
                push_event("ERR", f"ROS publish error: {e}")

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
            # 4) 오버레이 (여러 개 전부 그리기)
            # =========================
            for d in yolo_state.get("confirmed", []):
                bbox = d.get("bbox")
                center = d.get("center")
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


def start_worker_and_camera():
    threading.Thread(target=worker_loop, daemon=True).start()

    def _cam_safe():
        try:
            camera_loop()
        except Exception as e:
            push_event("ERR", f"camera_loop crashed: {e}")
            print("[camera_loop crashed]", e)

    threading.Thread(target=_cam_safe, daemon=True).start()
    push_event("SYSTEM", "GPT worker + camera loop started")
