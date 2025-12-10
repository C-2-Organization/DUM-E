#!/usr/bin/env python3
"""
Motion detect node - YOLOWorld를 사용해 객체 감지 후 좌표 변환
Services:
- /detect_object (perception_interfaces/srv/DetectObject) : 이미지에서 특정 객체 감지 후 base 좌표 반환
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from ultralytics import YOLOWorld
from utils.realsense import ImgNode

import DR_init
# ============================================================
# 설정
# ============================================================
ㅂ
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
HOME_JOINT = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# 경로 설정 (워크스페이스 상대)
# parents: [0]=motion/motion, [1]=motion, [2]=src/motion, [3]=src, [4]=ros2_ws/src,
# [5]=ros2_ws, [6]=repo root (DUM-E-feat-implement-llm-agent)
_ROOT = Path(__file__).resolve().parents[6]
MODEL_PATH = (_ROOT / "models" / "yolov8s-worldv2.pt").as_posix()
YOLO_CLASSES = ["person", "cup", "scissors", "box cutter", "bottle", "laptop", "hammer"]
YOLO_CONF_TH = 0.3

# 보정 파일 경로
GRIPPER2CAM_PATH = (_ROOT / "ros2_ws" / "src" / "perception" / "config" / "T_gripper2camera.npy").as_posix()


class DetectMotionNode(Node):
    """
    YOLOWorld 기반 감지 + 단일 기준 포즈에서 Joint5 스윕.
    - 기준 posx 지정 (기본: (367.69, 7.38, 425.09, 83.88, 179.96, 83.73))
    - 기본 대기자세, 예외처리자세 (하드코딩 조인트)
    - Joint5만 조작하는 move_joint5 유틸
    """

    def __init__(self):
        super().__init__("detect_motion_node")

        # 기준 posx (XYZRXRYRZ)
        self.ref_posx = (367.69, 7.38, 425.09, 83.88, 179.96, 83.73)

        # 조인트 하드코딩 포즈
        self.wait_joints = [-0.02, -90.32, 88.932, 4.74, 91.99, 90.43]  # 기본 대기자세
        self.exception_joints = [-0.02, -48.42, 84.33, -0.67, 117.11, 90.43]

        self.get_logger().info(f"🎯 기준 posx 설정: {self.ref_posx}")
        self.get_logger().info("📍 대기/예외 포즈 로드 완료")

        # 1) RealSense ImgNode 초기화
        self.img_node = ImgNode()

        # 2) intrinsics 대기
        self.intrinsics = None
        retry_count = 0
        while rclpy.ok() and self.intrinsics is None and retry_count < 50:
            self.get_logger().info("📷 camera intrinsics 대기 중...")
            rclpy.spin_once(self.img_node, timeout_sec=0.1)
            self.intrinsics = self.img_node.get_camera_intrinsic()
            retry_count += 1

        if self.intrinsics is None:
            self.get_logger().error("❌ 카메라 intrinsics를 가져오지 못했습니다.")
            raise RuntimeError("camera intrinsics not available")

        self.get_logger().info(f"📷 camera intrinsics 수신 완료: {self.intrinsics}")

        # 3) 변환 행렬 로드 (gripper ↔ camera)
        if os.path.exists(GRIPPER2CAM_PATH):
            self.gripper2cam = np.load(GRIPPER2CAM_PATH)
            self.get_logger().info(f"🔧 Loaded T_gripper2camera from {GRIPPER2CAM_PATH}")
        else:
            self.get_logger().warn(f"⚠️ {GRIPPER2CAM_PATH} 파일 없음. 항등 행렬 사용.")
            self.gripper2cam = np.eye(4)

        # 5) YOLOWorld 로드
        self.yolo_model = YOLOWorld(MODEL_PATH)
        self.yolo_model.set_classes(YOLO_CLASSES)
        self.conf_th = YOLO_CONF_TH
        self.get_logger().info("✅ YOLOWorld 모델 로드 완료")

        # 6) DSR_ROBOT2 import
        from DSR_ROBOT2 import movej, posj, get_current_posj
        self.movej = movej
        self.posj = posj
        self.get_current_posj = get_current_posj

        self.get_logger().info("✅ DetectMotionNode 준비 완료")

    def detect_target_object(self, color_img: np.ndarray, target_name: str):
        """
        color_img: BGR 이미지
        target_name: 감지할 클래스 이름 (예: "scissors")
        return: (cx, cy, conf) 또는 None, annotated_img
        """
        results = self.yolo_model.predict(
            source=color_img,
            conf=self.conf_th,
            imgsz=640,
            verbose=False,
        )
        res = results[0]
        boxes = res.boxes
        annotated = res.plot()

        if boxes is None or len(boxes) == 0:
            self.get_logger().info("[YOLO] 감지된 객체 없음")
            return None, annotated

        names = res.names
        candidates = []

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id]
            conf = float(box.conf[0].item())

            if cls_name.lower() == target_name.lower() and conf >= self.conf_th:
                candidates.append((conf, i, cls_name))

        if not candidates:
            self.get_logger().info(f"[YOLO] '{target_name}' 클래스 탐지 실패")
            return None, annotated

        candidates.sort(reverse=True)
        best_conf, best_idx, best_name = candidates[0]
        best_box = boxes[best_idx]

        xyxy = best_box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

        self.get_logger().info(f"[YOLO] 감지: {best_name}, conf={best_conf:.2f}, pixel=({cx},{cy})")
        return (cx, cy, best_conf), annotated

    def get_depth_value(self, cx: int, cy: int, depth_frame: np.ndarray):
        """픽셀 좌표에서 depth 값 추출"""
        h, w = depth_frame.shape
        if 0 <= cx < w and 0 <= cy < h:
            depth_value = depth_frame[cy, cx]
            return depth_value if depth_value != 0 else None

        self.get_logger().warn(f"⚠️ depth out of range: ({cx}, {cy})")
        return None

    def get_camera_pos(self, center_x: int, center_y: int, center_z: float, intrinsics: dict):
        """픽셀 + depth → 카메라 좌표"""
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        ppx = intrinsics["ppx"]
        ppy = intrinsics["ppy"]

        camera_x = (center_x - ppx) * center_z / fx
        camera_y = (center_y - ppy) * center_z / fy
        camera_z = center_z

        return (camera_x, camera_y, camera_z)

    def get_robot_pose_matrix(self, x: float, y: float, z: float, rx: float, ry: float, rz: float):
        """로봇 posx → 4x4 변환행렬"""
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords: tuple):
        """카메라 좌표 → 로봇 base 좌표"""
        from DSR_ROBOT2 import get_current_posx

        try:
            current_posx = get_current_posx()
            if not current_posx or len(current_posx) == 0:
                self.get_logger().error("❌ 현재 로봇 포즈를 가져올 수 없음")
                return None

            current_pos = current_posx[0]
            x, y, z, rx, ry, rz = current_pos[:6]

            T_base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)
            T_gripper2camera = self.gripper2cam
            T_base2camera = T_base2gripper @ T_gripper2camera

            camera_coord = np.array([camera_coords[0], camera_coords[1], camera_coords[2], 1.0])
            base_coord = T_base2camera @ camera_coord

            return tuple(base_coord[:3])
        except Exception as e:
            self.get_logger().error(f"❌ 좌표 변환 실패: {e}")
            return None

    def move_to_wait_pose(self):
        """기본 대기자세로 이동"""
        try:
            target_pose = self.posj(*self.wait_joints)
            self.get_logger().info(f"⏳ 대기자세 이동: {self.wait_joints}")
            result = self.movej(target_pose, vel=20, acc=30)
            self.get_logger().info(f"✅ 대기자세 이동 요청 완료 (resp={result})")
            return True
        except Exception as e:
            self.get_logger().error(f"❌ 대기자세 이동 실패: {e}")
            return False

    def move_to_exception_pose(self):
        """예외처리 자세로 이동"""
        try:
            target_pose = self.posj(*self.exception_joints)
            self.get_logger().info(f"⚠️ 예외자세 이동: {self.exception_joints}")
            result = self.movej(target_pose, vel=20, acc=30)
            self.get_logger().info(f"✅ 예외자세 이동 요청 완료 (resp={result})")
            return True
        except Exception as e:
            self.get_logger().error(f"❌ 예외자세 이동 실패: {e}")
            return False

    def move_joint5(self, target_deg: float):
        """Joint5 각도만 조정 (deg)"""
        try:
            cur = self.get_current_posj()
            if not cur or len(cur) == 0:
                self.get_logger().error("❌ 현재 조인트 상태를 가져올 수 없음")
                return False

            # 반환 구조가 [ [j1..j6] ] 또는 [j1..j6] 인 경우 모두 대응
            if isinstance(cur[0], (list, tuple, np.ndarray)):
                base_joints = cur[0]
            else:
                base_joints = cur

            if len(base_joints) < 6:
                self.get_logger().error(f"❌ 조인트 길이 이상: {base_joints}")
                return False

            # numpy 타입을 파이썬 float으로 변환해 posj에 전달
            joints = [float(x) for x in base_joints[:6]]
            self.get_logger().debug(f"현재 조인트: {joints}")
            joints[4] = target_deg  # Joint5 (0-based index 4)

            target_pose = self.posj(*joints)
            self.get_logger().info(f"🔄 Joint5 이동 -> {target_deg} deg (기타 유지)")
            result = self.movej(target_pose, vel=15, acc=25)
            self.get_logger().info(f"✅ Joint5 이동 요청 완료 (resp={result})")
            if result is None:
                self.get_logger().warn("movej 응답 None (서비스 연결/로봇 연결 상태 확인 필요)")
            return True
        except Exception as e:
            self.get_logger().error(f"❌ Joint5 이동 실패: {e}")
            return False

    def sweep_joint5(
        self,
        start_deg: float = 111.52,
        end_deg: float = 59.76,
        step_deg: float = -5.0,
        step_callback=None,
    ):
        """Joint5 범위 스윕 (기본: 111.52 -> 59.76).
        step_callback가 주어지면 각 포인트 도달 후 호출.
        """
        try:
            if step_deg == 0:
                self.get_logger().error("❌ step_deg는 0일 수 없습니다")
                return False

            # 방향 자동 보정
            if start_deg < end_deg and step_deg < 0:
                step_deg = abs(step_deg)
            if start_deg > end_deg and step_deg > 0:
                step_deg = -abs(step_deg)

            current = start_deg
            reached_any = False
            while (step_deg < 0 and current >= end_deg) or (step_deg > 0 and current <= end_deg):
                if not self.move_joint5(current):
                    return False
                reached_any = True
                if step_callback:
                    step_callback(current)
                current += step_deg

            # 마지막 목표가 step을 넘어갔다면 end_deg에 맞춰 정렬
            if reached_any and ((step_deg < 0 and current + step_deg < end_deg) or (step_deg > 0 and current + step_deg > end_deg)):
                if not self.move_joint5(end_deg):
                    return False
                if step_callback:
                    step_callback(end_deg)

            return True
        except Exception as e:
            self.get_logger().error(f"❌ Joint5 스윕 실패: {e}")
            return False

    def detect_and_get_coords(self, target_name: str) -> dict:
        """
        이미지에서 객체 감지 후 base 좌표 반환
        return: {"success": bool, "x": float, "y": float, "z": float,
                 "conf": float, "annotated": ndarray}
        """
        # 1) 현재 이미지 프레임 가져오기
        color_img = self.img_node.get_color_frame()
        depth_frame = self.img_node.get_depth_frame()

        if color_img is None or depth_frame is None:
            self.get_logger().error("❌ 카메라 프레임을 가져올 수 없음")
            return {"success": False, "message": "camera frame not available"}

        # 2) YOLO 감지
        detect_result = self.detect_target_object(color_img, target_name)
        if detect_result[0] is None:
            self.get_logger().warn(f"❌ '{target_name}' 객체를 감지하지 못했습니다")
            return {"success": False, "message": f"object '{target_name}' not detected"}

        cx, cy, conf = detect_result[0]
        annotated = detect_result[1]

        # 3) Depth 값 추출
        z = self.get_depth_value(cx, cy, depth_frame)
        if z is None or z == 0:
            self.get_logger().warn(f"❌ 유효한 depth 값을 얻지 못함: ({cx}, {cy})")
            return {"success": False, "message": "invalid depth value"}

        # 4) 카메라 좌표 계산
        camera_pos = self.get_camera_pos(cx, cy, z, self.intrinsics)
        self.get_logger().info(f"📷 Camera pos: {camera_pos}")

        # 5) Base 좌표로 변환
        base_pos = self.transform_to_base(camera_pos)
        if base_pos is None:
            return {"success": False, "message": "coordinate transformation failed"}

        self.get_logger().info(f"🤖 Base pos: {base_pos}")

        return {
            "success": True,
            "x": float(base_pos[0]),
            "y": float(base_pos[1]),
            "z": float(base_pos[2]),
            "conf": float(conf),
            "annotated": annotated,
        }

    def detect_during_sweep(self, target_name: str = "scissors"):
        """대기자세에서 Joint5 스윕하며 감지 시도."""
        self.get_logger().info("⏳ 대기자세 이동 후 스캔 시작")
        if not self.move_to_wait_pose():
            return False

        def _cb(_deg):
            res = self.detect_and_get_coords(target_name)
            if res.get("success"):
                self.get_logger().info(
                    f"🎯 감지 성공 @Joint5={_deg:.2f}: xyz=({res['x']:.1f}, {res['y']:.1f}, {res['z']:.1f}), conf={res['conf']:.2f}"
                )
            else:
                self.get_logger().debug(f"미감지 @Joint5={_deg:.2f}: {res.get('message')}")

        ok = self.sweep_joint5(step_callback=_cb)
        if not ok:
            self.get_logger().warn("⚠️ 스윕 중단")
            return False

        # 스윕 끝까지 갔는데 성공한 적 없으면 예외자세로 이동
        self.get_logger().info("감지 실패 → 예외자세 이동")
        self.move_to_exception_pose()
        return True


def main():
    rclpy.init()

    # Doosan 제어용 노드 먼저 생성하여 DR_init에 주입 (DSR_ROBOT2 내부 g_node 필요)
    dsr_node = rclpy.create_node("dsr_example_demo_py", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    node = DetectMotionNode()

    # 시작 시 바로 대기자세로 이동
    if not node.move_to_wait_pose():
        node.get_logger().error("대기자세 이동 실패 - 종료")
        node.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()
        return

    try:
        print("\n명령 입력: 'p' → 스캔/감지, 'q' → 종료\n")
        while rclpy.ok():
            cmd = input("명령(p:scan, q:quit)> ").strip().lower()
            if cmd == 'q':
                break
            if cmd == 'p':
                node.detect_during_sweep("scissors")
            else:
                print("알 수 없는 명령입니다. p 또는 q를 입력하세요.")

        # 종료 시 HOME_JOINT로 복귀 (keyboard UI와 동일 로직)
        node.get_logger().info("종료 명령 수신 → HOME_JOINT 복귀 후 종료")
        try:
            from DSR_ROBOT2 import movej, mwait

            resp = movej(HOME_JOINT, vel=30, acc=30)
            mwait()
            node.get_logger().info(f"HOME_JOINT 복귀 완료 (resp={resp})")
        except Exception as e:
            node.get_logger().warn(f"HOME_JOINT 복귀 실패: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
