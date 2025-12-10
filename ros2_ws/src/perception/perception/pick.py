#!/usr/bin/env python3
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from scipy.spatial.transform import Rotation

import DR_init
from utils.realsense import ImgNode
from utils.onrobot import RG
from ultralytics import YOLOWorld
import os
from ament_index_python.packages import get_package_share_directory

from perception_interfaces.srv import PickObject


# ==========================
# 사용자 설정
# ==========================

ROBOT_ID = "dsr01"

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502

MODEL_PATH = "/home/rokey/DUM-E/models/yolov8s-worldv2.pt"
YOLO_CLASSES = ["person", "cup", "scissors", "box cutter", "bottle", "laptop", "hammer"]

YOLO_CONF_TH = 0.3   # YOLO 후보 인정 최소 conf
PICK_CONF_TH = 0.3   # 실제 pick 실행 최소 conf (원하면 조절)

GRIPPER2CAM_PATH = "/home/rokey/DUM-E/calib/T_gripper2camera.npy"  # <- 네가 저장한 T 파일 경로


class VisionPickNode(Node):
    """
    LLM/다른 노드에서:
      /pick_object (perception/srv/PickObject) 서비스 호출
        - request.object_name = "scissors"
      이 노드는:
        1) RealSense frame 한 장 가져옴
        2) YOLOWorld로 object_name 클래스 디텍션
        3) bbox center 픽셀 → depth → camera 좌표 → base 좌표
        4) 바로 pick 동작 수행
        5) base 좌표 + conf를 응답으로 돌려줌
    """

    def __init__(self, img_node: ImgNode):
        super().__init__("vision_pick_node")

        # 1) RealSense ImgNode 생성
        self.img_node = img_node

        pkg_share = get_package_share_directory('perception')
        calib_path = os.path.join(pkg_share, 'config', 'T_gripper2camera.npy')
        self.gripper2cam = np.load(calib_path)

        # 2) intrinsics가 올 때까지 잠깐 대기
        self.intrinsics = None
        while rclpy.ok() and self.intrinsics is None:
            self.get_logger().info("📷 camera intrinsics 대기 중...")
            rclpy.spin_once(self.img_node, timeout_sec=0.1)
            self.intrinsics = self.img_node.get_camera_intrinsic()

        if self.intrinsics is None:
            self.get_logger().error("❌ 카메라 intrinsics 를 가져오지 못했습니다.")
            raise RuntimeError("camera intrinsics not available")

        self.get_logger().info(f"📷 camera intrinsics 수신 완료: {self.intrinsics}")

        # 3) gripper ↔ camera 변환행렬 로드
        self.gripper2cam = np.load(calib_path)
        self.get_logger().info(f"🔧 Loaded T_gripper2camera from {calib_path}")

        # 4) 그리퍼 / 로봇 파라미터
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

        self.LIN_VEL = [150.0, 300.0]
        self.LIN_ACC = [150.0, 150.0]

        self.JNT_VEL = 150.0
        self.JNT_ACC = 300.0

        self.CUSTOM_HOME_JOINT = [0, 0, 90, 0, 90, 0]

        # 5) YOLOWorld 로드
        self.yolo_model = YOLOWorld(MODEL_PATH)
        self.yolo_model.set_classes(YOLO_CLASSES)
        self.conf_th = YOLO_CONF_TH

        # 6) 서비스 서버 생성
        self.srv = self.create_service(
            PickObject,
            "pick_object",
            self.handle_pick_object,
        )
        self.get_logger().info("✅ VisionPickNode ready. Service: /pick_object")

    # ============================================
    # YOLO로 원하는 클래스 감지
    # ============================================
    def detect_target_object(self, color_img, target_name: str):
        """
        color_img: BGR 이미지
        target_name: YOLO 클래스 이름 (예: "scissors")
        return: (cx, cy, conf) 또는 None
        """
        results = self.yolo_model.predict(
            source=color_img,
            conf=self.conf_th,
            imgsz=640,
            verbose=False,
        )
        res = results[0]
        boxes = res.boxes
        annotated = res.plot()  # 디버깅용

        if boxes is None or len(boxes) == 0:
            self.get_logger().info("[YOLO] 박스 없음.")
            return None, annotated

        names = res.names
        candidates = []

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id]
            conf = float(box.conf[0].item())

            # target_name 과 동일한 클래스만 후보
            if cls_name.lower() == target_name.lower() and conf >= self.conf_th:
                candidates.append((conf, i, cls_name))

        if not candidates:
            self.get_logger().info(f"[YOLO] '{target_name}' 클래스 탐지 실패.")
            return None, annotated

        # conf 높은 순으로 정렬 후 하나 선택
        candidates.sort(reverse=True)
        best_conf, best_idx, best_name = candidates[0]
        best_box = boxes[best_idx]

        xyxy = best_box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

        print(f"[YOLO] Target: {best_name}, conf={best_conf:.2f}, pixel=({cx},{cy})")
        return (cx, cy, best_conf), annotated

    # ============================================
    # 픽셀 → depth
    # ============================================
    def get_depth_value(self, cx, cy, depth_frame):
        h, w = depth_frame.shape
        if 0 <= cx < w and 0 <= cy < h:
            depth_value = depth_frame[cy, cx]
            return depth_value if depth_value != 0 else None

        self.get_logger().warn(f"⚠️ depth out of range: ({cx}, {cy})")
        return None

    # ============================================
    # 픽셀 + depth → 카메라 좌표
    # ============================================
    def get_camera_pos(self, center_x, center_y, center_z, intrinsics):
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        ppx = intrinsics["ppx"]
        ppy = intrinsics["ppy"]

        camera_x = (center_x - ppx) * center_z / fx
        camera_y = (center_y - ppy) * center_z / fy
        camera_z = center_z

        return (camera_x, camera_y, camera_z)

    # ============================================
    # 로봇 posx → 4x4 변환행렬
    # ============================================
    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    # ============================================
    # 카메라 좌표 → 로봇 base 좌표
    # ============================================
    def transform_to_base(self, camera_coords):
        from DSR_ROBOT2 import get_current_posx  # DR_init 설정 후 import 가능

        coord = np.append(np.array(camera_coords), 1.0)

        # 현재 TCP 포즈 (base → gripper)
        tcp_pose = get_current_posx()[0]  # [x, y, z, rx, ry, rz]
        base2gripper = self.get_robot_pose_matrix(*tcp_pose)

        # base2cam = base2gripper @ gripper2cam
        base2cam = base2gripper @ self.gripper2cam
        td_coord = base2cam @ coord

        return td_coord[:3]

    # ============================================
    # 실제 Pick 동작
    # ============================================
    def pick_and_drop(self, x, y, z):
        from DSR_ROBOT2 import (
            movej,
            movel,
            wait,
            DR_MV_MOD_ABS,
            DR_MV_RA_DUPLICATE,
            get_current_posx,
        )
        from DR_common2 import posx

        print(f"[MOVE] Pick → base({x:.3f}, {y:.3f}, {z:.3f})")

        current_pos = get_current_posx()[0]

        approach_pos = posx([
            x,
            y,
            z + 205.0,
            current_pos[3],
            current_pos[4],
            current_pos[5],
        ])

        movel(
            approach_pos,
            vel=self.LIN_VEL,
            acc=self.LIN_ACC,
            mod=DR_MV_MOD_ABS,
            ra=DR_MV_RA_DUPLICATE,
        )

        self.gripper.close_gripper()
        wait(1)

        movej(
            self.CUSTOM_HOME_JOINT,
            vel=self.JNT_VEL,
            acc=self.JNT_ACC,
            mod=DR_MV_MOD_ABS,
            ra=DR_MV_RA_DUPLICATE,
        )

        self.gripper.open_gripper()
        wait(1)

    # ============================================
    # /pick_object 서비스 콜백
    # ============================================
    def handle_pick_object(self, request, response):
        target_name = request.object_name.strip()
        if not target_name:
            response.success = False
            response.message = "object_name is empty"
            response.x = response.y = response.z = 0.0
            response.confidence = 0.0
            return response

        self.get_logger().info(f"🔔 pick_object 요청: '{target_name}'")

        # 1) RealSense에서 최신 프레임 한 장 가져오기
        color_img = None
        depth_frame = None
        for _ in range(10):  # 최대 10번 정도 시도
            color_img = self.img_node.get_color_frame()
            depth_frame = self.img_node.get_depth_frame()
            if color_img is not None and depth_frame is not None:
                break

        if color_img is None or depth_frame is None:
            self.get_logger().error("❌ RealSense frame 을 가져오지 못했습니다.")
            response.success = False
            response.message = "No camera frame available"
            response.x = response.y = response.z = 0.0
            response.confidence = 0.0
            return response

        # 2) YOLO로 타겟 탐지
        target_info, annotated = self.detect_target_object(color_img, target_name)
        # 디버깅용: 필요하면 show / save 가능
        # cv2.imshow("debug", annotated); cv2.waitKey(1)

        if target_info is None:
            response.success = False
            response.message = f"No '{target_name}' detected"
            response.x = response.y = response.z = 0.0
            response.confidence = 0.0
            return response

        cx, cy, conf = target_info

        if conf < PICK_CONF_TH:
            msg = f"conf={conf:.2f} < PICK_CONF_TH={PICK_CONF_TH:.2f}, pick skip"
            self.get_logger().warn(msg)
            response.success = False
            response.message = msg
            response.x = response.y = response.z = 0.0
            response.confidence = float(conf)
            return response

        # 3) depth → camera → base 좌표
        z = self.get_depth_value(cx, cy, depth_frame)
        if z is None:
            msg = "Depth invalid at target pixel, skip"
            self.get_logger().warn(msg)
            response.success = False
            response.message = msg
            response.x = response.y = response.z = 0.0
            response.confidence = float(conf)
            return response

        cam_pos = self.get_camera_pos(cx, cy, z, self.intrinsics)
        base_pos = self.transform_to_base(cam_pos)
        bx, by, bz = base_pos

        self.get_logger().info(
            f"[DEBUG] target='{target_name}', pixel=({cx},{cy}), depth={z:.1f}, "
            f"cam=({cam_pos[0]:.1f},{cam_pos[1]:.1f},{cam_pos[2]:.1f}), "
            f"base=({bx:.1f},{by:.1f},{bz:.1f}), conf={conf:.2f}"
        )

        # 4) 실제 pick 동작 수행
        try:
            self.pick_and_drop(bx, by, bz)
            response.success = True
            response.message = "OK"
        except Exception as e:
            self.get_logger().error(f"❌ pick_and_drop 중 예외: {e}")
            response.success = False
            response.message = f"pick_and_drop error: {e}"

        response.x = float(bx)
        response.y = float(by)
        response.z = float(bz)
        response.confidence = float(conf)
        return response


def main(args=None):
    rclpy.init(args=args)

    # 1) Doosan 제어용 노드 먼저 생성
    dsr_node = rclpy.create_node("dsr_example_demo_py", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    # 2) RealSense ImgNode 생성
    img_node = ImgNode()

    # 3) VisionPickNode에 img_node 주입
    vp_node = VisionPickNode(img_node)

    # 4) Executor에 세 노드 등록 후 spin
    executor = SingleThreadedExecutor()
    executor.add_node(dsr_node)
    executor.add_node(img_node)
    executor.add_node(vp_node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        dsr_node.destroy_node()
        img_node.destroy_node()
        vp_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
