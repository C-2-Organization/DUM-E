import sys
import os

# ROS2 패키지 경로 추가
ROS2_WS = os.path.dirname(os.path.abspath(__file__)) + "/ros2_ws"
if os.path.exists(ROS2_WS):
    sys.path.insert(0, os.path.join(ROS2_WS, "src/utils/utils"))
    sys.path.insert(0, os.path.join(ROS2_WS, "install/utils/lib/python3.10/site-packages"))

import rclpy
from rclpy.node import Node
import numpy as np
import cv2, time
from scipy.spatial.transform import Rotation   # ✅ 어제 잘 되던 좌표변환 방식 사용
import DR_init
try:
    from onrobot import RG
    GRIPPER_AVAILABLE = True
except ImportError:
    GRIPPER_AVAILABLE = False
    print("⚠️  Warning: onrobot 모듈을 로드할 수 없습니다. 그리퍼 기능이 비활성화됩니다.")

from realsense import ImgNode
from ultralytics import YOLOWorld
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# ============================================================
# 사용자 설정
# ============================================================

ROBOT_ID = "dsr01"
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502  # ✅ 수정: 문자열 "502" → 정수 502 (Modbus 포트는 int 여야 함)

# YOLO 관련 설정
MODEL_PATH = "yolov8s-worldv2.pt"  # ✅ 자동으로 다운로드됨 (인터넷 필요)
# 또는 로컬 경로: "/path/to/model/yolov8s-worldv2.pt"
YOLO_CLASSES = ["person", "cup", "scissors", "box cutter", "bottle", "laptop"]
TARGET_CLASSES = {"cup", "scissors", "box cutter"}  # pick&place 대상 클래스

YOLO_CONF_TH = 0.5   # YOLO에서 후보로 인정할 최소 conf
PICK_CONF_TH = 0.5   # 실제 pick_place 할 최소 conf (나중에 수치만 바꾸면 됨)


# ============================================================
# TestNode 클래스 정의
# ============================================================

class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")

        # 1) RealSense 노드 초기화
        self.img_node = ImgNode()

        # 🔹 intrinsics가 None이 아니게 될 때까지 기다리기
        self.intrinsics = None
        while self.intrinsics is None:
            self.get_logger().info("📷 camera intrinsics 대기 중...")
            rclpy.spin_once(self.img_node, timeout_sec=0.1)
            self.intrinsics = self.img_node.get_camera_intrinsic()

        self.get_logger().info(f"📷 camera intrinsics 수신 완료: {self.intrinsics}")

        # 2) 변환 행렬 로드 (gripper ↔ camera)
        # 현재 디렉토리 또는 패키지 경로에서 찾기
        calib_path = None
        possible_paths = [
            "T_gripper2camera.npy",
            os.path.join(ROS2_WS, "src/perception/config/T_gripper2camera.npy"),
            os.path.join(ROS2_WS, "install/perception/share/perception/config/T_gripper2camera.npy"),
            "/home/rokey/DUM-E/ros2_ws/src/perception/config/T_gripper2camera.npy",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                calib_path = path
                break
        
        if calib_path is None:
            self.get_logger().error("❌ T_gripper2camera.npy를 찾을 수 없습니다!")
            raise FileNotFoundError("T_gripper2camera.npy not found")
        
        self.gripper2cam = np.load(calib_path)
        self.get_logger().info(f"✅ 변환 행렬 로드됨: {calib_path}")

        # 3) 로봇 초기 자세 / 그리퍼 연결
        self.JReady = posj([0, 0, 90, 0, 90, -90])
        
        # 그리퍼 초기화 (사용 가능한 경우만)
        if GRIPPER_AVAILABLE:
            try:
                self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
            except Exception as e:
                print(f"⚠️  그리퍼 연결 실패: {e}")
                self.gripper = None
        else:
            self.gripper = None

        # 4) YOLOWorld 모델 로드
        self.yolo_model = YOLOWorld(MODEL_PATH)
        self.yolo_model.set_classes(YOLO_CLASSES)
        self.target_classes = TARGET_CLASSES
        self.conf_th = YOLO_CONF_TH
        
        self.LIN_VEL = [150.0, 300.0]
        self.LIN_ACC = [150.0, 150.0]

        self.JNT_VEL = 150.0
        self.JNT_ACC = 300.0

        self.CUSTOM_HOME_JOINT = [0, 0, 90, 0, 90, 0]

        # 5) RViz 마커 퍼블리셔 추가
        self.marker_pub = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)

    # ============================================================
    # RViz 마커 퍼블리시
    # ============================================================
    def publish_marker(self, base_pos, conf, cls_name):
        """
        감지된 객체를 RViz에 마커로 시각화
        base_pos: [x, y, z] (베이스 좌표)
        conf: 신뢰도
        cls_name: 클래스 이름
        """
        marker_array = MarkerArray()
        
        # Frame ID 설정 - base_link에 표시 (로봇 움직임과 함께 보임)
        frame_id = "base_link"  # 로봇 베이스 프레임
        
        # 1) 구(Sphere) 마커 - 객체 위치
        sphere = Marker()
        sphere.header.frame_id = frame_id
        sphere.header.stamp = self.get_clock().now().to_msg()
        sphere.ns = "detected_objects"
        sphere.id = 0  # 고정된 ID (매번 덮어쓰기)
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        
        sphere.pose.position.x = float(base_pos[0])  # 이미 m 단위
        sphere.pose.position.y = float(base_pos[1])
        sphere.pose.position.z = float(base_pos[2])
        sphere.pose.orientation.w = 1.0
        
        sphere.scale.x = 0.05
        sphere.scale.y = 0.05
        sphere.scale.z = 0.05
        
        # 신뢰도에 따라 색상 변경 (초록→노랑→빨강)
        if conf > 0.8:
            sphere.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green
        elif conf > 0.6:
            sphere.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)  # Yellow
        else:
            sphere.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red
        
        marker_array.markers.append(sphere)
        
        # 2) 텍스트 마커 - 객체 이름 및 신뢰도
        text = Marker()
        text.header.frame_id = frame_id
        text.header.stamp = self.get_clock().now().to_msg()
        text.ns = "detected_objects_text"
        text.id = 1  # 고정된 ID (매번 덮어쓰기)
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        
        text.pose.position.x = float(base_pos[0])  # 이미 m 단위
        text.pose.position.y = float(base_pos[1])
        text.pose.position.z = float(base_pos[2]) + 0.1
        text.pose.orientation.w = 1.0
        
        text.text = f"{cls_name}\n(conf: {conf:.2f})"
        text.scale.z = 0.05
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)  # White
        
        marker_array.markers.append(text)
        
        # 퍼블리시 (로그 없음)
        self.marker_pub.publish(marker_array)

    # ============================================================
    # YOLO로 타겟 객체 감지
    # ============================================================
    def detect_target_object(self, color_img):
        results = self.yolo_model.predict(
            source=color_img,
            conf=self.conf_th,   # YOLO 최소 conf
            imgsz=640,
            verbose=False,
        )

        res = results[0]
        boxes = res.boxes
        annotated = res.plot()  # 바운딩 박스 그려진 이미지

        if boxes is None or len(boxes) == 0:
            return None, annotated

        names = res.names
        candidates = []

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id]
            conf = float(box.conf[0].item())
            if cls_name in self.target_classes and conf >= self.conf_th:
                candidates.append((conf, i, cls_name))

        if not candidates:
            return None, annotated

        # conf 높은 순으로 정렬 후 가장 높은 한 개 선택
        candidates.sort(reverse=True)
        best_conf, best_idx, best_name = candidates[0]
        best_box = boxes[best_idx]

        xyxy = best_box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

        print(f"[YOLO] Target: {best_name}, conf={best_conf:.2f}, pixel=({cx},{cy})")
        # 🔹 conf와 cls_name까지 같이 반환
        return (cx, cy, best_conf, best_name), annotated

    # ============================================================
    # 픽셀 → 깊이값 변환
    # ============================================================
    def get_depth_value(self, cx, cy, depth_frame):
        height, width = depth_frame.shape
        if 0 <= cx < width and 0 <= cy < height:
            depth_value = depth_frame[cy, cx]
            return depth_value if depth_value != 0 else None

        self.get_logger().warn(f"⚠️ depth out of range: ({cx}, {cy})")
        return None

    # ============================================================
    # 픽셀 → 카메라 좌표
    # ============================================================
    def get_camera_pos(self, center_x, center_y, center_z, intrinsics):
        # intrinsics는 dict 형식이라고 가정: {"fx", "fy", "ppx", "ppy", ...}
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        ppx = intrinsics["ppx"]
        ppy = intrinsics["ppy"]

        # RealSense 깊이값은 mm 단위 → m 단위로 변환
        center_z_m = center_z / 1000.0
        
        camera_x = (center_x - ppx) * center_z_m / fx
        camera_y = (center_y - ppy) * center_z_m / fy
        camera_z = center_z_m

        return (camera_x, camera_y, camera_z)

    
    # ============================================================
    # 로봇 포즈(x,y,z,rx,ry,rz) → 4x4 변환행렬
    # ============================================================
    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    # ============================================================
    # 카메라 → 로봇 베이스 좌표
    # ============================================================
    def transform_to_base(self, camera_coords):
        """
        camera_coords: 카메라 좌표계 기준 (x, y, z)
        return: 로봇 base 좌표계 기준 (x, y, z)
        """
        coord = np.append(np.array(camera_coords), 1.0)  # homogeneous

        try:
            # 현재 TCP 포즈 가져와서 base→gripper 변환행렬 생성
            tcp_pose = get_current_posx()[0]   # [x, y, z, rx, ry, rz]
            base2gripper = self.get_robot_pose_matrix(*tcp_pose)
        except Exception as e:
            # 로봇이 연결되지 않은 경우, 기본 변환만 사용
            tcp_pose = [0, 0, 0, 0, 0, 0]  # 기본 위치
            base2gripper = self.get_robot_pose_matrix(*tcp_pose)

        # base2cam = base2gripper @ gripper2cam
        base2cam = base2gripper @ self.gripper2cam
        td_coord = base2cam @ coord

        return td_coord[:3]

    # ============================================================
    # 픽앤드롭 실행
    # ============================================================
    def pick_and_drop(self, x, y, z):
        print(f"[MOVE] Pick&Place → base({x:.3f}, {y:.3f}, {z:.3f})")

        try:
            current_pos = get_current_posx()[0]
        except (IndexError, NameError, AttributeError) as e:
            print(f"⚠️  로봇 연결 실패: {e}")
            print("   로봇 연결 후 다시 시도하세요: ./connect_real_robot.sh")
            return

        pick_pos = posx([
            x,
            y,
            z + 205.0,
            current_pos[3],
            current_pos[4],
            current_pos[5],
        ])

        movel(
            pick_pos,
            vel=self.LIN_VEL,
            acc=self.LIN_ACC,
            mod=DR_MV_MOD_ABS,
            ra=DR_MV_RA_DUPLICATE,
        )

        # 그리퍼 사용 (사용 가능한 경우만)
        if self.gripper is not None:
            self.gripper.close_gripper()
            wait(1)
        else:
            print("⚠️  그리퍼 미사용")
            wait(1)

        movej(
            self.CUSTOM_HOME_JOINT,
            vel=self.JNT_VEL,
            acc=self.JNT_ACC,
            mod=DR_MV_MOD_ABS,
            ra=DR_MV_RA_DUPLICATE,
        )

        # 그리퍼 열기 (사용 가능한 경우만)
        if self.gripper is not None:
            self.gripper.open_gripper()
            wait(1)
        else:
            print("⚠️  그리퍼 미사용")
            wait(1)


    # ============================================================
    # 메인 루프: 'p'를 눌렀을 때만 pick&place 실행
    # ============================================================
    def run(self):  # ✅ 수정: 기존 while 루프를 메소드로 이동
        cv2.namedWindow("YOLO_PickPlace")
        print("▶ YOLO_PickPlace 실행 중... 'p' 누르면 픽앤플레이스, ESC 누르면 종료")
        
        frame_count = 0
        last_detection_log = -10  # 로그 간격 조절용

        while rclpy.ok():
            # 최신 이미지/뎁스 갱신
            rclpy.spin_once(self.img_node, timeout_sec=0.1)
            color_img = self.img_node.get_color_frame()
            depth_frame = self.img_node.get_depth_frame()
            if color_img is None or depth_frame is None:
                continue

            # YOLO로 타겟 탐지 (1회만!)
            target_info, annotated = self.detect_target_object(color_img)
            cv2.imshow("YOLO_PickPlace", annotated)

            # 🎯 타겟 감지 시 RViz에 마커 발행
            if target_info is not None:
                cx, cy, conf, cls_name = target_info
                z = self.get_depth_value(cx, cy, depth_frame)
                
                if z is not None:
                    cam_pos = self.get_camera_pos(cx, cy, z, self.intrinsics)
                    base_pos = self.transform_to_base(cam_pos)
                    
                    # 타겟 감지 시 마커 발행
                    self.publish_marker(base_pos, conf, cls_name)
                    
                    # 5프레임마다 로그만 출력
                    if frame_count - last_detection_log >= 5:
                        print(f"[MARKER] {cls_name} @ base({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})")
                        last_detection_log = frame_count

            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC 종료
                print("▶ ESC 입력, 종료합니다.")
                break

            # 'p' 눌렀을 때 + 타겟 있고 + conf 기준 만족하면 pick_place
            if key == ord("p"):
                if target_info is None:
                    print("⚠️ 현재 프레임에서 대상 객체를 찾지 못했습니다. (p 무시)")
                    continue

                cx, cy, conf, cls_name = target_info

                if conf < PICK_CONF_TH:
                    print(f"⚠️ conf={conf:.2f} < {PICK_CONF_TH:.2f} → pick_place 스킵")
                    continue

                z = self.get_depth_value(cx, cy, depth_frame)
                if z is None:
                    print("⚠️ Depth 불량, 스킵")
                    continue

                cam_pos = self.get_camera_pos(cx, cy, z, self.intrinsics)
                base_pos = self.transform_to_base(cam_pos)
                self.pick_and_drop(*base_pos)
            
            frame_count += 1

        cv2.destroyAllWindows()
        self.get_logger().info("YOLO_PickPlace 종료")


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("dsr_example_demo_py", namespace=ROBOT_ID)

    # 1) Doosan용 node 등록
    DR_init.__dsr__node = node

    # 2) 이제서야 DSR_ROBOT2 / DR_common2 import (노드가 준비된 상태)
    from DSR_ROBOT2 import (
        get_current_posx,
        movej,
        movel,
        wait,
        DR_MV_MOD_ABS,
        DR_MV_RA_DUPLICATE,
    )
    from DR_common2 import posx, posj

    # 3) TestNode 생성 후 run() 호출
    test_node = TestNode()
    test_node.run()
