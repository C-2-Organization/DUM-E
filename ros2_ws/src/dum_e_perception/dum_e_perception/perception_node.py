# dum_e_perception/perception_node.py
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

from dum_e_interfaces.srv import GetObjectPose
from .remote_detector import RemoteVisionDetector
from .pose_estimator import PoseEstimator
from dum_e_utils.realsense import ImgNode


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('dum_e_perception')

        # ----------------------------
        # Params
        # ----------------------------
        self.declare_parameter("remote_endpoint", "http://3.39.1.188:8000/detect")
        self.declare_parameter("remote_top_k", 5)
        self.declare_parameter("remote_box_threshold", 0.35)
        self.declare_parameter("remote_text_threshold", 0.25)

        self.remote_endpoint = self.get_parameter("remote_endpoint").value
        self.remote_top_k = int(self.get_parameter("remote_top_k").value)
        self.remote_box_threshold = float(self.get_parameter("remote_box_threshold").value)
        self.remote_text_threshold = float(self.get_parameter("remote_text_threshold").value)

        self.get_logger().info(f"[Perception] Endpoint={self.remote_endpoint}")
        self.get_logger().info(f"[Perception] Mode=Center Point Tracking (Lucas-Kanade)")

        # ----------------------------
        # Remote detector init
        # ----------------------------
        self.detector_remote = RemoteVisionDetector(self.remote_endpoint)

        # ----------------------------
        # Point Tracking State (Optical Flow)
        # ----------------------------
        self.prev_gray = None       # 이전 프레임 (Gray)
        self.track_point = None     # 추적 중인 중심점 좌표 (numpy array)
        self.track_wh = None        # 추적 중인 물체의 크기 (w, h) - Depth 계산용
        self.tracking_object_name = None 
        
        # Lucas-Kanade 파라미터
        self.lk_params = dict(
            winSize=(21, 21), 
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # RealSense init
        self.camera = ImgNode(self)
        self.estimator = None
        self.bridge = CvBridge()

        # Service
        self.srv = self.create_service(GetObjectPose, 'get_object_pose', self.handle_get_object_pose)

    def _ensure_estimator(self):
        if self.estimator is not None:
            return True
        intr = self.camera.get_camera_intrinsic()
        if intr is None:
            return False
        self.estimator = PoseEstimator(intrinsics=intr)
        self.get_logger().info(f"PoseEstimator initialized: {intr}")
        return True

    def _norm_to_xywh(self, bbox_norm, w, h):
        """Normalized [x1, y1, x2, y2] -> Pixel [x, y, w, h]"""
        x1, y1, x2, y2 = bbox_norm
        x = max(0, min(int(x1 * w), w - 1))
        y = max(0, min(int(y1 * h), h - 1))
        ww = max(1, min(int((x2 - x1) * w), w - x))
        hh = max(1, min(int((y2 - y1) * h), h - y))
        return x, y, ww, hh

    def _xywh_to_norm(self, bbox_xywh, w, h):
        """Pixel [x, y, w, h] -> Normalized [x1, y1, x2, y2]"""
        x, y, ww, hh = bbox_xywh
        x1 = x / w
        y1 = y / h
        x2 = (x + ww) / w
        y2 = (y + hh) / h
        return [x1, y1, x2, y2]

    def _detect_remote_best(self, color_bgr, object_name: str):
        try:
            detections = self.detector_remote.detect(
                color_bgr,
                text_prompt=object_name,
                top_k=self.remote_top_k,
                box_threshold=self.remote_box_threshold,
                text_threshold=self.remote_text_threshold,
            )
            if not detections:
                return None
            return max(detections, key=lambda d: d["confidence"])
        except Exception as e:
            self.get_logger().error(f"Remote detection error: {e}")
            return None

    def handle_get_object_pose(self, request, response):
        target_name = request.object_name
        use_tracking = getattr(request, 'use_tracking', True)

        if not self._ensure_estimator():
            response.success = False
            response.message = "Camera intrinsics not ready"
            return response

        color, depth = self.camera.get_frame()
        if color is None or depth is None:
            response.success = False
            response.message = "Frames not ready"
            return response
        
        h_img, w_img = color.shape[:2]
        gray_frame = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        bbox_norm = None
        source_method = "NONE"
        confidence = 0.0

        # ---------------------------------------------------------
        # 1. Point Tracking 시도 (Optical Flow)
        # ---------------------------------------------------------
        # 조건: 이전 포인트 존재, 이전 프레임 존재, 타겟 이름 일치, 트래킹 요청
        if (self.track_point is not None and 
            self.prev_gray is not None and
            self.tracking_object_name == target_name and 
            use_tracking):

            # Lucas-Kanade Optical Flow 계산
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray_frame, self.track_point, None, **self.lk_params
            )

            # st[0] == 1 이면 추적 성공
            if st[0] == 1:
                # 새 좌표 업데이트
                new_cx, new_cy = p1[0].ravel()
                
                # 경계 밖으로 나갔는지 체크
                if 0 <= new_cx < w_img and 0 <= new_cy < h_img:
                    self.track_point = p1 # 상태 업데이트
                    
                    # 3D Depth 계산을 위해 BBox 복원 (중심점은 이동, 크기는 고정 가정)
                    saved_w, saved_h = self.track_wh
                    t_x = int(new_cx - saved_w / 2)
                    t_y = int(new_cy - saved_h / 2)
                    
                    # Normalized BBox 생성
                    bbox_norm = self._xywh_to_norm((t_x, t_y, saved_w, saved_h), w_img, h_img)
                    source_method = "TRACK_POINT"
                    confidence = 1.0
                else:
                    self.track_point = None # 화면 밖으로 나감
            else:
                self.track_point = None # 추적 실패 (가려짐 등)

        # ---------------------------------------------------------
        # 2. Tracking 실패 또는 초기 진입 -> Remote Detection
        # ---------------------------------------------------------
        if bbox_norm is None:
            # 트래킹 상태 리셋
            self.track_point = None 
            
            best_det = self._detect_remote_best(color, target_name)
            
            if best_det is not None:
                bbox_norm = best_det.get("bbox")
                confidence = float(best_det.get("confidence", 0.0))
                source_method = "REMOTE_GDINO"

                # Point Tracking 초기화
                if bbox_norm is not None:
                    # Normalized -> Pixel
                    px, py, pw, ph = self._norm_to_xywh(bbox_norm, w_img, h_img)
                    
                    # 중심점 계산
                    cx = px + pw / 2.0
                    cy = py + ph / 2.0
                    
                    # 상태 저장
                    self.track_point = np.array([[cx, cy]], dtype=np.float32)
                    self.track_wh = (pw, ph)
                    self.tracking_object_name = target_name
                    self.get_logger().info(f"Point Tracking Init: '{target_name}' at ({cx:.1f}, {cy:.1f})")

        # 현재 프레임을 '이전 프레임'으로 저장 (다음 루프용)
        self.prev_gray = gray_frame.copy()

        # ---------------------------------------------------------
        # 3. 3D Pose Estimation
        # ---------------------------------------------------------
        if bbox_norm is None:
            response.success = False
            response.message = f"Object '{target_name}' not found"
            return response

        pose = self.estimator.bbox_to_3d_heuristic(
            bbox_norm,
            depth,
            roi_expand=0.08,
            z_min=150.0,
            z_max=2000.0,
            median_band=30,
        )

        if pose is None:
            response.success = False
            response.message = "Invalid depth (z=0)"
            return response

        x, y, z = pose
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "camera_link"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z

        response.success = True
        response.message = f"ok ({source_method})"
        response.pose = pose_msg
        response.confidence = confidence
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    node.get_logger().info("=== dum_e_perception (Point Tracking) Started ===")
    
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
