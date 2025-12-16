# dum_e_perception/perception_node.py
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from dum_e_interfaces.srv import GetObjectPose
from .yolo_detector import YOLODetector
from .remote_detector import RemoteVisionDetector
from .pose_estimator import PoseEstimator
from dum_e_utils.realsense import ImgNode
from ament_index_python.packages import get_package_share_directory

from .hand_detector import MediaPipeHandDetector
import numpy as np

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('dum_e_perception')

        # ----------------------------
        # Params
        # ----------------------------
        self.declare_parameter("detector_mode", "remote_gdino")  # local_yolo | remote_gdino
        self.declare_parameter("remote_endpoint", "http://3.39.1.188:8000/detect")
        self.declare_parameter("remote_top_k", 5)
        self.declare_parameter("remote_box_threshold", 0.35)
        self.declare_parameter("remote_text_threshold", 0.25)

        self.detector_mode = self.get_parameter("detector_mode").value
        self.remote_endpoint = self.get_parameter("remote_endpoint").value
        self.remote_top_k = int(self.get_parameter("remote_top_k").value)
        self.remote_box_threshold = float(self.get_parameter("remote_box_threshold").value)
        self.remote_text_threshold = float(self.get_parameter("remote_text_threshold").value)

        self.get_logger().info(f"[Perception] detector_mode={self.detector_mode}")
        self.get_logger().info(f"[Perception] remote_endpoint={self.remote_endpoint}")

        # ----------------------------
        # Local YOLO init (keep for local testing)
        # ----------------------------
        share_dir = get_package_share_directory("dum_e_perception")
        model_path = os.path.join(share_dir, "models", "yolov8s-worldv2.pt")

        self.detector_local = None
        if self.detector_mode == "local_yolo":
            if os.path.exists(model_path):
                self.get_logger().info(f"Loading YOLO model: {model_path}")
                self.detector_local = YOLODetector(model_path)  # 이제 cpu fallback도 됨 :contentReference[oaicite:5]{index=5}
            else:
                self.get_logger().warn(f"YOLO model not found: {model_path} (local_yolo mode will fail)")
        else:
            self.get_logger().info("Skip local YOLO init (detector_mode != local_yolo)")

        # else:
        #     self.get_logger().warn(f"YOLO model not found: {model_path} (local_yolo mode will fail)")

        # ----------------------------
        # Remote detector init
        # ----------------------------
        self.detector_remote = RemoteVisionDetector(self.remote_endpoint)

        # RealSense init
        self.camera = ImgNode(self)
        self.estimator = None
        self.bridge = CvBridge()
        
                # ----------------------------
        # MediaPipe Hands (for handover)
        # ----------------------------
        self.declare_parameter("handover_min_det_conf", 0.6)
        self.declare_parameter("handover_min_track_conf", 0.5)
        self.declare_parameter("handover_z_min_mm", 150.0)
        self.declare_parameter("handover_z_max_mm", 2000.0)
        self.declare_parameter("handover_roi_px", 18)          # depth 샘플 ROI half-size
        self.declare_parameter("handover_min_valid_px", 30)    # depth 유효 픽셀 최소
        self.declare_parameter("handover_median_band_mm", 30.0)

        self.handover_min_det_conf = float(self.get_parameter("handover_min_det_conf").value)
        self.handover_min_track_conf = float(self.get_parameter("handover_min_track_conf").value)
        self.handover_z_min_mm = float(self.get_parameter("handover_z_min_mm").value)
        self.handover_z_max_mm = float(self.get_parameter("handover_z_max_mm").value)
        self.handover_roi_px = int(self.get_parameter("handover_roi_px").value)
        self.handover_min_valid_px = int(self.get_parameter("handover_min_valid_px").value)
        self.handover_median_band_mm = float(self.get_parameter("handover_median_band_mm").value)

        self.hand_detector = MediaPipeHandDetector(
            max_num_hands=1,
            min_detection_confidence=self.handover_min_det_conf,
            min_tracking_confidence=self.handover_min_track_conf,
        )

        # Service 등록
        self.srv = self.create_service(GetObjectPose, 'get_object_pose', self.handle_get_object_pose)

    def _point_to_3d_heuristic(self, u: int, v: int, depth_image):
        """
        u,v (pixel) 주변 ROI에서 유효 depth의 median을 잡아서 3D로 변환.
        depth는 RealSense aligned depth라서 mm일 가능성이 높고,
        기존 bbox_to_3d_heuristic도 z_min=150~2000(mm) 기준이라 동일 기준 사용. :contentReference[oaicite:4]{index=4}
        """
        h, w = depth_image.shape[:2]
        u = max(0, min(w - 1, int(u)))
        v = max(0, min(h - 1, int(v)))

        r = self.handover_roi_px
        x1 = max(0, u - r)
        x2 = min(w, u + r + 1)
        y1 = max(0, v - r)
        y2 = min(h, v + r + 1)

        roi = depth_image[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # 유효 depth 마스크 (mm 기준)
        valid = np.isfinite(roi) & (roi > self.handover_z_min_mm) & (roi < self.handover_z_max_mm)
        if int(valid.sum()) < self.handover_min_valid_px:
            return None

        vals = roi[valid].astype(np.float32)
        z_med = float(np.median(vals))

        # median band 안에서 ROI 중심에 가까운 픽셀 선택 (bbox_to_3d_heuristic 방식과 동일 컨셉) :contentReference[oaicite:5]{index=5}
        band = valid & (np.abs(roi - z_med) <= self.handover_median_band_mm)
        if int(band.sum()) < max(10, self.handover_min_valid_px // 3):
            band = valid

        ys, xs = np.where(band)
        if len(xs) == 0:
            return None

        rc = (roi.shape[0] - 1) * 0.5
        cc = (roi.shape[1] - 1) * 0.5
        d2 = (ys - rc) ** 2 + (xs - cc) ** 2
        idx = int(np.argmin(d2))

        uu = x1 + int(xs[idx])
        vv = y1 + int(ys[idx])
        z = float(depth_image[vv, uu])
        if not np.isfinite(z) or z <= 0.0:
            return None

        # intrinsics는 PoseEstimator와 동일 키로 들어옴 (fx,fy,ppx/ppy) :contentReference[oaicite:6]{index=6}
        intr = self.camera.get_camera_intrinsic()
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr.get("cx", intr.get("ppx")))
        cy = float(intr.get("cy", intr.get("ppy")))

        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        return float(x), float(y), float(z)

    def _ensure_estimator(self):
        if self.estimator is not None:
            return True

        intr = self.camera.get_camera_intrinsic()
        if intr is None:
            return False

        self.estimator = PoseEstimator(intrinsics=intr)
        self.get_logger().info(f"PoseEstimator initialized with intrinsics: {intr}")
        return True

    def _detect_best(self, color_bgr, object_name: str):
        """
        Returns best detection dict that contains:
          - bbox: [x1,y1,x2,y2] normalized
          - confidence
          - class_name
        """
        if self.detector_mode == "local_yolo":
            if self.detector_local is None:
                raise RuntimeError("Local YOLO detector not initialized (model missing?)")

            detections = self.detector_local.detect(color_bgr, classes=[object_name], conf_threshold=0.3)
            candidates = [d for d in detections if d["class_name"] == object_name]
            if not candidates:
                return None
            return max(candidates, key=lambda d: d["confidence"])

        elif self.detector_mode == "remote_gdino":
            detections = self.detector_remote.detect(
                color_bgr,
                text_prompt=object_name,
                top_k=self.remote_top_k,
                box_threshold=self.remote_box_threshold,
                text_threshold=self.remote_text_threshold,
            )
            if not detections:
                return None
            # remote는 phrase가 정확히 object_name과 다를 수 있으니 score 기준
            return max(detections, key=lambda d: d["confidence"])

        else:
            raise ValueError(f"Unknown detector_mode: {self.detector_mode}")

    def handle_get_object_pose(self, request, response):
        object_name = request.object_name

        if not self._ensure_estimator():
            self.get_logger().warn("Camera intrinsics not ready yet")
            response.success = False
            response.message = "Camera intrinsics not ready yet"
            return response

        color, depth = self.camera.get_frame()
        if color is None or depth is None:
            self.get_logger().warn("Camera frames not ready yet")
            response.success = False
            response.message = "Camera frames not ready yet"
            return response

        # ✅ HANDOVER 전용: MediaPipe Hands
        if object_name.lower() == "handover":
            det = None
            self.get_logger().info(f"[HANDOVER] color shape={getattr(color, 'shape', None)}, dtype={getattr(color, 'dtype', None)}")
            try:
                det = self.hand_detector.detect(color)
            except Exception as e:
                self.get_logger().error(f"[HANDOVER] MediaPipe error: {e}")
                response.success = False
                response.message = f"MediaPipe error: {e}"
                return response

            if det is None:
                response.success = False
                response.message = "No hand detected (mediapipe)"
                response.confidence = 0.0
                return response

            xyz = self._point_to_3d_heuristic(det.u, det.v, depth)
            if xyz is None:
                response.success = False
                response.message = "Invalid depth around hand (z=0 or insufficient valid pixels)"
                response.confidence = float(det.confidence)
                return response

            x, y, z = xyz
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "camera_link"
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.pose.position.x = x
            pose_msg.pose.position.y = y
            pose_msg.pose.position.z = z

            response.success = True
            response.message = f"ok (mediapipe:{det.handedness})"
            response.pose = pose_msg
            response.confidence = float(det.confidence)
            return response

        try:
            best = self._detect_best(color, object_name)
        except Exception as e:
            self.get_logger().error(f"Detection error: {e}")
            response.success = False
            response.message = f"Detection error: {e}"
            return response

        if best is None:
            response.success = False
            response.message = f"No object '{object_name}' detected"
            return response

        bbox_norm = best.get("bbox", None)
        if bbox_norm is None or len(bbox_norm) != 4:
            response.success = False
            response.message = "Detection missing normalized bbox"
            return response

        pose = self.estimator.bbox_to_3d_heuristic(
            best["bbox"],
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
        response.message = f"ok ({best.get('source', 'detector')})"
        response.pose = pose_msg
        response.confidence = float(best.get("confidence", 0.0))
        return response

def test(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()

    node.get_logger().info("=== Perception Test Mode ===")
    node.get_logger().info("Waiting for camera intrinsics & frames...")

    # 1) intrinsics & frame 준비될 때까지 대기
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

        intr = node.camera.get_camera_intrinsic()
        color, depth = node.camera.get_frame()

        if intr is not None and color is not None and depth is not None:
            node.get_logger().info(f"Camera ready. Intrinsics: {intr}")
            break

    # 2) 이제 서비스 클라이언트로 자기 자신 호출
    from dum_e_interfaces.srv import GetObjectPose
    client = node.create_client(GetObjectPose, 'get_object_pose')

    node.get_logger().info("Waiting for 'get_object_pose' service...")
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info("Service not available, waiting...")

    node.get_logger().info("=== Perception Test: Searching for 'scissors' ===")
    req = GetObjectPose.Request()
    req.object_name = "scissors"
    req.use_tracking = False

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future)

    node.get_logger().info(f"Service result: {future.result()}")

    node.destroy_node()
    rclpy.shutdown()

def main(args=None):
    """정식 Perception 노드: 서비스만 띄우고 spin."""
    rclpy.init(args=args)
    node = PerceptionNode()
    node.get_logger().info("=== dum_e_perception node started ===")
    node.get_logger().info("Service: /get_object_pose")

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
