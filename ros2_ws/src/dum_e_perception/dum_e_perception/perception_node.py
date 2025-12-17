# dum_e_perception/perception_node.py
import os
import cv2
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
        if os.path.exists(model_path):
            self.get_logger().info(f"Loading YOLO model: {model_path}")
            self.detector_local = YOLODetector(model_path)
        else:
            self.get_logger().warn(f"YOLO model not found: {model_path} (local_yolo mode will fail)")

        # ----------------------------
        # Remote detector init
        # ----------------------------
        self.detector_remote = RemoteVisionDetector(self.remote_endpoint)

        # RealSense init
        self.camera = ImgNode(self)
        self.estimator = None
        self.bridge = CvBridge()

        # Service 등록
        self.srv = self.create_service(GetObjectPose, 'get_object_pose', self.handle_get_object_pose)

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
