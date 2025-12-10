# perception_node.py
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from dum_e_interfaces.srv import GetObjectPose
from .yolo_detector import YOLODetector
from .pose_estimator import PoseEstimator
from dum_e_utils.realsense import ImgNode
from ament_index_python.packages import get_package_share_directory

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('dum_e_perception')

        # 1) Get model path
        share_dir = get_package_share_directory("dum_e_perception")
        model_path = os.path.join(share_dir, "models", "yolov8s-worldv2.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.get_logger().info(f"Loading YOLO model: {model_path}")

        # YOLO 모델 초기화
        self.detector = YOLODetector(model_path)

        # RealSense 초기화
        self.camera = ImgNode(self)
        self.estimator = None

        self.bridge = CvBridge()

        # Service 등록
        self.srv = self.create_service(
            GetObjectPose,
            'get_object_pose',
            self.handle_get_object_pose
        )

    def handle_get_object_pose(self, request, response):
        object_name = request.object_name

        # intrinsics 준비됐는지 확인
        if self.estimator is None:
            intr = self.camera.get_camera_intrinsic()
            if intr is None:
                self.get_logger().warn("Camera intrinsics not ready yet")
                response.success = False
                response.message = "Camera intrinsics not ready yet"
                return response
            self.estimator = PoseEstimator(intrinsics=intr)
            self.get_logger().info(f"PoseEstimator initialized with intrinsics: {intr}")

        color, depth = self.camera.get_frame()
        if color is None or depth is None:
            self.get_logger().warn("Camera frames not ready yet")
            response.success = False
            response.message = "Camera frames not ready yet"
            return response

        # YOLO 탐색
        detections = self.detector.detect(color)
        candidates = [d for d in detections if d["class_name"] == object_name]

        if len(candidates) == 0:
            response.success = False
            response.message = f"No object '{object_name}' detected"
            return response

        best = max(candidates, key=lambda d: d["confidence"])

        # 3D 변환
        pose = self.estimator.bbox_to_3d(best["bbox"], depth)

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
        response.message = "ok"
        response.pose = pose_msg
        response.confidence = float(best["confidence"])
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
