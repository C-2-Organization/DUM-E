# dum_e_perception/tracking_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import requests

from dum_e_interfaces.srv import GetObjectPose
from .pose_estimator import PoseEstimator
from dum_e_utils.realsense import ImgNode

# =========================================================
# 내장된 RemoteDetector
# =========================================================
class RemoteDetector:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def detect(self, image, text_prompt, top_k=5, box_threshold=0.35, text_threshold=0.25):
        _, img_encoded = cv2.imencode(".jpg", image)
        
        # [수정 1] 키 이름 "image"로 통일
        files = {
            "image": ("image.jpg", img_encoded.tobytes(), "image/jpeg")
        }
        data = {
            "text_prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold
        }

        try:
            response = requests.post(self.endpoint, files=files, data=data, timeout=3.0)
            if response.status_code == 200:
                json_resp = response.json()
                # [수정 2] 리스트/딕셔너리 구조 안전 처리
                if isinstance(json_resp, list):
                    return json_resp
                elif isinstance(json_resp, dict):
                    if "results" in json_resp: return json_resp["results"]
                    if "detections" in json_resp: return json_resp["detections"]
                    # 단일 객체 감지 대응
                    if "bbox_xyxy" in json_resp or "box" in json_resp:
                        return [json_resp]
                    return []
                else:
                    return []
            else:
                print(f"[Remote] Error {response.status_code}: {response.text}")
                return []
        except Exception as e:
            print(f"[Remote] Request failed: {e}")
            return []

class TrackingNode(Node):
    def __init__(self):
        super().__init__('dum_e_tracking')

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

        self.get_logger().info(f"[Tracking] Initialized. Remote: {self.remote_endpoint}")

        # ----------------------------
        # Modules Init
        # ----------------------------
        self.detector_remote = RemoteDetector(self.remote_endpoint)
        self.camera = ImgNode(self)
        self.estimator = None # Intrinsic 대기
        self.bridge = CvBridge()

        # ----------------------------
        # Tracker State
        # ----------------------------
        self.tracker = None
        self.tracking_object_name = None
        self.tracker_initialized = False

        # Service Server
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

    def _detect_remote(self, color_bgr, object_name: str):
        """느리지만 정확한 Deep Learning 감지 + 좌표 정규화"""
        detections = self.detector_remote.detect(
            color_bgr,
            text_prompt=object_name,
            top_k=self.remote_top_k,
            box_threshold=self.remote_box_threshold,
            text_threshold=self.remote_text_threshold,
        )
        if not detections:
            return None

        best = None
        best_score = -1.0
        h, w = color_bgr.shape[:2]

        for d in detections:
            if not isinstance(d, dict): continue
            
            # 키 이름 처리 (confidence vs score / bbox vs bbox_xyxy)
            conf = float(d.get("score", d.get("confidence", 0.0)))
            
            # 박스 키 찾기
            raw_box = d.get("bbox_xyxy")
            if raw_box is None: raw_box = d.get("bbox")
            if raw_box is None: raw_box = d.get("box")

            if raw_box and conf > best_score:
                best_score = conf
                best = d
                # 찾은 박스를 정규화된 bbox(0~1)로 변환하여 저장
                # 서버가 픽셀(>1.0)을 주면 정규화하고, 정규화된 값을 주면 그대로 씀
                is_pixel = any(v > 1.0 for v in raw_box)
                if is_pixel:
                     best['bbox_norm'] = [
                         raw_box[0] / w, raw_box[1] / h,
                         raw_box[2] / w, raw_box[3] / h
                     ]
                else:
                    best['bbox_norm'] = raw_box

        return best

    def _init_tracker(self, color_bgr, bbox_norm):
        """OpenCV Tracker 초기화"""
        h, w = color_bgr.shape[:2]
        
        # 정규 좌표(0~1)를 픽셀 좌표로 변환
        x_min = int(bbox_norm[0] * w)
        y_min = int(bbox_norm[1] * h)
        x_max = int(bbox_norm[2] * w)
        y_max = int(bbox_norm[3] * h)
        
        # 안전장치: 이미지 범위 벗어나지 않게
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        # bbox format: (x, y, w, h)
        bw = x_max - x_min
        bh = y_max - y_min
        
        if bw <= 0 or bh <= 0:
            self.get_logger().warn(f"Invalid tracker bbox: {bw}x{bh}")
            return

        bbox_cv = (x_min, y_min, bw, bh)

        self.tracker = cv2.TrackerCSRT_create() 
        self.tracker.init(color_bgr, bbox_cv)
        self.tracker_initialized = True
        self.get_logger().info(f"Tracker Initialized (CSRT): {bbox_cv}")

    def _update_tracker(self, color_bgr):
        """Tracker 업데이트"""
        if self.tracker is None or not self.tracker_initialized:
            return False, None

        success, bbox_cv = self.tracker.update(color_bgr)
        if not success:
            return False, None

        # cv bbox (x, y, w, h) -> norm bbox (x1, y1, x2, y2)
        h, w = color_bgr.shape[:2]
        x, y, bw, bh = bbox_cv
        
        # 경계 체크
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        bbox_norm = [x1/w, y1/h, x2/w, y2/h]
        return True, bbox_norm

    def handle_get_object_pose(self, request, response):
        object_name = request.object_name
        use_tracking = request.use_tracking

        if not self._ensure_estimator():
            response.success = False
            response.message = "Camera intrinsics not ready"
            return response

        color, depth = self.camera.get_frame()
        if color is None or depth is None:
            response.success = False
            response.message = "Camera frames not ready"
            return response

        bbox_norm = None
        source = "none"
        confidence = 0.0

        # ----------------------------------------------------
        # 1. Tracking Mode (OpenCV)
        # ----------------------------------------------------
        # 요청이 '추적 모드'이고, 이전에 같은 물체를 추적 중이었다면
        if use_tracking and self.tracker_initialized and self.tracking_object_name == object_name:
            success, trk_bbox = self._update_tracker(color)
            if success:
                bbox_norm = trk_bbox
                source = "tracker_csrt"
                confidence = 1.0 
            else:
                self.get_logger().warn("Tracker lost object. Fallback to detection.")
                self.tracker_initialized = False

        # ----------------------------------------------------
        # 2. Detection Mode (Remote) - 초기화 또는 실패 시
        # ----------------------------------------------------
        if bbox_norm is None:
            best = self._detect_remote(color, object_name)
            if best and 'bbox_norm' in best:
                # 위 _detect_remote에서 정규화해둔 bbox_norm 사용
                bbox_norm = best['bbox_norm']
                confidence = float(best.get("score", best.get("confidence", 0.0)))
                source = "remote_gdino"
                
                # Tracking을 위해 찾은 박스로 Tracker 초기화
                self._init_tracker(color, bbox_norm)
                self.tracking_object_name = object_name
            else:
                self.tracker_initialized = False
                response.success = False
                response.message = f"Object '{object_name}' not found"
                return response

        # ----------------------------------------------------
        # 3. 3D Pose Calculation
        # ----------------------------------------------------
        # bbox_norm은 이제 항상 0~1 사이의 값임을 보장함
        pose_3d = self.estimator.bbox_to_3d_heuristic(
            bbox_norm,
            depth,
            roi_expand=0.08,
            z_min=150.0,
            z_max=2000.0,
            median_band=30,
        )

        if pose_3d is None:
            response.success = False
            response.message = "Invalid depth (z=0 or out of range)"
            return response

        x, y, z = pose_3d
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "camera_link"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z

        # response 채우기
        response.success = True
        response.message = f"ok ({source})"
        response.pose = pose_msg
        response.confidence = confidence
        response.bbox_norm = [float(v) for v in bbox_norm]
        return response


def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    node.get_logger().info("=== dum_e_tracking node started ===")
    node.get_logger().info("Service: /get_object_pose")

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
