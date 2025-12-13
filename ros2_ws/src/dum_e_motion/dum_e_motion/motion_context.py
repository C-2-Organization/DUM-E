# dum_e_motion/motion_context.py
import json
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from scipy.spatial.transform import Rotation

from dum_e_interfaces.srv import GetObjectPose
from dsr_msgs2.srv import MoveStop, MovePause, MoveResume

ROBOT_ID = "dsr01"

class MotionCancelled(Exception):
    """
    유저가 STOP을 요청해서 모션이 취소되었음을 나타내는 예외.
    """
    pass


class MotionContext:
    """
    스킬들이 공통으로 사용하는 모션/좌표/서비스 유틸 모음.

    - node: SkillManagerNode (로그, clock 등 사용)
    - gripper: RG 인스턴스
    - gripper2cam: (4x4) np.ndarray (T_gripper2camera)
    """

    def __init__(self, node: Node, gripper, gripper2cam: np.ndarray):
        self.node = node
        self.gripper = gripper
        self.gripper2cam = gripper2cam

        # 모션 파라미터 (필요하면 나중에 param으로 뺄 수 있음)
        self.LIN_VEL = [150.0, 300.0]
        self.LIN_ACC = [150.0, 150.0]
        self.JNT_VEL = 150.0
        self.JNT_ACC = 300.0
        self.CUSTOM_HOME_JOINT = [0, 0, 90, 0, 90, 0]

        self._cancel_flag = False
        self._cancel_lock = threading.Lock()

        stop_srv_name   = f"/{ROBOT_ID}/motion/move_stop"
        pause_srv_name  = f"/{ROBOT_ID}/motion/move_pause"
        resume_srv_name = f"/{ROBOT_ID}/motion/move_resume"

        self.stop_client   = node.create_client(MoveStop,   stop_srv_name)
        self.pause_client  = node.create_client(MovePause,  pause_srv_name)
        self.resume_client = node.create_client(MoveResume, resume_srv_name)

        self.node.get_logger().info(f"[SAFETY] Using STOP   : {stop_srv_name}")
        self.node.get_logger().info(f"[SAFETY] Using PAUSE  : {pause_srv_name}")
        self.node.get_logger().info(f"[SAFETY] Using RESUME : {resume_srv_name}")

        self.motion = MotionAPI(self)

    # ------------------------------------------------------------------
    # perception에 pose 요청 (임시 노드 사용)
    # ------------------------------------------------------------------
    def request_object_pose(self, object_name: str) -> GetObjectPose.Response | None:
        """
        PerceptionNode의 /get_object_pose 서비스 동기 호출.
        콜백 안에서 self를 spin하지 않기 위해 임시 노드를 사용한다.
        """
        tmp_node = rclpy.create_node("pose_client_tmp")
        client = tmp_node.create_client(GetObjectPose, "get_object_pose")

        self.node.get_logger().info(
            f"Waiting for /get_object_pose service (object='{object_name}')..."
        )
        if not client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error("❌ /get_object_pose 서비스가 준비되지 않았습니다. (timeout)")
            tmp_node.destroy_node()
            return None

        req = GetObjectPose.Request()
        req.object_name = object_name
        req.use_tracking = False

        future = client.call_async(req)
        rclpy.spin_until_future_complete(tmp_node, future)

        if future.result() is None:
            self.node.get_logger().error("❌ get_object_pose 호출 실패 (future 결과 없음)")
            tmp_node.destroy_node()
            return None

        resp = future.result()
        tmp_node.destroy_node()
        return resp

    # ------------------------------------------------------------------
    # posx → 4x4 변환행렬 (base → gripper)
    # ------------------------------------------------------------------
    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        # Doosan ZYZ Euler (deg)
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    # ------------------------------------------------------------------
    # camera_link 좌표 → base 좌표
    # ------------------------------------------------------------------
    def transform_camera_to_base(self, cam_pose: PoseStamped) -> np.ndarray:
        """
        cam_pose: camera_link 기준 PoseStamped
        return: base 좌표계 (x, y, z)
        """
        from DSR_ROBOT2 import get_current_posx  # lazy import

        cx = cam_pose.pose.position.x
        cy = cam_pose.pose.position.y
        cz = cam_pose.pose.position.z

        coord_cam = np.array([cx, cy, cz, 1.0])

        # 현재 TCP 포즈 (base → gripper)
        tcp_pose = get_current_posx()[0]  # [x, y, z, rx, ry, rz]
        base2gripper = self.get_robot_pose_matrix(*tcp_pose)

        # base2cam = base2gripper @ gripper2cam
        base2cam = base2gripper @ self.gripper2cam

        coord_base = base2cam @ coord_cam
        return coord_base[:3]

    # ------------------------------------------------------------------
    # final_pose 생성 헬퍼
    # ------------------------------------------------------------------
    def make_final_pose(self, x: float, y: float, z: float) -> PoseStamped:
        final_pose = PoseStamped()
        final_pose.header.frame_id = "base"
        final_pose.header.stamp = self.node.get_clock().now().to_msg()
        final_pose.pose.position.x = float(x)
        final_pose.pose.position.y = float(y)
        final_pose.pose.position.z = float(z)
        final_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return final_pose

    # ------------------------------------------------------------------
    # DOOSAN SAFETY SERVICE CALLS
    # ------------------------------------------------------------------
    def _call_move_stop(self):
        if not self.stop_client.wait_for_service(timeout_sec=0.5):
            self.node.get_logger().error("[SAFETY] move_stop unavailable")
            return
        req = MoveStop.Request()
        req.stop_mode = 2 # quick: 1, slow: 2
        self.stop_client.call_async(req)
        self.node.get_logger().error("[SAFETY] move_stop EXECUTED")

    def _call_move_pause(self):
        if not self.pause_client.wait_for_service(timeout_sec=0.5):
            self.node.get_logger().error("[SAFETY] move_pause unavailable")
            return
        req = MovePause.Request()
        self.pause_client.call_async(req)
        self.node.get_logger().warn("[SAFETY] move_pause EXECUTED")

    def _call_move_resume(self):
        if not self.resume_client.wait_for_service(timeout_sec=0.5):
            self.node.get_logger().error("[SAFETY] move_resume unavailable")
            return
        req = MoveResume.Request()
        self.resume_client.call_async(req)
        self.node.get_logger().info("[SAFETY] move_resume EXECUTED")

    # ------------------------------------------------------------------
    # stop / cancel 관련 유틸
    # ------------------------------------------------------------------
    def request_cancel(self):
        with self._cancel_lock:
            if self._cancel_flag:
                return
            self._cancel_flag = True

        self.node.get_logger().warn("[STOP] soft stop requested (move_pause + cancel_flag)")
        try:
            self._call_move_pause()
        except Exception as e:
            self.node.get_logger().error(f"[STOP] move_pause 호출 중 예외: {e}")

    def clear_cancel(self):
        with self._cancel_lock:
            self._cancel_flag = False

    def is_cancelled(self) -> bool:
        with self._cancel_lock:
            return self._cancel_flag

class MotionAPI:
    """
    Doosan 모션/그리퍼 API를 래핑해서
    - 호출 전/후에 cancel 여부를 체크하고
    - 취소 시 MotionCancelled 예외를 던진다.

    새로운 스킬은 반드시 ctx.motion.* 만 사용하도록 해서
    cancel 로직이 한 곳에만 모이게 한다.
    """

    def __init__(self, ctx: MotionContext):
        self._ctx = ctx

    # 내부 공통 체크 함수
    def _check_cancel(self):
        if self._ctx.is_cancelled():
            self._ctx.node.get_logger().warn("[MotionAPI] cancel flag detected, raising MotionCancelled")
            raise MotionCancelled("Motion cancelled by user request")

    # ---- 래핑 함수들 ----

    def movej(self, *args, **kwargs):
        from DSR_ROBOT2 import movej
        self._check_cancel()
        res = movej(*args, **kwargs)
        self._check_cancel()
        return res

    def movel(self, *args, **kwargs):
        from DSR_ROBOT2 import movel
        self._check_cancel()
        res = movel(*args, **kwargs)
        self._check_cancel()
        return res

    def wait(self, *args, **kwargs):
        from DSR_ROBOT2 import wait
        self._check_cancel()
        res = wait(*args, **kwargs)
        self._check_cancel()
        return res

    def open_gripper(self):
        self._check_cancel()
        self._ctx.gripper.open_gripper()
        self._check_cancel()

    def close_gripper(self):
        self._check_cancel()
        self._ctx.gripper.close_gripper()
        self._check_cancel()
