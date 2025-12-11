# dum_e_motion/motion_context.py
import json
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from scipy.spatial.transform import Rotation

from dum_e_interfaces.srv import GetObjectPose


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
            f"[PICK] Waiting for /get_object_pose service (object='{object_name}')..."
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
    # Doosan + RG2 pick 모션
    # ------------------------------------------------------------------
    def execute_pick_motion(self, x, y, z):
        """
        접근 → 잡기 → 홈
        """
        from DSR_ROBOT2 import (
            movej,
            movel,
            wait,
            DR_MV_MOD_ABS,
            DR_MV_RA_DUPLICATE,
            get_current_posx,
        )
        from DR_common2 import posx

        self.node.get_logger().info(
            f"[MOVE] Pick → base({x:.3f}, {y:.3f}, {z:.3f})"
        )

        current_pos = get_current_posx()[0]

        approach_pos = posx([
            x,
            y,
            z,
            current_pos[3],
            current_pos[4],
            current_pos[5],
        ])

        # 접근
        movel(
            approach_pos,
            vel=self.LIN_VEL,
            acc=self.LIN_ACC,
            mod=DR_MV_MOD_ABS,
            ra=DR_MV_RA_DUPLICATE,
        )

        # 집기
        self.gripper.close_gripper()
        wait(1)

        # 홈으로
        movej(
            self.CUSTOM_HOME_JOINT,
            vel=self.JNT_VEL,
            acc=self.JNT_ACC,
            mod=DR_MV_MOD_ABS,
            ra=DR_MV_RA_DUPLICATE,
        )

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
