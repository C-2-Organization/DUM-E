#!/usr/bin/env python3
import os
import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from geometry_msgs.msg import PoseStamped, Quaternion
from scipy.spatial.transform import Rotation
from ament_index_python.packages import get_package_share_directory

import DR_init

from dum_e_interfaces.srv import GetObjectPose, RunSkill
from dum_e_interfaces.msg import SkillCommand
from dum_e_utils.onrobot import RG

ROBOT_ID = "dsr01"

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502

PICK_CONF_TH = 0.5


class SkillManagerNode(Node):
    """
    DUM-E의 motion 스킬들을 관리하는 메인 노드.

    - 서비스: /run_skill (RunSkill.srv)
    - 현재 구현된 skill_type:
        SkillCommand.PICK (0)
    - 동작:
        1) PICK:
            - 필요 시 perception의 /get_object_pose 호출
            - camera_link 기준 pose → base 좌표 변환
            - Doosan + RG2로 pick 모션 수행
            - 최종 base pose를 final_pose로 응답
    """

    def __init__(self):
        super().__init__("skill_manager_node")

        # ---------------------------
        # Load T_gripper2camera.npy
        # ---------------------------
        share_dir = get_package_share_directory("dum_e_motion")
        calib_path = os.path.join(share_dir, "config", "T_gripper2camera.npy")

        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"T_gripper2camera not found: {calib_path}")

        self.gripper2cam = np.load(calib_path)

        self.get_logger().info(f"Loaded T_gripper2camera from: {calib_path}")

        # ======== 그리퍼 초기화 ========
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

        # 속도/가속도 파라미터 (필요하면 나중에 parameter로 뺄 수 있음)
        self.LIN_VEL = [150.0, 300.0]
        self.LIN_ACC = [150.0, 150.0]
        self.JNT_VEL = 150.0
        self.JNT_ACC = 300.0
        self.CUSTOM_HOME_JOINT = [0, 0, 90, 0, 90, 0]

        # ======== run_skill 서비스 서버 (/run_skill) ========
        self.skill_srv = self.create_service(
            RunSkill,
            "run_skill",
            self.handle_run_skill,
        )

        self.get_logger().info("✅ SkillManagerNode ready. Service: /run_skill")

    # ------------------------------------------------------------------
    # perception에 pose 요청 (공통 유틸)
    # ------------------------------------------------------------------
    def request_object_pose(self, object_name: str) -> GetObjectPose.Response | None:
        """
        PerceptionNode의 /get_object_pose 서비스 동기 호출.
        주의: 콜백 안에서 self를 spin하면 안 되기 때문에,
              별도의 임시 노드를 만들어 그 노드로만 spin_until_future_complete를 돌린다.
        """
        # 1) 임시 노드 생성
        tmp_node = rclpy.create_node("pose_client_tmp")
        client = tmp_node.create_client(GetObjectPose, "get_object_pose")

        # 2) 서비스 준비 대기
        self.get_logger().info("[PICK] Waiting for /get_object_pose service (tmp client)...")
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("❌ /get_object_pose 서비스가 준비되지 않았습니다. (timeout)")
            tmp_node.destroy_node()
            return None

        # 3) 요청 만들기
        req = GetObjectPose.Request()
        req.object_name = object_name
        req.use_tracking = False

        # 4) 비동기 호출 + 임시 노드로만 spin_until_future_complete
        future = client.call_async(req)
        rclpy.spin_until_future_complete(tmp_node, future)

        # 5) 결과 처리
        if future.result() is None:
            self.get_logger().error("❌ get_object_pose 호출 실패 (future 결과 없음)")
            tmp_node.destroy_node()
            return None

        resp = future.result()
        tmp_node.destroy_node()
        return resp

    # ------------------------------------------------------------------
    # posx → 4x4 변환행렬 (base → gripper)
    # ------------------------------------------------------------------
    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        # Doosan의 ZYZ Euler (deg) 기준
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    # ------------------------------------------------------------------
    # camera_link 좌표 → base 좌표 (공통 유틸)
    # ------------------------------------------------------------------
    def transform_camera_to_base(self, cam_pose: PoseStamped) -> np.ndarray:
        from DSR_ROBOT2 import get_current_posx
        """
        cam_pose: camera_link 기준 PoseStamped
        return: base 좌표계 (x, y, z)
        """
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
        return coord_base[:3]  # (x, y, z)

    # ------------------------------------------------------------------
    # 실제 Pick 동작 (스킬 본체)
    # ------------------------------------------------------------------
    def do_pick(
        self, object_name: str, target_pose: PoseStamped | None, params_json: str
    ) -> tuple[bool, str, float, PoseStamped]:
        """
        PICK 스킬:
          - target_pose가 유효하면 그걸 사용
          - 아니면 perception에서 pose 가져옴
          - base 좌표로 변환 후 로봇 모션 수행

        return: (success, message, confidence, final_pose)
        """
        confidence = 0.0

        # 1) target_pose가 이미 주어졌는지 확인 (frame_id가 비어있지 않으면 사용)
        if target_pose is not None and target_pose.header.frame_id != "":
            self.get_logger().info(
                f"[PICK] 외부에서 제공된 target_pose 사용 (frame_id={target_pose.header.frame_id})"
            )
            cam_pose = target_pose
            confidence = 1.0  # 외부가 신뢰한 값으로 가정
        else:
            # perception에 pose 요청
            self.get_logger().info(
                f"[PICK] perception에 pose 요청: object_name='{object_name}'"
            )
            pose_resp = self.request_object_pose(object_name)
            if pose_resp is None:
                msg = "get_object_pose call failed"
                self.get_logger().error(msg)
                dummy_pose = PoseStamped()  # 빈 pose
                return False, msg, 0.0, dummy_pose

            confidence = float(pose_resp.confidence)

            if not pose_resp.success:
                msg = f"get_object_pose 실패: {pose_resp.message}"
                self.get_logger().warn(msg)
                dummy_pose = PoseStamped()
                return False, msg, confidence, dummy_pose

            if confidence < PICK_CONF_TH:
                msg = (
                    f"conf={confidence:.2f} < "
                    f"PICK_CONF_TH={PICK_CONF_TH:.2f}, pick skip"
                )
                self.get_logger().warn(msg)
                dummy_pose = PoseStamped()
                return False, msg, confidence, dummy_pose

            cam_pose = pose_resp.pose

        # params_json은 지금은 안 쓰지만, 나중에 tilt angle, approach offset 같은 옵션에 사용 가능
        if params_json:
            try:
                params = json.loads(params_json)
                self.get_logger().info(f"[PICK] params_json = {params}")
            except json.JSONDecodeError:
                self.get_logger().warn(f"[PICK] params_json 파싱 실패: {params_json}")

        # 2) camera_link → base 좌표 변환
        base_xyz = self.transform_camera_to_base(cam_pose)
        bx, by, bz = base_xyz

        self.get_logger().info(
            f"[PICK DEBUG] target='{object_name}', "
            f"cam=({cam_pose.pose.position.x:.3f},"
            f"{cam_pose.pose.position.y:.3f},"
            f"{cam_pose.pose.position.z:.3f}), "
            f"base=({bx:.3f},{by:.3f},{bz:.3f}), "
            f"conf={confidence:.2f}"
        )

        # 3) 실제 로봇 동작
        try:
            self._pick_motion(bx, by, bz)
            success = True
            message = "OK"
        except Exception as e:
            success = False
            message = f"pick motion error: {e}"
            self.get_logger().error(f"❌ pick motion 중 예외: {e}")

        # 4) final_pose (base 기준 PoseStamped) 구성
        final_pose = PoseStamped()
        final_pose.header.frame_id = "base"
        final_pose.header.stamp = self.get_clock().now().to_msg()
        final_pose.pose.position.x = float(bx)
        final_pose.pose.position.y = float(by)
        final_pose.pose.position.z = float(bz)
        # orientation은 일단 identity. 필요하면 TCP orientation 써도 됨.
        final_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        return success, message, confidence, final_pose

    # ------------------------------------------------------------------
    # Doosan + RG2 로봇 모션 (자세한 path 정의)
    # ------------------------------------------------------------------
    def _pick_motion(self, x, y, z):
        """실제 로봇 모션 정의 (접근 → 집기 → 홈으로 → 놓기)"""
        from DSR_ROBOT2 import (
            movej,
            movel,
            wait,
            DR_MV_MOD_ABS,
            DR_MV_RA_DUPLICATE,
            get_current_posx,
        )
        from DR_common2 import posx
        self.get_logger().info(
            f"[MOVE] Pick → base({x:.3f}, {y:.3f}, {z:.3f})"
        )

        current_pos = get_current_posx()[0]

        approach_pos = posx([
            x,
            y,
            z + 205.0,  # 위에서 접근
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

        # 놓기
        self.gripper.open_gripper()
        wait(1)

    # ------------------------------------------------------------------
    # /run_skill 서비스 콜백
    # ------------------------------------------------------------------
    def handle_run_skill(self, request, response):
        cmd: SkillCommand = request.command

        # 기본값 준비
        response.success = False
        response.message = ""
        response.confidence = 0.0
        response.final_pose = PoseStamped()

        # 어떤 스킬인지 분기
        if cmd.skill_type == SkillCommand.PICK:
            self.get_logger().info(
                f"🔔 RunSkill 요청: PICK, object_name='{cmd.object_name}'"
            )

            # target_pose는 frame_id가 비어있으면 무시
            target_pose = cmd.target_pose if cmd.target_pose.header.frame_id else None

            success, message, confidence, final_pose = self.do_pick(
                cmd.object_name,
                target_pose,
                cmd.params_json,
            )

            response.success = success
            response.message = message
            response.confidence = confidence
            response.final_pose = final_pose
            return response

        else:
            # 아직 구현되지 않은 스킬 타입
            msg = f"skill_type={cmd.skill_type} 은(는) 아직 구현되지 않았습니다."
            self.get_logger().warn(msg)
            response.success = False
            response.message = msg
            response.confidence = 0.0
            response.final_pose = PoseStamped()
            return response


def main(args=None):
    rclpy.init(args=args)

    # 1) Doosan 제어용 노드 생성 (기존 DR_init 패턴 그대로)
    dsr_node = rclpy.create_node("dsr_example_demo_py", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    # 2) SkillManagerNode 생성
    skill_node = SkillManagerNode()

    # 3) Executor에 두 노드 등록 후 spin
    executor = SingleThreadedExecutor()
    executor.add_node(dsr_node)
    executor.add_node(skill_node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        dsr_node.destroy_node()
        skill_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
