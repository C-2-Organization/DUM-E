# dum_e_motion/skill_manager_node.py
#!/usr/bin/env python3
import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory

import DR_init

from dum_e_interfaces.srv import RunSkill
from dum_e_interfaces.msg import SkillCommand
from dum_e_utils.onrobot import RG
from dum_e_motion.motion_context import MotionContext
from dum_e_motion.skills import pick

ROBOT_ID = "dsr01"

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502


class SkillManagerNode(Node):
    """
    DUM-E의 motion 스킬들을 관리하는 메인 노드.

    - 서비스: /run_skill (RunSkill.srv)
    - 스킬 실행은 dum_e_motion.skills.* 모듈에 위임
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

        gripper2cam = np.load(calib_path)
        self.get_logger().info(f"Loaded T_gripper2camera from: {calib_path}")

        # ======== 그리퍼 초기화 ========
        gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

        # ======== MotionContext 생성 (스킬들이 사용할 컨텍스트) ========
        self.ctx = MotionContext(self, gripper, gripper2cam)

        # ======== run_skill 서비스 서버 ========
        self.skill_srv = self.create_service(
            RunSkill,
            "run_skill",
            self.handle_run_skill,
        )

        self.get_logger().info("✅ SkillManagerNode ready. Service: /run_skill")

    # ------------------------------------------------------------------
    # /run_skill 서비스 콜백
    # ------------------------------------------------------------------
    def handle_run_skill(self, request, response):
        cmd: SkillCommand = request.command

        # 기본값
        response.success = False
        response.message = ""
        response.confidence = 0.0
        response.final_pose = PoseStamped()

        if cmd.skill_type == SkillCommand.PICK:
            self.get_logger().info(
                f"🔔 RunSkill 요청: PICK, object_name='{cmd.object_name}'"
            )

            success, message, confidence, final_pose = pick.run_pick_skill(
                cmd, self.ctx
            )

            response.success = success
            response.message = message
            response.confidence = confidence
            response.final_pose = final_pose
            return response

        # 새로운 스킬 추가시 분기 추가
        # elif cmd.skill_type == SkillCommand.PLACE:

        else:
            msg = f"skill_type={cmd.skill_type} 은(는) 아직 구현되지 않았습니다."
            self.get_logger().warn(msg)
            response.success = False
            response.message = msg
            response.confidence = 0.0
            response.final_pose = PoseStamped()
            return response


def main(args=None):
    rclpy.init(args=args)

    # 1) Doosan 제어용 노드 생성
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
