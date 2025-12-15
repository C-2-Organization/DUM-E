# dum_e_motion/skill_manager_node.py
#!/usr/bin/env python3
import os
import numpy as np
from std_msgs.msg import String

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory

import DR_init

from dum_e_interfaces.srv import RunSkill
from dum_e_interfaces.msg import SkillCommand
from dum_e_utils.onrobot import RG
from dum_e_motion.motion_context import MotionContext, MotionCancelled
from dum_e_motion.skills import pick, find, look_at

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

        # ============ 콜백 그룹 설정 ============
        self.service_group = MutuallyExclusiveCallbackGroup()
        self.control_group = ReentrantCallbackGroup()

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
            "/run_skill",
            self.handle_run_skill,
            callback_group=self.service_group,
        )

        self.control_sub = self.create_subscription(
            String,
            "/dum_e_control",
            self.control_callback,
            10,
            callback_group=self.control_group,
        )

        self.get_logger().info("✅ SkillManagerNode ready. Service: /run_skill")

    def control_callback(self, msg: String):
        cmd = msg.data.strip().lower()
        self.get_logger().info(f"[CONTROL] Received cmd='{cmd}'")
        if cmd == "stop":
            self.get_logger().error("[CONTROL] STOP received -> hold motion")
            self.ctx.request_cancel()

    # ------------------------------------------------------------------
    # /run_skill 서비스 콜백
    # ------------------------------------------------------------------
    def handle_run_skill(self, request, response):
        cmd: SkillCommand = request.command

        self.ctx.clear_cancel()

        # 기본값
        response.success = False
        response.message = ""
        response.confidence = 0.0
        response.final_pose = PoseStamped()

        try:
            if cmd.skill_type == SkillCommand.PICK:
                self.get_logger().info(
                    f"🔔 RunSkill 요청: PICK, object_name='{cmd.object_name}'"
                )

                # 1차 시도: 바로 PICK
                pick_success, pick_msg, pick_conf, pick_pose = pick.run_pick_skill(
                    cmd, self.ctx
                )

                # 성공하면 그대로 반환
                if pick_success:
                    response.success = True
                    response.message = pick_msg
                    response.confidence = pick_conf
                    response.final_pose = pick_pose
                    return response

                # ------------------------
                # 여기부터는 "픽 실패" 후 리커버리 로직
                # ------------------------
                # 예: 메시지나 confidence 기준으로 "디텍션 실패"만 골라서 처리해도 됨
                self.get_logger().warn(
                    f"[PICK] 1차 시도 실패(message='{pick_msg}', conf={pick_conf:.2f}), "
                    f"FIND로 자세를 조정 후 재시도합니다."
                )

                # 2) FIND 시도 (같은 object_name)
                find_cmd = SkillCommand()
                find_cmd.skill_type = SkillCommand.FIND
                find_cmd.object_name = cmd.object_name
                find_cmd.target_pose = PoseStamped()  # Find는 pose 안 씀
                # 필요하면 params_json으로 검색 시간 지정 가능
                find_cmd.params_json = '{"max_search_time": 30.0, "scan_interval": 1.0}'

                find_success, find_msg, find_conf, _ = find.run_find_skill(
                    find_cmd, self.ctx
                )

                if not find_success:
                    # FIND도 실패 → 최종 실패
                    msg = (
                        f"PICK failed and FIND also failed. "
                        f"pick_msg='{pick_msg}', find_msg='{find_msg}'"
                    )
                    self.get_logger().warn(f"[PICK] {msg}")
                    response.success = False
                    response.message = msg
                    response.confidence = max(pick_conf, find_conf)
                    response.final_pose = PoseStamped()
                    return response

                # 3) FIND 성공했으니, 다시 한 번 PICK 재시도
                self.get_logger().info(
                    f"[PICK] FIND 성공(conf={find_conf:.2f}), PICK 재시도"
                )

                pick2_success, pick2_msg, pick2_conf, pick2_pose = pick.run_pick_skill(
                    cmd, self.ctx
                )

                response.success = pick2_success
                response.message = pick2_msg
                response.confidence = pick2_conf
                response.final_pose = pick2_pose if pick2_success else PoseStamped()
                return response

            elif cmd.skill_type == SkillCommand.FIND:
                self.get_logger().info(
                    f"🔔 RunSkill 요청: FIND, object_name='{cmd.object_name}'"
                )

                success, message, confidence, final_pose = find.run_find_skill(
                    cmd, self.ctx
                )

                response.success = success
                response.message = message
                response.confidence = confidence
                response.final_pose = final_pose
                return response
            
            elif cmd.skill_type == SkillCommand.LOOK_AT:
                self.get_logger().info("🔔 RunSkill 요청: LOOK_AT")
                success, message, confidence, final_pose = look_at.run_look_at_skill(cmd, self.ctx)
                response.success = success
                response.message = message
                response.confidence = confidence
                response.final_pose = final_pose
                return response


            else:
                msg = f"skill_type={cmd.skill_type} 은(는) 아직 구현되지 않았습니다."
                self.get_logger().warn(msg)
                response.success = False
                response.message = msg
                response.confidence = 0.0
                response.final_pose = PoseStamped()
                return response

        except MotionCancelled:
            # 유저가 멈춰! 라고 해서 중간에 끊긴 케이스
            self.get_logger().warn("[SkillManager] Motion cancelled by user request")
            self.ctx.clear_cancel()
            response.success = False
            response.message = "Motion cancelled by user request"
            response.confidence = 0.0
            response.final_pose = PoseStamped()
            return response

        except Exception as e:
            self.get_logger().error(f"[SkillManager] Unexpected error: {e}")
            response.success = False
            response.message = f"Unexpected error: {e}"
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
    executor = MultiThreadedExecutor(num_threads=4)
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
