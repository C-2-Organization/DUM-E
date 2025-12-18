# dum_e_motion/skill_manager_node.py
#!/usr/bin/env python3
import os
import numpy as np
from std_msgs.msg import String
import traceback

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
from dum_e_motion.skills import pick, find, home, drop, place, tracking, handover, swip, dump

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
            "run_skill",
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

    def _run_handover_with_person_fallback(self, cmd: SkillCommand):
        """
        1차 HANDOVER 시도
        실패하면: FIND(person) 실행 후, 다시 HANDOVER 시도
        """
        # 1차 시도
        success, message, confidence, final_pose = handover.run_handover_skill(
            cmd, self.ctx
        )

        if success:
            return success, message, confidence, final_pose

        # 2) 실패 → FIND(person)로 사용자 위치 잡기
        self.get_logger().warn(
            f"[HANDOVER] 1차 시도 실패(message='{message}', conf={confidence:.2f}), "
            f"FIND(person)으로 사람 위치를 잡은 뒤 재시도합니다."
        )

        find_cmd = SkillCommand()
        find_cmd.skill_type = SkillCommand.FIND
        find_cmd.object_name = "person"
        find_cmd.target_pose = PoseStamped()  # FIND는 pose 안 씀
        find_cmd.params_json = (
            '{"max_search_time": 30.0, "scan_interval": 1.0, "search_region": "outside"}'
        )

        find_success, find_msg, find_conf, _ = find.run_find_skill(
            find_cmd, self.ctx
        )

        if not find_success:
            # FIND도 실패 → 최종 실패
            msg = (
                f"HANDOVER failed and FIND(person) also failed. "
                f"handover_msg='{message}', find_msg='{find_msg}'"
            )
            self.get_logger().warn(f"[HANDOVER] {msg}")
            return False, msg, max(confidence, find_conf), PoseStamped()

        # 3) FIND(person) 성공 → HANDOVER 재시도
        self.get_logger().info(
            f"[HANDOVER] FIND(person) 성공(conf={find_conf:.2f}), HANDOVER 재시도"
        )

        success2, message2, confidence2, final_pose2 = handover.run_handover_skill(
            cmd, self.ctx
        )
        return success2, message2, confidence2, final_pose2 if success2 else PoseStamped()

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

            elif cmd.skill_type == SkillCommand.HOME:
                self.get_logger().info(
                    "🔔 RunSkill 요청: HOME, 기본 자세로 돌아갑니다."
                )

                success, message, confidence, final_pose = home.run_home_skill(self.ctx)

                response.success = success
                response.message = message
                response.confidence = confidence
                response.final_pose = final_pose
                return response

            elif cmd.skill_type == SkillCommand.DROP:
                self.get_logger().info(
                    "🔔 RunSkill 요청: DROP, 잡고 있는 오브젝트를 떨어뜨립니다."
                )

                success, message, confidence = drop.run_drop_skill(self.ctx)

                response.success = success
                response.message = message
                response.confidence = confidence
                response.final_pose = PoseStamped()
                return response

            elif cmd.skill_type == SkillCommand.PLACE:
                self.get_logger().info(
                    "🔔 RunSkill 요청: PLACE, 잡고있는 오브젝트를 지정한 위치에 놓습니다."
                )

                # 1차 시도: 바로 PLACE
                place_success, place_msg, place_conf, place_pose = place.run_place_skill(
                    cmd, self.ctx
                )

                if place_success:
                    response.success = True
                    response.message = place_msg
                    response.confidence = place_conf
                    response.final_pose = place_pose
                    return response

                # ------------------------
                # 여기부터는 "PLACE 실패" 후 리커버리 로직
                # ------------------------
                self.get_logger().warn(
                    f"[PLACE] 1차 시도 실패(message='{place_msg}', conf={place_conf:.2f}), "
                    f"FIND로 타겟을 재탐색 후 PLACE를 재시도합니다."
                )

                # 2) FIND 시도 (같은 object_name)
                find_cmd = SkillCommand()
                find_cmd.skill_type = SkillCommand.FIND
                find_cmd.object_name = cmd.object_name
                find_cmd.target_pose = PoseStamped()  # Find는 pose 안 씀
                find_cmd.params_json = '{"max_search_time": 30.0, "scan_interval": 1.0}'

                find_success, find_msg, find_conf, _ = find.run_find_skill(
                    find_cmd, self.ctx
                )

                if not find_success:
                    # FIND도 실패 → 최종 실패
                    msg = (
                        f"PLACE failed and FIND also failed. "
                        f"place_msg='{place_msg}', find_msg='{find_msg}'"
                    )
                    self.get_logger().warn(f"[PLACE] {msg}")
                    response.success = False
                    response.message = msg
                    response.confidence = max(place_conf, find_conf)
                    response.final_pose = PoseStamped()
                    return response

                # 3) FIND 성공했으니, 다시 한 번 PLACE 재시도
                self.get_logger().info(
                    f"[PLACE] FIND 성공(conf={find_conf:.2f}), PLACE 재시도"
                )

                place2_success, place2_msg, place2_conf, place2_pose = place.run_place_skill(
                    cmd, self.ctx
                )

                response.success = place2_success
                response.message = place2_msg
                response.confidence = place2_conf
                response.final_pose = place2_pose if place2_success else PoseStamped()
                return response

            elif cmd.skill_type == SkillCommand.TRACKING:
                self.get_logger().info(
                    f"🔔 RunSkill: TRACKING, {cmd.object_name}을 추적합니다."
                )

                # 1차 시도: 바로 TRACKING
                tracking_success, tracking_msg, tracking_conf, tracking_pose = (
                    tracking.run_tracking_skill(cmd, self.ctx)
                )

                if tracking_success:
                    response.success = True
                    response.message = tracking_msg
                    response.confidence = tracking_conf
                    response.final_pose = tracking_pose
                    return response

                # ------------------------
                # 여기부터는 "TRACKING 실패" 후 리커버리 로직
                # ------------------------
                self.get_logger().warn(
                    f"[TRACKING] 1차 시도 실패(message='{tracking_msg}', conf={tracking_conf:.2f}), "
                    f"FIND로 대상 재탐색 후 TRACKING을 재시도합니다."
                )

                # 2) FIND 시도 (같은 object_name)
                find_cmd = SkillCommand()
                find_cmd.skill_type = SkillCommand.FIND
                find_cmd.object_name = cmd.object_name
                find_cmd.target_pose = PoseStamped()
                find_cmd.params_json = '{"max_search_time": 30.0, "scan_interval": 1.0}'

                find_success, find_msg, find_conf, _ = find.run_find_skill(
                    find_cmd, self.ctx
                )

                if not find_success:
                    msg = (
                        f"TRACKING failed and FIND also failed. "
                        f"tracking_msg='{tracking_msg}', find_msg='{find_msg}'"
                    )
                    self.get_logger().warn(f"[TRACKING] {msg}")
                    response.success = False
                    response.message = msg
                    response.confidence = max(tracking_conf, find_conf)
                    response.final_pose = PoseStamped()
                    return response

                # 3) FIND 성공했으니, 다시 한 번 TRACKING 재시도
                self.get_logger().info(
                    f"[TRACKING] FIND 성공(conf={find_conf:.2f}), TRACKING 재시도"
                )

                tracking2_success, tracking2_msg, tracking2_conf, tracking2_pose = (
                    tracking.run_tracking_skill(cmd, self.ctx)
                )

                response.success = tracking2_success
                response.message = tracking2_msg
                response.confidence = tracking2_conf
                response.final_pose = tracking2_pose if tracking2_success else PoseStamped()
                return response

            elif cmd.skill_type == SkillCommand.HANDOVER:
                self.get_logger().info(
                    "🔔 RunSkill 요청: HANDOVER, 사람에게 물체를 건네줍니다."
                )

                # 0) 현재 그리퍼 상태 확인
                gripper_open = False
                try:
                    gripper_open = self.ctx.is_gripper_open()
                except Exception as e:
                    self.get_logger().warn(f"[HANDOVER] is_gripper_open() 체크 중 예외: {e}")

                # 🔹 CASE 1: 그리퍼가 열려 있음 → 아직 아무 것도 안 잡은 상태
                if gripper_open:
                    obj_name = (cmd.object_name or "").strip()
                    if not obj_name:
                        msg = (
                            "[HANDOVER] Gripper is open but no object_name specified. "
                            "Cannot pick anything for handover."
                        )
                        self.get_logger().warn(msg)
                        response.success = False
                        response.message = msg
                        response.confidence = 0.0
                        response.final_pose = PoseStamped()
                        return response

                    # 1-A) 먼저 PICK 시도
                    self.get_logger().info(
                        f"[HANDOVER] 그리퍼가 비어 있음 → 먼저 PICK('{obj_name}') 수행 후 HANDOVER 예정."
                    )

                    pick_cmd = SkillCommand()
                    pick_cmd.skill_type = SkillCommand.PICK
                    pick_cmd.object_name = obj_name
                    pick_cmd.target_pose = PoseStamped()
                    pick_cmd.params_json = "{}"

                    pick_success, pick_msg, pick_conf, pick_pose = pick.run_pick_skill(
                        pick_cmd, self.ctx
                    )

                    if not pick_success:
                        # 여기서도 PICK 쪽 리커버리(FIND 후 재-PICK)를 쓰고 싶다면
                        # 위의 PICK 분기와 동일한 패턴으로 확장할 수도 있음.
                        msg = (
                            f"[HANDOVER] Pre-PICK for handover failed: {pick_msg}"
                        )
                        self.get_logger().warn(msg)
                        response.success = False
                        response.message = msg
                        response.confidence = pick_conf
                        response.final_pose = PoseStamped()
                        return response

                    self.get_logger().info(
                        f"[HANDOVER] Pre-PICK 성공(conf={pick_conf:.2f}), 이제 HANDOVER 실행."
                    )

                    # 1-B) PICK 성공 후 HANDOVER (mediapipe 실패 시 FIND(person) fallback 포함)
                    success, message, confidence, final_pose = self._run_handover_with_person_fallback(
                        cmd
                    )

                    response.success = success
                    response.message = message
                    # 전체 신뢰도는 PICK과 HANDOVER 둘 중 작은 값을 써도 되고,
                    # 일단 HANDOVER 기준으로 사용
                    response.confidence = min(pick_conf, confidence)
                    response.final_pose = final_pose
                    return response

                # 🔹 CASE 2: 그리퍼가 닫혀 있음 → 이미 뭔가 집고 있다고 가정
                else:
                    self.get_logger().info(
                        "[HANDOVER] Gripper is closed → 이미 물체를 잡고 있다고 가정하고, 바로 HANDOVER 실행."
                    )

                    success, message, confidence, final_pose = self._run_handover_with_person_fallback(
                        cmd
                    )

                    response.success = success
                    response.message = message
                    response.confidence = confidence
                    response.final_pose = final_pose
                    return response


            elif cmd.skill_type == SkillCommand.SWIP:
                self.get_logger().info(
                    "🔔 RunSkill 요청: SWIP, 바닥을 닦습니다."
                )

                success, message, confidence = swip.run_swip_skill(self.ctx)

                response.success = success
                response.message = message
                response.confidence = confidence
                response.final_pose = PoseStamped()
                return response

            elif cmd.skill_type == SkillCommand.DUMP:
                self.get_logger().info(
                    "🔔 RunSkill 요청: DUMP, 쓰레기를 버립니다."
                )

                success, message, confidence = dump.run_dump_skill(self.ctx)

                response.success = success
                response.message = message
                response.confidence = confidence
                response.final_pose = PoseStamped()
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
            traceback.print_exc()
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
