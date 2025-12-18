# dum_e_motion/skills/handover.py

# ------------------------------------------------------------------
# Handover service call example
# ------------------------------------------------------------------
# ros2 service call /run_skill dum_e_interfaces/srv/RunSkill "{
#   command: {
#     skill_type: 6,
#     object_name: 'hammer',
#   }
# }"

import json
from typing import Tuple

from geometry_msgs.msg import PoseStamped

from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext

def _has_valid_external_pose(ps: PoseStamped) -> bool:
    """
    외부에서 실제 의미 있는 target_pose를 넣어준 경우에만 True.
    - frame_id가 비어있지 않고
    - position이 완전히 (0,0,0)인 기본값이 아닐 때만 인정.
    """
    if ps is None:
        return False

    if not ps.header.frame_id:
        return False

    p = ps.pose.position
    if abs(p.x) < 1e-6 and abs(p.y) < 1e-6 and abs(p.z) < 1e-6:
        # 기본 생성된 PoseStamped()라고 판단
        return False

    return True

# 기본 파라미터 (단위: mm / sec)
HANDOVER_CONF_TH = 0
DEFAULT_WAIT_SEC = 1
GRIPPER_OFFSET = 230

def execute_handover_motion(
    ctx: MotionContext,
    x: float,
    y: float,
    z: float,
    *,
    wait_sec: float,
) -> None:
    """
    Handover 모션 (단일 접근 버전):
    - approach(천천히 1번) → wait → open
    좌표 단위는 Doosan posx와 맞춰 mm 기준.
    """
    from DSR_ROBOT2 import (
        DR_MV_MOD_ABS,
        DR_MV_RA_DUPLICATE,
        get_current_posx,
    )
    from DR_common2 import posx

    ctx.node.get_logger().info(
        f"[HANDOVER] MOVE target base=({x:.3f}, {y:.3f}, {z:.3f})"
    )

    current_pos = get_current_posx()[0]
    rx, ry, rz = current_pos[3], current_pos[4], current_pos[5]

    target = posx([
        x,
        y,
        z,
        rx, ry, rz,
    ])

    # 1) target로 한 번만 이동 (사람 앞이니까 느리게)
    ctx.motion.movel(
        target,
        vel=[50.0, 80.0],
        acc=[50.0, 50.0],
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )

    # 2) wait
    ctx.motion.wait(wait_sec)

    # 3) release
    ctx.motion.open_gripper()
    ctx.motion.wait(0.5)


def run_handover_skill(
    cmd: SkillCommand,
    ctx: MotionContext,
) -> Tuple[bool, str, float, PoseStamped]:
    """
    HANDOVER 스킬 실행:
      - perception에 "handover" pose 요청 (또는 cmd.target_pose 사용)
      - camera_link → base 좌표 변환 (ctx.transform_camera_to_base)
      - 단일 접근/대기/오픈 수행
      - (success, message, confidence, final_pose) 반환
    """

    params_json = cmd.params_json

    # 1) target_pose가 이미 주어졌는지 확인
    if _has_valid_external_pose(cmd.target_pose):
        cam_pose = cmd.target_pose
        confidence = 1.0
        ctx.node.get_logger().info(
            f"[HANDOVER] 외부 제공 target_pose 사용 (frame_id={cam_pose.header.frame_id})"
        )
    else:
        # perception에 pose 요청 (object_name은 handover로 고정)
        pose_resp = ctx.request_object_pose("handover")
        if pose_resp is None:
            msg = "get_object_pose call failed"
            ctx.node.get_logger().error(f"[HANDOVER] {msg}")
            return False, msg, 0.0, PoseStamped()

        confidence = float(getattr(pose_resp, "confidence", 0.0))

        if not pose_resp.success:
            msg = f"get_object_pose 실패: {pose_resp.message}"
            ctx.node.get_logger().warn(f"[HANDOVER] {msg}")
            return False, msg, confidence, PoseStamped()

        if confidence < HANDOVER_CONF_TH:
            msg = f"conf={confidence:.2f} < HANDOVER_CONF_TH={HANDOVER_CONF_TH:.2f}, handover skip"
            ctx.node.get_logger().warn(f"[HANDOVER] {msg}")
            return False, msg, confidence, PoseStamped()

        cam_pose = pose_resp.pose

    # 2) params_json 파싱 (단일 파라미터만 유지)
    wait_sec = DEFAULT_WAIT_SEC

    if params_json:
        try:
            params = json.loads(params_json)
            ctx.node.get_logger().info(f"[HANDOVER] params_json = {params}")
            wait_sec = float(params.get("wait_sec", wait_sec))
        except json.JSONDecodeError:
            ctx.node.get_logger().warn(f"[HANDOVER] params_json 파싱 실패: {params_json}")

    # 3) camera_link → base 변환
    try:
        # 카메라 기준 손 위치(cam_pose)를 TCP 기준으로 보정
        tcp_cam_pose = PoseStamped()
        tcp_cam_pose.header = cam_pose.header
        tcp_cam_pose.pose.position.x = cam_pose.pose.position.x
        tcp_cam_pose.pose.position.y = cam_pose.pose.position.y
        tcp_cam_pose.pose.position.z = cam_pose.pose.position.z - GRIPPER_OFFSET
        tcp_cam_pose.pose.orientation = cam_pose.pose.orientation

        ctx.node.get_logger().info(
            f"[HANDOVER] GRIPPER_OFFSET 적용: "
            f"cam_z={cam_pose.pose.position.z:.1f} -> "
            f"tcp_cam_z={tcp_cam_pose.pose.position.z:.1f}"
        )

        base_xyz = ctx.transform_camera_to_base(tcp_cam_pose)
        bx, by, bz = base_xyz
    except Exception as e:
        msg = f"transform_camera_to_base failed: {e}"
        ctx.node.get_logger().error(f"[HANDOVER] {msg}")
        return False, msg, confidence, PoseStamped()

    ctx.node.get_logger().info(
        f"[HANDOVER DEBUG] cam=({cam_pose.pose.position.x:.3f},"
        f"{cam_pose.pose.position.y:.3f},"
        f"{cam_pose.pose.position.z:.3f}) -> base=({bx:.3f},{by:.3f},{bz:.3f}), conf={confidence:.2f}"
    )

    # 4) 모션 수행
    try:
        execute_handover_motion(
            ctx,
            bx, by, bz,
            wait_sec=wait_sec,
        )
        success = True
        message = "OK"
    except Exception as e:
        success = False
        message = f"handover motion error: {e}"
        ctx.node.get_logger().error(f"❌ [HANDOVER] motion 중 예외: {e}")

    # 5) final_pose 구성 (handover 타겟 기준)
    final_pose = ctx.make_final_pose(bx, by, bz)
    return success, message, confidence, final_pose
