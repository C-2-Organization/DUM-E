# dum_e_motion/skills/place.py
import json
from typing import Tuple

from geometry_msgs.msg import PoseStamped

from dum_e_interfaces.msg import SkillCommand
from dum_e_motion.motion_context import MotionContext


# ==============================================================================
# 설정 및 상수
# ==============================================================================
PLACE_CONF_TH = 0.3
GRIPPER_OFFSET = 250


# ------------------------------------------------------------------
# Doosan + RG2 place 모션
# ------------------------------------------------------------------
def execute_place_motion(ctx: MotionContext, x, y, z):
    """
    목표 좌표로 이동 → 그리퍼 오픈(드롭) → 홈
    """
    from DSR_ROBOT2 import (
        DR_MV_MOD_ABS,
        DR_MV_RA_DUPLICATE,
        get_current_posx,
    )
    from DR_common2 import posx

    ctx.node.get_logger().info(
        f"[MOVE] Place → base({x:.3f}, {y:.3f}, {z:.3f})"
    )

    current_pos = get_current_posx()[0]

    target_pos = posx([
        x,
        y,
        z,
        current_pos[3],
        current_pos[4],
        current_pos[5],
    ])

    # 1) 목표 지점으로 이동
    ctx.motion.movel(
        target_pos,
        vel=ctx.LIN_VEL,
        acc=ctx.LIN_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )

    # 2) 놓기 (그리퍼 오픈)
    ctx.motion.open_gripper()
    ctx.motion.wait(0.5)

    # 3) 살짝 위로 올리기 (충돌 방지)
    lift_pos = list(target_pos)
    lift_pos[2] += 80  # 80mm 상승
    lift_pos = posx(lift_pos)
    ctx.motion.movel(
        lift_pos,
        vel=ctx.LIN_VEL,
        acc=ctx.LIN_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )

    # 4) 홈으로 복귀
    ctx.motion.movej(
        ctx.CUSTOM_HOME_JOINT,
        vel=ctx.JNT_VEL,
        acc=ctx.JNT_ACC,
        mod=DR_MV_MOD_ABS,
        ra=DR_MV_RA_DUPLICATE,
    )


def run_place_skill(
    cmd: SkillCommand,
    ctx: MotionContext,
) -> Tuple[bool, str, float, PoseStamped]:
    """
    PLACE 스킬 실행:
      - object_name으로 목표 물체 인식
      - 해당 물체 위에 놓기
    """
    location = cmd.object_name.strip()

    if not location:
        msg = "location is empty"
        ctx.node.get_logger().warn(f"[PLACE] {msg}")
        return False, msg, 0.0, PoseStamped()

    ctx.node.get_logger().info(f"[PLACE] Start Skill: '{location}'")

    # ------------------------------------------------------------------
    # CASE A: 외부 Target Pose 사용 (직접 좌표 지정)
    # ------------------------------------------------------------------
    if cmd.target_pose.header.frame_id:
        ctx.node.get_logger().info("[PLACE] Using external target pose")

        cam_pose = cmd.target_pose
        confidence = 1.0

    # ------------------------------------------------------------------
    # CASE B: Perception으로 목표 위치 탐지
    # ------------------------------------------------------------------
    else:
        pose_resp = ctx.request_object_pose(location)
        if not pose_resp or not pose_resp.success:
            return False, f"Perception failed for '{location}'", 0.0, PoseStamped()

        confidence = float(pose_resp.confidence)
        if confidence < PLACE_CONF_TH:
            return False, f"Low confidence {confidence:.2f}", confidence, PoseStamped()

        cam_pose = pose_resp.pose
        ctx.node.get_logger().info(f"[PLACE] Detected '{location}' with confidence {confidence:.2f}")

    # ------------------------------------------------------------------
    # 공통: 좌표 변환 및 실행
    # ------------------------------------------------------------------

    # 1. 물체 위에 놓을 위치 계산 (높이 오프셋 적용)
    place_cam_pose = PoseStamped()
    place_cam_pose.header = cam_pose.header
    place_cam_pose.pose.position.x = cam_pose.pose.position.x
    place_cam_pose.pose.position.y = cam_pose.pose.position.y
    place_cam_pose.pose.position.z = cam_pose.pose.position.z - GRIPPER_OFFSET
    place_cam_pose.pose.orientation = cam_pose.pose.orientation

    # 2. Camera -> Base 좌표 변환
    base_xyz = ctx.transform_camera_to_base(place_cam_pose)
    bx, by, bz = base_xyz

    # 3. 안전 높이 확보 (너무 낮으면 위험)
    MIN_SAFE_Z = 50  # mm
    if bz < MIN_SAFE_Z:
        ctx.node.get_logger().warn(f"[PLACE] Z={bz:.1f} too low, adjusting to {MIN_SAFE_Z}")
        bz = MIN_SAFE_Z

    # 4. 모션 실행
    try:
        execute_place_motion(ctx, bx, by, bz)
        return True, "Success", confidence, ctx.make_final_pose(bx, by, bz)
    except Exception as e:
        ctx.node.get_logger().error(f"[PLACE] Motion Error: {e}")
        return False, f"Motion Error: {e}", confidence, PoseStamped()
